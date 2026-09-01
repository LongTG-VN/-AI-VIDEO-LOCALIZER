from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf

from app.models.project import DubbingOptions

logger = logging.getLogger(__name__)


@dataclass
class CueAudioSegment:
    start: float
    end: float
    audio_path: Path
    volume_linear: float = 1.0


class AudioMixer:
    """Handles audio speed stretching, crossfading, ducking, and final master mixing."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", sample_rate: int = 44100):
        self.ffmpeg_bin = ffmpeg_bin
        self.sample_rate = sample_rate

    def get_audio_duration(self, audio_path: Path) -> float:
        """Measures exact duration in seconds of an audio file."""
        info = sf.info(str(audio_path))
        return float(info.duration)

    def time_stretch_audio(
        self,
        input_path: Path,
        output_path: Path,
        speed: float,
    ) -> tuple[Path, float]:
        """Stretches audio to target speed factor using FFmpeg atempo filter (pitch-invariant)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        speed = max(0.5, min(2.0, speed))

        if 0.98 <= speed <= 1.02:
            # Within 2%, copy directly
            import shutil
            shutil.copy2(str(input_path), str(output_path))
            return output_path, self.get_audio_duration(output_path)

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(input_path),
            "-filter:a", f"atempo={speed:.4f}",
            "-ar", str(self.sample_rate),
            "-ac", "2",
            str(output_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg atempo stretch failed: {res.stderr.decode('utf-8', 'replace')}")

        new_dur = self.get_audio_duration(output_path)
        return output_path, new_dur

    def mix_dubbing_tracks(
        self,
        background_path: Path,
        vocal_path: Path | None,
        cue_segments: list[CueAudioSegment],
        output_mix_path: Path,
        options: DubbingOptions,
        target_duration: float | None = None,
    ) -> tuple[Path, float]:
        """Mixes background, optional attenuated original vocals, and Vietnamese TTS cues."""
        output_mix_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load Background Stem
        bg_data, sr = sf.read(str(background_path), dtype="float32")
        if sr != self.sample_rate:
            # Handle resampling if needed
            import scipy.signal
            num_samples = int(len(bg_data) * self.sample_rate / sr)
            bg_data = scipy.signal.resample(bg_data, num_samples)
        if bg_data.ndim == 1:
            bg_data = np.column_stack([bg_data, bg_data])

        total_samples = len(bg_data)
        if target_duration is not None:
            target_samples = int(round(target_duration * self.sample_rate))
            if target_samples > total_samples:
                pad = np.zeros((target_samples - total_samples, 2), dtype="float32")
                bg_data = np.vstack([bg_data, pad])
                total_samples = target_samples

        # 2. Build Smooth Ducking Envelope for Background Music
        # Default: 1.0 (0dB). During dialogue: duck to ducking_music_db (e.g. -3.5dB = ~0.67)
        bg_gain = np.ones(total_samples, dtype="float32")
        music_duck_factor = float(10.0 ** (options.ducking_music_db / 20.0))
        ramp_samples = int(round(0.040 * self.sample_rate))  # 40ms smooth ramp

        for seg in cue_segments:
            s_idx = max(0, int(round(seg.start * self.sample_rate)))
            e_idx = min(total_samples, int(round(seg.end * self.sample_rate)))
            if e_idx <= s_idx:
                continue

            # Apply smooth ramp down and up
            r_start = max(0, s_idx - ramp_samples)
            r_end = min(total_samples, e_idx + ramp_samples)

            # Core dialogue segment
            bg_gain[s_idx:e_idx] = np.minimum(bg_gain[s_idx:e_idx], music_duck_factor)

            # Attack ramp
            if s_idx > r_start:
                attack = np.linspace(1.0, music_duck_factor, s_idx - r_start, dtype="float32")
                bg_gain[r_start:s_idx] = np.minimum(bg_gain[r_start:s_idx], attack)

            # Release ramp
            if r_end > e_idx:
                release = np.linspace(music_duck_factor, 1.0, r_end - e_idx, dtype="float32")
                bg_gain[e_idx:r_end] = np.minimum(bg_gain[e_idx:r_end], release)

        bg_mixed = bg_data * bg_gain[:, np.newaxis]

        # 3. Optional Original Chinese Vocals Stem Handling
        vocal_mixed = np.zeros_like(bg_mixed)
        if vocal_path is not None and vocal_path.exists() and options.separation_engine == "demucs":
            voc_data, vsr = sf.read(str(vocal_path), dtype="float32")
            if vsr != self.sample_rate:
                import scipy.signal
                num_s = int(len(voc_data) * self.sample_rate / vsr)
                voc_data = scipy.signal.resample(voc_data, num_s)
            if voc_data.ndim == 1:
                voc_data = np.column_stack([voc_data, voc_data])

            voc_len = min(len(voc_data), total_samples)
            # Attenuate Chinese dialogue by ducking_dialogue_db (e.g. -24dB = 0.063)
            voc_gain = np.ones(voc_len, dtype="float32") * float(10.0 ** (options.ducking_dialogue_db / 20.0))

            # Suppress completely during Vietnamese dialogue
            for seg in cue_segments:
                s_idx = max(0, int(round(seg.start * self.sample_rate)))
                e_idx = min(voc_len, int(round(seg.end * self.sample_rate)))
                if e_idx > s_idx:
                    voc_gain[s_idx:e_idx] = float(10.0 ** ((options.ducking_dialogue_db - 12.0) / 20.0))

            vocal_mixed[:voc_len] = voc_data[:voc_len] * voc_gain[:, np.newaxis]

        # 4. Mix Vietnamese TTS Cues
        tts_track = np.zeros_like(bg_mixed)
        fade_samples = max(8, int(round((options.crossfade_ms / 1000.0) * self.sample_rate)))

        for seg in cue_segments:
            if not seg.audio_path.exists():
                continue
            c_data, csr = sf.read(str(seg.audio_path), dtype="float32")
            if csr != self.sample_rate:
                import scipy.signal
                num_s = int(len(c_data) * self.sample_rate / csr)
                c_data = scipy.signal.resample(c_data, num_s)
            if c_data.ndim == 1:
                c_data = np.column_stack([c_data, c_data])

            # Apply smooth 15ms fade-in and fade-out to prevent clicks
            c_len = len(c_data)
            if c_len > 2 * fade_samples:
                fade_in = np.linspace(0.0, 1.0, fade_samples, dtype="float32")
                fade_out = np.linspace(1.0, 0.0, fade_samples, dtype="float32")
                c_data[:fade_samples] *= fade_in[:, np.newaxis]
                c_data[-fade_samples:] *= fade_out[:, np.newaxis]

            # Place into TTS track at exact start timestamp
            s_idx = max(0, int(round(seg.start * self.sample_rate)))
            e_idx = min(total_samples, s_idx + c_len)
            insert_len = e_idx - s_idx
            if insert_len > 0:
                tts_track[s_idx:e_idx] += c_data[:insert_len] * seg.volume_linear

        # 5. Master Summation & Peak Normalization
        master = bg_mixed + vocal_mixed + tts_track

        # Peak normalization to prevent clipping (target <= -1.0 dBFS = ~0.891)
        max_peak = float(np.max(np.abs(master)))
        target_peak = 0.891  # -1 dBFS
        if max_peak > target_peak:
            master = master * (target_peak / max_peak)
            final_peak = float(np.max(np.abs(master)))
        else:
            final_peak = max_peak

        final_peak_db = 20.0 * math.log10(max(1e-6, final_peak))

        # Save to 16-bit PCM WAV
        sf.write(str(output_mix_path), master, self.sample_rate, subtype="PCM_16")
        logger.info("Mixed master audio saved to %s (peak=%.2fdB, dur=%.2fs)", output_mix_path, final_peak_db, len(master)/self.sample_rate)

        return output_mix_path, final_peak_db
