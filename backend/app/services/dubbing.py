from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path

from app.models.project import CharacterVoiceProfile, DubbingMetrics, DubbingOptions, Project
from app.services.audio_mixer import AudioMixer, CueAudioSegment
from app.services.audio_separator import AudioSeparator
from app.services.subtitles import normalize_render_cues
from app.services.tts.factory import get_tts_provider
from app.services.tts.normalizer import normalize_for_speech
from app.services.tts.voice_mapper import create_voice_profiles_for_project
from app.services.utterance_engine import UtteranceEngine

logger = logging.getLogger(__name__)


class DubbingService:
    """Orchestrates Text-to-Speech synthesis, speech normalization, timing fitting,

    stem separation, audio ducking, and video muxing for Vietnamese localization.
    """

    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin
        self.separator = AudioSeparator(ffmpeg_bin=ffmpeg_bin)
        self.mixer = AudioMixer(ffmpeg_bin=ffmpeg_bin)

    async def dub_project(
        self,
        project: Project,
        source_video_path: Path,
        output_video_path: Path,
        options: DubbingOptions | None = None,
        work_dir: Path | None = None,
    ) -> tuple[Path, DubbingMetrics]:
        """End-to-end dubbing pipeline generating FINAL_VI_DUBBED_V1.mp4."""
        if options is None:
            options = DubbingOptions()

        if work_dir is None:
            work_dir = output_video_path.parent / f".dub_tmp_{project.id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        tts_cache_dir = work_dir / "tts_cues"
        tts_cache_dir.mkdir(parents=True, exist_ok=True)

        metrics = DubbingMetrics()
        t_start = time.time()

        # 1. Resolve Voice Profiles for Characters
        if not options.voice_profiles:
            options.voice_profiles = create_voice_profiles_for_project(project.characters)

        # 2. Extract Spoken Render Subtitle Cues (V8 Normalized Utterance Units)
        engine = UtteranceEngine(max_line_chars=36)
        render_cues, _ = engine.process_cues(project.cues, translated=True)
        if not render_cues:
            # Fallback to normalize_render_cues
            render_cues = normalize_render_cues(project.cues, max_line_chars=36)

        metrics.total_cues = len(render_cues)
        tts_provider = get_tts_provider(options.tts_engine)

        # 3. Audio Separation
        audio_orig = work_dir / "orig_audio.wav"
        self.separator.extract_audio(source_video_path, audio_orig)
        voc_stem, bg_stem, sep_mode, sep_dur = self.separator.separate_stems(
            audio_path=audio_orig,
            output_dir=work_dir / "stems",
            engine=options.separation_engine,
        )
        metrics.separation_mode = sep_mode
        metrics.separation_duration_s = sep_dur

        # 4. Synthesize & Time-Fit Per-Cue Audio
        cue_segments: list[CueAudioSegment] = []
        speed_factors: list[float] = []

        for idx, cue in enumerate(render_cues):
            clean_speech_text = normalize_for_speech(cue.render_text)
            if not clean_speech_text:
                continue

            # Determine Character Voice Profile
            profile: CharacterVoiceProfile | None = None
            if cue.speaker_character_id and cue.speaker_character_id in options.voice_profiles:
                profile = options.voice_profiles[cue.speaker_character_id]
            else:
                # Find matching character
                for char in project.characters:
                    if cue.speaker_id in char.speaker_ids or char.name_vi in cue.render_text:
                        profile = options.voice_profiles.get(char.id)
                        break
                if profile is None:
                    # Default voice
                    profile = CharacterVoiceProfile(
                        character_id="default",
                        voice_id="vi-VN-HoaiMyNeural",
                        pitch_offset="+10Hz",
                        base_rate="+0%",
                    )

            raw_cue_path = tts_cache_dir / f"cue_{idx:03d}_raw.wav"
            stretched_cue_path = tts_cache_dir / f"cue_{idx:03d}_fit.wav"

            try:
                # Synthesize
                await tts_provider.synthesize(
                    text=clean_speech_text,
                    voice_id=profile.voice_id,
                    output_path=raw_cue_path,
                    rate=profile.base_rate,
                    pitch=profile.pitch_offset,
                    volume=profile.volume,
                )
                metrics.synthesized_cues += 1

                # Duration Measurement
                raw_dur = self.mixer.get_audio_duration(raw_cue_path)
                slot_dur = max(0.3, cue.end - cue.start)
                speed_ratio = raw_dur / slot_dur

                # Timing-Fit Policy
                fitted_path = raw_cue_path
                actual_speed = 1.0

                if speed_ratio > 1.05:
                    # Speech is longer than subtitle slot: time-stretch up to max_acceptable_speed
                    target_speed = min(options.max_acceptable_speed, speed_ratio)
                    fitted_path, _ = self.mixer.time_stretch_audio(
                        input_path=raw_cue_path,
                        output_path=stretched_cue_path,
                        speed=target_speed,
                    )
                    actual_speed = target_speed
                    metrics.time_stretched_cues += 1
                    if speed_ratio > options.max_acceptable_speed + 0.10:
                        metrics.still_overlong_cues += 1
                elif speed_ratio < 0.90:
                    # Speech is significantly shorter: slight deceleration for naturalness
                    target_speed = max(options.min_acceptable_speed, speed_ratio)
                    fitted_path, _ = self.mixer.time_stretch_audio(
                        input_path=raw_cue_path,
                        output_path=stretched_cue_path,
                        speed=target_speed,
                    )
                    actual_speed = target_speed
                    metrics.time_stretched_cues += 1

                speed_factors.append(actual_speed)
                cue_segments.append(
                    CueAudioSegment(
                        start=cue.start,
                        end=cue.start + (raw_dur / actual_speed),
                        audio_path=fitted_path,
                        volume_linear=1.0,
                    )
                )
                metrics.succeeded_cues += 1

            except Exception as e:
                logger.error("Failed to dub cue %d ('%s'): %s", idx, clean_speech_text, e)
                metrics.failed_cues += 1

        if speed_factors:
            metrics.avg_speed_factor = float(sum(speed_factors) / len(speed_factors))
            metrics.max_speed_factor = float(max(speed_factors))

        # 5. Audio Mixing & Ducking
        t_mix_start = time.time()
        final_mix_wav = work_dir / "final_dub_mix.wav"
        _, peak_db = self.mixer.mix_dubbing_tracks(
            background_path=bg_stem,
            vocal_path=voc_stem,
            cue_segments=cue_segments,
            output_mix_path=final_mix_wav,
            options=options,
            target_duration=project.duration,
        )
        metrics.mixing_duration_s = time.time() - t_mix_start
        metrics.final_peak_db = peak_db
        metrics.final_duration_s = time.time() - t_start

        # 6. Final Video Mux (Stream Copy Video + AAC Audio)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        mux_cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(source_video_path),
            "-i", str(final_mix_wav),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_video_path),
        ]
        res = subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg muxing failed: {res.stderr.decode('utf-8', 'replace')}")

        logger.info("Dubbing complete: %s in %.2fs", output_video_path, metrics.final_duration_s)
        return output_video_path, metrics
