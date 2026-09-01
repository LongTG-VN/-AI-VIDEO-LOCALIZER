from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class AudioSeparator:
    """Separates audio into dialogue (vocals) and background (music + SFX) stems using Demucs."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin
        self._separator = None

    def extract_audio(self, video_path: Path, output_audio_path: Path) -> Path:
        """Extracts 44.1kHz stereo WAV from video."""
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(output_audio_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {res.stderr.decode('utf-8', 'replace')}")
        return output_audio_path

    def separate_stems(
        self,
        audio_path: Path,
        output_dir: Path,
        engine: str = "demucs",
    ) -> tuple[Path | None, Path, str, float]:
        """Separates audio into vocals and background stems.

        Returns:
            (vocal_stem_path, background_stem_path, mode_used, duration_seconds)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        vocal_path = output_dir / "vocal_stem.wav"
        bg_path = output_dir / "background_stem.wav"

        t0 = time.time()
        if engine.lower() in ["demucs", "htdemucs"]:
            try:
                import demucs.api
                import soundfile as sf
                import numpy as np

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Initializing Demucs (htdemucs) on %s...", device)
                if self._separator is None:
                    self._separator = demucs.api.Separator(model="htdemucs", device=device)

                origin, separated = self._separator.separate_audio_file(audio_path)
                # separated contains: "drums", "bass", "other", "vocals"
                vocals = separated["vocals"]  # [channels, samples]
                background = separated["drums"] + separated["bass"] + separated["other"]

                # Save background and vocal stems
                demucs.api.save_audio(background, bg_path, samplerate=self._separator.samplerate)
                demucs.api.save_audio(vocals, vocal_path, samplerate=self._separator.samplerate)

                dur = time.time() - t0
                logger.info("Demucs audio separation completed in %.2fs: bg=%s, voc=%s", dur, bg_path, vocal_path)
                return vocal_path, bg_path, "demucs", dur
            except Exception as e:
                logger.warning("Demucs separation failed or unavailable: %s. Falling back to OVERDUB mode.", e)

        # Fallback mode: use original audio as background stem
        dur = time.time() - t0
        return None, audio_path, "overdub_fallback", dur
