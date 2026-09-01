from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from app.services.tts.base import TTSProvider

logger = logging.getLogger(__name__)


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge Neural TTS Provider for high-quality Vietnamese speech."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> Path:
        import edge_tts

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_mp3 = output_path.with_suffix(".mp3")

        for attempt in range(4):
            try:
                comm = edge_tts.Communicate(
                    text=text,
                    voice=voice_id,
                    rate=rate,
                    pitch=pitch,
                    volume=volume,
                )
                await comm.save(str(temp_mp3))
                break
            except Exception as e:
                if attempt < 3:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"EdgeTTS failed for text '{text[:30]}': {e}") from e

        # Convert to WAV with standardized 44.1kHz sample rate for mixing
        if output_path.suffix.lower() == ".wav":
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-i", str(temp_mp3),
                "-ar", "44100",
                "-ac", "2",
                str(output_path),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg conversion to WAV failed: {res.stderr.decode('utf-8', 'replace')}")
            if temp_mp3.exists() and temp_mp3 != output_path:
                temp_mp3.unlink()
            return output_path
        else:
            return temp_mp3
