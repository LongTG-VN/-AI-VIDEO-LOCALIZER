from __future__ import annotations

import math
import wave
import struct
from pathlib import Path
from app.services.tts.base import TTSProvider


class MockTTSProvider(TTSProvider):
    """Deterministic Mock TTS Provider for fast offline testing and CI."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        output_path: Path,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Duration proportional to character count: ~15 chars per second
        dur = max(0.4, len(text.strip()) * 0.07)
        num_samples = int(self.sample_rate * dur)

        # Generate a gentle sine tone or low-volume audio
        freq = 300.0 if "NamMinh" in voice_id else 450.0
        wav_path = output_path.with_suffix(".wav")

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            frames = bytearray()
            for i in range(num_samples):
                val = int(8000 * math.sin(2 * math.pi * freq * i / self.sample_rate))
                frames.extend(struct.pack("<hh", val, val))
            wf.writeframes(frames)

        return wav_path
