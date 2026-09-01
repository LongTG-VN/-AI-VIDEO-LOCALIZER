from __future__ import annotations

from app.services.tts.base import TTSProvider
from app.services.tts.edge import EdgeTTSProvider
from app.services.tts.mock import MockTTSProvider


def get_tts_provider(engine_name: str = "edge") -> TTSProvider:
    """Factory returning the configured TTSProvider instance."""
    engine = engine_name.lower().strip()
    if engine == "mock":
        return MockTTSProvider()
    elif engine in ["edge", "edge-tts", "edgetts"]:
        return EdgeTTSProvider()
    else:
        return EdgeTTSProvider()
