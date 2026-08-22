from typing import Any

from app.services.asr.base import ASREngine
from app.services.asr.funasr import FunASREngine
from app.services.asr.mock import MockASREngine


def create_asr_engine(name: str, **kwargs: Any) -> ASREngine:
    normalized = name.strip().lower()
    if normalized == "mock":
        return MockASREngine()
    if normalized == "funasr":
        return FunASREngine(**kwargs)
    raise ValueError(f"Unsupported ASR engine: {name}")
