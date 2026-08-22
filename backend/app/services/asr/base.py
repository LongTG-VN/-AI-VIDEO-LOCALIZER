from abc import ABC, abstractmethod
from pathlib import Path

from app.models.project import SubtitleCue


class ASREngine(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path, language: str = "zh") -> list[SubtitleCue]:
        raise NotImplementedError
