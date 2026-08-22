from abc import ABC, abstractmethod
from pathlib import Path

from app.models.project import SubtitleCue


class OCREngine(ABC):
    @abstractmethod
    def extract_subtitles(self, video_path: Path) -> list[SubtitleCue]:
        raise NotImplementedError
