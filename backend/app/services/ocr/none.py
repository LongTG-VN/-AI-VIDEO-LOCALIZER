from pathlib import Path

from app.models.project import SubtitleCue
from app.services.ocr.base import OCREngine


class NullOCREngine(OCREngine):
    def extract_subtitles(self, video_path: Path) -> list[SubtitleCue]:
        return []
