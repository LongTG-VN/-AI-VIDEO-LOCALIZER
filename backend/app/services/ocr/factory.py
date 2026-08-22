from app.services.ocr.base import OCREngine
from app.services.ocr.none import NullOCREngine
from app.services.ocr.paddle import PaddleSubtitleOCREngine


def create_ocr_engine(
    name: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    fps: float = 2.0,
    crop_top_ratio: float = 0.62,
) -> OCREngine:
    normalized = name.strip().lower()
    if normalized in {"", "none", "off"}:
        return NullOCREngine()
    if normalized == "paddle":
        return PaddleSubtitleOCREngine(
            ffmpeg_bin=ffmpeg_bin,
            fps=fps,
            crop_top_ratio=crop_top_ratio,
        )
    raise ValueError(f"Unsupported OCR engine: {name}")
