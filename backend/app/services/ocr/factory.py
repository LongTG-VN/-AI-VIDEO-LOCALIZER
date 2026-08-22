from app.services.ocr.base import OCREngine
from app.services.ocr.none import NullOCREngine
from app.services.ocr.paddle import PaddleSubtitleOCREngine


def create_ocr_engine(
    name: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    fps: float = 2.0,
    crop_top_ratio: float = 0.65,
    crop_bottom_ratio: float = 0.95,
    crop_left_ratio: float = 0.06,
    crop_right_ratio: float = 0.94,
    change_threshold: float = 16.0,
) -> OCREngine:
    normalized = name.strip().lower()
    if normalized in {"", "none", "off"}:
        return NullOCREngine()
    if normalized == "paddle":
        return PaddleSubtitleOCREngine(
            ffmpeg_bin=ffmpeg_bin,
            fps=fps,
            crop_top_ratio=crop_top_ratio,
            crop_bottom_ratio=crop_bottom_ratio,
            crop_left_ratio=crop_left_ratio,
            crop_right_ratio=crop_right_ratio,
            change_threshold=change_threshold,
        )
    raise ValueError(f"Unsupported OCR engine: {name}")
