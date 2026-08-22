from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.models.project import SubtitleCue
from app.services.ocr.base import OCREngine


@dataclass
class FrameText:
    timestamp: float
    text: str
    confidence: float


def _normalize(text: str) -> str:
    return "".join(text.split()).strip()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


class PaddleSubtitleOCREngine(OCREngine):
    """Hard-subtitle OCR for the bottom portion of a video.

    Frames are sampled with FFmpeg, cropped before OCR to reduce logos/background text,
    then temporally deduplicated into subtitle cues. This is intentionally simple and
    deterministic; scene-aware ROI detection can replace it later behind the same interface.
    """

    def __init__(self, ffmpeg_bin: str = "ffmpeg", fps: float = 2.0, crop_top_ratio: float = 0.62) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = max(0.25, fps)
        self.crop_top_ratio = min(max(crop_top_ratio, 0.0), 0.9)
        self._ocr: Any | None = None

    def _load_ocr(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OCR_ENGINE=paddle but PaddleOCR is not installed. Install backend/requirements-ocr.txt.") from exc
        self._ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
        return self._ocr

    def _extract_frames(self, video_path: Path, frame_dir: Path) -> None:
        crop_y = self.crop_top_ratio
        crop_h = 1.0 - crop_y
        vf = f"fps={self.fps},crop=iw:ih*{crop_h:.5f}:0:ih*{crop_y:.5f}"
        cmd = [self.ffmpeg_bin, "-v", "error", "-y", "-i", str(video_path), "-vf", vf, str(frame_dir / "%08d.jpg")]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is not installed or not on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.strip() or "OCR frame extraction failed") from exc

    @staticmethod
    def _read_prediction(result: Any) -> tuple[str, float]:
        data = result.json if hasattr(result, "json") else result
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
        texts = list(data.get("rec_texts", [])) if isinstance(data, dict) else []
        scores = list(data.get("rec_scores", [])) if isinstance(data, dict) else []
        pairs = [(str(text).strip(), float(scores[index]) if index < len(scores) else 0.0) for index, text in enumerate(texts) if str(text).strip()]
        if not pairs:
            return "", 0.0
        return "\n".join(text for text, _ in pairs), sum(score for _, score in pairs) / len(pairs)

    def _group(self, frames: list[FrameText]) -> list[SubtitleCue]:
        if not frames:
            return []
        frame_duration = 1.0 / self.fps
        groups: list[list[FrameText]] = []
        for frame in frames:
            if not frame.text:
                continue
            if groups and _similar(groups[-1][-1].text, frame.text) >= 0.82:
                groups[-1].append(frame)
            else:
                groups.append([frame])
        cues: list[SubtitleCue] = []
        for group in groups:
            representative = max(group, key=lambda item: item.confidence)
            cues.append(SubtitleCue(start=group[0].timestamp, end=group[-1].timestamp + frame_duration, source_text=representative.text, ocr_confidence=representative.confidence, confidence=representative.confidence))
        return cues

    def extract_subtitles(self, video_path: Path) -> list[SubtitleCue]:
        if not video_path.exists():
            raise RuntimeError(f"Video not found: {video_path}")
        ocr = self._load_ocr()
        root = Path(tempfile.mkdtemp(prefix="ai-video-localizer-ocr-"))
        try:
            self._extract_frames(video_path, root)
            frames: list[FrameText] = []
            for index, image_path in enumerate(sorted(root.glob("*.jpg")), start=0):
                text = ""
                confidence = 0.0
                predictions = ocr.predict(str(image_path))
                for result in predictions:
                    candidate, score = self._read_prediction(result)
                    if score > confidence:
                        text, confidence = candidate, score
                if text:
                    frames.append(FrameText(timestamp=index / self.fps, text=text, confidence=confidence))
            return self._group(frames)
        finally:
            shutil.rmtree(root, ignore_errors=True)
