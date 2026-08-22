from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.project import SubtitleCue
from app.services.ocr.base import OCREngine

logger = logging.getLogger(__name__)

# Single Chinese characters that represent valid dialogue and must not be dropped
VALID_SINGLE_CHAR_CHINESE = {
    "不", "好", "妈", "爸", "哥", "姐", "弟", "妹", "走", "看", "听", "去", "等",
    "快", "对", "是", "谁", "这", "那", "有", "要", "行", "能", "会", "请", "喂", "嗯"
}

# Known noise text patterns produced by watermarks or timestamps
KNOWN_NOISE_PATTERNS = {
    "10.5o", "10:50", "MILK", "MILK MILK", "LAA", "Y", "T", "CN", "10:CN",
    "西", "西 ", "IN-CN", "CN-IN", "10:59", "755135"
}


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
    """High-performance Hard-subtitle OCR Engine with frame change detection.

    Optimizations:
    1. Subtitle ROI crop: isolates the bottom subtitle band, removing background noise.
    2. High-contrast text binarization: extracts text masks independent of video background.
    3. Visual change detection: skips PaddleOCR for identical consecutive subtitle frames,
       extending existing cue duration.
    4. Empty frame skip: bypasses OCR during silence or pauses with no subtitle text.
    5. Multiline merge: joins vertically stacked subtitle lines into a single coherent cue.
    6. Noise filter: removes stray non-Chinese artifacts while preserving short valid words.
    7. Detailed metrics: tracks runtime, skip rates, and merges.
    """

    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        fps: float = 2.0,
        crop_top_ratio: float = 0.65,
        crop_bottom_ratio: float = 0.95,
        crop_left_ratio: float = 0.06,
        crop_right_ratio: float = 0.94,
        change_threshold: float = 16.0,
        min_text_pixels: int = 350,
    ) -> None:
        self.ffmpeg_bin = ffmpeg_bin
        self.fps = max(0.25, fps)
        self.crop_top_ratio = min(max(crop_top_ratio, 0.0), 0.95)
        self.crop_bottom_ratio = min(max(crop_bottom_ratio, self.crop_top_ratio + 0.05), 1.0)
        self.crop_left_ratio = min(max(crop_left_ratio, 0.0), 0.5)
        self.crop_right_ratio = min(max(crop_right_ratio, self.crop_left_ratio + 0.1), 1.0)
        self.change_threshold = change_threshold
        self.min_text_pixels = min_text_pixels
        self._ocr: Any | None = None
        self.last_metrics: dict[str, Any] = {}

    def _load_ocr(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OCR_ENGINE=paddle but PaddleOCR is not installed. Install backend/requirements-ocr.txt.") from exc
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        return self._ocr

    def _filter_and_merge_predictions(
        self, predictions: Any
    ) -> tuple[str, float, int, int]:
        """Extracts, filters noise, sorts, and merges detected text lines in a frame."""
        text_lines: list[tuple[str, float, float]] = []
        noise_count = 0
        multiline_count = 0

        for res in predictions:
            data = res.json if hasattr(res, "json") else res
            if isinstance(data, dict) and "res" in data:
                data = data["res"]
            texts = list(data.get("rec_texts", [])) if isinstance(data, dict) else []
            scores = list(data.get("rec_scores", [])) if isinstance(data, dict) else []
            polys = list(data.get("dt_polys", [])) if isinstance(data, dict) else []

            for t, s, p in zip(texts, scores, polys):
                t_str = str(t).strip()
                if not t_str:
                    continue
                # 1. Filter known noise text / timestamps
                if t_str in KNOWN_NOISE_PATTERNS or any(p in t_str for p in ["10:5", "CN", "MILK"]):
                    noise_count += 1
                    continue
                # 2. Filter single character non-Chinese / punctuation
                if len(t_str) == 1:
                    if not ('\u4e00' <= t_str <= '\u9fff'):
                        noise_count += 1
                        continue
                    if t_str not in VALID_SINGLE_CHAR_CHINESE and float(s) < 0.85:
                        noise_count += 1
                        continue
                # 3. Filter low confidence short gibberish
                if len(t_str) <= 2 and float(s) < 0.60:
                    noise_count += 1
                    continue

                y_pos = float(p[0][1]) if hasattr(p, '__getitem__') and len(p) > 0 else 0.0
                text_lines.append((t_str, float(s), y_pos))

        if not text_lines:
            return "", 0.0, noise_count, multiline_count

        # Sort top-to-bottom for multiline merging
        text_lines.sort(key=lambda item: item[2])
        if len(text_lines) > 1:
            multiline_count = 1

        merged_text = "".join(t for t, _, _ in text_lines)
        avg_score = sum(s for _, s, _ in text_lines) / len(text_lines)
        return merged_text, avg_score, noise_count, multiline_count

    def extract_subtitles(self, video_path: Path) -> list[SubtitleCue]:
        if not video_path.exists():
            raise RuntimeError(f"Video not found: {video_path}")

        ocr = self._load_ocr()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(video_fps / self.fps)))
        frame_duration = 1.0 / self.fps

        frames_sampled = 0
        frames_skipped = 0
        ocr_calls = 0
        multiline_merges = 0
        noise_removed = 0

        prev_thumb: np.ndarray | None = None
        prev_bbox: tuple[int, int, int, int] | None = None
        current_cue: SubtitleCue | None = None
        raw_cues: list[SubtitleCue] = []

        t_start = time.time()
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frames_sampled += 1
                timestamp = frame_idx / video_fps
                h, w = frame.shape[:2]

                # 1. Crop Subtitle ROI
                y1 = int(h * self.crop_top_ratio)
                y2 = int(h * self.crop_bottom_ratio)
                x1 = int(w * self.crop_left_ratio)
                x2 = int(w * self.crop_right_ratio)
                sub_roi = frame[y1:y2, x1:x2]

                # 2. Text Binarization Mask
                gray = cv2.cvtColor(sub_roi, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                num_white = int(np.sum(clean > 0))

                # Check A: Empty text area (no subtitle on screen)
                if num_white < self.min_text_pixels:
                    if current_cue is not None:
                        raw_cues.append(current_cue)
                        current_cue = None
                    prev_thumb = None
                    prev_bbox = None
                    frames_skipped += 1
                    frame_idx += 1
                    continue

                pts = cv2.findNonZero(clean)
                if pts is not None:
                    bx, by, bw, bh = cv2.boundingRect(pts)
                    if bw > 15 and bh > 8:
                        crop = clean[by : by + bh, bx : bx + bw]
                        thumb = cv2.resize(crop, (64, 16), interpolation=cv2.INTER_AREA)
                    else:
                        thumb = None
                        bx, by, bw, bh = (0, 0, 0, 0)
                else:
                    thumb = None
                    bx, by, bw, bh = (0, 0, 0, 0)

                # Check B: Visual change detection against active subtitle
                if current_cue is not None and prev_thumb is not None and thumb is not None and prev_bbox is not None:
                    px, py, pw, ph = prev_bbox
                    dw = abs(bw - pw) / max(1, max(bw, pw))
                    diff = float(np.mean(np.abs(thumb.astype(float) - prev_thumb.astype(float))))

                    # If text length is stable and thumbnail diff is below threshold, extend active cue!
                    if dw < 0.35 and diff < self.change_threshold:
                        current_cue.end = round(timestamp + frame_duration, 2)
                        frames_skipped += 1
                        prev_thumb = thumb
                        prev_bbox = (bx, by, bw, bh)
                        frame_idx += 1
                        continue

                # Check C: Run PaddleOCR on new / changed subtitle frame
                ocr_calls += 1
                predictions = ocr.predict(sub_roi)
                text, score, noise_cnt, ml_cnt = self._filter_and_merge_predictions(predictions)
                noise_removed += noise_cnt
                multiline_merges += ml_cnt

                if text:
                    if current_cue is not None and _similar(current_cue.source_text, text) >= 0.85:
                        current_cue.end = round(timestamp + frame_duration, 2)
                        current_cue.ocr_confidence = max(current_cue.ocr_confidence or 0.0, score)
                        current_cue.confidence = current_cue.ocr_confidence
                    else:
                        if current_cue is not None:
                            raw_cues.append(current_cue)
                        current_cue = SubtitleCue(
                            start=round(timestamp, 2),
                            end=round(timestamp + frame_duration, 2),
                            source_text=text,
                            ocr_confidence=round(score, 4),
                            confidence=round(score, 4),
                        )
                    prev_thumb = thumb
                    prev_bbox = (bx, by, bw, bh)
                else:
                    if current_cue is not None:
                        raw_cues.append(current_cue)
                        current_cue = None
                    prev_thumb = None
                    prev_bbox = None

            frame_idx += 1

        if current_cue is not None:
            raw_cues.append(current_cue)

        cap.release()
        total_time = time.time() - t_start

        # 3. Post-process: merge adjacent cues with identical text and small gap (<= 0.45s)
        normalized_cues: list[SubtitleCue] = []
        for cue in raw_cues:
            if not normalized_cues:
                normalized_cues.append(cue)
                continue
            prev = normalized_cues[-1]
            if _similar(prev.source_text, cue.source_text) >= 0.85 and (cue.start - prev.end) <= 0.45:
                prev.end = max(prev.end, cue.end)
                prev.ocr_confidence = max(prev.ocr_confidence or 0.0, cue.ocr_confidence or 0.0)
                prev.confidence = prev.ocr_confidence
            else:
                normalized_cues.append(cue)

        self.last_metrics = {
            "frames_sampled": frames_sampled,
            "frames_skipped": frames_skipped,
            "ocr_calls": ocr_calls,
            "ocr_runtime": round(total_time, 2),
            "average_ocr_call_time": round(total_time / max(1, ocr_calls), 3),
            "skip_rate": round(frames_skipped / max(1, frames_sampled) * 100, 1),
            "raw_ocr_cues": len(raw_cues),
            "normalized_ocr_cues": len(normalized_cues),
            "noise_cues_removed": noise_removed,
            "multiline_merges": multiline_merges,
        }
        logger.info("OCR Extraction Completed: %s", self.last_metrics)
        return normalized_cues
