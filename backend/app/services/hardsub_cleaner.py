from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.project import SubtitleCue

logger = logging.getLogger(__name__)


class HardSubCleaner:
    """Removes or conceals burned-in hard subtitles from video using ROI text inpainting."""

    def __init__(
        self,
        crop_top_ratio: float = 0.65,
        crop_bottom_ratio: float = 0.95,
        crop_left_ratio: float = 0.06,
        crop_right_ratio: float = 0.94,
        mask_dilate_radius: int = 3,
        luminance_threshold: int = 200,
        inpaint_radius: int = 3,
    ) -> None:
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.crop_left_ratio = crop_left_ratio
        self.crop_right_ratio = crop_right_ratio
        self.mask_dilate_radius = mask_dilate_radius
        self.luminance_threshold = luminance_threshold
        self.inpaint_radius = inpaint_radius

    def build_text_mask(self, frame: np.ndarray) -> np.ndarray | None:
        """Extracts a dilated binary mask covering only Chinese subtitle characters."""
        h, w = frame.shape[:2]
        y1 = int(h * self.crop_top_ratio)
        y2 = int(h * self.crop_bottom_ratio)
        x1 = int(w * self.crop_left_ratio)
        x2 = int(w * self.crop_right_ratio)

        sub_roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(sub_roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.luminance_threshold, 255, cv2.THRESH_BINARY)

        num_white = int(np.sum(mask > 0))
        if num_white < 250:
            return None

        kernel_size = 2 * self.mask_dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask, kernel, iterations=2)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = dilated
        return full_mask

    def clean_frame(self, frame: np.ndarray, mode: str = "inpaint") -> tuple[np.ndarray, bool]:
        """Cleans a single video frame if hard subtitles are detected."""
        if mode in {"none", "off"}:
            return frame, False

        mask = self.build_text_mask(frame)
        if mask is None:
            return frame, False

        if mode == "cover":
            # Soft dark translucent band fallback
            h, w = frame.shape[:2]
            y1 = int(h * self.crop_top_ratio)
            y2 = int(h * self.crop_bottom_ratio)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, y1), (w, y2), (0, 0, 0), -1)
            cleaned = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
            return cleaned, True

        # Default: inpaint mode
        inpainted = cv2.inpaint(frame, mask, inpaintRadius=self.inpaint_radius, flags=cv2.INPAINT_TELEA)
        return inpainted, True

    def clean_video(
        self,
        source_path: Path,
        output_path: Path,
        cues: list[SubtitleCue] | None = None,
        mode: str = "inpaint",
    ) -> dict[str, Any]:
        """Processes entire video, removing hard Chinese subtitles while preserving non-subtitle regions."""
        if not source_path.exists():
            raise RuntimeError(f"Source video not found: {source_path}")

        if mode in {"none", "off"}:
            return {
                "mode": mode,
                "frames_processed": 0,
                "frames_inpainted": 0,
                "clean_runtime": 0.0,
                "output_path": str(source_path),
            }

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {source_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frames_processed = 0
        frames_inpainted = 0
        t0 = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames_processed += 1
            cleaned_frame, was_cleaned = self.clean_frame(frame, mode=mode)
            if was_cleaned:
                frames_inpainted += 1
            writer.write(cleaned_frame)

        cap.release()
        writer.release()
        total_time = time.time() - t0

        metrics = {
            "mode": mode,
            "frames_processed": frames_processed,
            "frames_inpainted": frames_inpainted,
            "inpaint_rate": round(frames_inpainted / max(1, frames_processed) * 100, 1),
            "clean_runtime": round(total_time, 2),
            "fps_speed": round(frames_processed / max(0.01, total_time), 1),
            "output_path": str(output_path),
        }
        logger.info("HardSub Cleaner completed: %s", metrics)
        return metrics
