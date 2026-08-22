from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.project import SubtitleCue

logger = logging.getLogger(__name__)

NOISE_FILTER = {"10.5o", "10:50", "MILK", "MILK MILK", "IN-CN", "CN-IN", "755135", "CN"}


class HardSubCleaner:
    """Removes or conceals burned-in hard subtitles using cue-gated ROI inpainting."""

    def __init__(
        self,
        crop_top_ratio: float = 0.66,
        crop_bottom_ratio: float = 0.95,
        crop_left_ratio: float = 0.06,
        crop_right_ratio: float = 0.94,
        mask_dilate_radius: int = 2,
        luminance_threshold: int = 195,
        inpaint_radius: int = 3,
    ) -> None:
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.crop_left_ratio = crop_left_ratio
        self.crop_right_ratio = crop_right_ratio
        self.mask_dilate_radius = mask_dilate_radius
        self.luminance_threshold = luminance_threshold
        self.inpaint_radius = inpaint_radius

    def extract_text_mask(self, sub_roi: np.ndarray) -> np.ndarray | None:
        """Extracts a tight binary mask covering Chinese subtitle glyphs and outlines."""
        gray = cv2.cvtColor(sub_roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.luminance_threshold, 255, cv2.THRESH_BINARY)

        num_white = int(np.sum(binary > 0))
        if num_white < 220:
            return None

        # Clean noise specks
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # Dilate mask to enclose black stroke borders and drop shadows
        k_size = 2 * self.mask_dilate_radius + 1
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(clean, k_dilate, iterations=2)
        return dilated

    def clean_frame(
        self,
        frame: np.ndarray,
        mode: str = "inpaint",
        is_subtitle_active: bool = True,
    ) -> tuple[np.ndarray, bool]:
        """Cleans frame only when Chinese subtitle is actively on screen."""
        if mode in {"none", "off"} or not is_subtitle_active:
            return frame, False

        h, w = frame.shape[:2]
        y1 = int(h * self.crop_top_ratio)
        y2 = int(h * self.crop_bottom_ratio)
        x1 = int(w * self.crop_left_ratio)
        x2 = int(w * self.crop_right_ratio)

        sub_roi = frame[y1:y2, x1:x2]
        mask_roi = self.extract_text_mask(sub_roi)
        if mask_roi is None:
            return frame, False

        if mode == "cover":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, y1), (w, y2), (0, 0, 0), -1)
            cleaned = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
            return cleaned, True

        # Inpaint mode: build full-frame mask and inpaint
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_roi
        inpainted = cv2.inpaint(frame, full_mask, inpaintRadius=self.inpaint_radius, flags=cv2.INPAINT_TELEA)
        return inpainted, True

    def clean_video(
        self,
        source_path: Path,
        output_path: Path,
        cues: list[SubtitleCue] | None = None,
        mode: str = "inpaint",
    ) -> dict[str, Any]:
        """Decoupled video cleaner: inpainting is strictly gated by Chinese subtitle intervals."""
        if not source_path.exists():
            raise RuntimeError(f"Source video not found: {source_path}")

        if mode in {"none", "off"}:
            return {
                "mode": mode,
                "frames_processed": 0,
                "frames_inpainted": 0,
                "frames_bypassed": 0,
                "clean_runtime": 0.0,
                "output_path": str(source_path),
            }

        # Build Chinese subtitle active intervals
        active_intervals: list[tuple[float, float]] = []
        if cues:
            for c in cues:
                src = (c.source_text or "").strip()
                if len(src) > 1 and src not in NOISE_FILTER:
                    # Give small 0.05s tolerance for speech-subtitle alignment
                    active_intervals.append((float(c.start) - 0.05, float(c.end) + 0.05))

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
        frames_bypassed = 0
        t0 = time.time()
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            frames_processed += 1

            # Check if current timestamp falls within an active Chinese subtitle
            is_active = True
            if active_intervals:
                is_active = any(start <= timestamp <= end for start, end in active_intervals)

            if not is_active:
                # Frame has no Chinese subtitle: write untouched original frame
                writer.write(frame)
                frames_bypassed += 1
            else:
                cleaned_frame, was_cleaned = self.clean_frame(frame, mode=mode, is_subtitle_active=True)
                if was_cleaned:
                    frames_inpainted += 1
                else:
                    frames_bypassed += 1
                writer.write(cleaned_frame)

            frame_idx += 1

        cap.release()
        writer.release()
        total_time = time.time() - t0

        metrics = {
            "mode": mode,
            "frames_processed": frames_processed,
            "frames_inpainted": frames_inpainted,
            "frames_bypassed": frames_bypassed,
            "inpaint_rate": round(frames_inpainted / max(1, frames_processed) * 100, 1),
            "bypass_rate": round(frames_bypassed / max(1, frames_processed) * 100, 1),
            "clean_runtime": round(total_time, 2),
            "fps_speed": round(frames_processed / max(0.01, total_time), 1),
            "output_path": str(output_path),
        }
        logger.info("Decoupled HardSub Cleaner completed: %s", metrics)
        return metrics
