from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.project import OCRRegion, SubtitleCue

logger = logging.getLogger(__name__)


def compute_geometry_signal(
    frame: np.ndarray,
    regions: list[OCRRegion] | list[dict[str, Any]],
    luminance_threshold: int = 135,
) -> float:
    """Calculates normalized text glyph energy within OCR bounding polygons."""
    if not regions or frame is None or frame.size == 0:
        return 0.0

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total_text_pixels = 0
    total_polygon_area = 0

    for r in regions:
        r_pts = r.points if hasattr(r, "points") else r.get("points", [])
        if not r_pts or len(r_pts) < 3:
            continue

        poly_pts = np.array(
            [
                [
                    int(round(max(0.0, min(1.0, float(p[0]))) * w)),
                    int(round(max(0.0, min(1.0, float(p[1]))) * h)),
                ]
                for p in r_pts
                if len(p) >= 2
            ],
            dtype=np.int32,
        )
        if len(poly_pts) < 3:
            continue

        rx, ry, rw, rh = cv2.boundingRect(poly_pts)
        if rw <= 4 or rh <= 4:
            continue

        pad_x, pad_y = 4, 4
        x1 = max(0, rx - pad_x)
        y1 = max(0, ry - pad_y)
        x2 = min(w, rx + rw + pad_x)
        y2 = min(h, ry + rh + pad_y)

        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            continue

        local_poly = poly_pts - np.array([x1, y1])
        local_mask = np.zeros(patch.shape, dtype=np.uint8)
        cv2.fillPoly(local_mask, [local_poly], 255)
        local_dilated = cv2.dilate(
            local_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )

        bright_text = (patch >= luminance_threshold) & (local_dilated > 0)
        total_text_pixels += int(np.sum(bright_text))
        total_polygon_area += max(1, int(np.sum(local_dilated > 0)))

    return float(total_text_pixels) / max(1.0, float(total_polygon_area))


class VisualBoundaryTracker:
    """Refines coarse 2 FPS OCR timestamps to exact frame boundaries with single-seek streaming."""

    def __init__(
        self,
        sample_fps: float = 10.0,
        search_window_seconds: float = 0.60,
        signal_threshold: float = 0.035,
        min_stable_frames: int = 2,
    ) -> None:
        self.sample_fps = max(5.0, sample_fps)
        self.search_window_seconds = max(0.20, search_window_seconds)
        self.signal_threshold = signal_threshold
        self.min_stable_frames = max(1, min_stable_frames)
        self.last_metrics: dict[str, Any] = {}

    def refine_cues(
        self,
        video_path: Path,
        cues: list[SubtitleCue],
    ) -> list[SubtitleCue]:
        if not video_path.exists() or not cues:
            return cues

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return cues

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / max(1.0, video_fps)

        step_frames = max(1, int(round(video_fps / self.sample_fps)))
        t0 = time.time()

        cues_processed = 0
        start_refined_count = 0
        end_refined_count = 0
        start_adjustments: list[float] = []
        end_adjustments: list[float] = []
        frames_scanned = 0

        refined_cues: list[SubtitleCue] = []

        for cue in cues:
            copy = cue.model_copy(deep=True)
            regions = copy.ocr_regions
            if not regions or copy.ocr_start is None or copy.ocr_end is None:
                refined_cues.append(copy)
                continue

            orig_start = float(copy.ocr_start)
            orig_end = float(copy.ocr_end)
            cues_processed += 1

            # Seek ONCE to segment start window
            start_search_from = max(0.0, orig_start - self.search_window_seconds)
            end_search_to = min(video_duration, orig_end + self.search_window_seconds)

            start_fidx = int(round(start_search_from * video_fps))
            end_fidx = int(round(end_search_to * video_fps))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_fidx)

            signals: list[tuple[float, float]] = []
            f_curr = start_fidx

            while f_curr <= end_fidx:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_scanned += 1
                t_curr = f_curr / video_fps

                if (f_curr - start_fidx) % step_frames == 0:
                    sig = compute_geometry_signal(frame, regions)
                    signals.append((t_curr, sig))

                f_curr += 1

            if not signals:
                refined_cues.append(copy)
                continue

            # Reference signal at midpoint
            mid_signals = [s for t, s in signals if (orig_start <= t <= orig_end)]
            mid_sig = float(np.mean(mid_signals)) if mid_signals else float(np.max([s for _, s in signals]))

            thresh = max(self.signal_threshold, mid_sig * 0.25)

            # Refine START: earliest stable appearance before orig_end
            best_start = orig_start
            for i in range(len(signals)):
                t_val, s_val = signals[i]
                if t_val > (orig_end - 0.15):
                    break
                if s_val >= thresh:
                    stable = True
                    for j in range(1, min(self.min_stable_frames, len(signals) - i)):
                        if signals[i + j][1] < (thresh * 0.70):
                            stable = False
                            break
                    if stable:
                        best_start = round(t_val, 2)
                        break

            # Refine END: first stable disappearance after best_start
            best_end = orig_end
            start_checking_end = False
            for i in range(len(signals)):
                t_val, s_val = signals[i]
                if t_val >= (best_start + 0.15):
                    start_checking_end = True
                if not start_checking_end:
                    continue

                if s_val < thresh:
                    disappeared = True
                    for j in range(1, min(self.min_stable_frames, len(signals) - i)):
                        if signals[i + j][1] >= thresh:
                            disappeared = False
                            break
                    if disappeared:
                        best_end = round(t_val, 2)
                        break
                else:
                    best_end = round(t_val + (step_frames / video_fps), 2)

            if abs(best_start - orig_start) >= 0.04:
                start_refined_count += 1
                start_adjustments.append(abs(best_start - orig_start) * 1000.0)
                copy.ocr_start = best_start

            if abs(best_end - orig_end) >= 0.04:
                end_refined_count += 1
                end_adjustments.append(abs(best_end - orig_end) * 1000.0)
                copy.ocr_end = best_end

            refined_cues.append(copy)

        cap.release()
        elapsed = time.time() - t0

        self.last_metrics = {
            "visual_timing_cues": cues_processed,
            "visual_timing_start_refined": start_refined_count,
            "visual_timing_end_refined": end_refined_count,
            "avg_start_adjustment_ms": round(
                float(np.mean(start_adjustments)), 1
            )
            if start_adjustments
            else 0.0,
            "max_start_adjustment_ms": round(
                float(np.max(start_adjustments)), 1
            )
            if start_adjustments
            else 0.0,
            "avg_end_adjustment_ms": round(
                float(np.mean(end_adjustments)), 1
            )
            if end_adjustments
            else 0.0,
            "max_end_adjustment_ms": round(
                float(np.max(end_adjustments)), 1
            )
            if end_adjustments
            else 0.0,
            "visual_tracker_frames_scanned": frames_scanned,
            "visual_tracker_runtime_s": round(elapsed, 2),
        }
        logger.info("Visual Boundary Refinement Completed: %s", self.last_metrics)
        return refined_cues
