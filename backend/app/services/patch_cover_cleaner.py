from __future__ import annotations

import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.project import (
    OCRRegion,
    PatchCoverConfig,
    SubtitleCue,
)

logger = logging.getLogger(__name__)


class PatchCoverCleaner:
    """Fast, feathered background patch-cover cleaner for hard subtitles.

    Target style:
    - Leaves surrounding background untouched.
    - Softly attenuates/covers Chinese glyphs so they become faint and unreadable.
    - Preserves lighting and scene texture without Telea inpainting smudges or hard rectangular borders.
    - Places Vietnamese ASS subtitles sharply on top.
    """

    def __init__(
        self,
        config: PatchCoverConfig | None = None,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.config = config or PatchCoverConfig()
        self.ffmpeg_bin = ffmpeg_bin
        self.last_metrics: dict[str, Any] = {}

    def extract_active_intervals(
        self,
        cues: list[SubtitleCue],
        fps: float,
    ) -> list[dict[str, Any]]:
        """Extract and bridge subtitle intervals with temporal persistence."""
        min_conf = self.config.min_ocr_confidence
        gap_fill_sec = self.config.temporal_gap_fill_frames / max(1.0, fps)
        persistence_sec = self.config.mask_persistence_frames / max(1.0, fps)

        raw_intervals: list[dict[str, Any]] = []

        for cue in cues:
            start_t = cue.ocr_start if cue.ocr_start is not None else cue.start
            end_t = cue.ocr_end if cue.ocr_end is not None else cue.end

            if end_t <= start_t:
                continue

            end_t += persistence_sec  # add persistence

            polygons: list[list[list[float]]] = []
            if cue.ocr_regions:
                for region in cue.ocr_regions:
                    if region.points and (region.confidence is None or region.confidence >= min_conf):
                        polygons.append(region.points)

            if not polygons and cue.ocr_evidence:
                for ev in cue.ocr_evidence:
                    for region in ev.regions:
                        if region.points and (region.confidence is None or region.confidence >= min_conf):
                            polygons.append(region.points)

            if not polygons:
                # Default bottom subtitle band
                polygons = [[[0.08, 0.76], [0.92, 0.76], [0.92, 0.94], [0.08, 0.94]]]

            raw_intervals.append({
                "start": start_t,
                "end": end_t,
                "polygons": polygons,
                "cue_id": cue.id,
            })

        if not raw_intervals:
            return []

        raw_intervals.sort(key=lambda x: x["start"])

        # Temporal bridging
        bridged: list[dict[str, Any]] = [raw_intervals[0]]
        for cur in raw_intervals[1:]:
            prev = bridged[-1]
            if cur["start"] - prev["end"] <= gap_fill_sec:
                prev["end"] = max(prev["end"], cur["end"])
                prev["polygons"].extend(cur["polygons"])
            else:
                bridged.append(cur)

        return bridged

    def create_feathered_mask(
        self,
        height: int,
        width: int,
        polygons: list[list[list[float]]],
    ) -> np.ndarray:
        """Create a smooth feathered alpha mask in range [0.0, 1.0]."""
        mask = np.zeros((height, width), dtype=np.uint8)

        for poly in polygons:
            pts = np.array([
                [int(p[0] * width), int(p[1] * height)]
                for p in poly
            ], dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)

        pad = self.config.padding_px
        if pad > 0:
            k_pad = max(1, pad * 2 + 1)
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_pad, k_pad))
            mask = cv2.dilate(mask, dilate_kernel)

        feather = self.config.feather_px
        if feather > 0:
            k_feat = max(1, feather * 2 + 1)
            mask = cv2.GaussianBlur(mask, (k_feat, k_feat), feather)

        return mask.astype(np.float32) / 255.0

    def apply_patch_cover(
        self,
        frame: np.ndarray,
        alpha_mask: np.ndarray,
        donor_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Blend donor/blur patch over subtitle region according to feathered alpha mask."""
        h, w = frame.shape[:2]
        opacity = self.config.patch_opacity
        sigma = self.config.blur_sigma

        # Priority 1: Temporal donor if available
        if donor_frame is not None and self.config.use_temporal_donor:
            patch = donor_frame
            if sigma > 0:
                k_blur = max(1, int(sigma * 2 + 1))
                if k_blur % 2 == 0:
                    k_blur += 1
                patch = cv2.GaussianBlur(patch, (k_blur, k_blur), sigma)
        else:
            # Priority 2/3: Local soft blur patch
            k_blur = max(1, int(sigma * 3 + 1))
            if k_blur % 2 == 0:
                k_blur += 1
            patch = cv2.GaussianBlur(frame, (k_blur, k_blur), sigma)

        # 3-channel alpha
        alpha_3d = np.repeat((alpha_mask * opacity)[:, :, np.newaxis], 3, axis=2)

        # Composite: (1 - alpha) * frame + alpha * patch
        out = (1.0 - alpha_3d) * frame.astype(np.float32) + alpha_3d * patch.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def clean_video(
        self,
        source_path: Path,
        output_path: Path,
        cues: list[SubtitleCue],
        lossless_intermediate: bool = True,
    ) -> dict[str, Any]:
        """Process video and apply fast feathered patch cover on all subtitle intervals."""
        t_start = time.time()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source video: {source_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        intervals = self.extract_active_intervals(cues, fps)

        # Setup VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        frame_idx = 0
        patched_frames_count = 0

        # Mask cache across frames
        cached_interval_id = None
        cached_mask = None

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            cur_time = frame_idx / fps

            # Find active interval
            active_item = None
            for item in intervals:
                if item["start"] <= cur_time <= item["end"]:
                    active_item = item
                    break

            if active_item is not None:
                patched_frames_count += 1
                item_key = (active_item["cue_id"], active_item["start"], active_item["end"])

                if item_key == cached_interval_id and cached_mask is not None:
                    alpha_mask = cached_mask
                else:
                    alpha_mask = self.create_feathered_mask(height, width, active_item["polygons"])
                    cached_interval_id = item_key
                    cached_mask = alpha_mask

                covered_frame = self.apply_patch_cover(frame, alpha_mask)
                writer.write(covered_frame)
            else:
                # Fast path: 100% untouched frame
                writer.write(frame)

            frame_idx += 1

        cap.release()
        writer.release()

        t_elapsed = time.time() - t_start
        metrics = {
            "cleanup_mode": "patch_cover",
            "cleanup_time_s": round(t_elapsed, 3),
            "total_frames": total_frames,
            "patched_frames": patched_frames_count,
            "intervals_count": len(intervals),
            "output_path": str(output_path),
        }
        self.last_metrics = metrics
        logger.info("PatchCover cleaned %d/%d frames in %.3fs", patched_frames_count, total_frames, t_elapsed)
        return metrics
