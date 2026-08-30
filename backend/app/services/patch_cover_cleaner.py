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

    def _is_valid_subtitle_polygon(self, points: list[list[float]]) -> bool:
        """Reject polygons that are too tall, too wide, or located outside the subtitle band."""
        if not points or len(points) < 3:
            return False
        ys = [p[1] for p in points]
        xs = [p[0] for p in points]
        min_y, max_y = min(ys), max(ys)
        min_x, max_x = min(xs), max(xs)
        # Subtitles must be in the bottom band: y in [0.68, 0.98]
        if max_y < 0.68 or min_y > 0.98:
            return False
        # Height must not exceed 14% of frame height (reject face/body boxes)
        if (max_y - min_y) > 0.14:
            return False
        # Width must not exceed 82% of frame width
        if (max_x - min_x) > 0.82:
            return False
        return True

    def extract_active_intervals(
        self,
        cues: list[SubtitleCue],
        fps: float,
    ) -> list[dict[str, Any]]:
        """Extract subtitle intervals with tight glyph polygons only."""
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
                        if self._is_valid_subtitle_polygon(region.points):
                            polygons.append(region.points)

            if not polygons and cue.ocr_evidence:
                for ev in cue.ocr_evidence:
                    for region in ev.regions:
                        if region.points and (region.confidence is None or region.confidence >= min_conf):
                            if self._is_valid_subtitle_polygon(region.points):
                                polygons.append(region.points)

            # If no valid OCR polygons exist, keep empty list (never insert giant screen-wide fallback)
            if not polygons:
                continue

            raw_intervals.append({
                "start": start_t,
                "end": end_t,
                "polygons": polygons,
                "cue_id": cue.id,
            })

        if not raw_intervals:
            return []

        raw_intervals.sort(key=lambda x: x["start"])

        # Temporal bridging: merge timing without creating oversized polygon unions
        bridged: list[dict[str, Any]] = [raw_intervals[0]]
        for cur in raw_intervals[1:]:
            prev = bridged[-1]
            if cur["start"] - prev["end"] <= gap_fill_sec:
                prev["end"] = max(prev["end"], cur["end"])
                for poly in cur["polygons"]:
                    if poly not in prev["polygons"]:
                        prev["polygons"].append(poly)
            else:
                bridged.append(cur)

        return bridged

    def create_feathered_mask(
        self,
        height: int,
        width: int,
        polygons: list[list[list[float]]],
    ) -> np.ndarray:
        """Create a smooth feathered alpha mask strictly in the bottom subtitle band."""
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

        # Hard Face / Body Safety Cutoff: zero out any mask pixels above y = 0.68
        safe_top_cutoff = int(height * 0.68)
        mask[:safe_top_cutoff, :] = 0

        # Max area safety check: ensure mask does not cover > 15% of frame area
        total_pixels = height * width
        masked_pixels = np.count_nonzero(mask)
        if total_pixels > 0 and (masked_pixels / total_pixels) > 0.15:
            # Downscale mask intensity if anomalous
            scale_factor = 0.15 / (masked_pixels / total_pixels)
            mask = (mask.astype(np.float32) * scale_factor).astype(np.uint8)

        return mask.astype(np.float32) / 255.0

    def build_render_cue_contexts(
        self,
        cues: list[SubtitleCue],
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        """Build exact render context for each RenderSubtitleCue with 1:1 synchronized backing and Chinese polygons."""
        from app.services.subtitles import UtteranceEngine

        engine = UtteranceEngine()
        render_cues, _ = engine.process_cues(cues, translated=True)
        cues_by_id = {
            (getattr(c, "id", None) or getattr(c, "render_id", None)): c
            for c in cues
            if (getattr(c, "id", None) or getattr(c, "render_id", None))
        }

        cue_y_centers: list[int] = []
        default_y_center = int(round(height * 0.862 - 4.0))

        for cue in render_cues:
            y_pts: list[float] = []
            for cid in getattr(cue, "source_cue_ids", []):
                orig = cues_by_id.get(cid)
                if not orig:
                    continue
                for r in getattr(orig, "ocr_regions", []) or []:
                    pts = getattr(r, "points", []) or []
                    if len(pts) >= 3:
                        ys = [p[1] for p in pts if len(p) >= 2]
                        if ys and min(ys) >= 0.70 and max(ys) <= 0.98 and (max(ys) - min(ys)) <= 0.14:
                            y_pts.extend(ys)
                for ev in getattr(orig, "ocr_evidence", []) or []:
                    for r in getattr(ev, "regions", []) or []:
                        pts = getattr(r, "points", []) or []
                        if len(pts) >= 3:
                            ys = [p[1] for p in pts if len(p) >= 2]
                            if ys and min(ys) >= 0.70 and max(ys) <= 0.98 and (max(ys) - min(ys)) <= 0.14:
                                y_pts.extend(ys)

            if y_pts:
                min_y = min(y_pts)
                max_y = max(y_pts)
                mid_y = (min_y + max_y) / 2.0 * height - 4.0
                clamped_y = int(round(max(height * 0.75, min(height * 0.90, mid_y))))
                cue_y_centers.append(clamped_y)
            else:
                cue_y_centers.append(default_y_center)

        if cue_y_centers:
            median_y = int(round(float(np.median(cue_y_centers))))
            smoothed_y_centers = []
            for y_val in cue_y_centers:
                if abs(y_val - median_y) <= 14:
                    smoothed_y_centers.append(median_y)
                else:
                    smoothed_y_centers.append(y_val)
            cue_y_centers = smoothed_y_centers

        center_x = width // 2
        font_size = max(18, int(round(height * 0.050)))
        pad_x = 22
        pad_y = 10

        contexts: list[dict[str, Any]] = []
        for idx, cue in enumerate(render_cues):
            lines = [l.strip() for l in cue.render_text.replace(r"\N", "\n").split("\n") if l.strip()]
            if not lines:
                continue
            line_count = len(lines)
            max_line_chars = max(len(l) for l in lines)
            text_w = int(round(max_line_chars * (font_size * 0.54)))
            text_h = int(round(line_count * (font_size * 1.25)))

            y_pos = cue_y_centers[idx] if idx < len(cue_y_centers) else default_y_center
            x1 = max(0, int(center_x - text_w / 2 - pad_x))
            x2 = min(width, int(center_x + text_w / 2 + pad_x))
            y1 = max(int(height * 0.68), int(y_pos - text_h / 2 - pad_y))
            y2 = min(int(height * 0.98), int(y_pos + text_h / 2 + pad_y))

            # Chinese polygons for this cue only
            chinese_polygons: list[list[list[float]]] = []
            for cid in getattr(cue, "source_cue_ids", []):
                orig = cues_by_id.get(cid)
                if not orig:
                    continue
                for r in getattr(orig, "ocr_regions", []) or []:
                    pts = getattr(r, "points", []) or []
                    if self._is_valid_subtitle_polygon(pts):
                        chinese_polygons.append(pts)
                if not chinese_polygons:
                    for ev in getattr(orig, "ocr_evidence", []) or []:
                        for r in getattr(ev, "regions", []) or []:
                            pts = getattr(r, "points", []) or []
                            if self._is_valid_subtitle_polygon(pts):
                                chinese_polygons.append(pts)

            contexts.append({
                "start": cue.start,
                "end": cue.end,
                "cue_id": getattr(cue, "render_id", str(idx)),
                "bbox": (x1, y1, x2, y2),
                "chinese_polygons": chinese_polygons,
            })

        return contexts

    def compute_vi_backing_intervals(
        self,
        cues: list[SubtitleCue],
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        """Backwards compatibility alias for build_render_cue_contexts."""
        return self.build_render_cue_contexts(cues, width, height)

    def create_rounded_rect_mask(
        self,
        height: int,
        width: int,
        bbox: tuple[int, int, int, int],
        radius: int = 10,
        feather: int = 8,
    ) -> np.ndarray:
        """Create a smooth feathered rounded-rectangle mask for VI text backing."""
        x1, y1, x2, y2 = bbox
        mask = np.zeros((height, width), dtype=np.uint8)
        radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

        cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
        cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)
        cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
        cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
        cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
        cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)

        if feather > 0:
            k = max(1, feather * 2 + 1)
            mask = cv2.GaussianBlur(mask, (k, k), feather)

        return mask.astype(np.float32) / 255.0

    def apply_patch_cover(
        self,
        frame: np.ndarray,
        alpha_mask: np.ndarray | None = None,
        vi_backing_mask: np.ndarray | None = None,
        donor_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Blend Chinese patch cover and VI subtitle backdrop blur over video frame."""
        out = frame.copy()
        h, w = frame.shape[:2]
        opacity = self.config.patch_opacity
        sigma = self.config.blur_sigma

        # Step 1: Chinese Hard-Sub Patch Cover (tight glyph mask)
        if alpha_mask is not None and np.any(alpha_mask > 0.001):
            if donor_frame is not None and self.config.use_temporal_donor:
                patch = donor_frame
                if sigma > 0:
                    k_blur = max(1, int(sigma * 2 + 1))
                    if k_blur % 2 == 0:
                        k_blur += 1
                    patch = cv2.GaussianBlur(patch, (k_blur, k_blur), sigma)
            else:
                k_blur = max(1, int(sigma * 3 + 1))
                if k_blur % 2 == 0:
                    k_blur += 1
                patch = cv2.GaussianBlur(out, (k_blur, k_blur), sigma)

            dark_tint = getattr(self.config, "dark_tint", 0.48)
            if dark_tint > 0:
                patch = (patch.astype(np.float32) * (1.0 - dark_tint)).astype(np.uint8)

            alpha_3d = np.repeat((alpha_mask * opacity)[:, :, np.newaxis], 3, axis=2)
            out = (1.0 - alpha_3d) * out.astype(np.float32) + alpha_3d * patch.astype(np.float32)
            out = np.clip(out, 0, 255).astype(np.uint8)

        # Step 2: VI Subtitle Backing (consistent soft backdrop blur + subtle dark tint)
        if vi_backing_mask is not None and np.any(vi_backing_mask > 0.001):
            blur_sigma_bg = 8.0
            k_bg = 25
            blurred_bg = cv2.GaussianBlur(out, (k_bg, k_bg), blur_sigma_bg)
            dark_tint_bg = 0.38  # moderate dark tint (~38%)
            blurred_bg = (blurred_bg.astype(np.float32) * (1.0 - dark_tint_bg)).astype(np.uint8)
            backing_opacity = 0.88  # distinct soft backdrop blend across all scene brightness levels
            alpha_bg_3d = np.repeat((vi_backing_mask * backing_opacity)[:, :, np.newaxis], 3, axis=2)
            out = (1.0 - alpha_bg_3d) * out.astype(np.float32) + alpha_bg_3d * blurred_bg.astype(np.float32)
            out = np.clip(out, 0, 255).astype(np.uint8)

        return out

    def clean_video(
        self,
        source_path: Path,
        output_path: Path,
        cues: list[SubtitleCue],
        lossless_intermediate: bool = True,
    ) -> dict[str, Any]:
        """Process video and apply 1:1 cue-synchronized patch cover and backdrop blur."""
        t_start = time.time()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source video: {source_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        contexts = self.build_render_cue_contexts(cues, width, height)

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

        # Mask cache across frames for the active cue
        cached_ctx_id = None
        cached_cn_mask = None
        cached_vi_mask = None

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            cur_time = frame_idx / fps

            # Find active render cue: EXACT 1:1 timestamp match with subtitle lifecycle
            active_ctx = None
            for ctx in contexts:
                if ctx["start"] <= cur_time < ctx["end"]:
                    active_ctx = ctx
                    break

            if active_ctx is not None:
                patched_frames_count += 1
                ctx_key = (active_ctx["cue_id"], active_ctx["start"], active_ctx["end"])

                if ctx_key == cached_ctx_id and cached_vi_mask is not None:
                    alpha_cn_mask = cached_cn_mask
                    alpha_vi_mask = cached_vi_mask
                else:
                    if active_ctx["chinese_polygons"]:
                        alpha_cn_mask = self.create_feathered_mask(height, width, active_ctx["chinese_polygons"])
                    else:
                        alpha_cn_mask = None
                    alpha_vi_mask = self.create_rounded_rect_mask(height, width, active_ctx["bbox"], radius=10, feather=8)
                    cached_ctx_id = ctx_key
                    cached_cn_mask = alpha_cn_mask
                    cached_vi_mask = alpha_vi_mask

                covered_frame = self.apply_patch_cover(frame, alpha_mask=alpha_cn_mask, vi_backing_mask=alpha_vi_mask)
                writer.write(covered_frame)
            else:
                # Fast path: NO active subtitle -> 100% UNTOUCHED FRAME (0 backing, 0 blur, 0 ghost)
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
            "intervals_count": len(contexts),
            "output_path": str(output_path),
        }
        self.last_metrics = metrics
        logger.info("PatchCover cleaned %d/%d frames in %.3fs", patched_frames_count, total_frames, t_elapsed)
        return metrics
