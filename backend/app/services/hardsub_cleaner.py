from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.project import OCRRegion, SubtitleCue

logger = logging.getLogger(__name__)

NOISE_FILTER = {"10.5o", "10:50", "MILK", "MILK MILK", "IN-CN", "CN-IN", "755135", "CN"}


class HardSubCleaner:
    """Remove burned-in subtitles conservatively using OCR geometry and temporal restoration.

    Architecture (Phase 5.1D):
    - *when* text is visible: exact OCR visual timestamps (`ocr_start`/`ocr_end`), avoiding ASR dialogue overhang;
    - *where* text is visible: tight OCR bounding polygons (`ocr_regions`) restricting search area;
    - *how* glyphs are detected: high-contrast white seed + edge-aware dark outline/shadow extension + temporal diff;
    - *what* background replaces it: aligned clean temporal donor with guarded Telea inpainting.

    Every pixel outside the extracted subtitle glyph/outline mask is preserved 100% untouched.
    """

    def __init__(
        self,
        crop_top_ratio: float = 0.65,
        crop_bottom_ratio: float = 0.95,
        crop_left_ratio: float = 0.06,
        crop_right_ratio: float = 0.94,
        mask_dilate_radius: int = 1,
        mask_dilate_iterations: int = 1,
        luminance_threshold: int = 185,
        local_contrast_threshold: int = 18,
        inpaint_radius: int = 2,
        max_mask_coverage: float = 0.12,
        scene_cut_threshold: float = 34.0,
        temporal_max_distance_frames: int = 45,
        temporal_difference_threshold: int = 14,
        temporal_local_score_threshold: float = 22.0,
        ocr_min_confidence: float = 0.35,
        timing_pad_seconds: float = 0.15,
        geometry_enabled: bool = True,
        geometry_padding_px: int = 4,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.crop_left_ratio = crop_left_ratio
        self.crop_right_ratio = crop_right_ratio
        self.mask_dilate_radius = mask_dilate_radius
        self.mask_dilate_iterations = mask_dilate_iterations
        self.luminance_threshold = luminance_threshold
        self.local_contrast_threshold = local_contrast_threshold
        self.inpaint_radius = inpaint_radius
        self.max_mask_coverage = max_mask_coverage
        self.scene_cut_threshold = scene_cut_threshold
        self.temporal_max_distance_frames = temporal_max_distance_frames
        self.temporal_difference_threshold = temporal_difference_threshold
        self.temporal_local_score_threshold = temporal_local_score_threshold
        self.ocr_min_confidence = ocr_min_confidence
        self.timing_pad_seconds = timing_pad_seconds
        self.geometry_enabled = geometry_enabled
        self.geometry_padding_px = geometry_padding_px
        self.ffmpeg_bin = ffmpeg_bin

        self._metrics: dict[str, Any] = {}

    def _roi_bounds(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        y1 = max(0, min(h - 1, int(h * self.crop_top_ratio)))
        y2 = max(y1 + 1, min(h, int(h * self.crop_bottom_ratio)))
        x1 = max(0, min(w - 1, int(w * self.crop_left_ratio)))
        x2 = max(x1 + 1, min(w, int(w * self.crop_right_ratio)))
        return x1, y1, x2, y2

    @staticmethod
    def _component_rows(
        components: list[tuple[int, int, int, int, int]],
    ) -> list[list[tuple[int, int, int, int, int]]]:
        if not components:
            return []
        median_h = float(np.median([c[3] for c in components]))
        tolerance = max(4.0, median_h * 0.8)
        rows: list[list[tuple[int, int, int, int, int]]] = []
        for component in sorted(components, key=lambda c: c[1] + c[3] / 2):
            cy = component[1] + component[3] / 2
            placed = False
            for row in rows:
                row_cy = float(np.mean([item[1] + item[3] / 2 for item in row]))
                if abs(cy - row_cy) <= tolerance:
                    row.append(component)
                    placed = True
                    break
            if not placed:
                rows.append([component])
        return rows

    def extract_text_mask(self, sub_roi: np.ndarray) -> np.ndarray | None:
        """Extract a compact, subtitle-shaped mask without swallowing bright scenery."""
        if sub_roi.size == 0:
            return None

        gray = cv2.cvtColor(sub_roi, cv2.COLOR_BGR2GRAY)
        local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.2, sigmaY=2.2)
        contrast = cv2.subtract(gray, local_mean)
        gradient = cv2.morphologyEx(
            gray,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        bright = gray >= self.luminance_threshold
        high_contrast = contrast >= self.local_contrast_threshold
        strong_edge = gradient >= max(24, self.local_contrast_threshold)
        seed = (bright & (high_contrast | strong_edge)).astype(np.uint8) * 255

        seed = cv2.morphologyEx(
            seed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
        roi_h, roi_w = gray.shape
        max_component_area = max(48, int(roi_h * roi_w * 0.025))
        max_component_width = max(24, int(roi_w * 0.12))
        max_component_height = max(12, int(roi_h * 0.32))

        components: list[tuple[int, int, int, int, int]] = []
        for label in range(1, num_labels):
            x, y, w, h, area = [int(v) for v in stats[label]]
            if area < 3 or area > max_component_area:
                continue
            if h < 3 or h > max_component_height:
                continue
            if w > max_component_width:
                continue
            if w / max(1, h) > 5.5:
                continue
            components.append((x, y, w, h, label))

        if not components:
            self._metrics["mask_rejected_no_components"] = int(
                self._metrics.get("mask_rejected_no_components", 0)
            ) + 1
            return None

        rows = self._component_rows(components)
        scored_rows: list[tuple[float, list[tuple[int, int, int, int, int]]]] = []
        for row in rows:
            if len(row) < 2:
                continue
            x_min = min(c[0] for c in row)
            x_max = max(c[0] + c[2] for c in row)
            median_h = float(np.median([c[3] for c in row]))
            horizontal_span = x_max - x_min
            if horizontal_span < max(10.0, median_h * 1.1):
                continue
            center_bias = 1.0 - min(
                1.0,
                abs(((x_min + x_max) / 2) - roi_w / 2) / max(1.0, roi_w / 2),
            )
            score = len(row) * 2.0 + horizontal_span / max(8.0, median_h) + center_bias
            scored_rows.append((score, row))

        if not scored_rows:
            self._metrics["mask_rejected_no_text_row"] = int(
                self._metrics.get("mask_rejected_no_text_row", 0)
            ) + 1
            return None

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        selected_rows = [row for _, row in scored_rows[:2]]
        selected_labels = {component[4] for row in selected_rows for component in row}

        filtered = np.zeros_like(seed)
        for label in selected_labels:
            filtered[labels == label] = 255

        if self.mask_dilate_radius > 0:
            k_size = 2 * self.mask_dilate_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            filtered = cv2.dilate(filtered, kernel, iterations=self.mask_dilate_iterations)

        coverage = float(np.mean(filtered > 0))
        if coverage <= 0.0 or coverage > self.max_mask_coverage:
            self._metrics["mask_rejected_coverage"] = int(
                self._metrics.get("mask_rejected_coverage", 0)
            ) + 1
            return None

        return filtered

    def extract_geometry_mask(
        self,
        frame: np.ndarray,
        regions: list[OCRRegion],
        temporal_donors: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Extract high-recall, high-precision Chinese subtitle glyph + outline mask strictly inside OCR polygons."""
        if not regions:
            return None, {}

        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        coverages: list[float] = []
        outline_added = 0
        bboxes: list[tuple[int, int, int, int]] = []
        pad_x = max(14, self.geometry_padding_px + 8)
        pad_y = max(10, self.geometry_padding_px + 4)

        for region in regions:
            r_pts = region.get("points") if isinstance(region, dict) else getattr(region, "points", [])
            if not r_pts or len(r_pts) < 3:
                continue

            poly_pts = np.array(
                [
                    [
                        int(round(max(0.0, min(1.0, float(pt[0]))) * w)),
                        int(round(max(0.0, min(1.0, float(pt[1]))) * h)),
                    ]
                    for pt in r_pts
                    if len(pt) >= 2
                ],
                dtype=np.int32,
            )
            if len(poly_pts) < 3:
                continue

            rx, ry, rw, rh = cv2.boundingRect(poly_pts)
            if rw <= 4 or rh <= 4:
                continue

            x1 = max(0, rx - pad_x)
            y1 = max(0, ry - pad_y)
            x2 = min(w, rx + rw + pad_x)
            y2 = min(h, ry + rh + pad_y)
            bboxes.append((x1, y1, x2, y2))

            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            ph, pw = patch.shape[:2]
            local_poly = poly_pts - np.array([x1, y1])
            local_poly_mask = np.zeros((ph, pw), dtype=np.uint8)
            cv2.fillPoly(local_poly_mask, [local_poly], 255)

            local_poly_dilated = cv2.dilate(
                local_poly_mask,
                cv2.getStructuringElement(cv2.MORPH_RECT, (15, 11)),
                iterations=1,
            )

            # 1. Stroke interior seed detection (solid glyph seed)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            seed = ((gray >= 135) & (local_poly_dilated > 0)).astype(np.uint8) * 255

            # 2. Outline / Dark Stroke Extension
            # Dilate solid seed by (7, 7) ellipse x 2 iterations to engulf full drop-shadow & outer contour
            dilated_seed = cv2.dilate(
                seed,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=2,
            )
            region_mask = cv2.bitwise_and(dilated_seed, local_poly_dilated)

            initial_white_count = int(np.sum(seed > 0))
            total_outline_count = int(np.sum(region_mask > 0))
            outline_added += max(0, total_outline_count - initial_white_count)

            # 3. Temporal difference guidance if clean donor candidate is available
            if temporal_donors:
                for donor in temporal_donors:
                    if donor.shape == frame.shape:
                        donor_patch = donor[y1:y2, x1:x2]
                        diff = cv2.absdiff(patch, donor_patch)
                        diff_gray = np.max(diff, axis=2)
                        changed = (diff_gray >= self.temporal_difference_threshold).astype(np.uint8) * 255
                        temporal_glyph = cv2.bitwise_and(changed, local_poly_mask)
                        region_mask = cv2.bitwise_or(region_mask, temporal_glyph)

            # Morphological cleanup inside region
            region_mask = cv2.morphologyEx(
                region_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            )
            region_mask = cv2.bitwise_and(region_mask, local_poly_dilated)

            # Coverage check per local OCR region (Guard against over-inpaint)
            cov = float(np.mean(region_mask > 0))
            if cov > 0.75:
                tight_seed = cv2.dilate(
                    seed,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
                region_mask = cv2.bitwise_and(region_mask, tight_seed)
                cov = float(np.mean(region_mask > 0))

            if cov > 0.0:
                coverages.append(cov)
                combined_mask[y1:y2, x1:x2] = cv2.bitwise_or(
                    combined_mask[y1:y2, x1:x2], region_mask
                )

        total_mask_pixels = int(np.sum(combined_mask > 0))
        if total_mask_pixels == 0:
            return None, {}

        stats = {
            "region_coverages": coverages,
            "avg_coverage": float(np.mean(coverages)) if coverages else 0.0,
            "max_coverage": float(np.max(coverages)) if coverages else 0.0,
            "outline_pixels_added": outline_added,
            "bboxes": bboxes,
        }
        return combined_mask, stats

    def build_active_intervals(
        self,
        cues: list[SubtitleCue] | None,
        pad_seconds: float | None = None,
    ) -> list[tuple[float, float]]:
        """Build visual hard-sub intervals, preferring preserved OCR timestamps.

        Fused cues normally use ASR start/end for spoken dialogue.  Using those timestamps
        for image cleanup leaves stale masks after the on-screen Chinese subtitle has gone.
        New projects therefore carry ``ocr_start``/``ocr_end``.  Legacy projects fall back
        to OCR-confidence-backed cue timing and are reported in metrics so callers can
        choose to re-run Analyze for exact visual timing.
        """
        if not cues:
            return []

        pad = self.timing_pad_seconds if pad_seconds is None else pad_seconds
        valid = [
            cue
            for cue in cues
            if len((cue.ocr_text or cue.source_text or "").strip()) > 1
            and (cue.ocr_text or cue.source_text or "").strip() not in NOISE_FILTER
        ]

        explicit_visual = [
            cue
            for cue in valid
            if cue.ocr_start is not None
            and cue.ocr_end is not None
            and (cue.ocr_confidence is None or cue.ocr_confidence >= self.ocr_min_confidence)
        ]
        legacy_ocr = [
            cue
            for cue in valid
            if cue.ocr_start is None
            and cue.ocr_end is None
            and cue.ocr_confidence is not None
            and cue.ocr_confidence >= self.ocr_min_confidence
        ]

        if explicit_visual:
            selected = explicit_visual
            self._metrics["timing_source_explicit_ocr"] = len(selected)
            self._metrics["timing_source_legacy_ocr"] = 0
            self._metrics["timing_source_fallback"] = 0
            intervals = [
                (
                    max(0.0, float(cue.ocr_start) - pad),
                    float(cue.ocr_end) + pad,
                )
                for cue in selected
                if cue.ocr_start is not None and cue.ocr_end is not None
            ]
        elif legacy_ocr:
            selected = legacy_ocr
            self._metrics["timing_source_explicit_ocr"] = 0
            self._metrics["timing_source_legacy_ocr"] = len(selected)
            self._metrics["timing_source_fallback"] = 0
            intervals = [
                (max(0.0, float(cue.start) - pad), float(cue.end) + pad)
                for cue in selected
            ]
        else:
            # Compatibility for imported SRT / pre-Phase-4 projects.  This path is not
            # preferred for hard-sub cleanup and should disappear after a fresh Analyze.
            selected = valid
            self._metrics["timing_source_explicit_ocr"] = 0
            self._metrics["timing_source_legacy_ocr"] = 0
            self._metrics["timing_source_fallback"] = len(selected)
            intervals = [
                (max(0.0, float(cue.start) - pad), float(cue.end) + pad)
                for cue in selected
            ]

        intervals.sort(key=lambda item: item[0])
        merged: list[tuple[float, float]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def _fast_inpaint_roi(self, roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(
            roi,
            mask,
            inpaintRadius=self.inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )

    @staticmethod
    def _local_ring(mask: np.ndarray) -> np.ndarray:
        outer = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        inner = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        return cv2.subtract(outer, inner)

    def _align_temporal_candidate(
        self,
        current_roi: np.ndarray,
        donor_roi: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray | None, float | None, float | None]:
        if current_roi.shape != donor_roi.shape:
            return None, None, None

        current_gray = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
        donor_gray = cv2.cvtColor(donor_roi, cv2.COLOR_BGR2GRAY)
        guard = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        valid = cv2.bitwise_not(guard)
        if int(np.sum(valid > 0)) < max(100, int(valid.size * 0.35)):
            return None, None, None

        initial_diff = cv2.absdiff(current_gray, donor_gray)
        initial_score = float(np.mean(initial_diff[valid > 0]))
        if initial_score > self.scene_cut_threshold * 2.0:
            self._metrics["temporal_scene_rejects"] = int(
                self._metrics.get("temporal_scene_rejects", 0)
            ) + 1
            return None, None, None

        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            35,
            1e-4,
        )
        try:
            cv2.findTransformECC(
                current_gray,
                donor_gray,
                warp,
                cv2.MOTION_TRANSLATION,
                criteria,
                inputMask=valid,
                gaussFiltSize=3,
            )
            aligned = cv2.warpAffine(
                donor_roi,
                warp,
                (current_roi.shape[1], current_roi.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
        except cv2.error:
            self._metrics["temporal_alignment_failures"] = int(
                self._metrics.get("temporal_alignment_failures", 0)
            ) + 1
            return None, None, None

        aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        aligned_diff = cv2.absdiff(current_gray, aligned_gray)
        global_score = float(np.mean(aligned_diff[valid > 0]))
        if global_score > self.scene_cut_threshold:
            self._metrics["temporal_scene_rejects"] = int(
                self._metrics.get("temporal_scene_rejects", 0)
            ) + 1
            return None, global_score, None

        ring = self._local_ring(mask)
        ring_pixels = ring > 0
        local_score = (
            float(np.mean(aligned_diff[ring_pixels]))
            if int(np.sum(ring_pixels)) >= 20
            else global_score
        )
        if local_score > self.temporal_local_score_threshold:
            self._metrics["temporal_local_motion_rejects"] = int(
                self._metrics.get("temporal_local_motion_rejects", 0)
            ) + 1
            return None, global_score, local_score

        # Donor local color/luma validation
        if int(np.sum(ring_pixels)) >= 20:
            curr_luma = current_gray[ring_pixels]
            donor_luma = aligned_gray[ring_pixels]
            luma_diff = abs(float(np.mean(curr_luma)) - float(np.mean(donor_luma)))
            if luma_diff > 16.0:
                self._metrics["donor_local_color_rejects"] = int(
                    self._metrics.get("donor_local_color_rejects", 0)
                ) + 1
                return None, global_score, local_score

        return aligned, global_score, local_score

    def _refine_mask_from_donor(
        self,
        current_roi: np.ndarray,
        aligned_donor: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> tuple[np.ndarray, bool, float]:
        """Keep only candidate glyph pixels that actually differ from a clean donor frame."""
        diff = cv2.absdiff(current_roi, aligned_donor)
        diff_gray = np.max(diff, axis=2).astype(np.uint8)
        changed = (diff_gray >= self.temporal_difference_threshold).astype(np.uint8) * 255
        changed = cv2.dilate(
            changed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        refined = cv2.bitwise_and(candidate_mask, changed)
        original_pixels = max(1, int(np.sum(candidate_mask > 0)))
        ratio = float(np.sum(refined > 0)) / original_pixels

        # A donor that explains almost none of the candidate mask is not trustworthy.
        # Keep the conservative source mask and let the normal quality guard/fallback work.
        if ratio < 0.30:
            self._metrics["temporal_mask_refine_rejects"] = int(
                self._metrics.get("temporal_mask_refine_rejects", 0)
            ) + 1
            return candidate_mask, False, ratio

        refined = cv2.dilate(
            refined,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        refined = cv2.bitwise_and(refined, candidate_mask)
        self._metrics["temporal_mask_refined"] = int(
            self._metrics.get("temporal_mask_refined", 0)
        ) + 1
        return refined, True, ratio

    def _quality_inpaint_roi(
        self,
        roi: np.ndarray,
        mask: np.ndarray,
        donors: list[np.ndarray],
    ) -> tuple[np.ndarray, bool, float | None]:
        best: np.ndarray | None = None
        best_global_score: float | None = None
        best_local_score: float | None = None

        for donor in donors:
            aligned, global_score, local_score = self._align_temporal_candidate(roi, donor, mask)
            if aligned is None or global_score is None or local_score is None:
                continue
            candidate_key = (local_score, global_score)
            best_key = (
                best_local_score if best_local_score is not None else float("inf"),
                best_global_score if best_global_score is not None else float("inf"),
            )
            if best is None or candidate_key < best_key:
                best = aligned
                best_global_score = global_score
                best_local_score = local_score

        if best is None or best_local_score is None:
            return self._fast_inpaint_roi(roi, mask), False, None

        refined_mask, _, _ = self._refine_mask_from_donor(roi, best, mask)
        fast = self._fast_inpaint_roi(roi, refined_mask)

        confidence = max(
            0.0,
            min(1.0, 1.0 - best_local_score / self.temporal_local_score_threshold),
        )
        # A clean aligned frame is a much stronger background source than spatial
        # inpainting.  Keep a small Telea contribution only to hide donor seams.
        temporal_weight = 0.88 + 0.09 * confidence
        hybrid = cv2.addWeighted(
            best,
            temporal_weight,
            fast,
            1.0 - temporal_weight,
            0.0,
        )

        alpha = cv2.GaussianBlur(
            refined_mask.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=0.65,
            sigmaY=0.65,
        )
        alpha = np.clip(alpha * 1.08, 0.0, 1.0)[..., None]
        result = np.clip(
            roi.astype(np.float32) * (1.0 - alpha)
            + hybrid.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

        # Black-blob guard: inspect cleaned mask area vs surroundings
        cleaned_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        mask_pixels = refined_mask > 0
        ring = self._local_ring(refined_mask)
        surround_pixels = ring > 0
        if int(np.sum(mask_pixels)) >= 15 and int(np.sum(surround_pixels)) >= 15:
            cleaned_mean = float(np.mean(cleaned_gray[mask_pixels]))
            surround_mean = float(np.mean(cleaned_gray[surround_pixels]))
            black_ratio_cleaned = float(np.mean(cleaned_gray[mask_pixels] < 25))
            black_ratio_surround = float(np.mean(cleaned_gray[surround_pixels] < 25))
            if (surround_mean - cleaned_mean > 30.0) or (black_ratio_cleaned > 0.35 and black_ratio_surround < 0.10):
                # Fallback to Telea to prevent black blobs
                result = fast
                self._metrics["black_blob_guard_rejects"] = int(
                    self._metrics.get("black_blob_guard_rejects", 0)
                ) + 1

        return result, True, best_local_score

    def clean_frame(
        self,
        frame: np.ndarray,
        mode: str = "inpaint",
        is_subtitle_active: bool = True,
        temporal_donors: list[np.ndarray] | None = None,
        ocr_regions: list[OCRRegion] | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Clean one frame while leaving every pixel outside the subtitle mask untouched."""
        if mode in {"none", "off"} or not is_subtitle_active:
            return frame, False

        h, w = frame.shape[:2]

        # 1. Primary Path: Geometry-Guided Cleanup (if regions are available)
        if self.geometry_enabled and ocr_regions:
            geo_mask, geo_stats = self.extract_geometry_mask(frame, ocr_regions, temporal_donors)
            if geo_mask is not None:
                self._metrics["geometry_frames"] = int(self._metrics.get("geometry_frames", 0)) + 1
                self._metrics["geometry_mask_success"] = int(self._metrics.get("geometry_mask_success", 0)) + 1
                self._metrics["outline_pixels_added"] = (
                    int(self._metrics.get("outline_pixels_added", 0))
                    + geo_stats.get("outline_pixels_added", 0)
                )
                if geo_stats.get("region_coverages"):
                    all_covs = list(self._metrics.get("_region_coverages_list", []))
                    all_covs.extend(geo_stats["region_coverages"])
                    self._metrics["_region_coverages_list"] = all_covs
                    self._metrics["max_region_mask_coverage"] = round(float(max(all_covs)), 3)
                    self._metrics["avg_region_mask_coverage"] = round(float(np.mean(all_covs)), 3)

                if mode == "cover":
                    cleaned = frame.copy()
                    for bx1, by1, bx2, by2 in geo_stats.get("bboxes", []):
                        overlay = cleaned.copy()
                        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                        cleaned = cv2.addWeighted(overlay, 0.45, cleaned, 0.55, 0)
                    return cleaned, True

                pts = cv2.findNonZero(geo_mask)
                if pts is None:
                    return frame, False

                bx, by, bw, bh = cv2.boundingRect(pts)
                pad_box = 16
                bx1 = max(0, bx - pad_box)
                by1 = max(0, by - pad_box)
                bx2 = min(w, bx + bw + pad_box)
                by2 = min(h, by + bh + pad_box)

                patch = frame[by1:by2, bx1:bx2]
                mask_patch = geo_mask[by1:by2, bx1:bx2]

                cleaned = frame.copy()
                if mode in {"quality", "auto"} and temporal_donors:
                    donor_patches = [
                        d[by1:by2, bx1:bx2] for d in temporal_donors if d.shape == frame.shape
                    ]
                    cleaned_patch, used_temporal, t_score = self._quality_inpaint_roi(
                        patch, mask_patch, donor_patches
                    )
                    if used_temporal:
                        self._metrics["geometry_temporal_frames"] = (
                            int(self._metrics.get("geometry_temporal_frames", 0)) + 1
                        )
                        self._metrics["temporal_frames"] = (
                            int(self._metrics.get("temporal_frames", 0)) + 1
                        )
                        if t_score is not None:
                            self._metrics["temporal_score_sum"] = (
                                float(self._metrics.get("temporal_score_sum", 0.0)) + t_score
                            )
                    else:
                        self._metrics["geometry_telea_frames"] = (
                            int(self._metrics.get("geometry_telea_frames", 0)) + 1
                        )
                        self._metrics["fallback_inpaint_frames"] = (
                            int(self._metrics.get("fallback_inpaint_frames", 0)) + 1
                        )
                else:
                    cleaned_patch = self._fast_inpaint_roi(patch, mask_patch)
                    self._metrics["geometry_telea_frames"] = (
                        int(self._metrics.get("geometry_telea_frames", 0)) + 1
                    )
                    if mode in {"quality", "auto"}:
                        self._metrics["fallback_inpaint_frames"] = (
                            int(self._metrics.get("fallback_inpaint_frames", 0)) + 1
                        )

                # Cleanup verification & second-pass refinement inside patch
                res_gray = cv2.cvtColor(cleaned_patch, cv2.COLOR_BGR2GRAY)
                res_local_mean = cv2.GaussianBlur(res_gray, (0, 0), sigmaX=2.0, sigmaY=2.0)
                res_edge = cv2.morphologyEx(
                    res_gray,
                    cv2.MORPH_GRADIENT,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                )
                res_dark = (res_gray < (res_local_mean.astype(float) - 10.0)) & (res_edge >= 12)
                res_bright = (res_gray >= 148) & (res_edge >= 12)
                residual_mask = (res_dark | res_bright).astype(np.uint8) * 255
                residual_mask = cv2.bitwise_and(residual_mask, mask_patch)

                if int(np.sum(residual_mask > 0)) >= 15:
                    residual_mask_dilated = cv2.dilate(
                        residual_mask,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                        iterations=1,
                    )
                    cleaned_patch = cv2.inpaint(
                        cleaned_patch,
                        residual_mask_dilated,
                        inpaintRadius=self.inpaint_radius,
                        flags=cv2.INPAINT_TELEA,
                    )
                    mask_patch = cv2.bitwise_or(mask_patch, residual_mask_dilated)
                    self._metrics["geometry_second_pass_frames"] = (
                        int(self._metrics.get("geometry_second_pass_frames", 0)) + 1
                    )

                # Exact outside mask preservation: bit-identical background outside mask
                mask_bool = mask_patch > 0
                patch_target = frame[by1:by2, bx1:bx2].copy()
                patch_target[mask_bool] = cleaned_patch[mask_bool]
                cleaned[by1:by2, bx1:bx2] = patch_target
                return cleaned, True
            else:
                self._metrics["geometry_mask_fail"] = int(self._metrics.get("geometry_mask_fail", 0)) + 1

        # 2. Fallback Path: V3 Heuristic ROI detection
        if is_subtitle_active:
            self._metrics["geometry_missing_fallback_frames"] = (
                int(self._metrics.get("geometry_missing_fallback_frames", 0)) + 1
            )

        x1, y1, x2, y2 = self._roi_bounds(frame)
        sub_roi = frame[y1:y2, x1:x2]
        mask_roi = self.extract_text_mask(sub_roi)
        if mask_roi is None:
            return frame, False

        if mode == "cover":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, y1), (frame.shape[1], y2), (0, 0, 0), -1)
            cleaned = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
            return cleaned, True

        cleaned = frame.copy()
        if mode in {"quality", "auto"} and temporal_donors:
            donor_rois = [
                donor[y1:y2, x1:x2]
                for donor in temporal_donors
                if donor.shape == frame.shape
            ]
            cleaned_roi, used_temporal, temporal_score = self._quality_inpaint_roi(
                sub_roi,
                mask_roi,
                donor_rois,
            )
            if used_temporal:
                self._metrics["temporal_frames"] = int(
                    self._metrics.get("temporal_frames", 0)
                ) + 1
                if temporal_score is not None:
                    self._metrics["temporal_score_sum"] = float(
                        self._metrics.get("temporal_score_sum", 0.0)
                    ) + temporal_score
            else:
                self._metrics["fallback_inpaint_frames"] = int(
                    self._metrics.get("fallback_inpaint_frames", 0)
                ) + 1
        else:
            cleaned_roi = self._fast_inpaint_roi(sub_roi, mask_roi)
            if mode in {"quality", "auto"}:
                self._metrics["fallback_inpaint_frames"] = int(
                    self._metrics.get("fallback_inpaint_frames", 0)
                ) + 1

        cleaned[y1:y2, x1:x2] = cleaned_roi
        return cleaned, True

    def _read_frame_at(self, cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
        if frame_idx < 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None

    @staticmethod
    def _frame_in_ranges(frame_idx: int, ranges: list[tuple[int, int]]) -> bool:
        return any(start <= frame_idx <= end for start, end in ranges)

    def _find_clean_donor(
        self,
        cap: cv2.VideoCapture,
        anchor_idx: int,
        direction: int,
        max_distance: int,
        blocked_ranges: list[tuple[int, int]] | None = None,
    ) -> tuple[int, np.ndarray] | None:
        if max_distance <= 0:
            return None
        blocked = blocked_ranges or []
        for distance in range(1, max_distance + 1):
            idx = anchor_idx + direction * distance
            if idx < 0:
                break
            if self._frame_in_ranges(idx, blocked):
                continue
            frame = self._read_frame_at(cap, idx)
            if frame is None:
                if direction > 0:
                    break
                continue
            x1, y1, x2, y2 = self._roi_bounds(frame)
            if self.extract_text_mask(frame[y1:y2, x1:x2]) is None:
                return idx, frame
        return None

    def _open_lossless_writer(
        self,
        output_path: Path,
        fps: float,
        width: int,
        height: int,
    ) -> subprocess.Popen[bytes]:
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.stdin is None:
            raise RuntimeError("Failed to open FFmpeg raw-video stdin")
        return proc

    @staticmethod
    def _write_lossless_frame(proc: subprocess.Popen[bytes], frame: np.ndarray) -> None:
        if proc.stdin is None:
            raise RuntimeError("FFmpeg writer stdin is closed")
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    @staticmethod
    def _close_lossless_writer(proc: subprocess.Popen[bytes]) -> None:
        if proc.stdin is not None:
            proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"Lossless intermediate encode failed: {stderr[-2000:]}")

    def clean_video(
        self,
        source_path: Path,
        output_path: Path,
        cues: list[SubtitleCue] | None = None,
        mode: str = "auto",
        lossless_intermediate: bool = True,
    ) -> dict[str, Any]:
        """Clean video using OCR visual intervals and guarded temporal reconstruction."""
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

        self._metrics = {}
        active_intervals = self.build_active_intervals(cues)

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {source_path}")
        donor_cap = cv2.VideoCapture(str(source_path)) if mode in {"quality", "auto"} else None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        active_frame_ranges = [
            (
                max(0, int(np.floor(start * fps))),
                min(max(0, total_frames - 1), int(np.ceil(end * fps))),
            )
            for start, end in active_intervals
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        lossless_proc: subprocess.Popen[bytes] | None = None
        cv_writer: cv2.VideoWriter | None = None
        if lossless_intermediate:
            lossless_proc = self._open_lossless_writer(output_path, fps, width, height)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            cv_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not cv_writer.isOpened():
                cap.release()
                if donor_cap is not None:
                    donor_cap.release()
                raise RuntimeError(f"Failed to open intermediate video writer: {output_path}")

        def write_frame(frame: np.ndarray) -> None:
            if lossless_proc is not None:
                self._write_lossless_frame(lossless_proc, frame)
            elif cv_writer is not None:
                cv_writer.write(frame)

        cues_list = cues or []
        geometry_cues = len([c for c in cues_list if c.ocr_regions])
        geometry_regions = sum(len(c.ocr_regions) for c in cues_list)
        self._metrics["geometry_cues"] = geometry_cues
        self._metrics["geometry_regions"] = geometry_regions

        frames_processed = 0
        frames_inpainted = 0
        frames_bypassed = 0
        interval_idx = 0
        current_donors: list[tuple[int, np.ndarray]] = []
        loaded_donor_interval = -1
        t0 = time.time()
        frame_idx = 0
        pad = self.timing_pad_seconds

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps
                frames_processed += 1

                while interval_idx < len(active_intervals) and timestamp > active_intervals[interval_idx][1]:
                    interval_idx += 1

                is_active = (
                    interval_idx < len(active_intervals)
                    and active_intervals[interval_idx][0] <= timestamp <= active_intervals[interval_idx][1]
                )

                if not is_active:
                    write_frame(frame)
                    frames_bypassed += 1
                    frame_idx += 1
                    continue

                # Find active OCR regions for this timestamp
                active_regions: list[OCRRegion] = []
                pad_eff = max(0.12, pad)
                for cue in cues_list:
                    c_start = min(float(cue.ocr_start), float(cue.start)) if cue.ocr_start is not None else float(cue.start)
                    c_end = max(float(cue.ocr_end), float(cue.end)) if cue.ocr_end is not None else float(cue.end)
                    if (c_start - pad_eff) <= timestamp <= (c_end + pad_eff):
                        for r in cue.ocr_regions:
                            r_pts = r.get("points") if isinstance(r, dict) else getattr(r, "points", [])
                            if r_pts:
                                active_regions.append(r)

                temporal_donors: list[np.ndarray] = []
                if donor_cap is not None:
                    if loaded_donor_interval != interval_idx:
                        start, end = active_intervals[interval_idx]
                        start_idx = max(0, int(np.floor(start * fps)))
                        end_idx = min(max(0, total_frames - 1), int(np.ceil(end * fps)))
                        prev_donor = self._find_clean_donor(
                            donor_cap,
                            start_idx,
                            -1,
                            self.temporal_max_distance_frames,
                            blocked_ranges=active_frame_ranges,
                        )
                        next_donor = self._find_clean_donor(
                            donor_cap,
                            end_idx,
                            1,
                            self.temporal_max_distance_frames,
                            blocked_ranges=active_frame_ranges,
                        )
                        current_donors = [
                            item for item in [prev_donor, next_donor] if item is not None
                        ]
                        loaded_donor_interval = interval_idx

                    temporal_donors = [
                        donor_frame
                        for donor_idx, donor_frame in current_donors
                        if abs(donor_idx - frame_idx) <= self.temporal_max_distance_frames
                    ]

                cleaned_frame, was_cleaned = self.clean_frame(
                    frame,
                    mode=mode,
                    is_subtitle_active=True,
                    temporal_donors=temporal_donors,
                    ocr_regions=active_regions if active_regions else None,
                )
                if was_cleaned:
                    frames_inpainted += 1
                else:
                    frames_bypassed += 1
                write_frame(cleaned_frame)
                frame_idx += 1
        finally:
            cap.release()
            if donor_cap is not None:
                donor_cap.release()
            if cv_writer is not None:
                cv_writer.release()
            if lossless_proc is not None:
                self._close_lossless_writer(lossless_proc)

        total_time = time.time() - t0
        temporal_frames = int(self._metrics.get("temporal_frames", 0))
        temporal_score_sum = float(self._metrics.get("temporal_score_sum", 0.0))
        metrics = {
            "mode": mode,
            "frames_processed": frames_processed,
            "frames_inpainted": frames_inpainted,
            "frames_bypassed": frames_bypassed,
            "inpaint_rate": round(frames_inpainted / max(1, frames_processed) * 100, 1),
            "bypass_rate": round(frames_bypassed / max(1, frames_processed) * 100, 1),
            "active_intervals": len(active_intervals),
            "timing_source_explicit_ocr": int(self._metrics.get("timing_source_explicit_ocr", 0)),
            "timing_source_legacy_ocr": int(self._metrics.get("timing_source_legacy_ocr", 0)),
            "timing_source_fallback": int(self._metrics.get("timing_source_fallback", 0)),
            "geometry_cues": geometry_cues,
            "geometry_regions": geometry_regions,
            "geometry_frames": int(self._metrics.get("geometry_frames", 0)),
            "geometry_missing_fallback_frames": int(
                self._metrics.get("geometry_missing_fallback_frames", 0)
            ),
            "geometry_mask_success": int(self._metrics.get("geometry_mask_success", 0)),
            "geometry_mask_fail": int(self._metrics.get("geometry_mask_fail", 0)),
            "geometry_temporal_frames": int(self._metrics.get("geometry_temporal_frames", 0)),
            "geometry_telea_frames": int(self._metrics.get("geometry_telea_frames", 0)),
            "avg_region_mask_coverage": float(self._metrics.get("avg_region_mask_coverage", 0.0)),
            "max_region_mask_coverage": float(self._metrics.get("max_region_mask_coverage", 0.0)),
            "outline_pixels_added": int(self._metrics.get("outline_pixels_added", 0)),
            "temporal_frames": temporal_frames,
            "fallback_inpaint_frames": int(self._metrics.get("fallback_inpaint_frames", 0)),
            "temporal_scene_rejects": int(self._metrics.get("temporal_scene_rejects", 0)),
            "temporal_local_motion_rejects": int(
                self._metrics.get("temporal_local_motion_rejects", 0)
            ),
            "temporal_alignment_failures": int(self._metrics.get("temporal_alignment_failures", 0)),
            "temporal_mask_refined": int(self._metrics.get("temporal_mask_refined", 0)),
            "temporal_mask_refine_rejects": int(
                self._metrics.get("temporal_mask_refine_rejects", 0)
            ),
            "avg_temporal_score": round(temporal_score_sum / max(1, temporal_frames), 2),
            "mask_rejected_coverage": int(self._metrics.get("mask_rejected_coverage", 0)),
            "mask_rejected_no_components": int(
                self._metrics.get("mask_rejected_no_components", 0)
            ),
            "mask_rejected_no_text_row": int(self._metrics.get("mask_rejected_no_text_row", 0)),
            "lossless_intermediate": lossless_intermediate,
            "clean_runtime": round(total_time, 2),
            "fps_speed": round(frames_processed / max(0.01, total_time), 1),
            "output_path": str(output_path),
        }
        logger.info("HardSub Cleaner completed: %s", metrics)
        return metrics
