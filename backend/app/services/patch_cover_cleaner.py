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
    get_final_vi_text,
)

logger = logging.getLogger(__name__)


class PatchCoverCleaner:
    """SourceSubtitleCoverPlate and fast background patch-cover cleaner for hard subtitles.

    Target visual style:
    - Base video -> compact dark/black blurred subtitle cover covering Chinese source text
    - Vietnamese white subtitle with thin black outline rendered directly on top.
    - Soft feathered rounded-rectangle plate (semi-transparent black tint ~50-60% + backdrop blur).
    - Lifecycle strictly matches active subtitle cue (no cover during subtitle gaps).
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

    def create_feathered_mask(
        self,
        height: int,
        width: int,
        polygons: list[list[list[float]]],
    ) -> np.ndarray:
        """Create a feathered binary mask from normalized polygon coordinates."""
        mask = np.zeros((height, width), dtype=np.uint8)
        if not polygons:
            return mask.astype(np.float32)

        pad = self.config.padding_px
        feather = self.config.feather_px

        for poly in polygons:
            pts = np.array([[int(p[0] * width), int(p[1] * height)] for p in poly], dtype=np.int32)
            if len(pts) < 3:
                continue
            if pad > 0:
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(width - x, w + 2 * pad)
                h = min(height - y, h + 2 * pad)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            else:
                cv2.fillPoly(mask, [pts], 255)

        if feather > 0:
            k = max(1, feather * 2 + 1)
            mask = cv2.GaussianBlur(mask, (k, k), feather)

        # Safety cutoff: enforce 0 alpha in upper screen
        cutoff_y = int(height * getattr(self.config, "face_safety_y_cutoff", 0.68))
        mask[:cutoff_y, :] = 0

        return mask.astype(np.float32) / 255.0

    def extract_active_intervals(
        self,
        cues: list[SubtitleCue],
        fps: float,
    ) -> list[dict[str, Any]]:
        """Extract active subtitle intervals with tight glyph polygons."""
        min_conf = self.config.min_ocr_confidence
        gap_fill_sec = self.config.temporal_gap_fill_frames / max(1.0, fps)
        persistence_sec = self.config.mask_persistence_frames / max(1.0, fps)

        raw_intervals: list[dict[str, Any]] = []

        for cue in cues:
            start_t = cue.ocr_start if cue.ocr_start is not None else cue.start
            end_t = cue.ocr_end if cue.ocr_end is not None else cue.end

            if end_t <= start_t:
                continue

            end_t += persistence_sec

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

        # Temporal gap bridging
        merged: list[dict[str, Any]] = []
        for interval in raw_intervals:
            if not merged:
                merged.append(interval)
                continue

            prev = merged[-1]
            if interval["start"] <= (prev["end"] + gap_fill_sec):
                prev["end"] = max(prev["end"], interval["end"])
                prev["polygons"].extend(interval["polygons"])
            else:
                merged.append(interval)

        return merged

    def build_render_cue_contexts(
        self,
        cues: list[SubtitleCue],
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        """Build exact SourceSubtitleCoverPlate render contexts for each active subtitle cue."""
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
            for cid in getattr(cue, "source_cue_ids", []) or [getattr(cue, "id", None)]:
                if not cid:
                    continue
                orig = cues_by_id.get(cid)
                if not orig:
                    continue
                for r in getattr(orig, "ocr_regions", []) or []:
                    pts = getattr(r, "points", []) or []
                    if len(pts) >= 3:
                        ys = [p[1] for p in pts if len(p) >= 2]
                        if ys and min(ys) >= 0.80 and max(ys) <= 0.98 and (max(ys) - min(ys)) <= 0.14:
                            y_pts.extend(ys)
                for ev in getattr(orig, "ocr_evidence", []) or []:
                    for r in getattr(ev, "regions", []) or []:
                        pts = getattr(r, "points", []) or []
                        if len(pts) >= 3:
                            ys = [p[1] for p in pts if len(p) >= 2]
                            if ys and min(ys) >= 0.80 and max(ys) <= 0.98 and (max(ys) - min(ys)) <= 0.14:
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
            if getattr(cue, "suppression_status", None) in {"SUPPRESSED_FILLER", "SUPPRESSED_NONSEMANTIC_DIALOGUE"}:
                continue
            text = getattr(cue, "render_text", None) or getattr(cue, "translated_text", None) or getattr(cue, "source_text", "")
            lines = [l.strip() for l in str(text).replace(r"\N", "\n").split("\n") if l.strip()]
            if not lines:
                continue
            line_count = len(lines)
            max_line_chars = max(len(l) for l in lines)
            text_w = int(round(max_line_chars * (font_size * 0.54)))
            text_h = int(round(line_count * (font_size * 1.15)))

            y_pos = cue_y_centers[idx] if idx < len(cue_y_centers) else default_y_center
            vi_x1 = max(0, int(center_x - text_w / 2 - pad_x))
            vi_x2 = min(width, int(center_x + text_w / 2 + pad_x))
            vi_y1 = max(int(height * 0.68), int(y_pos - text_h / 2 - pad_y))
            vi_y2 = min(int(height * 0.98), int(y_pos + text_h / 2 + pad_y))

            chinese_polygons: list[list[list[float]]] = []
            source_cids = getattr(cue, "source_cue_ids", []) or [getattr(cue, "id", None)]
            for cid in source_cids:
                orig = cues_by_id.get(cid) or (cue if getattr(cue, "id", None) == cid else None)
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

            if chinese_polygons:
                # Filter to bottom dialogue subtitle band polygons (y >= 0.80) to exclude shirt/badge text
                sub_band_polys = [p for p in chinese_polygons if any(pt[1] >= 0.80 for pt in p)]
                active_polys = sub_band_polys if sub_band_polys else chinese_polygons

                zh_xs = [int(p[0] * width) for poly in active_polys for p in poly]
                zh_ys = [int(p[1] * height) for poly in active_polys for p in poly]
                zh_x1, zh_x2 = min(zh_xs), max(zh_xs)
                zh_y1, zh_y2 = min(zh_ys), max(zh_ys)

                cover_x1 = max(0, min(vi_x1, zh_x1 - 20))
                cover_x2 = min(width, max(vi_x2, zh_x2 + 20))
                cover_y1 = max(int(height * 0.68), min(vi_y1, zh_y1 - 10))
                cover_y2 = min(int(height * 0.98), max(vi_y2, zh_y2 + 10))
            else:
                src_cues = [cues_by_id.get(cid) for cid in getattr(cue, "source_cue_ids", [])]
                src_texts = [getattr(c, "source_text", "") for c in src_cues if c]
                src_chars = max([len(t) for t in src_texts] + [len(getattr(cue, "source_text", ""))])
                est_zh_w = int(round(src_chars * (font_size * 0.90)))
                zh_x1 = max(0, int(center_x - est_zh_w / 2 - pad_x))
                zh_x2 = min(width, int(center_x + est_zh_w / 2 + pad_x))

                cover_x1 = max(0, min(vi_x1, zh_x1))
                cover_x2 = min(width, max(vi_x2, zh_x2))
                cover_y1, cover_y2 = vi_y1, vi_y2

            is_two_line = line_count > 1 or (chinese_polygons and (zh_y2 - zh_y1) > 65)
            max_plate_h = int(height * (0.145 if is_two_line else 0.115))
            if (cover_y2 - cover_y1) > max_plate_h:
                cover_y1 = max(int(height * 0.68), int(y_pos - max_plate_h / 2))
                cover_y2 = min(int(height * 0.98), int(y_pos + max_plate_h / 2))

            # Canonical OCR-aware cover interval calculation
            cover_start, cover_end, source_start, source_end = self.get_cover_interval(cue, cues)

            src_text_check = (getattr(cue, "source_text", "") or "").strip()
            if src_text_check in ["AL", "10:50"]:
                continue

            contexts.append({
                "start": cover_start,
                "end": cover_end,
                "vi_start": round(cue.start, 3),
                "vi_end": round(cue.end, 3),
                "source_start": source_start,
                "source_end": source_end,
                "cue_id": getattr(cue, "render_id", str(idx)),
                "source_cue_ids": getattr(cue, "source_cue_ids", []) or [getattr(cue, "id", None)],
                "bbox": (cover_x1, cover_y1, cover_x2, cover_y2),
                "chinese_polygons": chinese_polygons,
                "has_vi": True,
                "reason": "TRANSLATED_DIALOGUE",
            })

        # 2. Process Suppressed Source Dialogue Cues (Fillers / Interjections without VI)
        covered_source_cids = set()
        for ctx in contexts:
            for cid in ctx.get("source_cue_ids", []):
                if cid:
                    covered_source_cids.add(cid)

        for orig in cues:
            if orig.id in covered_source_cids:
                continue
            src_text = (getattr(orig, "source_text", "") or "").strip()
            # Non-speech non-dialogue tokens
            if not src_text or src_text in ["AL", "10:50"]:
                continue

            s_starts = [orig.start]
            s_ends = [orig.end]
            if getattr(orig, "ocr_start", None) is not None and 0.0 < orig.ocr_start:
                s_starts.append(orig.ocr_start)
            if getattr(orig, "ocr_end", None) is not None and 0.0 < orig.ocr_end:
                s_ends.append(orig.ocr_end)
            for ev in getattr(orig, "ocr_evidence", []) or []:
                if getattr(ev, "start", None) is not None and 0.0 < ev.start:
                    s_starts.append(ev.start)
                if getattr(ev, "end", None) is not None and 0.0 < ev.end:
                    s_ends.append(ev.end)

            source_start = min(s_starts)
            source_end = max(s_ends)
            pre_roll_s = 2.0 / 30.0
            post_roll_s = 2.0 / 30.0
            cover_start = max(0.0, source_start - pre_roll_s)
            cover_end = source_end + post_roll_s

            chinese_polygons = []
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

            if chinese_polygons:
                sub_band_polys = [p for p in chinese_polygons if any(pt[1] >= 0.80 for pt in p)]
                active_polys = sub_band_polys if sub_band_polys else chinese_polygons
                zh_xs = [int(p[0] * width) for poly in active_polys for p in poly]
                zh_ys = [int(p[1] * height) for poly in active_polys for p in poly]
                raw_x1, raw_x2 = min(zh_xs), max(zh_xs)
                raw_y1, raw_y2 = min(zh_ys), max(zh_ys)
                center_glyph_x = (raw_x1 + raw_x2) // 2
                center_glyph_y = (raw_y1 + raw_y2) // 2

                min_w = max(int(font_size * 2.2), 90)
                min_h = max(int(font_size * 1.2), 48)
                glyph_w = max(min_w, (raw_x2 - raw_x1) + 48)
                glyph_h = max(min_h, (raw_y2 - raw_y1) + 24)

                cover_x1 = max(0, center_glyph_x - glyph_w // 2)
                cover_x2 = min(width, center_glyph_x + glyph_w // 2)
                cover_y1 = max(int(height * 0.68), center_glyph_y - glyph_h // 2)
                cover_y2 = min(int(height * 0.98), center_glyph_y + glyph_h // 2)
            else:
                est_zh_w = int(round(len(src_text) * (font_size * 0.90)))
                cover_x1 = max(0, int(center_x - est_zh_w / 2 - pad_x))
                cover_x2 = min(width, int(center_x + est_zh_w / 2 + pad_x))
                cover_y1 = max(int(height * 0.68), int(default_y_center - font_size * 0.6 - pad_y))
                cover_y2 = min(int(height * 0.98), int(default_y_center + font_size * 0.6 + pad_y))

            contexts.append({
                "start": round(cover_start, 3),
                "end": round(cover_end, 3),
                "vi_start": round(orig.start, 3),
                "vi_end": round(orig.end, 3),
                "source_start": round(source_start, 3),
                "source_end": round(source_end, 3),
                "cue_id": f"suppressed_{orig.id[:8]}",
                "source_cue_ids": [orig.id],
                "bbox": (cover_x1, cover_y1, cover_x2, cover_y2),
                "chinese_polygons": chinese_polygons,
                "has_vi": False,
                "reason": "SUPPRESSED_FILLER",
            })

        # Sort all contexts by start time
        contexts.sort(key=lambda x: x["start"])

        # Clean handoffs between consecutive covers
        for i in range(1, len(contexts)):
            prev = contexts[i - 1]
            curr = contexts[i]
            if prev["end"] > curr["start"]:
                min_t = prev["vi_end"] if prev.get("has_vi") else prev["start"]
                max_t = curr["vi_start"] if curr.get("has_vi") else curr["end"]
                handoff_t = round(max(min_t, min(max_t, (prev["end"] + curr["start"]) / 2.0)), 3)
                prev["end"] = handoff_t
                curr["start"] = max(handoff_t, curr["start"])
            elif 0.0 < (curr["start"] - prev["end"]) <= 0.08:
                handoff_t = round((prev["end"] + curr["start"]) / 2.0, 3)
                prev["end"] = handoff_t
                curr["start"] = handoff_t

        return contexts

    @staticmethod
    def get_cover_interval(
        cue: SubtitleCue,
        all_cues: list[SubtitleCue],
        pre_roll_s: float = 2.0 / 30.0,
        post_roll_s: float = 2.0 / 30.0,
    ) -> tuple[float, float, float, float]:
        """Canonical accessor: returns (cover_start, cover_end, source_start, source_end)."""
        source_cids = set(getattr(cue, "source_cue_ids", []) or [getattr(cue, "id", None)])
        mapped_origs = []
        for orig in all_cues:
            if orig.id in source_cids or (orig.start < cue.end and orig.end > cue.start):
                mapped_origs.append(orig)

        s_starts = [cue.start]
        s_ends = [cue.end]

        for orig in mapped_origs:
            s_starts.append(orig.start)
            s_ends.append(orig.end)

            if getattr(orig, "ocr_start", None) is not None and 0.0 < orig.ocr_start:
                if 0.0 < (cue.start - orig.ocr_start) <= 2.0:
                    s_starts.append(orig.ocr_start)
            if getattr(orig, "ocr_end", None) is not None and 0.0 < orig.ocr_end:
                if 0.0 < (orig.ocr_end - cue.end) <= 2.0:
                    s_ends.append(orig.ocr_end)

            for ev in getattr(orig, "ocr_evidence", []) or []:
                ev_text = (getattr(ev, "text", "") or "").strip()
                if ev_text:
                    if getattr(ev, "start", None) is not None and 0.0 < ev.start:
                        if 0.0 < (cue.start - ev.start) <= 2.0:
                            s_starts.append(ev.start)
                    if getattr(ev, "end", None) is not None and 0.0 < ev.end:
                        if 0.0 < (ev.end - cue.end) <= 2.0:
                            s_ends.append(ev.end)

        source_start = min(s_starts)
        source_end = max(s_ends)

        if source_start < cue.start:
            cover_start = max(0.0, source_start - pre_roll_s)
        else:
            cover_start = cue.start

        if source_end > cue.end:
            cover_end = source_end + post_roll_s
        else:
            cover_end = cue.end

        return round(cover_start, 3), round(cover_end, 3), round(source_start, 3), round(source_end, 3)

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
        """Blend dark source subtitle cover plate and backdrop blur over video frame."""
        out = frame.copy()
        h, w = frame.shape[:2]

        if alpha_mask is not None and np.any(alpha_mask > 0.001):
            k_blur = 19
            patch = cv2.GaussianBlur(out, (k_blur, k_blur), 8.0)
            patch = (patch.astype(np.float32) * 0.20).astype(np.uint8)
            alpha_3d = np.repeat((alpha_mask * 0.95)[:, :, np.newaxis], 3, axis=2)
            out = (1.0 - alpha_3d) * out.astype(np.float32) + alpha_3d * patch.astype(np.float32)
            out = np.clip(out, 0, 255).astype(np.uint8)

        if vi_backing_mask is not None and np.any(vi_backing_mask > 0.001):
            blur_sigma_bg = 16.0
            k_bg = 35
            blurred_bg = cv2.GaussianBlur(out, (k_bg, k_bg), blur_sigma_bg)
            dark_tint_bg = 0.68  # 68% dark tint to cleanly obscure Chinese text
            blurred_bg = (blurred_bg.astype(np.float32) * (1.0 - dark_tint_bg)).astype(np.uint8)
            backing_opacity = 0.98
            alpha_bg_3d = np.repeat((vi_backing_mask * backing_opacity)[:, :, np.newaxis], 3, axis=2)
            out = (1.0 - alpha_bg_3d) * out.astype(np.float32) + alpha_bg_3d * blurred_bg.astype(np.float32)
            out = np.clip(out, 0, 255).astype(np.uint8)

        return out

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
            "-f",
            "rawvideo",
            "-vcodec",
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
            "ffv1" if output_path.suffix == ".mkv" else "libx264",
            *(["-preset", "ultrafast", "-crf", "0"] if output_path.suffix != ".mkv" else ["-level", "3"]),
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
            raise RuntimeError(f"Intermediate encode failed: {stderr[-2000:]}")

    def clean_video(
        self,
        source_path: Path,
        output_path: Path,
        cues: list[SubtitleCue],
        lossless_intermediate: bool = True,
    ) -> dict[str, Any]:
        """Process video and apply 1:1 cue-synchronized dark source cover plate and backdrop blur."""
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

        lossless_proc: subprocess.Popen[bytes] | None = None
        cv_writer: cv2.VideoWriter | None = None
        if lossless_intermediate or output_path.suffix == ".mkv":
            lossless_proc = self._open_lossless_writer(output_path, fps, width, height)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            cv_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        def write_frame(frame: np.ndarray) -> None:
            if lossless_proc is not None:
                self._write_lossless_frame(lossless_proc, frame)
            elif cv_writer is not None:
                cv_writer.write(frame)

        context_masks = []
        for ctx in contexts:
            bbox = ctx["bbox"]
            vi_mask = self.create_rounded_rect_mask(height, width, bbox, radius=10, feather=8)
            context_masks.append({
                "start": ctx["start"],
                "end": ctx["end"],
                "vi_mask": vi_mask,
                "alpha_mask": None,
            })

        patched_frames = 0
        frame_idx = 0

        while True:
            ret, cap_frame = cap.read()
            if not ret or cap_frame is None:
                break

            current_time = frame_idx / max(1.0, fps)

            active = [cm for cm in context_masks if cm["start"] <= current_time <= cm["end"]]
            if active:
                combined_vi_mask = np.maximum.reduce([cm["vi_mask"] for cm in active])
                out_frame = self.apply_patch_cover(
                    cap_frame,
                    vi_backing_mask=combined_vi_mask,
                )
                write_frame(out_frame)
                patched_frames += 1
            else:
                write_frame(cap_frame)

            frame_idx += 1

        cap.release()
        if lossless_proc is not None:
            self._close_lossless_writer(lossless_proc)
        elif cv_writer is not None:
            cv_writer.release()

        elapsed = time.time() - t_start
        self.last_metrics = {
            "cleanup_mode": "source_subtitle_cover_plate",
            "cleanup_time_s": round(elapsed, 3),
            "total_frames": total_frames,
            "patched_frames": patched_frames,
            "contexts_count": len(contexts),
            "output_path": str(output_path),
        }
        logger.info(
            "SourceSubtitleCoverPlate cleaned %d/%d frames in %.3fs",
            patched_frames,
            total_frames,
            elapsed,
        )
        return self.last_metrics
