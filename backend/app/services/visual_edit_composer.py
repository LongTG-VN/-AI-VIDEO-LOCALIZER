from __future__ import annotations

import logging
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models.project import (
    BlurConfig,
    OCRRegion,
    OverlayAnchor,
    OverlayConfig,
    Project,
    RenderOptions,
    SubtitleCue,
    VisualEditConfig,
    VisualEditMode,
)

logger = logging.getLogger(__name__)


def escape_filter_path(path: Path) -> str:
    """Escapes file paths for FFmpeg filter complexes on Windows."""
    value = path.resolve().as_posix().replace("'", r"\'")
    return value.replace(":", r"\:")


class VisualEditComposer:
    """Composes dynamic OCR-region blur, image overlays, and Vietnamese subtitles into final video."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe"):
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.last_metrics: dict[str, Any] = {}

    def extract_temporal_ocr_intervals(
        self,
        cues: list[SubtitleCue],
        min_confidence: float = 0.25,
        gap_fill_sec: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Extract and temporally bridge OCR subtitle intervals to prevent single-frame flashes."""
        raw_intervals: list[dict[str, Any]] = []

        for cue in cues:
            # Determine temporal bounds
            start_t = cue.ocr_start if cue.ocr_start is not None else cue.start
            end_t = cue.ocr_end if cue.ocr_end is not None else cue.end

            if end_t <= start_t:
                continue

            # Collect polygons
            polygons: list[list[list[float]]] = []
            if cue.ocr_regions:
                for region in cue.ocr_regions:
                    if region.points and (region.confidence is None or region.confidence >= min_confidence):
                        polygons.append(region.points)

            if not polygons and cue.ocr_evidence:
                for ev in cue.ocr_evidence:
                    for region in ev.regions:
                        if region.points and (region.confidence is None or region.confidence >= min_confidence):
                            polygons.append(region.points)

            # If no polygon points available, use default bottom subtitle band
            if not polygons:
                polygons = [[
                    [0.08, 0.76], [0.92, 0.76], [0.92, 0.94], [0.08, 0.94]
                ]]

            raw_intervals.append({
                "start": start_t,
                "end": end_t,
                "polygons": polygons,
                "cue_id": cue.id,
            })

        if not raw_intervals:
            return []

        # Sort by start time
        raw_intervals.sort(key=lambda x: x["start"])

        # Bridge temporal gaps shorter than gap_fill_sec
        bridged: list[dict[str, Any]] = [raw_intervals[0]]
        for cur in raw_intervals[1:]:
            prev = bridged[-1]
            gap = cur["start"] - prev["end"]
            if gap <= gap_fill_sec:
                # Merge interval
                prev["end"] = max(prev["end"], cur["end"])
                # Combine polygons
                prev["polygons"].extend(cur["polygons"])
            else:
                bridged.append(cur)

        return bridged

    def generate_dynamic_blur_mask_video(
        self,
        output_mask_path: Path,
        cues: list[SubtitleCue],
        width: int,
        height: int,
        fps: float,
        duration: float,
        blur_config: BlurConfig,
    ) -> tuple[Path, dict[str, Any]]:
        """Generate a lightweight grayscale video mask where subtitle regions are white (255) with feathered edges."""
        t0 = time.time()
        output_mask_path.parent.mkdir(parents=True, exist_ok=True)

        gap_fill_sec = blur_config.temporal_gap_fill_frames / max(1.0, fps)
        intervals = self.extract_temporal_ocr_intervals(
            cues,
            min_confidence=blur_config.min_ocr_confidence,
            gap_fill_sec=gap_fill_sec,
        )

        total_frames = int(math.ceil(duration * fps))
        if total_frames <= 0:
            total_frames = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_mask_path),
            fourcc,
            fps,
            (width, height),
            isColor=False,
        )

        padding = blur_config.padding_px
        feather = blur_config.feather_px
        k_feather = max(1, feather * 2 + 1)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, padding * 2 + 1), max(1, padding * 2 + 1)))

        masked_frames_count = 0

        for frame_idx in range(total_frames):
            cur_time = frame_idx / fps

            # Find active intervals
            active_polygons = []
            for item in intervals:
                if item["start"] <= cur_time <= item["end"]:
                    active_polygons.extend(item["polygons"])

            frame_mask = np.zeros((height, width), dtype=np.uint8)

            if active_polygons:
                masked_frames_count += 1
                for poly in active_polygons:
                    pts = np.array([
                        [int(p[0] * width), int(p[1] * height)]
                        for p in poly
                    ], dtype=np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(frame_mask, [pts], 255)

                if padding > 0:
                    frame_mask = cv2.dilate(frame_mask, dilate_kernel)

                if feather > 0:
                    frame_mask = cv2.GaussianBlur(frame_mask, (k_feather, k_feather), feather)

            writer.write(frame_mask)

        writer.release()
        t_cost = time.time() - t0

        metrics = {
            "mask_generation_time_s": round(t_cost, 3),
            "total_frames": total_frames,
            "masked_frames": masked_frames_count,
            "intervals_count": len(intervals),
            "mask_path": str(output_mask_path),
        }
        logger.info("Generated dynamic blur mask in %.3fs (%d frames)", t_cost, total_frames)
        return output_mask_path, metrics

    def compute_overlay_coordinates(
        self,
        overlay: OverlayConfig,
        video_width: int,
        video_height: int,
        cues: list[SubtitleCue] | None = None,
    ) -> tuple[str, str, int]:
        """Compute FFmpeg overlay x, y expressions and scaled width."""
        target_w = max(16, int(overlay.width * video_width))

        anchor = overlay.anchor
        if anchor == OverlayAnchor.TOP_LEFT:
            x_expr = "24"
            y_expr = "24"
        elif anchor == OverlayAnchor.TOP_RIGHT:
            x_expr = "W-w-24"
            y_expr = "24"
        elif anchor == OverlayAnchor.BOTTOM_LEFT:
            x_expr = "24"
            y_expr = "H-h-24"
        elif anchor == OverlayAnchor.BOTTOM_RIGHT:
            x_expr = "W-w-24"
            y_expr = "H-h-24"
        elif anchor == OverlayAnchor.CENTER:
            x_expr = "(W-w)/2"
            y_expr = "(H-h)/2"
        elif anchor == OverlayAnchor.SUBTITLE_REGION:
            x_expr = "(W-w)/2"
            y_expr = f"H*{round(overlay.y, 4)}-h/2"
        else:  # ABSOLUTE normalized
            x_expr = f"W*{round(overlay.x, 4)}-w/2"
            y_expr = f"H*{round(overlay.y, 4)}-h/2"

        return x_expr, y_expr, target_w

    def build_composition_filter_graph(
        self,
        video_width: int,
        video_height: int,
        subtitle_file: Path,
        visual_edit: VisualEditConfig,
        has_blur_mask: bool,
        options: RenderOptions,
    ) -> tuple[list[str], list[str], str]:
        """Build the full multi-layer FFmpeg filter complex string ensuring strict layer order:

        1. base video
        2. original-Chinese blur/clean layer
        3. user image/sticker overlays (sorted by z_index)
        4. Vietnamese ASS subtitles
        """
        filter_parts: list[str] = []
        extra_input_args: list[str] = []

        current_v_label = "0:v"

        # 1. Base + Blur Layer
        if visual_edit.mode in {VisualEditMode.BLUR, VisualEditMode.BLUR_OVERLAY} and has_blur_mask:
            sigma = visual_edit.blur.sigma
            filter_parts.append(f"[{current_v_label}]gblur=sigma={sigma}[blurred_base]")
            filter_parts.append(f"[{current_v_label}][blurred_base][1:v]maskedmerge[clean_base]")
            current_v_label = "clean_base"

        # 2. Overlays Layer (Sorted by z_index ascending)
        if visual_edit.mode == VisualEditMode.BLUR_OVERLAY and visual_edit.overlays:
            sorted_overlays = sorted(visual_edit.overlays, key=lambda o: o.z_index)

            input_offset = 2 if (visual_edit.mode in {VisualEditMode.BLUR, VisualEditMode.BLUR_OVERLAY} and has_blur_mask) else 1

            for idx, ov in enumerate(sorted_overlays):
                ov_path = Path(ov.path)
                if not ov_path.exists():
                    logger.warning("Overlay image file not found: %s", ov_path)
                    continue

                input_idx = input_offset + idx
                extra_input_args.extend(["-i", str(ov_path.resolve())])

                x_expr, y_expr, scaled_w = self.compute_overlay_coordinates(ov, video_width, video_height)

                ov_stream = f"ov_{idx}"
                fade_filters = [f"scale={scaled_w}:-1", "format=rgba"]

                if ov.opacity < 1.0:
                    fade_filters.append(f"colorchannelmixer=aa={ov.opacity}")

                if ov.fade_in_ms > 0:
                    fade_in_s = ov.fade_in_ms / 1000.0
                    fade_filters.append(f"fade=t=in:st={ov.start}:d={fade_in_s}:alpha=1")

                if ov.fade_out_ms > 0:
                    fade_out_s = ov.fade_out_ms / 1000.0
                    fade_start = max(ov.start, ov.end - fade_out_s)
                    fade_filters.append(f"fade=t=out:st={fade_start}:d={fade_out_s}:alpha=1")

                filter_parts.append(f"[{input_idx}:v]{','.join(fade_filters)}[{ov_stream}]")

                next_v_label = f"v_after_ov_{idx}"
                filter_parts.append(
                    f"[{current_v_label}][{ov_stream}]overlay={x_expr}:{y_expr}:enable='between(t,{ov.start},{ov.end})'[{next_v_label}]"
                )
                current_v_label = next_v_label

        # 3. Vietnamese Subtitles Layer (MUST BE ON TOP of Overlays)
        escaped_sub = escape_filter_path(subtitle_file)
        if options.subtitle_format == "ass" or subtitle_file.suffix == ".ass":
            sub_filter = f"ass='{escaped_sub}'"
        else:
            style = f"FontName={options.font_name},FontSize={options.font_size},Outline={options.outline_width},Shadow={options.shadow_depth},MarginV={options.margin_v}"
            sub_filter = f"subtitles='{escaped_sub}':force_style='{style}'"

        filter_parts.append(f"[{current_v_label}]{sub_filter}[final_out]")
        current_v_label = "final_out"

        return filter_parts, extra_input_args, current_v_label
