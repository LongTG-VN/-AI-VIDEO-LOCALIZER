from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from app.models.project import Project, RenderOptions, VisualEditMode
from app.services.hardsub_cleaner import HardSubCleaner
from app.services.subtitles import write_ass, write_srt
from app.services.visual_edit_composer import VisualEditComposer

logger = logging.getLogger(__name__)


class RenderError(RuntimeError):
    pass


def escape_filter_path(path: Path) -> str:
    """Escapes file paths for FFmpeg filter complexes on Windows."""
    value = path.resolve().as_posix().replace("'", r"\'")
    return value.replace(":", r"\:")


def build_subtitle_filter(srt_path: Path, options: RenderOptions) -> str:
    escaped = escape_filter_path(srt_path)
    style = f"FontName={options.font_name},FontSize={options.font_size},Outline={options.outline_width},Shadow={options.shadow_depth},MarginV={options.margin_v}"
    return f"subtitles='{escaped}':force_style='{style}'"


def check_nvenc_available(ffmpeg_bin: str = "ffmpeg") -> bool:
    """Checks if h264_nvenc hardware encoder is supported by FFmpeg and GPU."""
    try:
        res = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
        return "h264_nvenc" in res.stdout
    except Exception:
        return False


def get_media_info(path: Path, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Inspects rendered video file using ffprobe."""
    if not path.exists():
        raise RenderError(f"Rendered file not found for verification: {path}")

    cmd = [
        ffprobe_bin,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
    except Exception as exc:
        raise RenderError(f"ffprobe verification failed: {exc}") from exc

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    return {
        "duration": float(format_info.get("duration", 0.0)),
        "size_bytes": int(format_info.get("size", 0)),
        "video_codec": v_stream.get("codec_name") if v_stream else None,
        "width": int(v_stream.get("width", 0)) if v_stream else 0,
        "height": int(v_stream.get("height", 0)) if v_stream else 0,
        "fps": v_stream.get("r_frame_rate") if v_stream else None,
        "audio_codec": a_stream.get("codec_name") if a_stream else None,
        "audio_present": a_stream is not None,
    }


class Renderer:
    """Production Video Renderer with Chinese Hard-Sub cleanup, ASS Subtitles, and NVENC."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe"):
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.last_render_metrics: dict[str, Any] = {}

    def _create_cleaner(self, options: RenderOptions) -> HardSubCleaner:
        return HardSubCleaner(
            crop_top_ratio=options.hardsub_crop_top_ratio,
            crop_bottom_ratio=options.hardsub_crop_bottom_ratio,
            crop_left_ratio=options.hardsub_crop_left_ratio,
            crop_right_ratio=options.hardsub_crop_right_ratio,
            mask_dilate_radius=options.hardsub_mask_dilate_radius,
            mask_dilate_iterations=options.hardsub_mask_dilate_iterations,
            local_contrast_threshold=options.hardsub_local_contrast_threshold,
            inpaint_radius=options.hardsub_inpaint_radius,
            max_mask_coverage=options.hardsub_max_mask_coverage,
            scene_cut_threshold=options.hardsub_scene_cut_threshold,
            temporal_max_distance_frames=options.hardsub_temporal_max_distance_frames,
            temporal_difference_threshold=options.hardsub_temporal_difference_threshold,
            temporal_local_score_threshold=options.hardsub_temporal_local_score_threshold,
            ocr_min_confidence=options.hardsub_ocr_min_confidence,
            geometry_enabled=options.hardsub_geometry_enabled,
            geometry_padding_px=options.hardsub_geometry_padding_px,
            ffmpeg_bin=self.ffmpeg_bin,
        )

    def render(self, project: Project, output_path: Path, options: RenderOptions | None = None) -> Path:
        if options is None:
            options = RenderOptions()

        source = Path(project.source_video_path)
        if not source.exists():
            raise RenderError(f"Source video not found: {source}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = output_path.parent / f".{project.id}-render"
        work_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()
        cleaned_video_path = source
        cleanup_metrics: dict[str, Any] = {}

        try:
            visual_edit = options.visual_edit or project.visual_edit
            is_visual_edit_blur = visual_edit is not None and visual_edit.mode in {
                VisualEditMode.BLUR,
                VisualEditMode.BLUR_OVERLAY,
            }

            # 1. Chinese Hard-Sub Cleanup / Dynamic Blur Mask Preparation
            if not is_visual_edit_blur and options.hardsub_removal_mode not in {"none", "off"}:
                logger.info("Starting Chinese hard-sub cleanup (mode: %s)...", options.hardsub_removal_mode)
                cleaner = self._create_cleaner(options)
                intermediate_clean = work_dir / (
                    "cleaned_video.mkv" if options.hardsub_lossless_intermediate else "cleaned_video.mp4"
                )
                cleanup_metrics = cleaner.clean_video(
                    source_path=source,
                    output_path=intermediate_clean,
                    cues=project.cues,
                    mode=options.hardsub_removal_mode,
                    lossless_intermediate=options.hardsub_lossless_intermediate,
                )
                cleaned_video_path = intermediate_clean

            t_after_cleanup = time.time()

            # 2. Generate Subtitle File (.ass preferred, .srt fallback)
            v_width = project.width or 852
            v_height = project.height or 480

            if options.subtitle_format == "ass":
                sub_path = write_ass(
                    work_dir / "subtitles.ass",
                    project.cues,
                    options,
                    width=v_width,
                    height=v_height,
                    translated=True,
                )
            else:
                sub_path = write_srt(work_dir / "subtitles.srt", project.cues, translated=True)

            # 3. Build FFmpeg Filter Complex
            burned = work_dir / "burned.mp4"
            cmd = [self.ffmpeg_bin, "-y"]

            if is_visual_edit_blur:
                composer = VisualEditComposer(self.ffmpeg_bin, self.ffprobe_bin)
                mask_path = work_dir / "dynamic_blur_mask.mp4"
                v_fps = 30.0
                try:
                    meta = get_media_info(source, self.ffprobe_bin)
                    if meta.get("fps"):
                        num, den = meta["fps"].split("/") if "/" in meta["fps"] else (meta["fps"], "1")
                        v_fps = float(num) / float(den) if float(den) > 0 else 30.0
                except Exception:
                    pass

                _, mask_metrics = composer.generate_dynamic_blur_mask_video(
                    output_mask_path=mask_path,
                    cues=project.cues,
                    width=v_width,
                    height=v_height,
                    fps=v_fps,
                    duration=project.duration or 60.0,
                    blur_config=visual_edit.blur,
                )
                cleanup_metrics.update(mask_metrics)

                cmd.extend(["-i", str(source), "-i", str(mask_path)])

                filter_parts, extra_inputs, current_label = composer.build_composition_filter_graph(
                    video_width=v_width,
                    video_height=v_height,
                    subtitle_file=sub_path,
                    visual_edit=visual_edit,
                    has_blur_mask=True,
                    options=options,
                )
                cmd.extend(extra_inputs)
            else:
                cmd.extend(["-i", str(cleaned_video_path), "-i", str(source)])
                for sticker in options.stickers:
                    cmd.extend(["-i", sticker.path])

                escaped_sub = escape_filter_path(sub_path)
                if options.subtitle_format == "ass":
                    sub_filter = f"ass='{escaped_sub}'"
                else:
                    style = f"FontName={options.font_name},FontSize={options.font_size},Outline={options.outline_width},Shadow={options.shadow_depth},MarginV={options.margin_v}"
                    sub_filter = f"subtitles='{escaped_sub}':force_style='{style}'"

                # Overlays first, then subtitles on top
                current_label = "0:v"
                filter_parts = []
                for index, sticker in enumerate(options.stickers, start=2):
                    sticker_label = f"st{index}"
                    next_label = f"v{index}"
                    filter_parts.append(f"[{index}:v]scale={sticker.scale_width}:-1[{sticker_label}]")
                    filter_parts.append(
                        f"[{current_label}][{sticker_label}]overlay={sticker.x}:{sticker.y}:enable='between(t,{sticker.start},{sticker.end})'[{next_label}]"
                    )
                    current_label = next_label

                filter_parts.append(f"[{current_label}]{sub_filter}[final_out]")
                current_label = "final_out"

            # 4. Hardware Encoder Selection (NVENC vs libx264)
            encoder_used = "libx264"
            video_codec_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]

            if options.use_nvenc and check_nvenc_available(self.ffmpeg_bin):
                encoder_used = "h264_nvenc"
                video_codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-b:v", "0"]

            audio_map_idx = "0:a?" if is_visual_edit_blur else "1:a?"
            cmd.extend([
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                f"[{current_label}]",
                "-map",
                audio_map_idx,
                *video_codec_args,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(burned),
            ])

            logger.info("Executing FFmpeg render with encoder: %s", encoder_used)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                # Fallback to libx264 if NVENC failed
                if encoder_used == "h264_nvenc":
                    logger.warning("NVENC encode failed, falling back to libx264...")
                    encoder_used = "libx264 (fallback)"
                    video_codec_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
                    cmd_fallback = [self.ffmpeg_bin, "-y", "-i", str(cleaned_video_path), "-i", str(source)]
                    for sticker in options.stickers:
                        cmd_fallback.extend(["-i", sticker.path])
                    cmd_fallback.extend([
                        "-filter_complex",
                        ";".join(filter_parts),
                        "-map",
                        f"[{current_label}]",
                        "-map",
                        "1:a?",
                        *video_codec_args,
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-movflags",
                        "+faststart",
                        str(burned),
                    ])
                    proc = subprocess.run(cmd_fallback, capture_output=True, text=True)

                if proc.returncode != 0:
                    raise RenderError(proc.stderr[-3000:] or "FFmpeg render failed")

            t_after_encode = time.time()

            # 5. Intro / Outro Concatenation (if present)
            segments: list[Path] = []
            for candidate in [options.intro_path, str(burned), options.outro_path]:
                if candidate:
                    path = Path(candidate)
                    if path.exists():
                        segments.append(path)

            if len(segments) <= 1:
                shutil.copy2(burned, output_path)
            else:
                concat_cmd = [self.ffmpeg_bin, "-y"]
                for segment in segments:
                    concat_cmd.extend(["-i", str(segment)])
                concat_inputs = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(len(segments)))
                concat_filter = f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[v][a]"
                concat_cmd.extend([
                    "-filter_complex",
                    concat_filter,
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                    "-c:v",
                    "h264_nvenc" if encoder_used.startswith("h264_nvenc") else "libx264",
                    "-c:a",
                    "aac",
                    str(output_path),
                ])
                proc_concat = subprocess.run(concat_cmd, capture_output=True, text=True)
                if proc_concat.returncode != 0:
                    raise RenderError("Intro/outro concat failed: " + (proc_concat.stderr[-2000:] if proc_concat.stderr else ""))

            total_render_time = time.time() - t_start

            # 6. Automated Media Verification with ffprobe
            media_info = get_media_info(output_path, self.ffprobe_bin)

            self.last_render_metrics = {
                "output_path": str(output_path),
                "encoder": encoder_used,
                "hardsub_removal_mode": options.hardsub_removal_mode,
                "cleanup_time": cleanup_metrics.get("clean_runtime", 0.0),
                "encode_time": round(t_after_encode - t_after_cleanup, 2),
                "total_render_time": round(total_render_time, 2),
                "media_info": media_info,
                "cleanup_metrics": cleanup_metrics,
            }
            logger.info("Render completed successfully: %s", self.last_render_metrics)
            return output_path

        finally:
            # Clean up temporary work directory
            if work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)
