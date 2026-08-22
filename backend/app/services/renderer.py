from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from app.models.project import Project, RenderOptions
from app.services.subtitles import write_srt


class RenderError(RuntimeError):
    pass


def escape_filter_path(path: Path) -> str:
    value = path.resolve().as_posix().replace("'", r"\'")
    return value.replace(":", r"\:")


def build_subtitle_filter(srt_path: Path, options: RenderOptions) -> str:
    escaped = escape_filter_path(srt_path)
    style = f"FontName={options.font_name},FontSize={options.font_size},Outline=2,Shadow=0,MarginV={options.margin_v}"
    return f"subtitles='{escaped}':force_style='{style}'"


class Renderer:
    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin

    def render(self, project: Project, output_path: Path, options: RenderOptions) -> Path:
        source = Path(project.source_video_path)
        if not source.exists():
            raise RenderError(f"Source video not found: {source}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = output_path.parent / f".{project.id}-render"
        work_dir.mkdir(parents=True, exist_ok=True)
        srt_path = write_srt(work_dir / "translated.srt", project.cues, translated=True)
        burned = work_dir / "burned.mp4"
        cmd = [self.ffmpeg_bin, "-y", "-i", str(source)]
        for sticker in options.stickers:
            cmd.extend(["-i", sticker.path])
        filter_parts = [f"[0:v]{build_subtitle_filter(srt_path, options)}[base]"]
        current = "base"
        for index, sticker in enumerate(options.stickers, start=1):
            sticker_label = f"st{index}"
            next_label = f"v{index}"
            filter_parts.append(f"[{index}:v]scale={sticker.scale_width}:-1[{sticker_label}]")
            filter_parts.append(f"[{current}][{sticker_label}]overlay={sticker.x}:{sticker.y}:enable='between(t,{sticker.start},{sticker.end})'[{next_label}]")
            current = next_label
        cmd.extend(["-filter_complex", ";".join(filter_parts), "-map", f"[{current}]", "-map", "0:a?", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(burned)])
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise RenderError("ffmpeg is not installed or not on PATH") from exc
        except subprocess.CalledProcessError as exc:
            raise RenderError(exc.stderr[-5000:] or "FFmpeg render failed") from exc

        segments: list[Path] = []
        for candidate in [options.intro_path, str(burned), options.outro_path]:
            if candidate:
                path = Path(candidate)
                if path.exists():
                    segments.append(path)
        if len(segments) == 1:
            shutil.copy2(burned, output_path)
            return output_path
        concat_cmd = [self.ffmpeg_bin, "-y"]
        for segment in segments:
            concat_cmd.extend(["-i", str(segment)])
        concat_inputs = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(len(segments)))
        concat_filter = f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[v][a]"
        concat_cmd.extend(["-filter_complex", concat_filter, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac", str(output_path)])
        try:
            subprocess.run(concat_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise RenderError("Intro/outro concat failed. Normalize resolution/FPS/audio layout first. " + (exc.stderr[-3000:] if exc.stderr else "")) from exc
        return output_path
