from __future__ import annotations

import re
from pathlib import Path
import numpy as np

from app.models.project import RenderOptions, SubtitleCue
from app.services.utterance_engine import (
    NOISE_SUBTITLE_PATTERNS,
    RenderSubtitleCue,
    UtteranceEngine,
    clean_vietnamese_typography,
    semantic_line_break,
)

_TIME_RE = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})")


def parse_timestamp(value: str) -> float:
    match = _TIME_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    parts = {key: int(number) for key, number in match.groupdict().items()}
    return parts["h"] * 3600 + parts["m"] * 60 + parts["s"] + parts["ms"] / 1000


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def format_ass_timestamp(seconds: float) -> str:
    """Formats seconds into ASS timestamp: H:MM:SS.cs (centiseconds)."""
    centiseconds = max(0, round(seconds * 100))
    hours, centiseconds = divmod(centiseconds, 360000)
    minutes, centiseconds = divmod(centiseconds, 6000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"{hours:01}:{minutes:02}:{secs:02}.{centiseconds:02}"


def wrap_vietnamese_text(text: str, max_line_chars: int = 36) -> str:
    """Wrapper function preserving semantic line breaking."""
    return semantic_line_break(text, max_line_chars=max_line_chars)


def normalize_render_cues(
    cues: list[SubtitleCue],
    safe_gap: float = 0.03,
    min_duration: float = 0.25,
    translated: bool = True,
) -> list[RenderSubtitleCue]:
    """Generates non-overlapping single-active movie subtitle cues via UtteranceEngine."""
    engine = UtteranceEngine(safe_gap=safe_gap, min_display_duration=max(0.80, min_duration))
    render_cues, _ = engine.process_cues(cues, translated=translated)
    return render_cues


def parse_srt(text: str) -> list[SubtitleCue]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    cues: list[SubtitleCue] = []
    for block in re.split(r"\n\s*\n", normalized):
        lines = [line.rstrip() for line in block.splitlines()]
        if len(lines) < 2:
            continue
        timing_index = 1 if lines[0].strip().isdigit() else 0
        if timing_index >= len(lines) or "-->" not in lines[timing_index]:
            continue
        start_raw, end_raw = [item.strip().split(" ")[0] for item in lines[timing_index].split("-->")]
        body = "\n".join(lines[timing_index + 1 :]).strip()
        if not body:
            continue
        cues.append(SubtitleCue(start=parse_timestamp(start_raw), end=parse_timestamp(end_raw), source_text=body))
    return cues


def to_srt(cues: list[SubtitleCue], translated: bool = True) -> str:
    blocks: list[str] = []
    render_cues = normalize_render_cues(cues, safe_gap=0.03, min_duration=0.25, translated=translated)
    for index, cue in enumerate(render_cues, start=1):
        clean_text = cue.render_text.replace(r"\N", "\n")
        blocks.append(f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{clean_text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(path: Path, cues: list[SubtitleCue], translated: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_srt(cues, translated=translated), encoding="utf-8")
    return path


def to_ass(
    cues: list[SubtitleCue],
    options: RenderOptions,
    width: int = 852,
    height: int = 480,
    translated: bool = True,
) -> str:
    # Preset detection
    preset = "default"
    if options.visual_edit is not None:
        preset = getattr(options.visual_edit, "preset", "default") or "default"

    is_shortform_ref = preset == "shortform_reference"
    is_shortform_yellow = preset == "shortform_bold_yellow"
    is_shortform_white_black = preset in {"shortform_white_black_soft_bg", "shortform_soft_bg"}

    is_any_shortform = is_shortform_ref or is_shortform_yellow or is_shortform_white_black

    font_name = options.font_name or "Arial"
    font_size = 23 if is_any_shortform else (options.font_size or 26)
    # Vertical position: raise shortform subtitles (margin_v ~58px at 480p) to directly overlap the original hard-sub band
    if options.margin_v is not None:
        margin_v = options.margin_v
    elif is_any_shortform:
        margin_v = max(48, int(round(height * 0.122)))
    else:
        margin_v = 38
    shadow_d = 0.6 if is_shortform_white_black else (1.2 if is_shortform_yellow else (1.0 if is_shortform_ref else (options.shadow_depth or 0.8)))
    font_color = options.font_color or "&H00FFFFFF"
    bold_val = -1

    # Outline color & width
    if is_shortform_white_black:
        # Thin Black outline
        outline_color = "&H00000000"
        outline_w = 1.8
    elif is_shortform_yellow:
        # Golden Yellow outline in ASS BGR format (B=00, G=D7, R=FF)
        outline_color = "&H0000D7FF"
        outline_w = 2.4
    elif is_shortform_ref:
        outline_color = "&H00000000"
        outline_w = 2.8
    else:
        outline_color = options.outline_color or "&H00000000"
        outline_w = options.outline_width or 2.2

    # Check Subtitle Backing Plate (Layer 0)
    backing_cfg = getattr(options.visual_edit, "subtitle_backing", None) if options.visual_edit else None
    has_backing = (
        backing_cfg is not None and backing_cfg.enabled
    ) or is_shortform_yellow or is_shortform_white_black

    styles_list = []

    if has_backing:
        opacity = backing_cfg.opacity if backing_cfg else 0.60
        pad_x = backing_cfg.padding_x if backing_cfg else 18
        # Calculate hex alpha: 0.0=solid (&H00), 1.0=transparent (&HFF)
        alpha_int = int(max(0, min(255, round((1.0 - opacity) * 255))))
        backing_color = f"&H{alpha_int:02X}000000"
        styles_list.append(
            f"Style: BackingPlate,{font_name},{font_size},&HFF000000,&HFF000000,{backing_color},{backing_color},{bold_val},0,0,0,100,100,0,0,3,{pad_x},0,2,20,20,{margin_v},1"
        )

    styles_list.append(
        f"Style: Default,{font_name},{font_size},{font_color},&H000000FF,{outline_color},&H80000000,{bold_val},0,0,0,100,100,0,0,1,{outline_w},{shadow_d},2,20,20,{margin_v},1"
    )

    header = f"""[Script Info]
; Script generated by AI Video Localizer (Phase 5.1B Utterance Engine)
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{chr(10).join(styles_list)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    engine = UtteranceEngine(
        safe_gap=options.safe_gap if hasattr(options, "safe_gap") else 0.03,
        max_line_chars=options.max_line_chars,
    )
    render_cues, _ = engine.process_cues(cues, translated=translated)
    cues_by_id = {
        (getattr(c, "id", None) or getattr(c, "render_id", None)): c
        for c in cues
        if (getattr(c, "id", None) or getattr(c, "render_id", None))
    }

    # Precompute OCR-anchored vertical center positions for shortform mode
    cue_y_centers: list[int] = []
    default_y_center = int(round(height * 0.862 - 4.0))  # fallback ~410px on 480p

    for cue in render_cues:
        y_pts: list[float] = []
        for cid in getattr(cue, "source_cue_ids", []):
            orig = cues_by_id.get(cid)
            if not orig:
                continue
            for r in getattr(orig, "ocr_regions", []) or []:
                for pt in getattr(r, "points", []) or []:
                    if len(pt) >= 2 and pt[1] >= 0.65:  # Focus on bottom subtitle region
                        y_pts.append(pt[1])
            for ev in getattr(orig, "ocr_evidence", []) or []:
                for r in getattr(ev, "regions", []) or []:
                    for pt in getattr(r, "points", []) or []:
                        if len(pt) >= 2 and pt[1] >= 0.65:
                            y_pts.append(pt[1])

        if y_pts:
            min_y = min(y_pts)
            max_y = max(y_pts)
            mid_y = (min_y + max_y) / 2.0 * height - 4.0  # -4px visual overlap tuning
            clamped_y = int(round(max(height * 0.72, min(height * 0.90, mid_y))))
            cue_y_centers.append(clamped_y)
        else:
            cue_y_centers.append(default_y_center)

    # Anti-jitter: smooth y-centers within dialogue sequences
    if cue_y_centers:
        median_y = int(round(float(np.median(cue_y_centers)))) if "np" in globals() else default_y_center
        smoothed_y_centers = []
        for y_val in cue_y_centers:
            # If deviation is within 14px of median, anchor to median to avoid 1-frame height jumping
            if abs(y_val - median_y) <= 14:
                smoothed_y_centers.append(median_y)
            else:
                smoothed_y_centers.append(y_val)
        cue_y_centers = smoothed_y_centers

    center_x = width // 2

    for idx, cue in enumerate(render_cues):
        ass_text = cue.render_text.replace("{", r"\{").replace("}", r"\}")
        start_str = format_ass_timestamp(cue.start)
        end_str = format_ass_timestamp(cue.end)
        y_pos = cue_y_centers[idx] if idx < len(cue_y_centers) else default_y_center

        if is_any_shortform:
            if has_backing:
                blur_r = backing_cfg.blur_radius if backing_cfg else 8
                # Layer 0: Soft blurred backing plate centered directly over Chinese subtitle region
                events.append(f"Dialogue: 0,{start_str},{end_str},BackingPlate,,0,0,0,,{{\\an5\\pos({center_x},{y_pos})\\blur{blur_r}\\alpha&HFF&}}{ass_text}")
                # Layer 1: Crisp text centered directly over Chinese subtitle region
                events.append(f"Dialogue: 1,{start_str},{end_str},Default,,0,0,0,,{{\\an5\\pos({center_x},{y_pos})}}{ass_text}")
            else:
                events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{{\\an5\\pos({center_x},{y_pos})}}{ass_text}")
        else:
            if has_backing:
                blur_r = backing_cfg.blur_radius if backing_cfg else 8
                events.append(f"Dialogue: 0,{start_str},{end_str},BackingPlate,,0,0,0,,{{\\blur{blur_r}\\alpha&HFF&}}{ass_text}")
                events.append(f"Dialogue: 1,{start_str},{end_str},Default,,0,0,0,,{ass_text}")
            else:
                events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{ass_text}")

    return header + "\n".join(events) + "\n"


def write_ass(
    path: Path,
    cues: list[SubtitleCue],
    options: RenderOptions,
    width: int = 852,
    height: int = 480,
    translated: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_ass(cues, options, width, height, translated=translated), encoding="utf-8")
    return path
