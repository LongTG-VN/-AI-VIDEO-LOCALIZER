from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.models.project import RenderOptions, SubtitleCue

_TIME_RE = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})")

# Noise filter for non-dialogue artifacts
NOISE_SUBTITLE_PATTERNS = {"10.5o", "10:50", "MILK", "MILK MILK", "IN-CN", "CN-IN", "755135", "CN", "..."}


@dataclass
class RenderCue:
    """Non-mutating render cue for single-active subtitle lane rendering."""
    id: str
    render_start: float
    render_end: float
    source_text: str
    translated_text: str
    speaker_id: str | None = None


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
    """Intelligently wraps Vietnamese subtitle text into at most 2 balanced lines."""
    clean = " ".join(text.replace("\r\n", " ").replace("\n", " ").split()).strip()
    if len(clean) <= max_line_chars:
        return clean

    words = clean.split()
    if len(words) <= 1:
        return clean

    mid_idx = len(clean) // 2
    best_split_pos = -1
    min_dist_to_center = float("inf")

    # Priority 1: Punctuation boundaries (. , ! ? ; : -)
    punc_matches = [m.end() for m in re.finditer(r"[,.!?;:—–-]\s+", clean)]
    for pos in punc_matches:
        dist = abs(pos - mid_idx)
        if dist < min_dist_to_center:
            min_dist_to_center = dist
            best_split_pos = pos

    # Priority 2: Space boundaries if no ideal punctuation split
    if best_split_pos == -1 or min_dist_to_center > max_line_chars // 2:
        for m in re.finditer(r"\s+", clean):
            pos = m.start()
            dist = abs(pos - mid_idx)
            if dist < min_dist_to_center:
                min_dist_to_center = dist
                best_split_pos = pos

    if best_split_pos > 0 and best_split_pos < len(clean):
        part1 = clean[:best_split_pos].strip()
        part2 = clean[best_split_pos:].strip()
        if part1 and part2:
            return part1 + r"\N" + part2

    return clean


def normalize_render_cues(
    cues: list[SubtitleCue],
    safe_gap: float = 0.03,
    min_duration: float = 0.25,
    translated: bool = True,
) -> list[RenderCue]:
    """Normalizes subtitle cues into non-overlapping, single-active render cues.

    Guarantees:
    1. MAX_ACTIVE_SUBTITLES = 1 (Zero concurrent subtitles).
    2. cue[i].render_end <= cue[i+1].render_start - safe_gap.
    3. Positive duration for every cue (duration > 0).
    4. Merges adjacent identical translations into a single continuous cue.
    5. Preserves original SubtitleCue objects without mutation.
    """
    valid: list[dict] = []
    for c in cues:
        raw_text = c.translated_text if translated and c.translated_text else c.source_text
        if not raw_text:
            continue
        txt = raw_text.strip()
        if txt in NOISE_SUBTITLE_PATTERNS or len(txt) <= 1:
            continue
        valid.append({
            "id": c.id,
            "start": float(c.start),
            "end": float(c.end),
            "source_text": c.source_text,
            "translated_text": txt,
            "speaker_id": c.speaker_id,
        })

    if not valid:
        return []

    # Sort strictly by start time
    valid.sort(key=lambda item: item["start"])

    # Merge adjacent identical translations (normalized whitespace and trailing punctuation)
    merged: list[dict] = []
    for item in valid:
        clean_text = item["translated_text"].strip().rstrip(".").rstrip(",")
        if not merged:
            merged.append(item)
            continue
        prev = merged[-1]
        prev_clean = prev["translated_text"].strip().rstrip(".").rstrip(",")
        if prev_clean == clean_text and (item["start"] - prev["end"]) <= 0.50:
            prev["end"] = max(prev["end"], item["end"])
        else:
            merged.append(item)

    # Enforce Single Active Subtitle Policy (MAX_ACTIVE_SUBTITLES = 1)
    render_cues: list[RenderCue] = []
    for i, cur in enumerate(merged):
        r_start = float(cur["start"])
        r_end = float(cur["end"])

        # Ensure start time does not overlap with previous cue's end
        if render_cues:
            prev_end = render_cues[-1].render_end
            if r_start < prev_end + safe_gap:
                r_start = round(prev_end + safe_gap, 3)

        # Trim end time so it expires before next cue begins
        if i < len(merged) - 1:
            next_start = float(merged[i + 1]["start"])
            max_allowed_end = round(next_start - safe_gap, 3)
            if r_end > max_allowed_end:
                r_end = max_allowed_end

        # Ensure valid positive duration
        if r_end <= r_start:
            r_end = round(r_start + min_duration, 3)

        render_cues.append(
            RenderCue(
                id=str(cur["id"]),
                render_start=r_start,
                render_end=r_end,
                source_text=cur["source_text"],
                translated_text=cur["translated_text"],
                speaker_id=cur.get("speaker_id"),
            )
        )

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
        blocks.append(f"{index}\n{format_timestamp(cue.render_start)} --> {format_timestamp(cue.render_end)}\n{cue.translated_text}")
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
    """Generates a complete ASS v4.00+ subtitle file with single-active subtitle timeline."""
    font_name = options.font_name or "Arial"
    font_size = options.font_size or 24
    margin_v = options.margin_v or 32
    outline_w = options.outline_width or 2.8
    shadow_d = options.shadow_depth or 1.0
    font_color = options.font_color or "&H00FFFFFF"
    outline_color = options.outline_color or "&H00000000"

    header = f"""[Script Info]
; Script generated by AI Video Localizer (Phase 5.1 Single-Active Lane)
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{font_color},&H000000FF,{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,{outline_w},{shadow_d},2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    # Normalize render cues to guarantee zero concurrent active subtitles
    render_cues = normalize_render_cues(cues, safe_gap=0.03, min_duration=0.25, translated=translated)

    for cue in render_cues:
        # Smart Vietnamese line wrapping
        wrapped_text = wrap_vietnamese_text(cue.translated_text, max_line_chars=options.max_line_chars)
        ass_text = wrapped_text.replace("{", r"\{").replace("}", r"\}")

        start_str = format_ass_timestamp(cue.render_start)
        end_str = format_ass_timestamp(cue.render_end)
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
