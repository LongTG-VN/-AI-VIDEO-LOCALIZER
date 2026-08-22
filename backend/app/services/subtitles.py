from __future__ import annotations

import re
from pathlib import Path

from app.models.project import SubtitleCue

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
    for index, cue in enumerate(sorted(cues, key=lambda item: item.start), start=1):
        text = cue.translated_text if translated and cue.translated_text else cue.source_text
        blocks.append(f"{index}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_srt(path: Path, cues: list[SubtitleCue], translated: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_srt(cues, translated=translated), encoding="utf-8")
    return path
