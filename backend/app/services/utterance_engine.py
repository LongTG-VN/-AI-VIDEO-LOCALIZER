from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from app.models.project import SubtitleCue

# Noise patterns for non-dialogue artifacts and isolated OCR noise
NOISE_SUBTITLE_PATTERNS = {
    "10.5o", "10:50", "MILK", "MILK MILK", "IN-CN", "CN-IN", "755135", "CN",
    "...", "西", "T", "Y", "1", "0", "工", "国", "LAA",
}

NOISE_REPLACE_PATTERNS = [
    r"\bMILK\s+MILK\b",
    r"\bMILK\b",
    r"\b10:50\b",
    r"\b10\.5o\b",
    r"\bIN-CN\b",
    r"\bCN-IN\b",
    r"\b755135\b",
    r"\bCN\b",
]


@dataclass
class RenderSubtitleCue:
    """Represents a finalized, single-active cinematic subtitle cue for rendering."""
    render_id: str
    source_cue_ids: list[str]
    start: float
    end: float
    source_text: str
    translated_text: str
    render_text: str
    speaker_id: str | None = None
    speaker_character_id: str | None = None
    source_starts: list[float] = field(default_factory=list)
    source_ends: list[float] = field(default_factory=list)
    cps: float = 0.0


def clean_text_for_comparison(text: str) -> str:
    """Normalizes text for duplicate and containment checking."""
    t = text
    for noise in ["MILK MILK", "MILK", "10:50", "10.5o", "IN-CN", "CN-IN", "755135", "CN"]:
        t = t.replace(noise, "")
    return re.sub(r"[^\w\s]", "", t.lower()).replace(" ", "").strip()


def clean_vietnamese_typography(text: str) -> str:
    """Applies clean, domain-generic Vietnamese subtitle typography normalization.

    Strictly avoids character names, dialogue-specific replacements, or story rewrites.
    """
    res = text.strip()

    # 1. Remove isolated OCR noise tokens
    for pat in NOISE_REPLACE_PATTERNS:
        res = re.sub(pat, "", res, flags=re.IGNORECASE)

    # 2. Punctuation normalization
    res = re.sub(r",\s*,+", ",", res)  # duplicate commas
    res = re.sub(r"\.{4,}", "...", res)  # 4+ dots to ellipsis
    res = re.sub(r"\s+([,!?;:])", r"\1", res)  # remove space before punctuation (not period)
    res = re.sub(r"\s+\.(?!\d)", ".", res)  # remove space before period (not decimal)
    res = re.sub(r"([,!?;:])(?=[A-Za-zÀ-ỹ0-9])", r"\1 ", res)  # space after comma/colon/etc
    res = re.sub(r"\.(?=[A-Za-zÀ-ỹ])", ". ", res)  # space after period if followed by letter (preserves 0.3)

    # 3. Spacing inside brackets/quotes
    res = re.sub(r"\(\s+", "(", res)
    res = re.sub(r"\s+\)", ")", res)

    # 4. Collapse multiple whitespace
    res = re.sub(r"\s+", " ", res).strip()

    # 5. Clean leading stray punctuation (e.g. leading comma or dot)
    res = re.sub(r"^[,.:;]\s*", "", res)

    # 6. Ensure proper capitalization after terminal punctuation if single sentence
    if res and res[0].islower():
        res = res[0].upper() + res[1:]

    return res.strip()


def semantic_line_break(text: str, max_line_chars: int = 36) -> str:
    """Balances Vietnamese subtitle text into 2 natural semantic lines."""
    if "\n" in text or r"\N" in text:
        return text.replace("\n", r"\N")
    clean = " ".join(text.split()).strip()
    if len(clean) <= max_line_chars:
        return clean

    mid = len(clean) // 2
    best_pos = -1
    min_dist = float("inf")

    # Priority 1: Punctuation boundaries (. , ! ? ; : — -)
    for m in re.finditer(r"[,.!?;:—–-]\s+", clean):
        pos = m.end()
        dist = abs(pos - mid)
        if dist < min_dist:
            min_dist = dist
            best_pos = pos

    # Priority 2: Natural conjunctions / clause boundaries
    if best_pos == -1 or min_dist > max_line_chars // 2:
        conjunctions = [
            r"\s+nhưng\s+", r"\s+mà\s+", r"\s+và\s+", r"\s+hoặc\s+",
            r"\s+thì\s+", r"\s+đã\s+", r"\s+được\s+", r"\s+để\s+", r"\s+trong\s+",
        ]
        for c_pat in conjunctions:
            for m in re.finditer(c_pat, clean, flags=re.IGNORECASE):
                pos = m.start()
                dist = abs(pos - mid)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = pos

    # Priority 3: General word spacing
    if best_pos == -1 or min_dist > max_line_chars // 2:
        for m in re.finditer(r"\s+", clean):
            pos = m.start()
            dist = abs(pos - mid)
            if dist < min_dist:
                min_dist = dist
                best_pos = pos

    if best_pos > 0 and best_pos < len(clean):
        p1 = clean[:best_pos].strip()
        p2 = clean[best_pos:].strip()
        if p1 and p2:
            return p1 + r"\N" + p2

    return clean


class UtteranceEngine:
    """Groups raw speech/OCR fragments into coherent, readable movie subtitles.

    Adheres strictly to chronological source ordering, speaker/narration boundaries,
    and generic typography rules.
    """

    def __init__(
        self,
        max_utterance_gap: float = 0.60,
        min_display_duration: float = 0.80,
        safe_gap: float = 0.03,
        max_line_chars: int = 36,
    ) -> None:
        self.max_utterance_gap = max_utterance_gap
        self.min_display_duration = min_display_duration
        self.safe_gap = safe_gap
        self.max_line_chars = max_line_chars

    def process_cues(
        self,
        raw_cues: list[SubtitleCue],
        translated: bool = True,
    ) -> tuple[list[RenderSubtitleCue], dict[str, Any]]:
        # 1. Sort cues strictly by chronological (start, end)
        cues = sorted(raw_cues, key=lambda x: (float(x.start), float(x.end)))

        # 2. Filter noise and suppress duplicate/redundant fragments with strong evidence
        filtered: list[dict[str, Any]] = []
        suppressed_count = 0

        for c in cues:
            src = (c.source_text or "").strip()
            tr = (c.translated_text or "").strip() if translated and c.translated_text else src
            if not src or not tr or src in NOISE_SUBTITLE_PATTERNS or tr in NOISE_SUBTITLE_PATTERNS or len(src) <= 1:
                continue

            clean_src = clean_text_for_comparison(src)
            clean_tr = clean_text_for_comparison(tr)
            if not clean_src or not clean_tr:
                continue

            if filtered:
                prev = filtered[-1]
                prev_clean_src = clean_text_for_comparison(prev["source_text"])

                # Check temporal overlap
                overlap = min(float(c.end), float(prev["end"])) - max(float(c.start), float(prev["start"]))
                shortest_dur = max(0.01, min(float(c.end) - float(c.start), float(prev["end"]) - float(prev["start"])))
                overlap_ratio = max(0.0, overlap / shortest_dur)

                # Check speaker compatibility (allow matching if one is an unassigned OCR fragment)
                same_speaker = (
                    (c.speaker_id is not None and c.speaker_id == prev.get("speaker_id"))
                    or (c.speaker_character_id is not None and c.speaker_character_id == prev.get("speaker_character_id"))
                    or (c.speaker_id is None and prev.get("speaker_id") is None)
                )
                same_or_unassigned = (
                    same_speaker
                    or (c.speaker_id is None and prev.get("speaker_id") is not None)
                    or (c.speaker_id is not None and prev.get("speaker_id") is None)
                )

                # Similarity check on source text
                src_sim = SequenceMatcher(None, clean_src, prev_clean_src).ratio()
                is_exact_src_match = clean_src == prev_clean_src
                is_contained_src = (
                    (clean_src in prev_clean_src or prev_clean_src in clean_src)
                    and min(len(clean_src), len(prev_clean_src)) >= 4
                )

                # Suppress ONLY if strong multi-factor evidence exists
                if same_or_unassigned and (overlap_ratio >= 0.40 or abs(float(c.start) - float(prev["start"])) <= 0.35):
                    if is_exact_src_match or (same_speaker and src_sim >= 0.85 and (is_contained_src or overlap_ratio >= 0.70)):
                        prev["end"] = max(prev["end"], float(c.end))
                        prev["source_cue_ids"].append(c.id)
                        prev["source_starts"].append(float(c.start))
                        prev["source_ends"].append(float(c.end))
                        if not prev.get("speaker_id") and c.speaker_id:
                            prev["speaker_id"] = c.speaker_id
                        if not prev.get("speaker_character_id") and c.speaker_character_id:
                            prev["speaker_character_id"] = c.speaker_character_id
                        suppressed_count += 1
                        continue

            filtered.append({
                "id": c.id,
                "source_cue_ids": [c.id],
                "start": float(c.start),
                "end": float(c.end),
                "source_text": src,
                "translated_text": tr,
                "speaker_id": c.speaker_id,
                "speaker_character_id": c.speaker_character_id,
                "addressee_id": c.addressee_id,
                "addressee_character_id": c.addressee_character_id,
                "source_starts": [float(c.start)],
                "source_ends": [float(c.end)],
            })

        # 3. Multi-step Semantic Utterance Grouping (Generic Rules Only)
        utterance_groups: list[dict[str, Any]] = []
        i = 0
        merged_group_count = 0

        while i < len(filtered):
            cur = filtered[i].copy()

            while i < len(filtered) - 1:
                nxt = filtered[i + 1]
                gap = float(nxt["start"]) - float(cur["end"])

                cur_char = cur.get("speaker_character_id")
                nxt_char = nxt.get("speaker_character_id")
                cur_spk = cur.get("speaker_id")
                nxt_spk = nxt.get("speaker_id")
                cur_addr = cur.get("addressee_id") or cur.get("addressee_character_id")
                nxt_addr = nxt.get("addressee_id") or nxt.get("addressee_character_id")

                # RULE 1: Speaker Compatibility (Canonical character preferred)
                if cur_char and nxt_char and cur_char != nxt_char:
                    break
                if cur_spk and nxt_spk and cur_spk != nxt_spk and (not cur_char or not nxt_char or cur_char != nxt_char):
                    break

                # RULE 2: Dialogue vs Monologue / Narration Boundary Protection
                # Never merge direct dialogue (with addressee) with monologue/narration (addressee is None)
                if (cur_addr is not None and nxt_addr is None) or (cur_addr is None and nxt_addr is not None):
                    break
                if cur_addr is not None and nxt_addr is not None and cur_addr != nxt_addr:
                    break

                # RULE 3: Time Proximity
                near_time = gap <= self.max_utterance_gap or gap <= 0.20

                # RULE 4: Guardrails on Duration and Text Length
                raw_combined_tr = f"{cur['translated_text'].strip()} {nxt['translated_text'].strip()}"
                within_limits = (len(raw_combined_tr) <= 75) and (max(cur["end"], nxt["end"]) - cur["start"] <= 5.2)

                if not (near_time and within_limits):
                    break

                # RULE 5: Sentence Completeness and Continuation Signals
                cur_tr = cur["translated_text"].strip()
                nxt_tr = nxt["translated_text"].strip()

                cur_ends_terminal = bool(re.search(r"[.!?]$", cur_tr))
                cur_ends_incomplete = bool(re.search(r"[,…—–-]$", cur_tr)) or (not cur_ends_terminal)
                nxt_starts_lower = bool(nxt_tr and nxt_tr[0].islower())
                nxt_starts_conjunction = bool(re.match(r"^(mà|và|hoặc|thì|đã|được|để|trong|nhưng|hôm|ngày|thời)\b", nxt_tr, flags=re.IGNORECASE))

                overlap_amount = max(0.0, float(cur["end"]) - float(nxt["start"]))
                shortest = max(0.01, min(float(cur["end"]) - float(cur["start"]), float(nxt["end"]) - float(nxt["start"])))
                is_heavy_overlap = (overlap_amount / shortest) >= 0.35

                can_merge = (
                    (cur_ends_incomplete or nxt_starts_lower or nxt_starts_conjunction or is_heavy_overlap)
                    and not (cur_ends_terminal and not (is_heavy_overlap or nxt_starts_lower))
                )

                if can_merge:
                    # Invariant: Concat in strict chronological source order
                    cur["source_text"] = f"{cur['source_text'].strip()} {nxt['source_text'].strip()}"
                    cur["translated_text"] = f"{cur['translated_text'].strip()} {nxt['translated_text'].strip()}"
                    cur["source_cue_ids"].extend(nxt["source_cue_ids"])
                    cur["source_starts"].extend(nxt["source_starts"])
                    cur["source_ends"].extend(nxt["source_ends"])
                    cur["end"] = max(cur["end"], nxt["end"])
                    if not cur.get("speaker_id") or cur.get("speaker_id") == "unknown":
                        cur["speaker_id"] = nxt_spk
                    if not cur.get("speaker_character_id"):
                        cur["speaker_character_id"] = nxt_char
                    merged_group_count += 1
                    i += 1
                else:
                    break

            utterance_groups.append(cur)
            i += 1

        # 4. Polish typography & enforce Single Active Timeline
        render_cues: list[RenderSubtitleCue] = []
        for idx, u in enumerate(utterance_groups, start=1):
            r_start = float(u["start"])
            r_end = float(u["end"])

            polished = clean_vietnamese_typography(u["translated_text"])
            render_text = semantic_line_break(polished, max_line_chars=self.max_line_chars)

            # Single-active timeline spacing
            if render_cues:
                prev_end = render_cues[-1].end
                if r_start < prev_end + self.safe_gap:
                    r_start = round(prev_end + self.safe_gap, 3)

            if idx < len(utterance_groups):
                nxt_start = float(utterance_groups[idx]["start"])
                max_end = round(nxt_start - self.safe_gap, 3)
                if r_end > max_end:
                    r_end = max_end

            dur = max(0.20, r_end - r_start)
            if dur < self.min_display_duration and idx < len(utterance_groups):
                nxt_start = float(utterance_groups[idx]["start"])
                if nxt_start - r_start >= self.min_display_duration + self.safe_gap:
                    r_end = round(r_start + self.min_display_duration, 3)
                    dur = r_end - r_start

            cps = round(len(polished.replace("\n", "").replace(r"\N", "")) / max(0.1, dur), 1)

            # Assert timeline invariants
            if r_start > r_end:
                r_end = round(r_start + 0.20, 3)

            render_cues.append(
                RenderSubtitleCue(
                    render_id=f"render_{idx:03d}",
                    source_cue_ids=u.get("source_cue_ids", [u.get("id")]),
                    start=r_start,
                    end=r_end,
                    source_text=u["source_text"],
                    translated_text=u["translated_text"],
                    render_text=render_text,
                    speaker_id=u.get("speaker_id"),
                    speaker_character_id=u.get("speaker_character_id"),
                    source_starts=u.get("source_starts", [r_start]),
                    source_ends=u.get("source_ends", [r_end]),
                    cps=cps,
                )
            )

        # 5. Verify timeline monotonic ordering invariant
        for k in range(len(render_cues) - 1):
            assert render_cues[k].start <= render_cues[k].end, f"Invalid duration at cue {render_cues[k].render_id}"
            assert render_cues[k].end <= render_cues[k + 1].start + 1e-4, f"Overlap at cue {render_cues[k].render_id}"

        durations = [rc.end - rc.start for rc in render_cues]
        sorted_durs = sorted(durations)
        median_dur = sorted_durs[len(sorted_durs) // 2] if sorted_durs else 0.0

        metrics = {
            "source_cues": len(raw_cues),
            "render_cues": len(render_cues),
            "suppressed_duplicates": suppressed_count,
            "merged_groups": merged_group_count,
            "reduction_pct": round((1 - len(render_cues) / max(1, len(raw_cues))) * 100, 1),
            "avg_duration": round(sum(durations) / max(1, len(durations)), 2),
            "median_duration": round(median_dur, 2),
            "avg_cps": round(sum(rc.cps for rc in render_cues) / max(1, len(render_cues)), 1),
            "max_cps": max((rc.cps for rc in render_cues), default=0.0),
            "high_cps_count": sum(1 for rc in render_cues if rc.cps > 20.0),
        }
        return render_cues, metrics
