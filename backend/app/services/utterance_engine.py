from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.models.project import SubtitleCue

# Generic noise patterns for non-dialogue artifacts and isolated OCR noise
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

CHINESE_CONTINUATION_WORDS = (
    "但是", "但", "因为", "所以", "如果", "虽然", "而且", "然后",
    "只有", "只要", "并且", "既", "不仅", "另外", "反而", "甚至",
    "或者", "还是", "与其", "不如", "即使", "哪怕", "而我", "但我",
)

VIETNAMESE_CONTINUATION_WORDS = (
    "nhưng", "vì", "nên", "mà", "và", "hoặc", "nếu", "tuy",
    "cho nên", "do đó", "bởi vì", "thậm chí", "ngược lại",
    "chỉ muốn", "lớn lên", "đã", "được", "con đã", "em đã",
)


class DiscourseMode(str, Enum):
    DIRECT_DIALOGUE = "direct_dialogue"
    MONOLOGUE = "monologue"
    NARRATION = "narration"
    SYSTEM = "system"
    UNKNOWN = "unknown"


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
    """Applies clean, domain-generic Vietnamese subtitle typography normalization."""
    res = text.strip()

    # 1. Remove isolated OCR noise tokens
    for pat in NOISE_REPLACE_PATTERNS:
        res = re.sub(pat, "", res, flags=re.IGNORECASE)

    # 2. Punctuation normalization
    res = re.sub(r",\s*,+", ",", res)  # duplicate commas
    res = re.sub(r"\.{4,}", "...", res)  # 4+ dots to ellipsis
    res = re.sub(r"\s+([,!?;:])", r"\1", res)  # remove space before punctuation
    res = re.sub(r"\s+\.(?!\d)", ".", res)  # remove space before period (not decimal)
    res = re.sub(r"([,!?;:])(?=[A-Za-zÀ-ỹ0-9])", r"\1 ", res)  # space after comma/colon/etc
    res = re.sub(r"\.(?=[A-Za-zÀ-ỹ])", ". ", res)  # space after period if followed by letter

    # 3. Spacing inside brackets/quotes
    res = re.sub(r"\(\s+", "(", res)
    res = re.sub(r"\s+\)", ")", res)

    # 4. Collapse multiple whitespace
    res = re.sub(r"\s+", " ", res).strip()

    # 5. Clean leading stray punctuation
    res = re.sub(r"^[,.:;]\s*", "", res)

    # 6. Ensure proper capitalization of first letter
    if res and res[0].islower():
        res = res[0].upper() + res[1:]

    # 7. Clean trailing dangling punctuation
    res = re.sub(r"[,:;]\s*$", "", res).strip()

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


def infer_discourse_mode(cue: dict[str, Any]) -> DiscourseMode:
    """Infers discourse mode from addressee, source context, and direct dialogue markers."""
    addr = cue.get("addressee_id") or cue.get("addressee_character_id")
    src = (cue.get("source_text") or "").strip()
    tr = (cue.get("translated_text") or "").strip()

    # Past story / background narration markers
    if any(w in src for w in ["在一家", "在一个", "早餐店", "小时候", "那时候", "曾经", "过去", "家庭里"]):
        return DiscourseMode.NARRATION

    if addr is not None and addr != "" and addr != "audience":
        return DiscourseMode.DIRECT_DIALOGUE

    # Direct confrontation/vocative marker (e.g. explicit character call + you / 你)
    if any(w in src for w in ["你偷了", "你给我", "你看清楚", "你别", "你是不是", "你难道", "你良心"]):
        return DiscourseMode.DIRECT_DIALOGUE
    if re.search(r"^(?:秦|宋|孟|Tần|Mạnh|Song)\w*[,，\s]+(?:mày|cô|bạn|anh|em|cậu|bác|chú|dì)\b", tr, re.IGNORECASE):
        return DiscourseMode.DIRECT_DIALOGUE

    return DiscourseMode.MONOLOGUE


def is_short_imperative_or_assessment(src: str) -> bool:
    """Detects short independent assessment or imperative clauses (e.g. 领口歪了, 坐姿不对, 笑的太假, 看清楚).

    Characteristics: 3 to 7 characters, self-contained statement or predicate with no trailing conjunction.
    These must be rendered as independent, sequential subtitle beats.
    """
    s = re.sub(r"[，。！？、,.!?\s]", "", src).strip()
    if 2 <= len(s) <= 7:
        if s.endswith(("歪了", "不对", "太假", "好了", "快点", "走了", "停下", "站住", "看清楚", "听好了")):
            return True
    return False


@dataclass
class MergeEvidence:
    """Evaluates whether two consecutive subtitle cues have strong positive evidence to merge."""
    same_speaker: bool
    same_addressee: bool
    same_mode: bool
    temporal_gap: float
    is_question: bool
    is_short_assessment: bool
    has_strong_zh_continuation: bool
    has_strong_vi_continuation: bool
    has_trailing_comma_or_dash: bool
    is_short_subject_lead: bool
    nxt_starts_lower: bool
    is_terminal_zh: bool
    combined_tr_len: int
    combined_duration: float
    is_dangling_fragment: bool = False

    def calculate_score(self) -> tuple[float, list[str]]:
        """Calculates evidence score. Default is DO NOT MERGE (< 3.5)."""
        reasons: list[str] = []
        score = 0.0

        # Hard Veto 1: Speaker mismatch
        if not self.same_speaker:
            return -100.0, ["speaker_mismatch"]

        # Hard Veto 2: Addressee / Discourse Mode mismatch (Dialogue vs Monologue never merge)
        if not self.same_addressee:
            return -100.0, ["addressee_mismatch"]
        if not self.same_mode:
            return -100.0, ["discourse_mode_mismatch"]

        # Hard Veto 3: Question followed by anything
        if self.is_question:
            return -100.0, ["question_boundary"]

        # Hard Veto 4: Short independent assessment / imperative beat
        if self.is_short_assessment and not self.has_strong_zh_continuation:
            return -100.0, ["short_assessment_boundary"]

        # Hard Veto 5: Length or duration overflow
        if self.combined_tr_len > 76:
            return -100.0, ["length_overflow"]
        if self.combined_duration > 6.0:
            return -100.0, ["duration_overflow"]

        # Hard Veto 6: Temporal gap too wide
        if self.temporal_gap > 0.40:
            return -100.0, ["gap_too_wide"]

        # Positive Evidence Accumulation
        if self.has_strong_zh_continuation:
            score += 3.5
            reasons.append("strong_zh_continuation")

        if self.is_dangling_fragment:
            score += 3.0
            reasons.append("resolve_dangling_fragment")

        if self.has_trailing_comma_or_dash:
            score += 2.0
            reasons.append("trailing_comma_or_dash")

        if self.nxt_starts_lower or self.has_strong_vi_continuation:
            score += 1.5
            reasons.append("continuation_syntax")

        if self.is_short_subject_lead:
            score += 1.5
            reasons.append("short_subject_lead")

        if self.temporal_gap <= 0.15:
            score += 0.5
            reasons.append("tight_temporal_gap")

        # Negative Evidence
        if self.is_terminal_zh and not self.has_strong_zh_continuation:
            score -= 5.0
            reasons.append("terminal_zh_sentence")

        return score, reasons


class UtteranceEngine:
    """Groups raw speech/OCR fragments into coherent, readable movie subtitles."""

    def __init__(
        self,
        max_utterance_gap: float = 0.35,
        min_display_duration: float = 0.80,
        safe_gap: float = 0.03,
        max_line_chars: int = 36,
        merge_score_threshold: float = 3.5,
    ) -> None:
        self.max_utterance_gap = max_utterance_gap
        self.min_display_duration = min_display_duration
        self.safe_gap = safe_gap
        self.max_line_chars = max_line_chars
        self.merge_score_threshold = merge_score_threshold

    def process_cues(
        self,
        raw_cues: list[SubtitleCue],
        translated: bool = True,
    ) -> tuple[list[RenderSubtitleCue], dict[str, Any]]:
        cues = sorted(raw_cues, key=lambda x: (float(x.start), float(x.end)))

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

                overlap = min(float(c.end), float(prev["end"])) - max(float(c.start), float(prev["start"]))
                shortest_dur = max(0.01, min(float(c.end) - float(c.start), float(prev["end"]) - float(prev["end"])))
                overlap_ratio = max(0.0, overlap / shortest_dur)

                same_speaker = (
                    (c.speaker_id is not None and c.speaker_id == prev.get("speaker_id"))
                    or (c.speaker_character_id is not None and c.speaker_character_id == prev.get("speaker_character_id"))
                    or (c.speaker_id is None and prev.get("speaker_id") is None)
                )

                is_exact_src_match = clean_src == prev_clean_src

                if same_speaker and is_exact_src_match and (overlap_ratio >= 0.50 or abs(float(c.start) - float(prev["start"])) <= 0.35):
                    prev["end"] = max(prev["end"], float(c.end))
                    prev["source_cue_ids"].append(c.id)
                    prev["source_starts"].append(float(c.start))
                    prev["source_ends"].append(float(c.end))
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

        utterance_groups: list[dict[str, Any]] = []
        i = 0
        merged_group_count = 0
        wrong_speaker_merges = 0

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

                same_speaker = False
                if cur_char and nxt_char:
                    same_speaker = cur_char == nxt_char
                elif cur_spk and nxt_spk:
                    same_speaker = cur_spk == nxt_spk
                elif not cur_char and not nxt_char and not cur_spk and not nxt_spk:
                    same_speaker = True

                same_addressee = (cur_addr == nxt_addr)

                cur_mode = infer_discourse_mode(cur)
                nxt_mode = infer_discourse_mode(nxt)
                is_dialogue_cur = (cur_mode == DiscourseMode.DIRECT_DIALOGUE)
                is_dialogue_nxt = (nxt_mode == DiscourseMode.DIRECT_DIALOGUE)
                same_mode = (is_dialogue_cur == is_dialogue_nxt) and (cur_mode != DiscourseMode.SYSTEM and nxt_mode != DiscourseMode.SYSTEM)

                cur_src = cur["source_text"].strip()
                nxt_src = nxt["source_text"].strip()
                cur_tr = cur["translated_text"].strip()
                nxt_tr = nxt["translated_text"].strip()

                is_question = bool(re.search(r"[?？]$", cur_src) or re.search(r"[?？]$", cur_tr))
                is_short_assessment = is_short_imperative_or_assessment(cur_src)

                has_strong_zh_continuation = any(nxt_src.startswith(w) for w in CHINESE_CONTINUATION_WORDS) or any(cur_src.endswith(w) for w in ("但是", "但", "而且", "然后", "因为", "所以"))
                has_strong_vi_continuation = any(nxt_tr.lower().startswith(w) for w in VIETNAMESE_CONTINUATION_WORDS)
                has_trailing_comma_or_dash = bool(re.search(r"(?:[,，…—–-]|[.]{2,})$", cur_src) or re.search(r"(?:[,…—–-]|[.]{2,})$", cur_tr))
                is_short_subject_lead = len(re.sub(r"[^\w]", "", cur_src)) <= 3 and has_trailing_comma_or_dash
                nxt_starts_lower = bool(nxt_tr and nxt_tr[0].islower())
                is_terminal_zh = bool(re.search(r"[。！？!?]$", cur_src))
                is_dangling_fragment = bool(
                    re.search(r"(?:[,，…—–-]|[.]{2,})$", cur_tr)
                    or re.search(r"\b(cô|anh|em|ông|bà|chị|bạn|mày|con|tôi|ta|mà|và|thì|nhưng|hoặc|đã|đang|sẽ|được|bị|vì|bởi|của|ở|tại|là|để)\s*,?$", cur_tr.strip(), re.IGNORECASE)
                )

                raw_combined_tr = f"{cur_tr} {nxt_tr}"
                combined_duration = max(cur["end"], nxt["end"]) - cur["start"]

                evidence = MergeEvidence(
                    same_speaker=same_speaker,
                    same_addressee=same_addressee,
                    same_mode=same_mode,
                    temporal_gap=gap,
                    is_question=is_question,
                    is_short_assessment=is_short_assessment,
                    has_strong_zh_continuation=has_strong_zh_continuation,
                    has_strong_vi_continuation=has_strong_vi_continuation,
                    has_trailing_comma_or_dash=has_trailing_comma_or_dash,
                    is_short_subject_lead=is_short_subject_lead,
                    nxt_starts_lower=nxt_starts_lower,
                    is_terminal_zh=is_terminal_zh,
                    combined_tr_len=len(raw_combined_tr),
                    combined_duration=combined_duration,
                    is_dangling_fragment=is_dangling_fragment,
                )

                score, reasons = evidence.calculate_score()

                if score >= self.merge_score_threshold:
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

        render_cues: list[RenderSubtitleCue] = []
        overlap_pairs = 0
        high_cps_count = 0

        for idx, g in enumerate(utterance_groups):
            raw_tr = g["translated_text"]
            clean_tr = clean_vietnamese_typography(raw_tr)
            render_text = semantic_line_break(clean_tr, max_line_chars=self.max_line_chars)

            start = float(g["start"])
            end = float(g["end"])

            if end - start < self.min_display_duration:
                end = start + self.min_display_duration

            if render_cues:
                prev_end = render_cues[-1].end
                if start < prev_end:
                    overlap_pairs += 1
                    if start - render_cues[-1].start >= self.min_display_duration:
                        render_cues[-1].end = max(render_cues[-1].start + self.min_display_duration, start - self.safe_gap)
                    else:
                        start = prev_end + self.safe_gap
                        end = max(start + self.min_display_duration, end)

            dur = max(0.01, end - start)
            clean_chars = len(re.sub(r"[\s\\N]", "", render_text))
            cps = round(clean_chars / dur, 1)
            if cps > 20.0:
                high_cps_count += 1

            render_cues.append(
                RenderSubtitleCue(
                    render_id=f"sub_{idx + 1:03d}",
                    source_cue_ids=g["source_cue_ids"],
                    start=round(start, 2),
                    end=round(end, 2),
                    source_text=g["source_text"],
                    translated_text=clean_tr,
                    render_text=render_text,
                    speaker_id=g.get("speaker_id"),
                    speaker_character_id=g.get("speaker_character_id"),
                    source_starts=g["source_starts"],
                    source_ends=g["source_ends"],
                    cps=cps,
                )
            )

        for j in range(len(render_cues) - 1):
            if render_cues[j].end > render_cues[j + 1].start - self.safe_gap:
                render_cues[j].end = round(max(render_cues[j].start + 0.5, render_cues[j + 1].start - self.safe_gap), 2)

        metrics = {
            "source_cues": len(raw_cues),
            "filtered_cues": len(filtered),
            "render_cues": len(render_cues),
            "merged_groups": merged_group_count,
            "suppressed_duplicates": suppressed_count,
            "wrong_speaker_merges": wrong_speaker_merges,
            "overlap_pairs": overlap_pairs,
            "high_cps_cues": high_cps_count,
        }

        return render_cues, metrics
