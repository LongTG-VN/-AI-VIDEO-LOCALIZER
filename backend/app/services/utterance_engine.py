from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.models.project import SubtitleCue
from app.services.semantic_context import normalize_discourse_mode

# Generic noise patterns for non-dialogue artifacts and isolated OCR noise.
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

# These are generic connective particles, not title-specific dialogue.
CHINESE_CONTINUATION_WORDS = (
    "但是", "但", "因为", "所以", "如果", "虽然", "而且", "然后",
    "只有", "只要", "并且", "既", "不仅", "另外", "反而", "甚至",
    "或者", "还是", "与其", "不如", "即使", "哪怕", "而", "但我", "而我",
)

VIETNAMESE_CONTINUATION_WORDS = (
    "nhưng", "vì", "nên", "mà", "và", "hoặc", "nếu", "tuy",
    "cho nên", "do đó", "bởi vì", "thậm chí", "ngược lại",
)


class DiscourseMode(str, Enum):
    DIRECT_DIALOGUE = "direct_dialogue"
    MONOLOGUE = "monologue"
    NARRATION = "narration"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class RenderSubtitleCue:
    """Finalized single-active subtitle cue used by SRT/ASS rendering."""

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
    discourse_mode: str = DiscourseMode.UNKNOWN.value


def clean_text_for_comparison(text: str) -> str:
    """Normalize text for duplicate/containment checks."""
    normalized = text or ""
    for noise in ["MILK MILK", "MILK", "10:50", "10.5o", "IN-CN", "CN-IN", "755135", "CN"]:
        normalized = normalized.replace(noise, "")
    return re.sub(r"[^\w\s]", "", normalized.lower()).replace(" ", "").strip()


def clean_vietnamese_typography(text: str) -> str:
    """Apply domain-generic Vietnamese subtitle typography cleanup only."""
    result = (text or "").strip()

    for pattern in NOISE_REPLACE_PATTERNS:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    result = re.sub(r",\s*,+", ",", result)
    result = re.sub(r"\.{4,}", "...", result)
    result = re.sub(r"\s+([,!?;:])", r"\1", result)
    result = re.sub(r"\s+\.(?!\d)", ".", result)
    result = re.sub(r"([,!?;:])(?=[A-Za-zÀ-ỹ0-9])", r"\1 ", result)
    result = re.sub(r"\.(?=[A-Za-zÀ-ỹ])", ". ", result)
    result = re.sub(r"\(\s+", "(", result)
    result = re.sub(r"\s+\)", ")", result)
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"^[,.:;]\s*", "", result)

    # Fix accidental mid-sentence capitalization: e.g. "cô Còn" -> "cô còn", "em Nhưng" -> "em nhưng"
    def _lower_mid_sentence(match: re.Match) -> str:
        return match.group(1) + match.group(2).lower()

    result = re.sub(
        r"([a-zà-ỹ0-9,;:—–-]\s+)(Còn|Nhưng|Mà|Và|Nếu|Tuy|Vì|Nên|Thì|Để|Cho|Với|Không|Đã|Đang|Sẽ|Được|Bị|Cũng|Thậm chí|Ngược lại)\b",
        _lower_mid_sentence,
        result
    )

    # Clean duplicate address pronoun at question end: "cô còn ăn được, cô?" -> "cô còn ăn được sao?"
    result = re.sub(
        r"\b(cô|anh|em|chị|bạn|ông|bà)\b(.*?)\s*,\s*\1\s*([?？])?$",
        r"\1\2?",
        result,
        flags=re.IGNORECASE
    )

    if result and result[0].islower():
        result = result[0].upper() + result[1:]

    result = re.sub(r"[,:;]\s*$", "", result).strip()
    return result


def merge_vietnamese_clauses(left: str, right: str) -> str:
    """Merge two Vietnamese subtitle clauses while preserving proper capitalization and grammar."""
    l = (left or "").strip()
    r = (right or "").strip()
    if not l:
        return r
    if not r:
        return l

    # If left ends with sentence-terminating punctuation, keep right capitalized
    if re.search(r"[.!?。！？]\s*$", l):
        return f"{l} {r}"

    # Lowercase the first character of right if it is a common word
    words = r.split(" ", 1)
    first_word = words[0]
    rest = (" " + words[1]) if len(words) > 1 else ""

    if first_word and first_word[0].isupper() and not first_word.isupper():
        # Do not lowercase if it is likely a capital abbreviation
        first_word = first_word[0].lower() + first_word[1:]
        r = first_word + rest

    # Strip trailing punctuation from left that would conflict with right
    l_clean = re.sub(r"[,;:\s]+$", "", l)
    combined = f"{l_clean} {r}".strip()
    return clean_vietnamese_typography(combined)


def validate_render_cue_entity_ownership(
    render_cues: list[RenderSubtitleCue],
    source_cues_by_id: dict[str, Any],
    known_name_pairs: list[dict[str, str]],
) -> list[str]:
    """Verify that no canonical character name migrated into a render cue whose source cues lack that name."""
    violations: list[str] = []
    for rc in render_cues:
        combined_zh = " ".join(
            (getattr(source_cues_by_id.get(cid), "source_text", None) or (source_cues_by_id.get(cid) or {}).get("source_text") or "")
            for cid in rc.source_cue_ids
            if cid in source_cues_by_id
        )
        rc_vi_lower = rc.render_text.lower()
        for pair in known_name_pairs:
            zh_name = pair.get("zh", "").strip()
            vi_name = pair.get("vi", "").strip().lower()
            if vi_name and len(vi_name) >= 3 and zh_name:
                if re.search(rf"\b{re.escape(vi_name)}\b", rc_vi_lower):
                    if zh_name not in combined_zh:
                        violations.append(
                            f"Entity ownership violation in render cue {rc.render_id}: "
                            f"name '{pair.get('vi')}' present in subtitle, but source '{combined_zh}' does not contain '{zh_name}'."
                        )
    return violations


def semantic_line_break(text: str, max_line_chars: int = 36) -> str:
    """Balance a subtitle across at most two semantic lines."""
    if "\n" in text or r"\N" in text:
        return text.replace("\n", r"\N")

    clean = " ".join((text or "").split()).strip()
    if len(clean) <= max_line_chars:
        return clean

    mid = len(clean) // 2
    best_pos = -1
    min_dist = float("inf")

    for match in re.finditer(r"[,.!?;:—–-]\s+", clean):
        pos = match.end()
        distance = abs(pos - mid)
        if distance < min_dist:
            min_dist = distance
            best_pos = pos

    if best_pos == -1 or min_dist > max_line_chars // 2:
        for pattern in [
            r"\s+nhưng\s+", r"\s+mà\s+", r"\s+và\s+", r"\s+hoặc\s+",
            r"\s+thì\s+", r"\s+để\s+", r"\s+trong\s+",
        ]:
            for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
                pos = match.start()
                distance = abs(pos - mid)
                if distance < min_dist:
                    min_dist = distance
                    best_pos = pos

    if best_pos == -1 or min_dist > max_line_chars // 2:
        for match in re.finditer(r"\s+", clean):
            pos = match.start()
            distance = abs(pos - mid)
            if distance < min_dist:
                min_dist = distance
                best_pos = pos

    if 0 < best_pos < len(clean):
        first = clean[:best_pos].strip()
        second = clean[best_pos:].strip()
        if first and second:
            return first + r"\N" + second
    return clean


def infer_discourse_mode(cue: dict[str, Any]) -> DiscourseMode:
    """Resolve discourse mode, preferring persisted context-analysis metadata.

    Old projects may not have discourse metadata. The fallback remains deliberately
    conservative and generic: a concrete addressee means direct dialogue; otherwise the
    cue is unknown/monologue-like rather than guessed from title-specific words or names.
    """
    explicit = normalize_discourse_mode(str(cue.get("discourse_mode") or "unknown"))
    if explicit != DiscourseMode.UNKNOWN.value:
        return DiscourseMode(explicit)

    speaker = str(cue.get("speaker_id") or "").lower()
    if any(token in speaker for token in ("system", "alarm", "announcement")):
        return DiscourseMode.SYSTEM

    addressee = cue.get("addressee_character_id") or cue.get("addressee_id")
    if addressee not in (None, "", "audience"):
        return DiscourseMode.DIRECT_DIALOGUE

    # Backward-compatible fallback. We intentionally do not infer narration from lexical
    # story words here; context analysis owns that decision for new projects.
    return DiscourseMode.MONOLOGUE


def is_short_imperative_or_assessment(source_text: str) -> bool:
    """Conservatively detect short self-contained Chinese beats.

    The heuristic uses grammatical/punctuation shape rather than named characters or a
    particular benchmark sentence. It is only a veto against aggressive merging.
    """
    raw = (source_text or "").strip()
    stripped = re.sub(r"[，。！？、,.!?\s]", "", raw)
    if not (2 <= len(stripped) <= 8):
        return False
    if raw.endswith(("。", "！", "？", "!", "?")):
        return True
    return bool(re.search(r"(了|不对|太.+|好了|住|清楚)$", stripped))


def _has_dangling_vi_tail(text: str) -> bool:
    clean = (text or "").strip()
    if not clean:
        return False
    patterns = [
        r"\b(cô|anh|em|ông|bà|chị|bạn|mày|con|tôi|ta)\s*[,;:]?$",
        r"\b(mà|và|thì|nhưng|hoặc|đã|đang|sẽ|được|bị|vì|bởi|của|ở|tại|là|để|cho|với)\s*[,;:]?$",
        r"\b(muốn|nghĩ|biết|cần)\s*[,;:]$",
    ]
    return any(re.search(pattern, clean, re.IGNORECASE) for pattern in patterns)


def _starts_with_continuation(text: str, words: tuple[str, ...]) -> bool:
    lowered = (text or "").strip().lower()
    return any(lowered == word.lower() or lowered.startswith(word.lower() + " ") for word in words)


@dataclass
class MergeEvidence:
    """Positive-evidence merge policy; default remains NO MERGE."""

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
        reasons: list[str] = []
        score = 0.0

        if not self.same_speaker:
            return -100.0, ["speaker_mismatch"]
        if not self.same_addressee:
            return -100.0, ["addressee_mismatch"]
        if not self.same_mode:
            return -100.0, ["discourse_mode_mismatch"]
        if self.is_question:
            return -100.0, ["question_boundary"]
        if self.is_short_assessment and not self.has_strong_zh_continuation:
            return -100.0, ["short_assessment_boundary"]
        if self.combined_tr_len > 76:
            return -100.0, ["length_overflow"]
        if self.combined_duration > 6.0:
            return -100.0, ["duration_overflow"]
        if self.temporal_gap > 0.40:
            return -100.0, ["gap_too_wide"]

        if self.has_strong_zh_continuation:
            score += 3.5
            reasons.append("strong_zh_continuation")
        if self.is_dangling_fragment:
            score += 3.0
            reasons.append("resolve_dangling_fragment")
        if self.has_trailing_comma_or_dash:
            score += 2.0
            reasons.append("trailing_comma_or_dash")
        if self.has_strong_vi_continuation or self.nxt_starts_lower:
            score += 1.5
            reasons.append("continuation_syntax")
        if self.is_short_subject_lead:
            score += 1.5
            reasons.append("short_subject_lead")
        if self.temporal_gap <= 0.15:
            score += 0.5
            reasons.append("tight_temporal_gap")

        if self.is_terminal_zh and not self.has_strong_zh_continuation:
            score -= 5.0
            reasons.append("terminal_zh_sentence")

        return score, reasons


class UtteranceEngine:
    """Create readable render cues without allowing timing to override semantics."""

    def __init__(
        self,
        max_utterance_gap: float = 0.35,
        min_display_duration: float = 0.80,
        safe_gap: float = 0.03,
        max_line_chars: int = 36,
        merge_score_threshold: float = 3.5,
        max_source_cues_per_group: int = 2,
    ) -> None:
        self.max_utterance_gap = max_utterance_gap
        self.min_display_duration = min_display_duration
        self.safe_gap = safe_gap
        self.max_line_chars = max_line_chars
        self.merge_score_threshold = merge_score_threshold
        self.max_source_cues_per_group = max(1, max_source_cues_per_group)

    @staticmethod
    def _same_speaker(cur: dict[str, Any], nxt: dict[str, Any]) -> bool:
        cur_char = cur.get("speaker_character_id")
        nxt_char = nxt.get("speaker_character_id")
        if cur_char and nxt_char:
            return cur_char == nxt_char

        cur_speaker = cur.get("speaker_id")
        nxt_speaker = nxt.get("speaker_id")
        if cur_speaker and nxt_speaker:
            return cur_speaker == nxt_speaker

        return not any((cur_char, nxt_char, cur_speaker, nxt_speaker))

    @staticmethod
    def _addressee_key(cue: dict[str, Any]) -> str | None:
        return cue.get("addressee_character_id") or cue.get("addressee_id")

    def process_cues(
        self,
        raw_cues: list[SubtitleCue],
        translated: bool = True,
    ) -> tuple[list[RenderSubtitleCue], dict[str, Any]]:
        cues = sorted(raw_cues, key=lambda cue: (float(cue.start), float(cue.end)))

        filtered: list[dict[str, Any]] = []
        suppressed_count = 0

        for cue in cues:
            source = (cue.source_text or "").strip()
            translated_text = (
                (cue.translated_text or "").strip()
                if translated and cue.translated_text
                else source
            )
            if (
                not source
                or not translated_text
                or source in NOISE_SUBTITLE_PATTERNS
                or translated_text in NOISE_SUBTITLE_PATTERNS
                or len(source) <= 1
            ):
                continue

            clean_source = clean_text_for_comparison(source)
            clean_translation = clean_text_for_comparison(translated_text)
            if not clean_source or not clean_translation:
                continue

            cid = getattr(cue, "id", None) or getattr(cue, "render_id", "cue")
            current = {
                "id": cid,
                "source_cue_ids": list(getattr(cue, "source_cue_ids", [cid])),
                "start": float(cue.start),
                "end": float(cue.end),
                "source_text": source,
                "translated_text": translated_text,
                "speaker_id": getattr(cue, "speaker_id", None),
                "speaker_character_id": getattr(cue, "speaker_character_id", None),
                "addressee_id": getattr(cue, "addressee_id", None),
                "addressee_character_id": getattr(cue, "addressee_character_id", None),
                "discourse_mode": normalize_discourse_mode(getattr(cue, "discourse_mode", None)),
                "source_starts": list(getattr(cue, "source_starts", [float(cue.start)])),
                "source_ends": list(getattr(cue, "source_ends", [float(cue.end)])),
            }

            if filtered:
                prev = filtered[-1]
                prev_clean_source = clean_text_for_comparison(prev["source_text"])
                is_exact_match = clean_source == prev_clean_source
                same_speaker = self._same_speaker(prev, current)
                same_mode = infer_discourse_mode(prev) == infer_discourse_mode(current)

                overlap = min(current["end"], prev["end"]) - max(current["start"], prev["start"])
                cue_duration = max(0.01, current["end"] - current["start"])
                prev_duration = max(0.01, prev["end"] - prev["start"])
                shortest_duration = min(cue_duration, prev_duration)
                overlap_ratio = max(0.0, overlap / shortest_duration)

                if (
                    same_speaker
                    and same_mode
                    and is_exact_match
                    and (overlap_ratio >= 0.50 or abs(current["start"] - prev["start"]) <= 0.35)
                ):
                    prev["end"] = max(prev["end"], current["end"])
                    prev["source_cue_ids"].append(cue.id)
                    prev["source_starts"].append(current["start"])
                    prev["source_ends"].append(current["end"])
                    suppressed_count += 1
                    continue

            filtered.append(current)

        utterance_groups: list[dict[str, Any]] = []
        merged_group_count = 0
        i = 0

        while i < len(filtered):
            cur = filtered[i].copy()

            while i < len(filtered) - 1:
                if len(cur["source_cue_ids"]) >= self.max_source_cues_per_group:
                    break

                nxt = filtered[i + 1]
                gap = float(nxt["start"]) - float(cur["end"])
                if gap > self.max_utterance_gap:
                    break

                same_speaker = self._same_speaker(cur, nxt)
                same_addressee = self._addressee_key(cur) == self._addressee_key(nxt)
                cur_mode = infer_discourse_mode(cur)
                nxt_mode = infer_discourse_mode(nxt)
                same_mode = cur_mode == nxt_mode

                cur_source = cur["source_text"].strip()
                nxt_source = nxt["source_text"].strip()
                cur_translation = cur["translated_text"].strip()
                nxt_translation = nxt["translated_text"].strip()

                is_question = bool(
                    re.search(r"[?？]\s*$", cur_source)
                    or re.search(r"[?？]\s*$", cur_translation)
                )
                is_short_assessment = is_short_imperative_or_assessment(cur_source)
                has_strong_zh_continuation = (
                    _starts_with_continuation(nxt_source, CHINESE_CONTINUATION_WORDS)
                    or any(cur_source.endswith(word) for word in CHINESE_CONTINUATION_WORDS)
                )
                has_strong_vi_continuation = _starts_with_continuation(
                    nxt_translation,
                    VIETNAMESE_CONTINUATION_WORDS,
                )
                has_trailing_comma_or_dash = bool(
                    re.search(r"(?:[,，…—–-]|[.]{2,})\s*$", cur_source)
                    or re.search(r"(?:[,，…—–-]|[.]{2,})\s*$", cur_translation)
                )
                is_short_subject_lead = (
                    len(re.sub(r"[^\w]", "", cur_source)) <= 3
                    and has_trailing_comma_or_dash
                )
                nxt_starts_lower = bool(
                    nxt_translation and nxt_translation[0].isalpha() and nxt_translation[0].islower()
                )
                is_terminal_zh = bool(re.search(r"[。！？!?]\s*$", cur_source))
                is_dangling_fragment = _has_dangling_vi_tail(cur_translation)

                raw_combined_translation = f"{cur_translation} {nxt_translation}".strip()
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
                    combined_tr_len=len(raw_combined_translation),
                    combined_duration=combined_duration,
                    is_dangling_fragment=is_dangling_fragment,
                )
                score, _ = evidence.calculate_score()

                if score < self.merge_score_threshold:
                    break

                cur["source_text"] = f"{cur_source} {nxt_source}".strip()
                cur["translated_text"] = merge_vietnamese_clauses(cur_translation, nxt_translation)
                cur["source_cue_ids"].extend(nxt["source_cue_ids"])
                cur["source_starts"].extend(nxt["source_starts"])
                cur["source_ends"].extend(nxt["source_ends"])
                cur["end"] = max(cur["end"], nxt["end"])
                merged_group_count += 1
                i += 1

            utterance_groups.append(cur)
            i += 1

        render_cues: list[RenderSubtitleCue] = []
        overlap_pairs = 0
        high_cps_count = 0

        for index, group in enumerate(utterance_groups):
            clean_translation = clean_vietnamese_typography(group["translated_text"])
            render_text = semantic_line_break(clean_translation, max_line_chars=self.max_line_chars)

            start = float(group["start"])
            end = float(group["end"])
            if end - start < self.min_display_duration:
                end = start + self.min_display_duration

            if render_cues:
                previous = render_cues[-1]
                if start < previous.end:
                    overlap_pairs += 1
                    if start - previous.start >= self.min_display_duration:
                        previous.end = max(
                            previous.start + self.min_display_duration,
                            start - self.safe_gap,
                        )
                    else:
                        start = previous.end + self.safe_gap
                        end = max(start + self.min_display_duration, end)

            duration = max(0.01, end - start)
            visible_chars = len(re.sub(r"[\s\\N]", "", render_text))
            cps = round(visible_chars / duration, 1)
            if cps > 20.0:
                high_cps_count += 1

            mode = infer_discourse_mode(group).value
            render_cues.append(
                RenderSubtitleCue(
                    render_id=f"sub_{index + 1:03d}",
                    source_cue_ids=group["source_cue_ids"],
                    start=round(start, 2),
                    end=round(end, 2),
                    source_text=group["source_text"],
                    translated_text=clean_translation,
                    render_text=render_text,
                    speaker_id=group.get("speaker_id"),
                    speaker_character_id=group.get("speaker_character_id"),
                    source_starts=group["source_starts"],
                    source_ends=group["source_ends"],
                    cps=cps,
                    discourse_mode=mode,
                )
            )

        # Final single-active invariant. This is timing cleanup only and never merges text.
        for index in range(len(render_cues) - 1):
            current = render_cues[index]
            nxt = render_cues[index + 1]
            if current.end > nxt.start - self.safe_gap:
                current.end = round(
                    max(current.start + 0.5, nxt.start - self.safe_gap),
                    2,
                )

        metrics = {
            "source_cues": len(raw_cues),
            "filtered_cues": len(filtered),
            "render_cues": len(render_cues),
            "merged_groups": merged_group_count,
            "suppressed_duplicates": suppressed_count,
            "wrong_speaker_merges": 0,
            "overlap_pairs": overlap_pairs,
            "high_cps_cues": high_cps_count,
        }
        return render_cues, metrics
