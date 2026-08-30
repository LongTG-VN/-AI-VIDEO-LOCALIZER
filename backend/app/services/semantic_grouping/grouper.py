from __future__ import annotations

import logging
import re
from typing import Any

from app.models.project import Project, SubtitleCue
from app.services.semantic_grouping.models import (
    SemanticGroupingConfig,
    SemanticTranslationGroup,
)

logger = logging.getLogger(__name__)

# Indicators that a Chinese source cue is syntactically incomplete / continuing
INCOMPLETE_ZH_ENDINGS = [
    r"[，、：；(（\-\s]$",
    r"(在|和|与|或|或者|及|把|将|被|让|给|使|令|因为|由于|如果|要是|虽然|即使|但|但是|却|而且|并且|不仅|不但|而|的|地|得|着|了|过|是|像|比如|唯一|能够|可以|想要|准备|开始|正在|刚|一)$",
]

# Punctuation indicating a hard terminal sentence boundary
TERMINAL_ZH_PUNCTUATION = re.compile(r"[。！？!?…]+$")


class SemanticGrouper:
    """Groups adjacent SubtitleCues into SemanticTranslationGroups when they form one coherent thought.
    
    POLICY:
    - DEFAULT: 1 SourceCue = 1 SemanticTranslationGroup.
    - Only groups adjacent cues when strong continuation evidence exists.
    - HARD STOPS:
      * speaker change
      * addressee change
      * discourse change (narration vs dialogue)
      * terminal punctuation (。！？)
      * question -> answer
      * character introduction beat
      * temporal gap > max_gap_seconds
      * group size reaches max_group_size (default 3)
    """

    def __init__(self, config: SemanticGroupingConfig | None = None):
        self.config = config or SemanticGroupingConfig()

    def create_groups(
        self,
        cues: list[SubtitleCue],
    ) -> list[SemanticTranslationGroup]:
        if not cues:
            return []

        groups: list[SemanticTranslationGroup] = []
        current_cues: list[SubtitleCue] = []

        for cue in cues:
            if not current_cues:
                current_cues.append(cue)
                continue

            prev = current_cues[-1]
            can_group, reason = self._can_group_adjacent(prev, cue, len(current_cues))

            if can_group:
                current_cues.append(cue)
            else:
                groups.append(self._build_group(current_cues, reason if len(current_cues) > 1 else "single_cue_default"))
                current_cues = [cue]

        if current_cues:
            groups.append(self._build_group(current_cues, "single_cue_default" if len(current_cues) == 1 else "multi_cue_continuation"))

        return groups

    def _can_group_adjacent(
        self,
        prev: SubtitleCue,
        curr: SubtitleCue,
        current_group_size: int,
    ) -> tuple[bool, str]:
        # 1. Size Limit
        if current_group_size >= self.config.max_group_size:
            return False, "max_group_size_reached"

        # 2. Hard Stop: Speaker Change
        prev_spk = prev.speaker_character_id or prev.speaker_id
        curr_spk = curr.speaker_character_id or curr.speaker_id
        if prev_spk != curr_spk:
            return False, "speaker_change"

        # 3. Hard Stop: Addressee Change
        prev_addr = prev.addressee_character_id or prev.addressee_id
        curr_addr = curr.addressee_character_id or curr.addressee_id
        if prev_addr != curr_addr:
            return False, "addressee_change"

        # 4. Hard Stop: Discourse Mode Change
        prev_mode = getattr(prev, "discourse_mode", "direct_dialogue")
        curr_mode = getattr(curr, "discourse_mode", "direct_dialogue")
        if prev_mode != curr_mode:
            return False, "discourse_mode_change"

        # 5. Hard Stop: Temporal Gap
        gap = curr.start - prev.end
        if gap > self.config.max_gap_seconds or gap < -0.2:
            return False, f"temporal_gap_{gap:.2f}s"

        # 6. Hard Stop: Terminal Punctuation on previous cue (。！？)
        prev_src = (prev.source_text or "").strip()
        curr_src = (curr.source_text or "").strip()

        if TERMINAL_ZH_PUNCTUATION.search(prev_src):
            return False, "terminal_sentence_boundary"

        # 7. Hard Stop: Question-Answer boundary
        if "？" in prev_src or "?" in prev_src:
            return False, "question_boundary"

        # 8. Hard Stop: Character Introduction Beat (e.g. 这是苏棠 followed by action/speech)
        if re.search(r"^(这是|这位是|我是)[^，,。！？!?]{2,8}$", prev_src):
            return False, "character_intro_boundary"

        # 9. Positive Continuation Evidence:
        # (A) Explicit incomplete ending in prev_src
        is_incomplete = any(re.search(pat, prev_src) for pat in INCOMPLETE_ZH_ENDINGS)
        # (B) No punctuation in prev_src and short clause length
        has_no_punct = not re.search(r"[，。！？!?；;:：]", prev_src) and len(prev_src) < 14
        # (C) Incomplete connector in curr_src
        curr_starts_continuation = bool(re.search(r"^(里|中|上|下|的|得|地|传给|带上|来看|去|到|是)", curr_src))

        if is_incomplete or curr_starts_continuation or has_no_punct:
            return True, "syntactic_clause_continuation"

        return False, "no_strong_continuation_evidence"

    def _build_group(
        self,
        cues: list[SubtitleCue],
        reason: str,
    ) -> SemanticTranslationGroup:
        c_first = cues[0]
        c_last = cues[-1]
        gid = f"grp_{c_first.id}" if len(cues) == 1 else f"grp_{c_first.id[:8]}_{c_last.id[:8]}"
        
        combined_text = "".join((c.source_text or "").strip() for c in cues)

        return SemanticTranslationGroup(
            group_id=gid,
            source_cue_ids=[c.id for c in cues],
            source_texts=[(c.source_text or "").strip() for c in cues],
            combined_source_text=combined_text,
            start=c_first.start,
            end=c_last.end,
            speaker_id=c_first.speaker_id,
            speaker_character_id=c_first.speaker_character_id,
            addressee_id=c_first.addressee_id,
            addressee_character_id=c_first.addressee_character_id,
            discourse_mode=getattr(c_first, "discourse_mode", "direct_dialogue"),
            grouping_reason=reason,
            confidence=1.0,
        )
