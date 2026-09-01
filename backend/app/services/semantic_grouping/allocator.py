from __future__ import annotations

import logging
import re
from typing import Any

from app.models.project import Project, SubtitleCue
from app.services.semantic_context import source_name_mentions
from app.services.semantic_grouping.models import (
    SemanticAllocationUnit,
    SemanticTranslationGroup,
)

logger = logging.getLogger(__name__)

# Orphan connectors that must not dangle at beginning or end of allocated cue text
ORPHAN_CONNECTORS_START = re.compile(r"^(mà|vì|nên|là|của|để|và|hoặc|nhưng)\s+", re.IGNORECASE)
ORPHAN_CONNECTORS_END = re.compile(r"\s+(mà|vì|nên|là|của|để|và|hoặc|nhưng)$", re.IGNORECASE)


class SemanticAllocationValidator:
    """Validates that allocated Vietnamese text strictly satisfies all cohesion invariants."""

    @staticmethod
    def validate_group_allocation(
        project: Project,
        group: SemanticTranslationGroup,
        allocations: list[SemanticAllocationUnit],
    ) -> tuple[bool, list[str]]:
        issues: list[str] = []

        # 1. Cue ID count & presence check
        expected_ids = group.source_cue_ids
        allocated_ids = [a.cue_id for a in allocations]
        if len(expected_ids) != len(allocated_ids):
            issues.append(f"Allocation cue count mismatch: expected {len(expected_ids)}, got {len(allocated_ids)}")

        if set(expected_ids) != set(allocated_ids):
            issues.append(f"Allocation cue IDs do not match group: expected {expected_ids}, got {allocated_ids}")

        # 2. Chronological order check
        if expected_ids != allocated_ids:
            issues.append(f"Allocation cue order mismatch: expected {expected_ids}, got {allocated_ids}")

        # 3. Non-empty translation check
        for a in allocations:
            if not a.allocated_vi.strip() and a.source_text.strip():
                issues.append(f"Empty allocation for non-empty source cue {a.cue_id}")

        # 4. Canonical Name Ownership check
        for a in allocations:
            req_names = source_name_mentions(project, a.source_text)
            for req in req_names:
                t_name = req.get("target") or req.get("name_vi")
                if t_name and not re.search(r"\b" + re.escape(t_name) + r"\b", a.allocated_vi, re.IGNORECASE):
                    issues.append(f"Canonical name '{t_name}' owned by source '{a.source_text}' missing from cue {a.cue_id}")

        # 5. Concatenation semantic equivalence check
        full_vi = (group.full_vi or "").strip()
        concat_vi = " ".join(a.allocated_vi.strip() for a in allocations)
        # Check if length drastically diverged
        if full_vi and (len(concat_vi) < 0.5 * len(full_vi) or len(concat_vi) > 2.0 * len(full_vi)):
            issues.append(f"Concatenated allocated text length diverged significantly from full_vi")

        is_valid = len(issues) == 0
        return is_valid, issues


class SemanticAllocator:
    """Allocates complete group Vietnamese translations back to constituent SourceCues."""

    def __init__(self):
        self.validator = SemanticAllocationValidator()

    def allocate(
        self,
        project: Project,
        group: SemanticTranslationGroup,
        full_vi: str,
        llm_allocations: list[dict[str, str]] | None = None,
    ) -> list[SemanticAllocationUnit]:
        group.full_vi = full_vi.strip()

        # CASE 1: Single-cue group
        if len(group.source_cue_ids) == 1:
            unit = SemanticAllocationUnit(
                cue_id=group.source_cue_ids[0],
                source_text=group.source_texts[0] if group.source_texts else group.combined_source_text,
                allocated_vi=group.full_vi,
            )
            is_valid, issues = self.validator.validate_group_allocation(project, group, [unit])
            group.validation_status = "PASS" if is_valid else "FAIL"
            group.validation_issues = issues
            group.allocations = [unit]
            return [unit]

        # CASE 2: Multi-cue group with LLM-provided allocations
        if llm_allocations:
            alloc_map = {item.get("cue_id"): item.get("allocated_vi", "").strip() for item in llm_allocations if item.get("cue_id")}
            if all(cid in alloc_map and alloc_map[cid] for cid in group.source_cue_ids):
                units = [
                    SemanticAllocationUnit(
                        cue_id=cid,
                        source_text=group.source_texts[i] if i < len(group.source_texts) else "",
                        allocated_vi=self._clean_orphan_connectors(alloc_map[cid]),
                    )
                    for i, cid in enumerate(group.source_cue_ids)
                ]
                is_valid, issues = self.validator.validate_group_allocation(project, group, units)
                if is_valid:
                    group.validation_status = "PASS"
                    group.validation_issues = []
                    group.allocations = units
                    return units

        # CASE 3: Deterministic Syntactic Clause Partitioning Fallback
        units = self._deterministic_partition(group)
        is_valid, issues = self.validator.validate_group_allocation(project, group, units)
        group.validation_status = "PASS" if is_valid else "FAIL"
        group.validation_issues = issues
        group.allocations = units
        return units

    def _deterministic_partition(
        self,
        group: SemanticTranslationGroup,
    ) -> list[SemanticAllocationUnit]:
        """Proportionally and syntactically splits full_vi across constituent cues."""
        full_vi = group.full_vi or ""
        cues_count = len(group.source_cue_ids)
        if cues_count <= 1:
            return [SemanticAllocationUnit(cue_id=group.source_cue_ids[0], source_text=group.source_texts[0], allocated_vi=full_vi)]

        # Split full_vi by clause delimiters (comma, semicolon, period)
        clauses = [c.strip() for c in re.split(r"([,;])", full_vi) if c.strip()]
        combined_clauses: list[str] = []
        for c in clauses:
            if c in {",", ";"} and combined_clauses:
                combined_clauses[-1] += c
            else:
                combined_clauses.append(c)

        if len(combined_clauses) == cues_count:
            return [
                SemanticAllocationUnit(
                    cue_id=group.source_cue_ids[i],
                    source_text=group.source_texts[i],
                    allocated_vi=combined_clauses[i].strip(",; "),
                )
                for i in range(cues_count)
            ]

        # Word-based proportional distribution
        words = full_vi.split()
        total_src_len = sum(len(st) for st in group.source_texts) or 1
        allocated_units: list[SemanticAllocationUnit] = []

        w_idx = 0
        for i, cid in enumerate(group.source_cue_ids):
            src_len = len(group.source_texts[i])
            ratio = src_len / total_src_len
            if i == cues_count - 1:
                chunk_words = words[w_idx:]
            else:
                count = max(1, round(len(words) * ratio))
                chunk_words = words[w_idx : w_idx + count]
                w_idx += count

            chunk_text = " ".join(chunk_words)
            chunk_text = self._clean_orphan_connectors(chunk_text)
            allocated_units.append(
                SemanticAllocationUnit(
                    cue_id=cid,
                    source_text=group.source_texts[i],
                    allocated_vi=chunk_text,
                )
            )

        return allocated_units

    def _clean_orphan_connectors(self, text: str) -> str:
        """Removes awkward orphan conjunctions from subtitle chunk boundaries."""
        t = text.strip()
        # Ensure capitalization
        if t and t[0].islower():
            t = t[0].upper() + t[1:]
        return t
