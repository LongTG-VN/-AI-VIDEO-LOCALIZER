from __future__ import annotations

import logging
from app.services.source_integrity.models import SourceCue, SourceIntegrityStatus

logger = logging.getLogger(__name__)


class SourceIntegrityValidator:
    """Validates source cue invariants prior to downstream translation."""

    def validate_source_cues(
        self,
        cues: list[SourceCue],
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []
        
        if not cues:
            return True, errors

        for i, cue in enumerate(cues):
            # 1. Non-empty source text
            if not cue.source_text or not cue.source_text.strip():
                errors.append(f"Cue {cue.cue_id} has empty source_text")

            # 2. Timing validity
            if cue.start < 0 or cue.end <= cue.start:
                errors.append(f"Cue {cue.cue_id} has invalid timing: {cue.start} -> {cue.end}")

            # 3. Chronological timeline ordering
            if i > 0 and cue.start < cues[i-1].start - 0.05:
                errors.append(f"Cue {cue.cue_id} starts before preceding cue: {cue.start} < {cues[i-1].start}")

            # 4. Provenance tracking
            if not cue.original_source_cue_ids:
                errors.append(f"Cue {cue.cue_id} has missing original_source_cue_ids provenance")

        is_valid = len(errors) == 0
        logger.info("Source Integrity Validation: PASS=%s (errors=%d)", is_valid, len(errors))
        return is_valid, errors
