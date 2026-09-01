from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from app.models.project import Project, SubtitleCue
from app.services.semantic_grouping.grouper import SemanticGrouper
from app.services.semantic_grouping.models import (
    SemanticGroupingConfig,
    SemanticTranslationGroup,
)
from app.services.semantic_grouping.translator import GroupTranslator

logger = logging.getLogger(__name__)


class SemanticGroupingPipeline:
    """Coordinates Semantic Translation Grouping, Group Translation, and Cue Allocation."""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        config: SemanticGroupingConfig | None = None,
    ):
        self.config = config or SemanticGroupingConfig()
        self.grouper = SemanticGrouper(self.config)
        self.translator = GroupTranslator(base_url, api_key, model)

    def process_project(
        self,
        project: Project,
        save_trace_dir: Path | str | None = None,
    ) -> list[SemanticTranslationGroup]:
        cues = project.cues
        if not cues:
            return []

        # 1. Form Groups
        groups = self.grouper.create_groups(cues)
        logger.info(
            "Created %d SemanticTranslationGroups from %d cues (%d multi-cue groups)",
            len(groups),
            len(cues),
            sum(1 for g in groups if len(g.source_cue_ids) > 1),
        )

        # 2. Translate and Allocate
        groups = self.translator.translate_and_allocate_groups(project, groups)

        # 3. Apply Allocated VI back to SubtitleCue.translated_text
        cue_map = {c.id: c for c in cues}
        for grp in groups:
            for alloc in grp.allocations:
                if alloc.cue_id in cue_map:
                    target_cue = cue_map[alloc.cue_id]
                    target_cue.translated_text = alloc.allocated_vi
                    target_cue.draft_translation = alloc.allocated_vi
                    target_cue.final_translation = alloc.allocated_vi
                    target_cue.quality_status = "PASS"
                    target_cue.quality_version = self.config.semantic_translation_group_version

        # 4. Save Tracing Artifacts
        if save_trace_dir:
            out_dir = Path(save_trace_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            trace_data = [grp.model_dump() for grp in groups]
            with open(out_dir / "semantic_translation_groups.json", "w", encoding="utf-8") as f:
                json.dump(trace_data, f, ensure_ascii=False, indent=2)

        return groups
