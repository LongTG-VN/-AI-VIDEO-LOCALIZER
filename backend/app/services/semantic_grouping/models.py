from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class SemanticAllocationUnit(BaseModel):
    cue_id: str
    source_text: str
    allocated_vi: str


class SemanticTranslationGroup(BaseModel):
    group_id: str
    source_cue_ids: list[str]
    source_texts: list[str] = Field(default_factory=list)
    combined_source_text: str
    start: float
    end: float
    speaker_id: str | None = None
    speaker_character_id: str | None = None
    addressee_id: str | None = None
    addressee_character_id: str | None = None
    discourse_mode: str = "direct_dialogue"
    grouping_reason: str = "single_cue_default"
    confidence: float = 1.0
    full_vi: str | None = None
    allocations: list[SemanticAllocationUnit] = Field(default_factory=list)
    validation_status: Literal["PASS", "FAIL"] = "PASS"
    validation_issues: list[str] = Field(default_factory=list)


class SemanticGroupingConfig(BaseModel):
    max_group_size: int = 3
    max_gap_seconds: float = 0.85
    semantic_translation_group_version: str = "v1"
