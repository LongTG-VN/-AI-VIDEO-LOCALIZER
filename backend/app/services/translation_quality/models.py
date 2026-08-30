from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, Field


class QualitySeverity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class QualityIssue(BaseModel):
    type: str  # e.g. "cue.content_migration", "accuracy.mistranslation", "relationship.polysemy_mismatch", etc.
    severity: QualitySeverity = QualitySeverity.MAJOR
    message: str
    source_span: str | None = None
    target_span: str | None = None
    reviewer: str = "general"


class CueQualityResult(BaseModel):
    cue_id: str
    status: Literal["PASS", "FAIL", "NEEDS_REVIEW"] = "PASS"
    issues: list[QualityIssue] = Field(default_factory=list)
    attempts: int = 0
    original_translation: str = ""
    final_translation: str = ""
    confidence: float | None = None
    review_notes: str | None = None


class TranslationQualityReport(BaseModel):
    total_cues: int = 0
    passed_first_attempt: int = 0
    repaired: int = 0
    needs_review: int = 0
    issue_counts: dict[str, int] = Field(default_factory=dict)
    cue_results: dict[str, CueQualityResult] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class CharacterCard(BaseModel):
    character_id: str
    canonical_name: str
    name_zh: str | None = None
    name_vi: str | None = None
    aliases: list[str] = Field(default_factory=list)
    gender_if_known: str | None = None
    role: str | None = None
    description: str | None = None


class RelationshipCard(BaseModel):
    from_character_id: str
    to_character_id: str
    type: str
    preferred_vi_pronouns: dict[str, str] = Field(default_factory=dict)  # {"speaker": "...", "listener": "..."}
    confidence: float | None = None


class TranslationContextCard(BaseModel):
    story_summary: str | None = None
    genre: str | None = None
    tone: str | None = None
    characters: list[CharacterCard] = Field(default_factory=list)
    relationships: list[RelationshipCard] = Field(default_factory=list)
    terminology: dict[str, str] = Field(default_factory=dict)
    ambiguous_terms: list[str] = Field(default_factory=list)
    style_rules: dict[str, str] = Field(default_factory=dict)


class TranslationQualityConfig(BaseModel):
    enabled: bool = True
    context_card: bool = True
    cue_integrity: bool = True
    accuracy: bool = True
    relationships: bool = True
    targeted_repair: bool = True
    max_retries: int = 2
    naturalness: bool = True
    consistency: bool = True
    deterministic_validation: bool = True
