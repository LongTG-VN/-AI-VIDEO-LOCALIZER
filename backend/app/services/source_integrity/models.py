from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import uuid4
from pydantic import BaseModel, Field

from app.models.project import OCREvidence, OCRRegion, SubtitleCue


class SourceIntegrityStatus(str, Enum):
    PASS = "PASS"
    REPAIRED = "REPAIRED"
    SOURCE_NEEDS_REVIEW = "SOURCE_NEEDS_REVIEW"


class OCRInterval(BaseModel):
    """Normalized and deduplicated stable OCR subtitle interval."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    normalized_text: str
    raw_text: str
    confidence: float = Field(default=0.85, ge=0, le=1)
    geometry: list[list[float]] = Field(default_factory=list)
    regions: list[OCRRegion] = Field(default_factory=list)
    is_dialogue: bool = True
    match_score: float = 0.0


class SourceTokenCorrection(BaseModel):
    """Provenance tracking for an ASR span corrected via OCR / reconciliation."""

    original_asr: str
    corrected_text: str
    evidence: str
    confidence: float = Field(default=0.85, ge=0, le=1)
    source: str = "ocr_reconciliation"
    span_start: int | None = None
    span_end: int | None = None


class SourceCue(BaseModel):
    """Repaired and validated source cue ready for downstream translation."""

    cue_id: str = Field(default_factory=lambda: str(uuid4()))
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    source_text: str
    original_source_cue_ids: list[str] = Field(default_factory=list)
    segmentation_method: str | None = None
    source_integrity_status: SourceIntegrityStatus = SourceIntegrityStatus.PASS
    asr_text: str | None = None
    ocr_text: str | None = None
    speaker_id: str | None = None
    speaker_character_id: str | None = None
    discourse_mode: Literal["direct_dialogue", "monologue", "narration", "system", "unknown"] = "unknown"
    source_confidence: float = Field(default=0.85, ge=0, le=1)
    corrections: list[SourceTokenCorrection] = Field(default_factory=list)
    ocr_start: float | None = None
    ocr_end: float | None = None
    ocr_regions: list[OCRRegion] = Field(default_factory=list)
    ocr_evidence: list[OCREvidence] = Field(default_factory=list)
    translated_text: str | None = None
    word_timestamps: list[dict[str, Any]] = Field(default_factory=list)

    def to_subtitle_cue(self) -> SubtitleCue:
        return SubtitleCue(
            id=self.cue_id,
            start=self.start,
            end=self.end,
            speaker_id=self.speaker_id,
            speaker_character_id=self.speaker_character_id,
            discourse_mode=self.discourse_mode,
            source_text=self.source_text,
            translated_text=self.translated_text,
            confidence=self.source_confidence,
            ocr_start=self.ocr_start,
            ocr_end=self.ocr_end,
            ocr_text=self.ocr_text,
            ocr_regions=self.ocr_regions,
            ocr_evidence=self.ocr_evidence,
            original_source_cue_ids=self.original_source_cue_ids or [self.cue_id],
            source_integrity_status=self.source_integrity_status.value,
            source_confidence=self.source_confidence,
            segmentation_method=self.segmentation_method,
            source_corrections=[c.model_dump() for c in self.corrections],
            word_timestamps=self.word_timestamps,
        )


class SourceIntegrityConfig(BaseModel):
    enabled: bool = True
    source_integrity_version: str = "v1"
    min_duration_anomaly_s: float = 3.5
    max_ocr_turnover_per_cue: int = 1
    min_ocr_dialogue_confidence: float = 0.35
    dialogue_y_min: float = 0.55  # Subtitle band geometry: dialogue is in bottom half
    dialogue_y_max: float = 0.98
    split_similarity_threshold: float = 0.20


class SourceIntegrityReport(BaseModel):
    total_source_cues: int = 0
    passed: int = 0
    repaired: int = 0
    needs_review: int = 0
    overmerge_detected: int = 0
    split_count: int = 0
    ocr_corrections: int = 0
    speaker_boundary_splits: int = 0
    vad_boundary_splits: int = 0
    processing_time_s: float = 0.0
    source_integrity_version: str = "v1"
    details: list[dict[str, Any]] = Field(default_factory=list)
