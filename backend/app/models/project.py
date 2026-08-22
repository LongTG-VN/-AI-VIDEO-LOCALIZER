from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class SubtitleCue(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    speaker_id: str | None = None
    speaker_character_id: str | None = None
    addressee_id: str | None = None
    addressee_character_id: str | None = None
    source_text: str
    translated_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    asr_confidence: float | None = Field(default=None, ge=0, le=1)
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    # Visual hard-sub timing is intentionally preserved independently from the
    # ASR-backed dialogue timing used by the rest of the application.  The
    # cleaner must follow what is actually visible on screen, not how long the
    # spoken utterance lasts.
    ocr_start: float | None = Field(default=None, ge=0)
    ocr_end: float | None = Field(default=None, gt=0)
    ocr_text: str | None = None
    ocr_regions: list[OCRRegion] = Field(default_factory=list)
    translation_confidence: float | None = Field(default=None, ge=0, le=1)
    relationship_confidence: float | None = Field(default=None, ge=0, le=1)
    needs_review: bool = False
    review_notes: str | None = None
    critic_score: float | None = None
    critic_flags: list[str] = Field(default_factory=list)


class OCRRegion(BaseModel):
    text: str | None = None
    confidence: float | None = None
    points: list[list[float]] = Field(default_factory=list)  # normalized [0, 1] relative to full frame: [[x, y], ...]


class Character(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    name_zh: str | None = None
    name_vi: str | None = None
    aliases: list[str] = Field(default_factory=list)
    gender: str | None = None
    role: str | None = None
    description: str | None = None
    speaker_ids: list[str] = Field(default_factory=list)
    confidence: float | None = None
    notes: str | None = None


class RelationshipRule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    from_character_id: str
    to_character_id: str
    relationship: str
    relationship_type: str | None = None
    valid_from: float = 0
    valid_until: float | None = None
    vi_self: str | None = None
    vi_other: str | None = None
    vi_self_pronoun: str | None = None
    vi_target_pronoun: str | None = None
    en_register: str | None = None
    confidence: float | None = None
    notes: str | None = None


class GlossaryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    target: str
    category: str | None = None
    confidence: float | None = None
    note: str | None = None


class Scene(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    scene_id: str | None = None
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    summary: str
    tone: str | None = None
    characters: list[str] = Field(default_factory=list)


class StickerOverlay(BaseModel):
    path: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    x: str = "W-w-24"
    y: str = "24"
    scale_width: int = Field(default=180, ge=16, le=4096)


class RenderOptions(BaseModel):
    intro_path: str | None = None
    outro_path: str | None = None
    stickers: list[StickerOverlay] = Field(default_factory=list)
    font_name: str = "Arial"
    font_size: int = Field(default=22, ge=8, le=96)
    margin_v: int = Field(default=32, ge=0, le=400)
    hardsub_removal_mode: Literal["none", "inpaint", "quality", "cover", "auto"] = "auto"
    hardsub_crop_top_ratio: float = Field(default=0.65, ge=0.0, lt=1.0)
    hardsub_crop_bottom_ratio: float = Field(default=0.95, gt=0.0, le=1.0)
    hardsub_crop_left_ratio: float = Field(default=0.06, ge=0.0, lt=1.0)
    hardsub_crop_right_ratio: float = Field(default=0.94, gt=0.0, le=1.0)
    hardsub_mask_dilate_radius: int = Field(default=1, ge=0, le=6)
    hardsub_mask_dilate_iterations: int = Field(default=1, ge=1, le=4)
    hardsub_inpaint_radius: int = Field(default=2, ge=1, le=8)
    hardsub_local_contrast_threshold: int = Field(default=18, ge=0, le=255)
    hardsub_max_mask_coverage: float = Field(default=0.12, gt=0.0, le=0.5)
    hardsub_scene_cut_threshold: float = Field(default=34.0, gt=0.0, le=255.0)
    hardsub_temporal_max_distance_frames: int = Field(default=45, ge=0, le=300)
    hardsub_temporal_difference_threshold: int = Field(default=14, ge=1, le=255)
    hardsub_temporal_local_score_threshold: float = Field(default=22.0, gt=0.0, le=255.0)
    hardsub_ocr_min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    hardsub_geometry_enabled: bool = True
    hardsub_geometry_padding_px: int = Field(default=4, ge=0, le=32)
    hardsub_lossless_intermediate: bool = True
    subtitle_format: Literal["ass", "srt"] = "ass"
    font_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    outline_width: float = Field(default=2.5, ge=0.0, le=10.0)
    shadow_depth: float = Field(default=1.0, ge=0.0, le=10.0)
    max_line_chars: int = Field(default=36, ge=10, le=100)
    use_nvenc: bool = True


class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    source_video_path: str
    source_language: str = "zh"
    target_language: Literal["vi", "en"] = "vi"
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scenes: list[Scene] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)
    relationships: list[RelationshipRule] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    cues: list[SubtitleCue] = Field(default_factory=list)


class ProjectPatch(BaseModel):
    name: str | None = None
    target_language: Literal["vi", "en"] | None = None
    scenes: list[Scene] | None = None
    characters: list[Character] | None = None
    relationships: list[RelationshipRule] | None = None
    glossary: list[GlossaryEntry] | None = None
    cues: list[SubtitleCue] | None = None
