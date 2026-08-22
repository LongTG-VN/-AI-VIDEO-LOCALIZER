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
    translation_confidence: float | None = Field(default=None, ge=0, le=1)
    relationship_confidence: float | None = Field(default=None, ge=0, le=1)
    needs_review: bool = False
    review_notes: str | None = None
    critic_score: float | None = None
    critic_flags: list[str] = Field(default_factory=list)


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
    margin_v: int = Field(default=36, ge=0, le=400)


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
