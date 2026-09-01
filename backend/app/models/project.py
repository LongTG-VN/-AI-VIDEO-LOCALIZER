from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class OCRRegion(BaseModel):
    text: str | None = None
    confidence: float | None = None
    points: list[list[float]] = Field(default_factory=list)  # normalized [0, 1] relative to full frame: [[x, y], ...]


class OCREvidence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    confidence: float | None = None
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    regions: list[OCRRegion] = Field(default_factory=list)
    matched_span_start: int | None = None
    matched_span_end: int | None = None
    match_score: float | None = None


class SubtitleCue(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    speaker_id: str | None = None
    speaker_character_id: str | None = None
    addressee_id: str | None = None
    addressee_character_id: str | None = None
    discourse_mode: Literal["direct_dialogue", "monologue", "narration", "system", "unknown"] = "unknown"
    source_text: str
    translated_text: str | None = None
    draft_translation: str | None = None
    quality_status: Literal["PASS", "REPAIRED", "NEEDS_REVIEW", "PENDING"] = "PENDING"
    repaired_translation: str | None = None
    final_translation: str | None = None
    quality_version: str | None = "v2"
    confidence: float | None = Field(default=None, ge=0, le=1)
    asr_confidence: float | None = Field(default=None, ge=0, le=1)
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    # Visual hard-sub timing is intentionally preserved independently from the
    # ASR-backed dialogue timing used by the rest of the application. The
    # cleaner must follow what is actually visible on screen, not how long the
    # spoken utterance lasts.
    ocr_start: float | None = Field(default=None, ge=0)
    ocr_end: float | None = Field(default=None, gt=0)
    ocr_text: str | None = None
    ocr_regions: list[OCRRegion] = Field(default_factory=list)
    ocr_evidence: list[OCREvidence] = Field(default_factory=list)
    translation_confidence: float | None = Field(default=None, ge=0, le=1)
    needs_review: bool = False
    review_notes: str | None = None
    suppression_status: str | None = None
    suppression_reason: str | None = None
    critic_score: float | None = None
    critic_flags: list[str] = Field(default_factory=list)
    original_source_cue_ids: list[str] = Field(default_factory=list)
    source_integrity_status: Literal["PASS", "REPAIRED", "UNRESOLVED", "MANUAL_SPLIT"] = "PASS"
    source_integrity_reason: str | None = None
    source_confidence: float | None = Field(default=None, ge=0, le=1)
    segmentation_method: str | None = None
    source_corrections: list[dict[str, Any]] = Field(default_factory=list)
    word_timestamps: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def get_final_vi_text(cue: SubtitleCue) -> str:
    """Canonical accessor for finalized Vietnamese subtitle translation.

    Contract:
    - If cue has suppression_status in SUPPRESSED_FILLER or SUPPRESSED_NONSEMANTIC_DIALOGUE, returns ""
    - If cue is non-speech noise (e.g. 'AL', '10:50'), returns ""
    - Returns final_translation if non-empty and valid Vietnamese
    - Otherwise returns repaired_translation if non-empty and valid Vietnamese
    - Otherwise returns translated_text if non-empty, not identical to source_text, and valid Vietnamese
    - Otherwise returns draft_translation if non-empty and valid Vietnamese
    - NEVER silently returns Chinese source when translated text is requested
    """
    if getattr(cue, "suppression_status", None) in {"SUPPRESSED_FILLER", "SUPPRESSED_NONSEMANTIC_DIALOGUE"}:
        return ""

    raw_src = (getattr(cue, "source_text", "") or "").strip()
    if raw_src in {"AL", "10:50", "..."}:
        return ""

    for val in [
        getattr(cue, "final_translation", None),
        getattr(cue, "repaired_translation", None),
        getattr(cue, "translated_text", None),
        getattr(cue, "draft_translation", None),
    ]:
        if val is not None and str(val).strip():
            s = str(val).strip()
            if s == (cue.source_text or "").strip():
                continue
            import re
            has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
            has_latin = bool(re.search(r"[a-zA-Zà-ỹÀ-Ỹ]", s))
            if has_chinese and not has_latin:
                continue
            return s
    return ""


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


class VisualEditMode(str, Enum):
    CLEAN = "clean"
    BLUR = "blur"
    PATCH_COVER = "patch_cover"
    BLUR_OVERLAY = "blur_overlay"


class OverlayAnchor(str, Enum):
    ABSOLUTE = "absolute"
    SUBTITLE_REGION = "subtitle_region"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"


class BlurConfig(BaseModel):
    enabled: bool = True
    sigma: float = Field(default=18.0, ge=1.0, le=60.0)
    padding_px: int = Field(default=8, ge=0, le=64)
    feather_px: int = Field(default=6, ge=0, le=32)
    min_ocr_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    temporal_gap_fill_frames: int = Field(default=5, ge=0, le=30)


class PatchCoverConfig(BaseModel):
    enabled: bool = True
    patch_opacity: float = Field(default=0.98, ge=0.5, le=1.0)
    dark_tint: float = Field(default=0.48, ge=0.0, le=1.0)  # effective suppression of bright glyphs
    padding_px: int = Field(default=7, ge=0, le=32)
    feather_px: int = Field(default=8, ge=0, le=32)
    blur_sigma: float = Field(default=10.0, ge=1.0, le=30.0)
    min_ocr_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    temporal_gap_fill_frames: int = Field(default=6, ge=0, le=30)
    mask_persistence_frames: int = Field(default=3, ge=0, le=15)
    use_temporal_donor: bool = True
    use_spatial_donor: bool = True


class OverlayConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    path: str
    start: float = Field(default=0.0, ge=0.0)
    end: float = Field(default=10.0, gt=0.0)
    x: float = Field(default=0.5, ge=0.0, le=1.0)  # normalized x coordinate [0.0, 1.0]
    y: float = Field(default=0.5, ge=0.0, le=1.0)  # normalized y coordinate [0.0, 1.0]
    width: float = Field(default=0.25, gt=0.0, le=1.0)  # normalized width relative to video width
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    fade_in_ms: int = Field(default=200, ge=0, le=5000)
    fade_out_ms: int = Field(default=200, ge=0, le=5000)
    z_index: int = Field(default=10, ge=0)
    anchor: OverlayAnchor = OverlayAnchor.ABSOLUTE


class SubtitleBackingConfig(BaseModel):
    enabled: bool = True
    color: str = "&H00000000"  # black
    opacity: float = Field(default=0.72, ge=0.0, le=1.0)  # slightly increased for bright scenes
    padding_x: int = Field(default=20, ge=0, le=64)
    padding_y: int = Field(default=10, ge=0, le=32)
    corner_radius: int = Field(default=10, ge=0, le=32)
    blur_radius: int = Field(default=8, ge=0, le=32)


class VisualEditConfig(BaseModel):
    mode: VisualEditMode = VisualEditMode.CLEAN
    blur: BlurConfig = Field(default_factory=BlurConfig)
    patch_cover: PatchCoverConfig = Field(default_factory=PatchCoverConfig)
    subtitle_backing: SubtitleBackingConfig = Field(default_factory=SubtitleBackingConfig)
    overlays: list[OverlayConfig] = Field(default_factory=list)
    preset: Literal[
        "default",
        "shortform_reference",
        "shortform_bold_yellow",
        "shortform_white_black_soft_bg",
        "shortform_soft_bg",
    ] = "default"


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
    visual_edit: VisualEditConfig | None = None
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
    visual_edit: VisualEditConfig | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scenes: list[Scene] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)
    relationships: list[RelationshipRule] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    cues: list[SubtitleCue] = Field(default_factory=list)
    translation_quality: dict[str, Any] | None = None
    source_integrity: dict[str, Any] | None = None


class CharacterVoiceProfile(BaseModel):
    character_id: str
    voice_id: str = "vi-VN-HoaiMyNeural"
    gender_style: str | None = None
    age_style: str | None = None
    base_rate: str = "+0%"
    pitch_offset: str = "+0Hz"
    volume: str = "+0%"


class DubbingOptions(BaseModel):
    tts_engine: str = "edge"
    separation_engine: str = "demucs"  # "demucs" | "overdub_fallback"
    ducking_dialogue_db: float = Field(default=-24.0, le=0.0)
    ducking_music_db: float = Field(default=-3.5, le=0.0)
    crossfade_ms: int = Field(default=20, ge=5, le=100)
    target_lufs: float = -16.0
    max_acceptable_speed: float = Field(default=1.25, ge=1.0, le=1.6)
    min_acceptable_speed: float = Field(default=0.90, ge=0.7, le=1.0)
    voice_profiles: dict[str, CharacterVoiceProfile] = Field(default_factory=dict)


class DubbingMetrics(BaseModel):
    total_cues: int = 0
    synthesized_cues: int = 0
    succeeded_cues: int = 0
    failed_cues: int = 0
    time_stretched_cues: int = 0
    llm_compressed_cues: int = 0
    still_overlong_cues: int = 0
    avg_speed_factor: float = 1.0
    max_speed_factor: float = 1.0
    separation_mode: str = "demucs"
    separation_duration_s: float = 0.0
    mixing_duration_s: float = 0.0
    final_duration_s: float = 0.0
    final_peak_db: float = 0.0


class ProjectPatch(BaseModel):
    name: str | None = None
    target_language: Literal["vi", "en"] | None = None
    scenes: list[Scene] | None = None
    characters: list[Character] | None = None
    relationships: list[RelationshipRule] | None = None
    glossary: list[GlossaryEntry] | None = None
    cues: list[SubtitleCue] | None = None
    translation_quality: dict[str, Any] | None = None
