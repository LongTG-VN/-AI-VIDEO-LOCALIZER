from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class PipelineVersionManifest:
    """Central manifest recording the active version of every pipeline component."""

    pipeline_version: str = "stable-v1"
    source_integrity_version: str = "v1"
    semantic_translation_group_version: str = "v1"
    translation_quality_version: str = "v2"
    source_cover_version: str = "v11"
    subtitle_layout_version: str = "v12"
    renderer_version: str = "v12"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class SourceIntegrityPolicy:
    enabled: bool = True
    split_merged_cues: bool = True
    recover_lost_lead_ins: bool = True
    filter_non_dialogue_ocr: bool = True
    vad_temporal_tolerance_s: float = 0.20


@dataclass
class SemanticGroupingPolicy:
    enabled: bool = True
    max_source_cues_per_group: int = 2
    max_utterance_gap_s: float = 0.35
    continuation_boost: float = 1.5


@dataclass
class TranslationQualityPolicy:
    enabled: bool = True
    multi_pass_review: bool = True
    idiom_candidate_ranking: bool = True
    targeted_repair_enabled: bool = True
    enforce_relationship_consistency: bool = True
    deterministic_validation: bool = True


@dataclass
class SubtitleLayoutPolicy:
    font_name: str = "Arial"
    font_size: int = 22
    max_line_chars: int = 34
    max_backing_width_ratio: float = 0.70
    padding_x: int = 12
    padding_y: int = 8
    wrap_natural_vietnamese: bool = True


@dataclass
class SourceCoverPolicy:
    enabled: bool = True
    plate_opacity: float = 0.62
    blur_radius: int = 8
    frame_local_glyph_tracking: bool = True
    suppressed_filler_cover: bool = True
    zero_gap_handoff: bool = True


@dataclass
class RendererPolicy:
    preferred_encoder: Literal["h264_nvenc", "libx264"] = "h264_nvenc"
    pixel_format: str = "yuv420p"
    video_profile: str = "Main"
    faststart: bool = True
    nvenc_preset: str = "p4"
    nvenc_cq: int = 20
    libx264_preset: str = "medium"
    libx264_crf: int = 18


@dataclass
class QAPolicy:
    strict_invariants: bool = True
    verify_pixel_contrast: bool = True
    max_uncovered_frames: int = 0
    max_readable_chinese_frames: int = 0
    max_stale_vi_frames: int = 0


@dataclass
class LocalizationPolicy:
    """Master runtime policy configuration for the canonical localization pipeline."""

    profile: Literal["stable", "debug"] = "stable"
    version_manifest: PipelineVersionManifest = field(default_factory=PipelineVersionManifest)
    source_integrity: SourceIntegrityPolicy = field(default_factory=SourceIntegrityPolicy)
    semantic_grouping: SemanticGroupingPolicy = field(default_factory=SemanticGroupingPolicy)
    translation_quality: TranslationQualityPolicy = field(default_factory=TranslationQualityPolicy)
    subtitle_layout: SubtitleLayoutPolicy = field(default_factory=SubtitleLayoutPolicy)
    source_cover: SourceCoverPolicy = field(default_factory=SourceCoverPolicy)
    renderer: RendererPolicy = field(default_factory=RendererPolicy)
    qa: QAPolicy = field(default_factory=QAPolicy)

    def compute_config_hash(self) -> str:
        """Deterministic SHA-256 hash of active policy configuration."""
        data_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()[:16]


# Cache Dependency and Invalidation Graph
CACHE_DEPENDENCY_GRAPH = {
    "import": [],
    "asr": ["import"],
    "ocr": ["import"],
    "fusion": ["asr", "ocr"],
    "source_integrity": ["fusion"],
    "semantic_grouping": ["source_integrity"],
    "context_graph": ["source_integrity"],
    "translation": ["semantic_grouping", "context_graph"],
    "translation_quality": ["translation"],
    "source_cover_timeline": ["source_integrity", "ocr"],
    "subtitle_layout": ["translation_quality"],
    "renderer": ["source_cover_timeline", "subtitle_layout"],
    "qa": ["renderer"],
}

INVALIDATION_RULES = {
    "asr_changed": ["fusion", "source_integrity", "semantic_grouping", "context_graph", "translation", "translation_quality", "source_cover_timeline", "subtitle_layout", "renderer", "qa"],
    "ocr_changed": ["fusion", "source_integrity", "semantic_grouping", "context_graph", "translation", "translation_quality", "source_cover_timeline", "subtitle_layout", "renderer", "qa"],
    "source_integrity_changed": ["semantic_grouping", "context_graph", "translation", "translation_quality", "source_cover_timeline", "subtitle_layout", "renderer", "qa"],
    "translation_changed": ["translation_quality", "subtitle_layout", "renderer", "qa"],
    "cover_or_layout_changed": ["renderer", "qa"],
    "renderer_changed": ["qa"],
}
