from __future__ import annotations

import logging
import re
from typing import Any

from app.models.project import Project, SubtitleCue
from app.services.translation_quality.models import (
    FillerReviewResult,
    QualityIssue,
    QualitySeverity,
    TranslationContextCard,
)

logger = logging.getLogger(__name__)

# Standard Chinese short conversational fillers
CHINESE_FILLERS = {"嗯", "啊", "哦", "诶", "哎", "欸", "哈", "哇", "呀", "唔", "咦"}

FILLER_VI_MAP = {
    "嗯": "Ừm.",
    "哦": "Ồ.",
    "啊": "À.",
    "诶": "Ơ.",
    "哎": "Ôi.",
    "欸": "Này.",
    "哈": "Ha.",
    "哇": "Oa.",
    "呀": "Á.",
}


class FillerHandler:
    """Pass 5: Conversational Filler Reviewer and Normalizer.
    
    Ensures short conversational fillers are translated naturally into Vietnamese,
    rejects translations where fillers become unrelated multi-word sentences,
    and supports explicit filler suppression when appropriate.
    """

    def evaluate_cues(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None = None,
    ) -> tuple[dict[str, list[QualityIssue]], dict[str, FillerReviewResult], dict[str, str]]:
        """Returns (issues_by_id, filler_reviews, normalized_filler_translations)."""
        issues_by_id: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}
        filler_reviews: dict[str, FillerReviewResult] = {}
        normalized_translations: dict[str, str] = {}

        for cue in cues:
            src = (cue.source_text or "").strip()
            vi = (cue.translated_text or "").strip()

            # Clean Chinese text of punctuation
            src_clean = re.sub(r"[，。！？、“”《》；：,.!?\s]", "", src)

            # Check if cue is a standalone Chinese filler
            if src_clean in CHINESE_FILLERS:
                expected_vi = FILLER_VI_MAP.get(src_clean, "Ừm.")

                is_valid_filler_vi = bool(re.search(r"\b(ừm|ừ|vâng|dạ|ồ|à|ơi|này|ôi|ha|oa|á)\b", vi, re.IGNORECASE))
                if len(vi.split()) > 2 or not is_valid_filler_vi:
                    issues_by_id[cue.id].append(
                        QualityIssue(
                            type="filler.unrelated_lexical_translation",
                            severity=QualitySeverity.CRITICAL,
                            message=f"Short filler '{src}' was translated into unrelated lexical sentence: '{vi}'",
                            source_span=src,
                            target_span=vi,
                            reviewer="fillers",
                        )
                    )
                    filler_reviews[cue.id] = FillerReviewResult(
                        cue_id=cue.id,
                        filler_token=src_clean,
                        is_filler=True,
                        action="TRANSLATE",
                        translated_vi=expected_vi,
                        suppression_reason=None,
                    )
                    normalized_translations[cue.id] = expected_vi
                else:
                    # Normalized standard filler
                    if not vi or len(vi.split()) > 2:
                        normalized_translations[cue.id] = expected_vi
                    filler_reviews[cue.id] = FillerReviewResult(
                        cue_id=cue.id,
                        filler_token=src_clean,
                        is_filler=True,
                        action="TRANSLATE",
                        translated_vi=normalized_translations.get(cue.id, vi),
                        suppression_reason=None,
                    )

        return issues_by_id, filler_reviews, normalized_translations
