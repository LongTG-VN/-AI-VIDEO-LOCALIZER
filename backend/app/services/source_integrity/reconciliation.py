from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from app.models.project import OCRRegion
from app.services.fusion import align_and_correct_span, clean_chinese_text, text_similarity
from app.services.source_integrity.models import (
    OCRInterval,
    SourceIntegrityConfig,
    SourceTokenCorrection,
)

logger = logging.getLogger(__name__)


class SourceReconciler:
    """Reconciles ASR source text with high-confidence dialogue OCR text."""

    def __init__(self, config: SourceIntegrityConfig | None = None):
        self.config = config or SourceIntegrityConfig()

    def filter_dialogue_ocr_intervals(
        self,
        intervals: list[OCRInterval],
    ) -> list[OCRInterval]:
        """Filters out non-dialogue OCR (watermarks, top titles, UI text)."""
        valid_dialogue: list[OCRInterval] = []
        for interval in intervals:
            # 1. Geometry check
            is_in_dialogue_band = True
            if interval.geometry:
                # Check average y of points
                ys = [p[1] for p in interval.geometry if len(p) > 1]
                if ys:
                    avg_y = sum(ys) / len(ys)
                    if avg_y < self.config.dialogue_y_min or avg_y > self.config.dialogue_y_max:
                        is_in_dialogue_band = False
            elif interval.regions:
                all_ys = [p[1] for r in interval.regions for p in r.points if len(p) > 1]
                if all_ys:
                    avg_y = sum(all_ys) / len(all_ys)
                    if avg_y < self.config.dialogue_y_min or avg_y > self.config.dialogue_y_max:
                        is_in_dialogue_band = False

            # 2. Confidence check
            if interval.confidence < self.config.min_ocr_dialogue_confidence:
                is_in_dialogue_band = False

            interval.is_dialogue = is_in_dialogue_band
            if is_in_dialogue_band:
                valid_dialogue.append(interval)
            else:
                logger.debug("Filtered out non-dialogue OCR interval: '%s' (conf=%.2f)", interval.raw_text, interval.confidence)

        return valid_dialogue

    def reconcile_segment_text(
        self,
        asr_text: str,
        matched_ocr: list[OCRInterval],
    ) -> tuple[str, list[SourceTokenCorrection]]:
        """Reconciles ASR text against matched dialogue OCR intervals."""
        corrections: list[SourceTokenCorrection] = []
        if not matched_ocr or not asr_text:
            return asr_text, corrections

        current_text = asr_text
        for ocr in matched_ocr:
            if not ocr.is_dialogue or ocr.confidence < self.config.min_ocr_dialogue_confidence:
                continue

            ocr_clean = clean_chinese_text(ocr.normalized_text)
            if len(ocr_clean) < 2:
                continue

            sim = text_similarity(current_text, ocr.normalized_text)

            # CASE A: ASR and OCR already strictly identical
            if ocr_clean == clean_chinese_text(current_text):
                continue

            # CASE B: OCR is high confidence and temporally aligned phonetic/dialogue correction
            # Try fuzzy span alignment
            matcher = SequenceMatcher(None, current_text, ocr.normalized_text)
            match = matcher.find_longest_match(0, len(current_text), 0, len(ocr.normalized_text))
            
            # If high confidence OCR aligns with the cue interval
            if ocr.confidence >= 0.80 and (match.size >= 1 or sim >= self.config.split_similarity_threshold or len(clean_chinese_text(current_text)) <= len(ocr_clean) + 4):
                corr = SourceTokenCorrection(
                    original_asr=asr_text,
                    corrected_text=ocr.normalized_text,
                    evidence=f"OCR interval [{ocr.start:.2f}s - {ocr.end:.2f}s]: '{ocr.raw_text}' (conf={ocr.confidence:.2f})",
                    confidence=ocr.confidence,
                    source="ocr_reconciliation",
                )
                if corr.corrected_text != current_text:
                    current_text = corr.corrected_text
                    corrections.append(corr)

        return current_text, corrections
