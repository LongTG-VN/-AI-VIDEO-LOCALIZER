from __future__ import annotations

import logging
import re
from typing import Any
from uuid import uuid4

from app.models.project import OCREvidence, OCRRegion, SubtitleCue
from app.services.fusion import clean_chinese_text, text_similarity
from app.services.source_integrity.models import (
    OCRInterval,
    SourceCue,
    SourceIntegrityConfig,
    SourceIntegrityStatus,
    SourceTokenCorrection,
)
from app.services.source_integrity.reconciliation import SourceReconciler

logger = logging.getLogger(__name__)


class SourceSegmenter:
    """Detects suspicious over-merged source cues and re-segments them along multi-modal boundaries."""

    def __init__(self, config: SourceIntegrityConfig | None = None):
        self.config = config or SourceIntegrityConfig()
        self.reconciler = SourceReconciler(self.config)

    def extract_stable_ocr_intervals(
        self,
        ocr_cues: list[SubtitleCue] | list[dict[str, Any]],
    ) -> list[OCRInterval]:
        """Builds deduplicated, chronological OCR intervals from raw OCR cues."""
        raw_intervals: list[OCRInterval] = []
        for c in ocr_cues:
            st = float(c.start if hasattr(c, "start") else c.get("start", 0))
            en = float(c.end if hasattr(c, "end") else c.get("end", 0))
            txt = str(c.source_text if hasattr(c, "source_text") else c.get("source_text", c.get("text", ""))).strip()
            conf = float(c.ocr_confidence if hasattr(c, "ocr_confidence") and c.ocr_confidence is not None else (c.get("ocr_confidence") or c.get("confidence", 0.85)))
            regions = getattr(c, "ocr_regions", []) if hasattr(c, "ocr_regions") else [OCRRegion(**r) for r in c.get("ocr_regions", [])]
            
            pts = []
            for r in regions:
                if r.points:
                    pts.extend(r.points)
            
            if txt and en > st:
                raw_intervals.append(
                    OCRInterval(
                        id=str(getattr(c, "id", uuid4())),
                        start=st,
                        end=en,
                        normalized_text=txt,
                        raw_text=txt,
                        confidence=conf,
                        geometry=pts,
                        regions=regions,
                    )
                )

        # Deduplicate overlapping intervals with same normalized text
        raw_intervals.sort(key=lambda x: (x.start, x.end))
        deduped: list[OCRInterval] = []
        for iv in raw_intervals:
            if not deduped:
                deduped.append(iv)
                continue
            prev = deduped[-1]
            if clean_chinese_text(prev.normalized_text) == clean_chinese_text(iv.normalized_text) and iv.start <= prev.end + 0.50:
                prev.end = max(prev.end, iv.end)
                prev.confidence = max(prev.confidence, iv.confidence)
                if iv.regions and not prev.regions:
                    prev.regions = iv.regions
            else:
                deduped.append(iv)

        return self.reconciler.filter_dialogue_ocr_intervals(deduped)

    def is_suspicious_overmerge(
        self,
        cue: SubtitleCue,
        ocr_intervals: list[OCRInterval],
    ) -> tuple[bool, list[str], list[OCRInterval]]:
        """Evaluates whether a source cue is suspicious of containing multiple turns."""
        reasons: list[str] = []
        duration = max(0.01, cue.end - cue.start)
        
        # 1. Collect overlapping OCR intervals
        matched_ocr: list[OCRInterval] = []
        for iv in ocr_intervals:
            if not iv.is_dialogue:
                continue
            overlap = max(0.0, min(cue.end, iv.end) - max(cue.start, iv.start))
            if overlap >= 0.35 or (iv.start >= cue.start - 0.20 and iv.end <= cue.end + 0.20):
                matched_ocr.append(iv)

        matched_ocr.sort(key=lambda x: (x.start, x.end))

        # Check distinct non-overlapping OCR strings
        distinct_ocr = []
        for iv in matched_ocr:
            c_txt = clean_chinese_text(iv.normalized_text)
            if len(c_txt) >= 2 and not any(c_txt in clean_chinese_text(d.normalized_text) or clean_chinese_text(d.normalized_text) in c_txt for d in distinct_ocr):
                distinct_ocr.append(iv)

        # Multi-signal combined evidence evaluation:
        # Signal A: Multiple distinct OCR lines (turnover >= 2)
        has_ocr_turnover = len(distinct_ocr) >= 2
        if has_ocr_turnover:
            reasons.append(f"ocr_turnover_{len(distinct_ocr)}_distinct_lines")

        # Signal B: Duration anomaly (> 3.2s with multiple clauses)
        clauses = [s for s in re.split(r"[，。！？；\s]", cue.source_text) if len(s.strip()) >= 2]
        has_duration_anomaly = duration >= self.config.min_duration_anomaly_s and len(clauses) >= 2
        if has_duration_anomaly:
            reasons.append(f"duration_anomaly_{duration:.1f}s_with_{len(clauses)}_clauses")

        # Signal C: OCR lines >= 3 inside single ASR cue (strong overmerge)
        has_strong_ocr_overmerge = len(distinct_ocr) >= 3
        if has_strong_ocr_overmerge:
            reasons.append("strong_ocr_overmerge_3_lines")

        # Signal D: Long cue duration (> 4.5s) with multiple distinct OCR
        if duration >= 4.5 and len(distinct_ocr) >= 2:
            reasons.append(f"excessive_duration_{duration:.1f}s")

        # A cue is suspicious ONLY if multiple independent signals agree AND duration is substantial (>= 2.5s)
        # Never over-split short phrases (< 2.5s) unless strong 3-line overmerge exists
        is_suspicious = False
        if duration >= 2.5 and has_ocr_turnover and (has_duration_anomaly or has_strong_ocr_overmerge or duration >= 4.0):
            is_suspicious = True
        elif has_strong_ocr_overmerge and duration >= 2.0:
            is_suspicious = True

        return is_suspicious, reasons, distinct_ocr

    def segment_suspicious_cue(
        self,
        cue: SubtitleCue,
        distinct_ocr: list[OCRInterval],
        reasons: list[str],
    ) -> list[SourceCue]:
        """Splits an overmerged source cue into clean, chronologically aligned SourceCues."""
        split_cues: list[SourceCue] = []
        
        if not distinct_ocr:
            # Cannot split without OCR boundary evidence
            return [
                SourceCue(
                    cue_id=cue.id,
                    start=cue.start,
                    end=cue.end,
                    source_text=cue.source_text,
                    original_source_cue_ids=[cue.id],
                    segmentation_method="unmodified",
                    source_integrity_status=SourceIntegrityStatus.PASS,
                    asr_text=cue.source_text,
                    speaker_id=cue.speaker_id,
                    speaker_character_id=cue.speaker_character_id,
                    discourse_mode=cue.discourse_mode,
                    source_confidence=cue.confidence or 0.85,
                    ocr_start=cue.ocr_start,
                    ocr_end=cue.ocr_end,
                    ocr_regions=cue.ocr_regions,
                    ocr_evidence=cue.ocr_evidence,
                    translated_text=cue.translated_text,
                )
            ]

        # Allocate OCR intervals to split segments
        for idx, ocr in enumerate(distinct_ocr):
            seg_start = max(cue.start, ocr.start)
            seg_end = min(cue.end, ocr.end)
            if seg_end <= seg_start:
                seg_end = seg_start + max(0.80, (ocr.end - ocr.start))

            # Match text portion
            seg_text = ocr.normalized_text
            corr_list: list[SourceTokenCorrection] = []
            
            # Reconcile segment text
            corr_list.append(
                SourceTokenCorrection(
                    original_asr=cue.source_text,
                    corrected_text=seg_text,
                    evidence=f"Split from merged cue {cue.id} via OCR turnover: '{ocr.raw_text}'",
                    confidence=ocr.confidence,
                    source="ocr_segmentation",
                )
            )

            ev = OCREvidence(
                id=ocr.id,
                text=ocr.raw_text,
                confidence=ocr.confidence,
                start=ocr.start,
                end=ocr.end,
                regions=ocr.regions,
                match_score=1.0,
            )

            split_cues.append(
                SourceCue(
                    cue_id=f"{cue.id}_seg{idx+1}",
                    start=seg_start,
                    end=seg_end,
                    source_text=seg_text,
                    original_source_cue_ids=[cue.id],
                    segmentation_method="ocr_turnover_split",
                    source_integrity_status=SourceIntegrityStatus.REPAIRED,
                    asr_text=cue.source_text,
                    ocr_text=ocr.raw_text,
                    speaker_id=cue.speaker_id,
                    speaker_character_id=cue.speaker_character_id,
                    discourse_mode=cue.discourse_mode,
                    source_confidence=ocr.confidence,
                    corrections=corr_list,
                    ocr_start=ocr.start,
                    ocr_end=ocr.end,
                    ocr_regions=ocr.regions,
                    ocr_evidence=[ev],
                )
            )

        split_cues.sort(key=lambda c: (c.start, c.end))
        return split_cues
