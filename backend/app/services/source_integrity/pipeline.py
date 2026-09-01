from __future__ import annotations

import logging
import time
from typing import Any

from app.models.project import Project, SubtitleCue
from app.services.source_integrity.models import (
    OCRInterval,
    SourceCue,
    SourceIntegrityConfig,
    SourceIntegrityReport,
    SourceIntegrityStatus,
)
from app.services.source_integrity.reconciliation import SourceReconciler
from app.services.source_integrity.segmentation import SourceSegmenter
from app.services.source_integrity.validators import SourceIntegrityValidator

logger = logging.getLogger(__name__)


class SourceIntegrityPipeline:
    """Pipeline orchestrating source cue validation, suspicious overmerge detection, and re-segmentation."""

    def __init__(self, config: SourceIntegrityConfig | None = None):
        self.config = config or SourceIntegrityConfig()
        self.segmenter = SourceSegmenter(self.config)
        self.reconciler = SourceReconciler(self.config)
        self.validator = SourceIntegrityValidator()
        self.last_report: SourceIntegrityReport | None = None

    def run_pipeline(
        self,
        fused_cues: list[SubtitleCue],
        ocr_cues: list[SubtitleCue] | list[dict[str, Any]] | None = None,
    ) -> tuple[list[SubtitleCue], SourceIntegrityReport]:
        t0 = time.time()
        report = SourceIntegrityReport(total_source_cues=len(fused_cues))
        
        if not self.config.enabled or not fused_cues:
            report.passed = len(fused_cues)
            report.processing_time_s = round(time.time() - t0, 3)
            self.last_report = report
            return fused_cues, report

        # 1. Extract and deduplicate stable dialogue OCR intervals
        ocr_intervals: list[OCRInterval] = []
        if ocr_cues:
            ocr_intervals = self.segmenter.extract_stable_ocr_intervals(ocr_cues)

        # Also incorporate OCR evidence already attached to fused cues
        for c in fused_cues:
            for ev in c.ocr_evidence:
                ocr_intervals.append(
                    OCRInterval(
                        id=ev.id,
                        start=ev.start,
                        end=ev.end,
                        normalized_text=ev.text,
                        raw_text=ev.text,
                        confidence=ev.confidence or 0.85,
                        regions=ev.regions,
                    )
                )
        ocr_intervals = self.reconciler.filter_dialogue_ocr_intervals(ocr_intervals)

        # 2. Process each fused cue
        repaired_source_cues: list[SourceCue] = []
        
        for cue in fused_cues:
            is_suspicious, reasons, distinct_ocr = self.segmenter.is_suspicious_overmerge(cue, ocr_intervals)
            
            if is_suspicious and distinct_ocr:
                report.overmerge_detected += 1
                logger.info("Suspicious overmerged cue detected: ID=%s duration=%.2fs reasons=%s", cue.id, cue.end - cue.start, reasons)
                
                # Split overmerged cue
                split_segments = self.segmenter.segment_suspicious_cue(cue, distinct_ocr, reasons)
                report.split_count += len(split_segments)
                report.repaired += len(split_segments)
                report.speaker_boundary_splits += 1
                repaired_source_cues.extend(split_segments)
                
                report.details.append({
                    "original_cue_id": cue.id,
                    "reasons": reasons,
                    "action": "split",
                    "split_count": len(split_segments),
                    "split_ids": [s.cue_id for s in split_segments],
                })
            else:
                # Reconcile text against matched OCR if available
                reconciled_text, corrections = self.reconciler.reconcile_segment_text(cue.source_text, distinct_ocr)
                status = SourceIntegrityStatus.REPAIRED if corrections else SourceIntegrityStatus.PASS
                if corrections:
                    report.ocr_corrections += len(corrections)
                    report.repaired += 1
                else:
                    report.passed += 1

                sc = SourceCue(
                    cue_id=cue.id,
                    start=cue.start,
                    end=cue.end,
                    source_text=reconciled_text,
                    original_source_cue_ids=cue.original_source_cue_ids or [cue.id],
                    segmentation_method="reconciled" if corrections else "unmodified",
                    source_integrity_status=status,
                    asr_text=cue.source_text,
                    ocr_text=cue.ocr_text,
                    speaker_id=cue.speaker_id,
                    speaker_character_id=cue.speaker_character_id,
                    discourse_mode=cue.discourse_mode,
                    source_confidence=cue.confidence or 0.85,
                    corrections=corrections,
                    ocr_start=cue.ocr_start,
                    ocr_end=cue.ocr_end,
                    ocr_regions=cue.ocr_regions,
                    ocr_evidence=cue.ocr_evidence,
                    translated_text=cue.translated_text if not corrections else None,
                    word_timestamps=cue.word_timestamps,
                )
                repaired_source_cues.append(sc)

        # 3. Sort chronologically
        repaired_source_cues.sort(key=lambda c: (c.start, c.end))

        # 4. Validate output source cues
        is_valid, val_errors = self.validator.validate_source_cues(repaired_source_cues)
        if not is_valid:
            logger.warning("Source integrity validation reported warnings: %s", val_errors)

        # 5. Convert to standard SubtitleCue list for downstream stages
        final_subtitle_cues = [sc.to_subtitle_cue() for sc in repaired_source_cues]
        
        report.processing_time_s = round(time.time() - t0, 3)
        self.last_report = report
        logger.info(
            "Source Integrity Pipeline finished: total=%d, passed=%d, repaired=%d, overmerge_splits=%d (took %.3fs)",
            len(final_subtitle_cues),
            report.passed,
            report.repaired,
            report.split_count,
            report.processing_time_s,
        )
        return final_subtitle_cues, report
