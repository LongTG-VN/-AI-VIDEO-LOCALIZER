from __future__ import annotations

import pytest
from app.models.project import OCREvidence, OCRRegion, Project, SubtitleCue
from app.services.source_integrity.models import (
    OCRInterval,
    SourceCue,
    SourceIntegrityConfig,
    SourceIntegrityReport,
    SourceIntegrityStatus,
    SourceTokenCorrection,
)
from app.services.source_integrity.pipeline import SourceIntegrityPipeline
from app.services.source_integrity.reconciliation import SourceReconciler
from app.services.source_integrity.segmentation import SourceSegmenter
from app.services.source_integrity.validators import SourceIntegrityValidator
from app.services.utterance_engine import UtteranceEngine


# 1. Overmerge detection: cue with distinct OCR intervals flagged suspicious
def test_overmerge_detection_flags_suspicious():
    segmenter = SourceSegmenter(SourceIntegrityConfig(min_duration_anomaly_s=3.0))
    cue = SubtitleCue(
        id="cue_long",
        start=10.0,
        end=15.0,
        source_text="今天入职以后跟你们一个项目组大家好我刚准备动筷",
    )
    ocr_intervals = [
        OCRInterval(start=10.0, end=11.5, normalized_text="今天入职", raw_text="今天入职", confidence=0.99),
        OCRInterval(start=11.6, end=13.0, normalized_text="以后跟你们一个项目组", raw_text="以后跟你们一个项目组", confidence=0.98),
        OCRInterval(start=13.2, end=14.8, normalized_text="大家好", raw_text="大家好", confidence=0.95),
    ]
    is_suspicious, reasons, distinct = segmenter.is_suspicious_overmerge(cue, ocr_intervals)
    assert is_suspicious is True
    assert len(distinct) == 3
    assert any("ocr_turnover" in r for r in reasons)


# 2. Short normal cue NOT flagged as suspicious
def test_short_normal_cue_not_flagged():
    segmenter = SourceSegmenter(SourceIntegrityConfig(min_duration_anomaly_s=3.0))
    cue = SubtitleCue(
        id="cue_short",
        start=10.0,
        end=11.5,
        source_text="大家好",
    )
    ocr_intervals = [
        OCRInterval(start=10.0, end=11.5, normalized_text="大家好", raw_text="大家好", confidence=0.99),
    ]
    is_suspicious, reasons, distinct = segmenter.is_suspicious_overmerge(cue, ocr_intervals)
    assert is_suspicious is False


# 3. Word-boundary / OCR-boundary splitting
def test_segment_suspicious_cue_splits_correctly():
    segmenter = SourceSegmenter()
    cue = SubtitleCue(
        id="cue_merged",
        start=10.0,
        end=15.0,
        source_text="唯一让我在这个写字楼里有点存在感的是我妈传给我的手艺做饭",
    )
    distinct_ocr = [
        OCRInterval(start=10.0, end=11.5, normalized_text="唯一让我在这个", raw_text="唯一让我在这个", confidence=0.99),
        OCRInterval(start=11.5, end=13.0, normalized_text="写字楼里有点存在感的", raw_text="写字楼里有点存在感的", confidence=0.99),
        OCRInterval(start=13.0, end=15.0, normalized_text="是我妈传给我的手艺做饭", raw_text="是我妈传给我的手艺做饭", confidence=0.99),
    ]
    splits = segmenter.segment_suspicious_cue(cue, distinct_ocr, ["ocr_turnover_3_distinct_lines"])
    assert len(splits) == 3
    assert splits[0].source_text == "唯一让我在这个"
    assert splits[1].source_text == "写字楼里有点存在感的"
    assert splits[2].source_text == "是我妈传给我的手艺做饭"
    for s in splits:
        assert s.original_source_cue_ids == ["cue_merged"]
        assert s.source_integrity_status == SourceIntegrityStatus.REPAIRED


# 4. Phonetic repair: ASR phonetic corruption replaced with high-confidence OCR
def test_phonetic_repair_with_high_confidence_ocr():
    reconciler = SourceReconciler()
    asr_text = "我妈传给我的手印做饭"
    ocr_intervals = [
        OCRInterval(
            start=10.0,
            end=12.5,
            normalized_text="是我妈传给我的手艺做饭",
            raw_text="是我妈传给我的手艺做饭",
            confidence=0.98,
        )
    ]
    reconciled_text, corrections = reconciler.reconcile_segment_text(asr_text, ocr_intervals)
    assert reconciled_text == "是我妈传给我的手艺做饭"
    assert len(corrections) == 1
    assert corrections[0].corrected_text == "是我妈传给我的手艺做饭"


# 5. Non-dialogue OCR rejection: decorative top title ignored
def test_non_dialogue_decorative_ocr_rejection():
    reconciler = SourceReconciler(SourceIntegrityConfig(dialogue_y_min=0.55, dialogue_y_max=0.98))
    top_watermark = OCRInterval(
        start=1.0,
        end=5.0,
        normalized_text="独播短剧",
        raw_text="独播短剧",
        confidence=0.99,
        geometry=[[0.1, 0.05], [0.3, 0.05], [0.3, 0.15], [0.1, 0.15]],
    )
    valid_dialogue = reconciler.filter_dialogue_ocr_intervals([top_watermark])
    assert len(valid_dialogue) == 0
    assert top_watermark.is_dialogue is False


# 6. Low confidence OCR rejected
def test_low_confidence_ocr_rejected():
    reconciler = SourceReconciler(SourceIntegrityConfig(min_ocr_dialogue_confidence=0.50))
    noisy_ocr = OCRInterval(
        start=1.0,
        end=3.0,
        normalized_text="乱七八糟",
        raw_text="乱七八糟",
        confidence=0.20,
    )
    valid = reconciler.filter_dialogue_ocr_intervals([noisy_ocr])
    assert len(valid) == 0


# 7. Provenance retention: original_source_cue_ids preserved
def test_provenance_retention():
    pipeline = SourceIntegrityPipeline()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好"),
        SubtitleCue(id="c2", start=2.5, end=3.5, source_text="再见"),
    ]
    repaired, report = pipeline.run_pipeline(cues)
    assert len(repaired) == 2
    assert repaired[0].original_source_cue_ids == ["c1"]
    assert repaired[1].original_source_cue_ids == ["c2"]


# 8. Chronological ordering enforced by validator
def test_validator_detects_out_of_order_cues():
    validator = SourceIntegrityValidator()
    cues = [
        SourceCue(cue_id="c1", start=5.0, end=6.0, source_text="Sentence 1", original_source_cue_ids=["c1"]),
        SourceCue(cue_id="c2", start=2.0, end=3.0, source_text="Sentence 2", original_source_cue_ids=["c2"]),
    ]
    is_valid, errors = validator.validate_source_cues(cues)
    assert is_valid is False
    assert any("starts before preceding cue" in e for e in errors)


# 9. Empty text rejected by validator
def test_validator_rejects_empty_source_text():
    validator = SourceIntegrityValidator()
    cues = [
        SourceCue(cue_id="c1", start=1.0, end=2.0, source_text="", original_source_cue_ids=["c1"]),
    ]
    is_valid, errors = validator.validate_source_cues(cues)
    assert is_valid is False
    assert any("empty source_text" in e for e in errors)


# 10. End-to-end integration: Synthetic overmerged dialogue -> SourceIntegrity -> UtteranceEngine
def test_synthetic_overmerged_pipeline_integration():
    pipeline = SourceIntegrityPipeline()
    fused_cue = SubtitleCue(
        id="merged_dialogue",
        start=100.0,
        end=106.0,
        source_text="这是苏棠今天入职以后跟你们一个项目组大家好",
    )
    ocr_cues = [
        SubtitleCue(id="ocr1", start=100.0, end=101.5, source_text="这是苏棠", ocr_confidence=0.99),
        SubtitleCue(id="ocr2", start=101.5, end=103.0, source_text="今天入职", ocr_confidence=0.99),
        SubtitleCue(id="ocr3", start=103.0, end=104.5, source_text="以后跟你们一个项目组", ocr_confidence=0.98),
        SubtitleCue(id="ocr4", start=104.5, end=106.0, source_text="大家好", ocr_confidence=0.99),
    ]

    repaired_cues, report = pipeline.run_pipeline([fused_cue], ocr_cues=ocr_cues)
    assert report.overmerge_detected == 1
    assert len(repaired_cues) == 4

    # Provide translated text for UtteranceEngine
    repaired_cues[0].translated_text = "Đây là Tô Đường"
    repaired_cues[1].translated_text = "hôm nay nhận việc"
    repaired_cues[2].translated_text = "sau này cùng nhóm dự án với các bạn"
    repaired_cues[3].translated_text = "chào mọi người"

    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(repaired_cues, translated=True)
    assert len(render_cues) >= 1
    # Verify chronological ordering
    for i in range(1, len(render_cues)):
        assert render_cues[i].start >= render_cues[i-1].start
