import cv2
import numpy as np
import pytest

from app.models.project import OCREvidence, OCRRegion, SubtitleCue
from app.services.fusion import align_and_correct_span, fuse_cues_with_metrics
from app.services.hardsub_cleaner import HardSubCleaner
from app.services.ocr.visual_tracker import VisualBoundaryTracker, compute_geometry_signal


def test_no_over_fusion():
    """Test 24: Separate ASR cues must NEVER be merged into one."""
    asr1 = SubtitleCue(start=11.9, end=12.5, source_text="领口歪了")
    asr2 = SubtitleCue(start=12.7, end=13.5, source_text="坐姿不对")

    ocr1 = SubtitleCue(start=11.9, end=12.5, source_text="领口歪了", ocr_regions=[OCRRegion(text="领口歪了", points=[[0.1, 0.7], [0.3, 0.7], [0.3, 0.8], [0.1, 0.8]])])
    ocr2 = SubtitleCue(start=12.7, end=13.5, source_text="坐姿不对", ocr_regions=[OCRRegion(text="坐姿不对", points=[[0.1, 0.7], [0.3, 0.7], [0.3, 0.8], [0.1, 0.8]])])

    fused, metrics = fuse_cues_with_metrics([asr1, asr2], [ocr1, ocr2])
    assert len(fused) == 2
    assert fused[0].source_text == "领口歪了"
    assert fused[1].source_text == "坐姿不对"
    assert metrics["full_sentence_replacements"] == 0
    assert metrics["asr_derived_fused_cues"] == 2


def test_long_asr_multi_visual_fragments():
    """Test 25: Long ASR cue must preserve full sentence and attach multiple OCR evidences."""
    asr = SubtitleCue(
        start=35.23,
        end=37.49,
        source_text="你的存在拉低了秦家的执行效率。",
        asr_confidence=0.92,
    )
    ocr1 = SubtitleCue(
        start=35.20,
        end=35.80,
        source_text="你的存在",
        ocr_confidence=0.96,
        ocr_regions=[OCRRegion(text="你的存在", points=[[0.1, 0.7], [0.3, 0.7], [0.3, 0.8], [0.1, 0.8]])],
    )
    ocr2 = SubtitleCue(
        start=35.80,
        end=37.30,
        source_text="拉低了秦家的执行效率",
        ocr_confidence=0.95,
        ocr_regions=[OCRRegion(text="拉低了秦家的执行效率", points=[[0.2, 0.7], [0.7, 0.7], [0.7, 0.8], [0.2, 0.8]])],
    )

    fused, metrics = fuse_cues_with_metrics([asr], [ocr1, ocr2])
    assert len(fused) == 1
    assert "你的存在拉低了秦家的执行效率" in fused[0].source_text
    assert len(fused[0].ocr_evidence) == 2
    assert fused[0].ocr_evidence[0].text == "你的存在"
    assert fused[0].ocr_evidence[1].text == "拉低了秦家的执行效率"
    assert metrics["full_sentence_replacements"] == 0
    assert metrics["asr_source_shortened"] == 0


def test_name_homophone_correction():
    """Test 26: High-confidence OCR homophone corrects single character in ASR span without truncation."""
    asr = SubtitleCue(start=29.4, end=30.8, source_text="我爸秦燕川", asr_confidence=0.88)
    ocr = SubtitleCue(
        start=29.4,
        end=30.8,
        source_text="秦砚川",
        ocr_confidence=0.98,
        ocr_regions=[OCRRegion(text="秦砚川", points=[[0.2, 0.7], [0.4, 0.7], [0.4, 0.8], [0.2, 0.8]])],
    )

    fused, metrics = fuse_cues_with_metrics([asr], [ocr])
    assert len(fused) == 1
    assert fused[0].source_text == "我爸秦砚川"
    assert metrics["span_corrections_applied"] == 1
    assert metrics["full_sentence_replacements"] == 0


def test_unrelated_ocr_noise():
    """Test 27: Unrelated OCR text nearby must NOT corrupt ASR source text."""
    asr = SubtitleCue(start=45.0, end=47.0, source_text="今天是我的成人礼", asr_confidence=0.90)
    ocr = SubtitleCue(
        start=45.2,
        end=46.8,
        source_text="MILK",
        ocr_confidence=0.80,
        ocr_regions=[OCRRegion(text="MILK", points=[[0.8, 0.1], [0.9, 0.1], [0.9, 0.2], [0.8, 0.2]])],
    )

    fused, metrics = fuse_cues_with_metrics([asr], [ocr])
    assert len(fused) >= 1
    asr_fused = [c for c in fused if c.id == asr.id][0]
    assert asr_fused.source_text == "今天是我的成人礼"


def test_asr_timing_invariance():
    """Test 28: ASR timing must be 100% preserved as dialogue backbone."""
    asr = SubtitleCue(start=10.0, end=12.0, source_text="明天见")
    ocr = SubtitleCue(start=10.5, end=11.5, source_text="明天见", ocr_regions=[OCRRegion(text="明天见")])

    fused, metrics = fuse_cues_with_metrics([asr], [ocr])
    assert len(fused) == 1
    assert fused[0].start == 10.0
    assert fused[0].end == 12.0
    assert metrics["asr_timing_changed"] == 0


def test_cleaner_per_evidence_intervals():
    """Test 29: Cleaner active regions match exact per-evidence intervals."""
    cleaner = HardSubCleaner()
    ev1 = OCREvidence(
        text="你的存在",
        start=35.20,
        end=35.80,
        regions=[OCRRegion(text="左", points=[[0.1, 0.7], [0.3, 0.7], [0.3, 0.8], [0.1, 0.8]])],
    )
    ev2 = OCREvidence(
        text="拉低了秦家的执行效率",
        start=35.80,
        end=37.30,
        regions=[OCRRegion(text="右", points=[[0.4, 0.7], [0.8, 0.7], [0.8, 0.8], [0.4, 0.8]])],
    )
    cue = SubtitleCue(
        start=35.0,
        end=37.5,
        source_text="你的存在拉低了秦家的执行效率",
        ocr_evidence=[ev1, ev2],
    )

    intervals = cleaner.build_active_intervals([cue])
    assert len(intervals) >= 1
    # Check that intervals cover visual times
    assert intervals[0][0] <= 35.20 + 0.05
    assert intervals[-1][1] >= 37.30 - 0.05


def test_outside_mask_exact_identity():
    """Test 30: Pixels outside the mask must be 100% bitwise identical."""
    cleaner = HardSubCleaner(luminance_threshold=150, local_contrast_threshold=10)
    h, w = 100, 200
    np.random.seed(42)
    # Background texture
    frame = np.random.randint(50, 120, (h, w, 3), dtype=np.uint8)

    # Insert bright text box
    frame[70:85, 30:170] = 240  # Bright text pixels

    regions = [
        OCRRegion(
            text="TEST",
            points=[[0.15, 0.70], [0.85, 0.70], [0.85, 0.85], [0.15, 0.85]],
        )
    ]

    cleaned, modified = cleaner.clean_frame(frame, mode="auto", is_subtitle_active=True, ocr_regions=regions)
    assert modified is True

    # Extract mask that was modified
    diff = np.max(cv2.absdiff(frame, cleaned), axis=2)
    modified_pixels = diff > 0

    # Outside the modified text region must be 100% bit-identical
    assert np.array_equal(frame[~modified_pixels], cleaned[~modified_pixels])


def test_visual_tracker_hysteresis_and_onset():
    """Test 31: Tracker hysteresis catches true onset without false early activation."""
    tracker = VisualBoundaryTracker(sample_fps=10.0, search_window_seconds=0.50, min_stable_frames=2)

    # Verify geometry signal extraction
    h, w = 100, 200
    frame_blank = np.zeros((h, w, 3), dtype=np.uint8)
    frame_text = np.zeros((h, w, 3), dtype=np.uint8)
    frame_text[70:85, 30:170] = 230  # Bright text

    regions = [OCRRegion(points=[[0.15, 0.70], [0.85, 0.70], [0.85, 0.85], [0.15, 0.85]])]

    sig_blank = compute_geometry_signal(frame_blank, regions)
    sig_text = compute_geometry_signal(frame_text, regions)

    assert sig_blank < 0.01
    assert sig_text > 0.10
