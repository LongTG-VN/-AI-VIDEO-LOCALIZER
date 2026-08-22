from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models.project import OCRRegion, SubtitleCue
from app.services.fusion import (
    clean_chinese_text,
    fuse_cues,
    fuse_cues_with_metrics,
    stitch_fragments,
    text_similarity,
)
from app.services.ocr.visual_tracker import (
    VisualBoundaryTracker,
    compute_geometry_signal,
)


def test_test_a_multi_ocr_fragments_preserve_full_asr_sentence():
    """Test A: Multiple OCR fragments combined with ASR sentence preserving full meaning."""
    asr = SubtitleCue(
        id="asr-1",
        start=35.0,
        end=37.5,
        source_text="你的存在拉低了秦家的执行效率",
        asr_confidence=0.95,
    )
    ocr_a = SubtitleCue(
        id="ocr-a",
        start=35.0,
        end=35.8,
        source_text="你的存在",
        ocr_start=35.0,
        ocr_end=35.8,
        ocr_confidence=0.98,
        ocr_regions=[
            OCRRegion(text="你的存在", confidence=0.98, points=[[0.3, 0.8], [0.5, 0.8], [0.5, 0.9], [0.3, 0.9]])
        ],
    )
    ocr_b = SubtitleCue(
        id="ocr-b",
        start=35.8,
        end=37.3,
        source_text="拉低了秦家的执行效率",
        ocr_start=35.8,
        ocr_end=37.3,
        ocr_confidence=0.99,
        ocr_regions=[
            OCRRegion(text="拉低了秦家的执行效率", confidence=0.99, points=[[0.3, 0.8], [0.7, 0.8], [0.7, 0.9], [0.3, 0.9]])
        ],
    )

    fused, metrics = fuse_cues_with_metrics([asr], [ocr_a, ocr_b])

    assert len(fused) == 1
    assert "你的存在" in fused[0].source_text
    assert "拉低了秦家的执行效率" in fused[0].source_text
    assert fused[0].start == 35.0
    assert fused[0].end == 37.5
    assert metrics["asr_with_multi_ocr"] == 1
    assert metrics["multi_ocr_fragments_consumed"] == 2
    assert metrics["unmatched_ocr_cues"] == 0
    # Visual regions from both fragments must be preserved
    assert len(fused[0].ocr_regions) >= 1


def test_test_b_partial_ocr_homophone_correction():
    """Test B: Named entity / homophone correction in ASR sentence."""
    asr = SubtitleCue(
        id="asr-2",
        start=29.0,
        end=31.0,
        source_text="我爸秦燕川",
        asr_confidence=0.92,
    )
    ocr = SubtitleCue(
        id="ocr-name",
        start=29.5,
        end=30.8,
        source_text="秦砚川",
        ocr_start=29.5,
        ocr_end=30.8,
        ocr_confidence=0.99,
        ocr_regions=[
            OCRRegion(text="秦砚川", confidence=0.99, points=[[0.4, 0.8], [0.6, 0.8], [0.6, 0.9], [0.4, 0.9]])
        ],
    )

    fused, metrics = fuse_cues_with_metrics([asr], [ocr])

    assert len(fused) == 1
    assert fused[0].source_text == "我爸秦砚川"
    assert fused[0].start == 29.0
    assert fused[0].end == 31.0
    assert metrics["partial_ocr_corrections"] == 1


def test_test_c_duplicate_ocr_fragment_stitching():
    """Test C: Stitching overlapping OCR fragments without duplicate repetitions."""
    fragments = ["你的存在", "你的存在拉低了", "拉低了秦家的执行效率"]
    stitched = stitch_fragments(fragments)
    assert stitched == "你的存在拉低了秦家的执行效率"


def test_test_d_all_matched_ocr_ids_marked_used():
    """Test D: All associated OCR fragments are marked used; unmatched are kept."""
    asr = SubtitleCue(id="asr-1", start=10.0, end=13.0, source_text="我们在开会")
    ocr_1 = SubtitleCue(id="ocr-1", start=10.2, end=11.5, source_text="我们在")
    ocr_2 = SubtitleCue(id="ocr-2", start=11.5, end=12.8, source_text="开会")
    ocr_unmatched = SubtitleCue(id="ocr-unmatched", start=20.0, end=22.0, source_text="屏幕外的字幕")

    fused, metrics = fuse_cues_with_metrics([asr], [ocr_1, ocr_2, ocr_unmatched])

    assert len(fused) == 2
    fused_ids = [c.id for c in fused]
    assert "asr-1" in fused_ids
    assert "ocr-unmatched" in fused_ids
    assert "ocr-1" not in fused_ids
    assert "ocr-2" not in fused_ids
    assert metrics["unmatched_ocr_cues"] == 1


def test_test_e_asr_timing_preserved_as_backbone():
    """Test E: ASR timing is never overridden by OCR timing."""
    asr = SubtitleCue(id="asr-1", start=10.0, end=14.0, source_text="完整的一句话")
    ocr = SubtitleCue(id="ocr-1", start=10.5, end=13.5, ocr_start=10.5, ocr_end=13.5, source_text="完整的一句话")

    fused = fuse_cues([asr], [ocr])

    assert len(fused) == 1
    assert fused[0].start == 10.0
    assert fused[0].end == 14.0
    assert fused[0].ocr_start == 10.5
    assert fused[0].ocr_end == 13.5


def test_test_f_synthetic_visual_boundary_tracker(tmp_path):
    """Test F: Synthetic video frames correctly refine subtitle onset and offset."""
    video_file = tmp_path / "synthetic.mp4"
    h, w = 240, 320
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, fps, (w, h))

    # 40 frames total (4.0s):
    # Frames 0..9 (0.0s - 0.9s): Black (no subtitle)
    # Frames 10..29 (1.0s - 2.9s): Subtitle active (white rectangle at y=190..210, x=100..220)
    # Frames 30..39 (3.0s - 3.9s): Black (no subtitle)
    for i in range(40):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if 10 <= i <= 29:
            # Draw white text glyph box
            cv2.rectangle(frame, (100, 190), (220, 210), (255, 255, 255), -1)
        out.write(frame)
    out.release()

    # Coarse OCR timing (e.g. from 2 FPS sampling): start=1.5s, end=2.5s
    region = OCRRegion(
        text="测试字幕",
        confidence=0.99,
        points=[[100 / w, 190 / h], [220 / w, 190 / h], [220 / w, 210 / h], [100 / w, 210 / h]],
    )
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=3.0,
        source_text="测试字幕",
        ocr_start=1.5,
        ocr_end=2.5,
        ocr_regions=[region],
    )

    tracker = VisualBoundaryTracker(sample_fps=10.0, search_window_seconds=0.80)
    refined = tracker.refine_cues(video_file, [cue])

    assert len(refined) == 1
    # Onset at 1.0s, offset at 3.0s
    assert abs(refined[0].ocr_start - 1.0) <= 0.15
    assert abs(refined[0].ocr_end - 3.0) <= 0.15
    assert tracker.last_metrics["visual_timing_start_refined"] >= 1
    assert tracker.last_metrics["visual_timing_end_refined"] >= 1
