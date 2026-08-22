import numpy as np
import pytest
import cv2

from app.models.project import OCRRegion, SubtitleCue
from app.services.fusion import fuse_cues
from app.services.hardsub_cleaner import HardSubCleaner


def test_ocr_region_normalization_and_serialization():
    region = OCRRegion(
        text='领口歪了',
        confidence=0.98,
        points=[[0.35, 0.82], [0.65, 0.82], [0.65, 0.88], [0.35, 0.88]]
    )
    cue = SubtitleCue(
        start=1.0,
        end=2.0,
        source_text='领口歪了',
        ocr_regions=[region]
    )
    dumped = cue.model_dump()
    assert len(dumped['ocr_regions']) == 1
    assert dumped['ocr_regions'][0]['text'] == '领口歪了'
    assert dumped['ocr_regions'][0]['points'][0] == [0.35, 0.82]


def test_fusion_keeps_geometry_metadata():
    region = OCRRegion(
        text='秦家十八年来唯一的千金',
        confidence=0.96,
        points=[[0.20, 0.80], [0.80, 0.80], [0.80, 0.86], [0.20, 0.86]]
    )
    ocr_cue = SubtitleCue(
        start=5.0,
        end=7.0,
        source_text='秦家十八年来唯一的千金',
        ocr_start=5.2,
        ocr_end=6.8,
        ocr_confidence=0.96,
        ocr_regions=[region]
    )
    asr_cue = SubtitleCue(
        start=4.9,
        end=7.1,
        source_text='秦家十八年来唯一千金',
        speaker_id='speaker_0',
        asr_confidence=0.90
    )
    fused = fuse_cues([asr_cue], [ocr_cue])
    assert len(fused) == 1
    assert fused[0].speaker_id == 'speaker_0'
    assert fused[0].ocr_start == 5.2
    assert fused[0].ocr_end == 6.8
    assert len(fused[0].ocr_regions) == 1
    assert fused[0].ocr_regions[0].text == '秦家十八年来唯一的千金'


def test_extract_geometry_mask_within_polygon():
    cleaner = HardSubCleaner(geometry_enabled=True, geometry_padding_px=4)
    frame = np.full((480, 852, 3), 30, dtype=np.uint8)

    # Put white text inside OCR region
    cv2.putText(frame, 'TEST SUBTITLE', (300, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # Put bright scenery outside OCR region (top left)
    cv2.circle(frame, (100, 100), 40, (255, 255, 255), -1)

    region = OCRRegion(
        text='TEST SUBTITLE',
        confidence=0.99,
        points=[[280 / 852, 390 / 480], [580 / 852, 390 / 480], [580 / 852, 440 / 480], [280 / 852, 440 / 480]]
    )

    mask, stats = cleaner.extract_geometry_mask(frame, [region])
    assert mask is not None
    assert int(np.sum(mask > 0)) > 0
    # Text area inside OCR polygon has mask
    assert np.any(mask[390:440, 280:580] > 0)
    # Bright scenery outside OCR polygon must NOT be masked!
    assert np.all(mask[60:140, 60:140] == 0)
    assert stats['avg_coverage'] < 0.70


def test_clean_frame_geometry_preserves_background():
    cleaner = HardSubCleaner(geometry_enabled=True, geometry_padding_px=4)
    # Frame with complex noise texture background
    np.random.seed(42)
    frame = np.random.randint(40, 90, (480, 852, 3), dtype=np.uint8)

    # Subtitle text
    cv2.putText(frame, 'CHINESE TEXT', (300, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 2)

    region = OCRRegion(
        text='CHINESE TEXT',
        confidence=0.95,
        points=[[280 / 852, 390 / 480], [550 / 852, 390 / 480], [550 / 852, 440 / 480], [280 / 852, 440 / 480]]
    )

    cleaned, was_cleaned = cleaner.clean_frame(frame, mode='inpaint', is_subtitle_active=True, ocr_regions=[region])
    assert was_cleaned is True
    # Background outside OCR bounding box must remain exactly identical
    assert np.array_equal(frame[0:300, :], cleaned[0:300, :])
    assert np.array_equal(frame[460:480, :], cleaned[460:480, :])


def test_clean_frame_fallback_for_legacy_cues():
    cleaner = HardSubCleaner(geometry_enabled=True)
    frame = np.full((480, 852, 3), 40, dtype=np.uint8)
    cv2.putText(frame, 'LEGACY TEXT', (300, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 2)

    # No ocr_regions provided -> should fallback to heuristic without crashing
    cleaned, was_cleaned = cleaner.clean_frame(frame, mode='inpaint', is_subtitle_active=True, ocr_regions=None)
    assert was_cleaned is True
    assert cleaner._metrics.get('geometry_missing_fallback_frames', 0) >= 1
