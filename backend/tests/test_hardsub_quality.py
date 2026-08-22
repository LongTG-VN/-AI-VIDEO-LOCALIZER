import cv2
import numpy as np

from app.models.project import RenderOptions, SubtitleCue
from app.services.hardsub_cleaner import HardSubCleaner


def _subtitle_like_roi(height: int = 140, width: int = 740) -> np.ndarray:
    roi = np.full((height, width, 3), 55, dtype=np.uint8)
    x = 260
    y = 58
    for _ in range(6):
        cv2.rectangle(roi, (x, y), (x + 11, y + 17), (245, 245, 245), -1)
        x += 22
    return roi


def _textured_frame(height: int = 480, width: int = 852) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = (40 + (xx % 90)).astype(np.uint8)
    frame[..., 1] = (60 + (yy % 100)).astype(np.uint8)
    frame[..., 2] = (80 + ((xx + yy) % 80)).astype(np.uint8)
    return frame


def test_render_options_default_to_quality_auto_mode():
    options = RenderOptions()
    assert options.hardsub_removal_mode == "auto"
    assert options.hardsub_mask_dilate_radius == 1
    assert options.hardsub_lossless_intermediate is True


def test_active_intervals_prefer_ocr_backed_cues():
    cleaner = HardSubCleaner()
    cues = [
        SubtitleCue(
            start=1.0,
            end=2.0,
            source_text="领口歪了",
            ocr_confidence=0.95,
            asr_confidence=0.9,
        ),
        SubtitleCue(
            start=3.0,
            end=4.0,
            source_text="只有语音没有屏幕字幕",
            asr_confidence=0.95,
            ocr_confidence=None,
        ),
    ]

    intervals = cleaner.build_active_intervals(cues, pad_seconds=0.0)
    assert intervals == [(1.0, 2.0)]


def test_active_intervals_fallback_when_ocr_metadata_missing():
    cleaner = HardSubCleaner()
    cues = [SubtitleCue(start=1.0, end=2.0, source_text="领口歪了")]
    assert cleaner.build_active_intervals(cues, pad_seconds=0.0) == [(1.0, 2.0)]


def test_mask_rejects_large_bright_background_patch():
    cleaner = HardSubCleaner()
    roi = np.full((140, 740, 3), 45, dtype=np.uint8)
    cv2.rectangle(roi, (120, 40), (620, 100), (245, 245, 245), -1)

    mask = cleaner.extract_text_mask(roi)
    assert mask is None


def test_mask_detects_compact_subtitle_row_without_covering_roi():
    cleaner = HardSubCleaner(max_mask_coverage=0.12)
    roi = _subtitle_like_roi()

    mask = cleaner.extract_text_mask(roi)
    assert mask is not None
    coverage = float(np.mean(mask > 0))
    assert 0.001 < coverage < 0.12


def test_fast_cleanup_preserves_pixels_outside_text_mask():
    cleaner = HardSubCleaner(
        crop_top_ratio=0.65,
        crop_bottom_ratio=0.95,
        crop_left_ratio=0.06,
        crop_right_ratio=0.94,
    )
    frame = _textured_frame()

    x1, y1, x2, y2 = cleaner._roi_bounds(frame)
    roi = frame[y1:y2, x1:x2]
    subtitle = _subtitle_like_roi(roi.shape[0], roi.shape[1])
    mask_seed = np.all(subtitle > 200, axis=2)
    roi[mask_seed] = 245

    mask = cleaner.extract_text_mask(roi)
    assert mask is not None
    before = frame.copy()
    cleaned, changed = cleaner.clean_frame(frame, mode="inpaint", is_subtitle_active=True)
    assert changed

    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    outside = full_mask == 0
    assert np.array_equal(cleaned[outside], before[outside])


def test_quality_mode_uses_clean_temporal_donor_when_alignment_is_safe():
    cleaner = HardSubCleaner(
        crop_top_ratio=0.65,
        crop_bottom_ratio=0.95,
        crop_left_ratio=0.06,
        crop_right_ratio=0.94,
        scene_cut_threshold=40.0,
    )
    donor = _textured_frame()
    current = donor.copy()
    x1, y1, x2, y2 = cleaner._roi_bounds(current)
    roi = current[y1:y2, x1:x2]
    synthetic = _subtitle_like_roi(roi.shape[0], roi.shape[1])
    glyphs = np.all(synthetic > 200, axis=2)
    roi[glyphs] = 245

    cleaned, changed = cleaner.clean_frame(
        current,
        mode="quality",
        is_subtitle_active=True,
        temporal_donors=[donor],
    )

    assert changed
    assert int(cleaner._metrics.get("temporal_frames", 0)) == 1
    assert np.mean(np.abs(cleaned.astype(np.int16) - donor.astype(np.int16))) < np.mean(
        np.abs(current.astype(np.int16) - donor.astype(np.int16))
    )
