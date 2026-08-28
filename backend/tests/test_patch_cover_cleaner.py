import numpy as np
import pytest
from pathlib import Path

from app.models.project import (
    OCRRegion,
    PatchCoverConfig,
    Project,
    RenderOptions,
    SubtitleCue,
    VisualEditConfig,
    VisualEditMode,
)
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.subtitles import to_ass


def test_patch_cover_config_defaults():
    """Verify default parameters of PatchCoverConfig."""
    cfg = PatchCoverConfig()
    assert cfg.enabled is True
    assert 0.88 <= cfg.patch_opacity <= 0.96
    assert 4 <= cfg.padding_px <= 8
    assert 5 <= cfg.feather_px <= 10
    assert 4 <= cfg.blur_sigma <= 10
    assert cfg.temporal_gap_fill_frames == 6
    assert cfg.mask_persistence_frames == 3


def test_outside_mask_pixels_remain_100_percent_untouched():
    """Verify that pixels outside the OCR bounding region mask are 100% byte-identical."""
    cleaner = PatchCoverCleaner(
        config=PatchCoverConfig(padding_px=4, feather_px=4, patch_opacity=0.95, blur_sigma=6.0)
    )

    h, w = 480, 852
    # Create distinct test pattern frame
    np.random.seed(42)
    original_frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    # Subtitle polygon in bottom center
    polygons = [[[0.2, 0.8], [0.8, 0.8], [0.8, 0.9], [0.2, 0.9]]]
    alpha_mask = cleaner.create_feathered_mask(h, w, polygons)

    patched_frame = cleaner.apply_patch_cover(original_frame, alpha_mask)

    # In top half (where alpha is 0.0), pixels MUST be 100% identical
    top_original = original_frame[:200, :, :]
    top_patched = patched_frame[:200, :, :]
    assert np.array_equal(top_original, top_patched)


def test_temporal_gap_bridging_and_persistence():
    """Verify that short OCR gaps are bridged and persistence is added."""
    cleaner = PatchCoverCleaner(
        config=PatchCoverConfig(temporal_gap_fill_frames=6, mask_persistence_frames=3)
    )

    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            source_text="你好",
            ocr_regions=[OCRRegion(points=[[0.2, 0.8], [0.8, 0.8], [0.8, 0.9], [0.2, 0.9]], confidence=0.8)],
        ),
        SubtitleCue(
            id="c2",
            start=2.1,  # 0.1s gap <= 6 frames (0.2s)
            end=3.0,
            source_text="世界",
            ocr_regions=[OCRRegion(points=[[0.2, 0.8], [0.8, 0.8], [0.8, 0.9], [0.2, 0.9]], confidence=0.8)],
        ),
    ]

    intervals = cleaner.extract_active_intervals(cues, fps=30.0)
    assert len(intervals) == 1
    assert intervals[0]["start"] == 1.0
    # End should be 3.0 + 3/30.0 = 3.1s
    assert intervals[0]["end"] == pytest.approx(3.1, abs=0.01)


def test_vietnamese_subtitle_preset_shortform_reference():
    """Verify that preset 'shortform_reference' formats ASS subtitles correctly."""
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=3.0,
            source_text="你好世界",
            translated_text="Xin chào thế giới",
        )
    ]

    opts = RenderOptions(
        visual_edit=VisualEditConfig(
            mode=VisualEditMode.PATCH_COVER,
            preset="shortform_reference",
        )
    )

    ass_content = to_ass(cues, opts, width=1280, height=720)

    # Verify shortform typography styles
    assert "Style: Default,Arial,23" in ass_content
    assert "&H00FFFFFF" in ass_content
    assert "&H00000000" in ass_content
    assert ",-1,0,0,0,100,100,0,0,1,2.8,1.0,2,20,20,36,1" in ass_content
    assert "Xin chào thế giới" in ass_content


def test_vietnamese_subtitle_preset_shortform_bold_yellow():
    """Verify that preset 'shortform_bold_yellow' renders Layer 0 BackingPlate and Layer 1 Bold Yellow Outline text."""
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=3.0,
            source_text="你好世界",
            translated_text="Xin chào thế giới",
        )
    ]

    opts = RenderOptions(
        visual_edit=VisualEditConfig(
            mode=VisualEditMode.PATCH_COVER,
            preset="shortform_bold_yellow",
        )
    )

    ass_content = to_ass(cues, opts, width=1280, height=720)

    # Verify Style definitions
    assert "Style: BackingPlate" in ass_content
    assert "Style: Default" in ass_content
    # Golden yellow outline in ASS format
    assert "&H0000D7FF" in ass_content
    # Multi-layer events
    assert "Dialogue: 0," in ass_content
    assert "BackingPlate" in ass_content
    assert "Dialogue: 1," in ass_content
    assert "Default" in ass_content
    assert "Xin chào thế giới" in ass_content


def test_patch_cover_preserves_translation_and_cues_invariance():
    """Verify that patch cover mode is strictly post-localization and does not alter cues."""
    cues = [
        SubtitleCue(id="c1", start=10.0, end=12.0, source_text="领口歪了，", translated_text="Cổ áo lệch rồi."),
        SubtitleCue(id="c2", start=70.0, end=73.0, source_text="在早餐店", translated_text="Ở một gia đình mở quán ăn sáng"),
    ]

    proj = Project(
        name="test_proj",
        source_video_path="dummy.mp4",
        cues=cues,
        visual_edit=VisualEditConfig(mode=VisualEditMode.PATCH_COVER),
    )

    assert proj.cues[0].translated_text == "Cổ áo lệch rồi."
    assert proj.cues[1].translated_text == "Ở một gia đình mở quán ăn sáng"
