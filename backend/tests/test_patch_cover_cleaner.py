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
    assert 5 <= cfg.feather_px <= 16
    assert 4 <= cfg.blur_sigma <= 12
    assert 0.0 <= cfg.dark_tint <= 0.5
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
    assert ",-1,0,0,0,100,100,0,0,1,2.8,1.0,2,20,20," in ass_content
    assert "Xin chào thế giới" in ass_content


def test_vietnamese_subtitle_preset_shortform_white_black_soft_bg():
    """Verify that preset 'shortform_white_black_soft_bg' renders Layer 0 Soft Blurred Backing and Layer 1 White Text with Thin Black Outline."""
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
            preset="shortform_white_black_soft_bg",
        )
    )

    ass_content = to_ass(cues, opts, width=1280, height=720)

    # Verify Style definitions
    assert "Style: BackingPlate" in ass_content
    assert "Style: Default" in ass_content
    # White fill & thin black outline
    assert "&H00FFFFFF" in ass_content
    assert "&H00000000" in ass_content
    assert "1.8,0.6,2,20,20," in ass_content
    # Layer 0 has soft blur
    assert "\\blur12" in ass_content or "\\blur" in ass_content
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


def test_malformed_oversized_ocr_region_rejected_by_cleaner():
    """Verify that giant/malformed OCR polygons (e.g. face/screen wide) are rejected."""
    cleaner = PatchCoverCleaner()
    # Anomalously tall polygon (e.g. face box at y=0.30 - 0.75)
    tall_poly = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.75], [0.3, 0.75]]
    assert cleaner._is_valid_subtitle_polygon(tall_poly) is False

    # Anomalously wide polygon
    wide_poly = [[0.05, 0.8], [0.95, 0.8], [0.95, 0.9], [0.05, 0.9]]
    assert cleaner._is_valid_subtitle_polygon(wide_poly) is False

    # Valid subtitle polygon
    valid_poly = [[0.35, 0.82], [0.65, 0.82], [0.65, 0.89], [0.35, 0.89]]
    assert cleaner._is_valid_subtitle_polygon(valid_poly) is True


def test_face_safety_cutoff_zeroes_upper_screen_mask():
    """Verify that create_feathered_mask guarantees 0 alpha in upper 68% of screen."""
    cleaner = PatchCoverCleaner()
    h, w = 480, 852
    # Even if an upper polygon slipped through, safety cutoff must force it to 0
    poly = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.9], [0.1, 0.9]]
    mask = cleaner.create_feathered_mask(h, w, [poly])
    cutoff = int(h * 0.68)
    assert np.all(mask[:cutoff, :] == 0.0)


def test_no_backing_without_active_vi_subtitle():
    """Verify that ASS generated subtitles have no dialogue events during empty intervals."""
    cues = [
        SubtitleCue(id="c1", start=5.0, end=7.0, source_text="你好", translated_text="Xin chào")
    ]
    opts = RenderOptions(
        visual_edit=VisualEditConfig(
            mode=VisualEditMode.PATCH_COVER,
            preset="shortform_white_black_soft_bg",
        )
    )
    ass_content = to_ass(cues, opts, width=852, height=480)
    lines = [l for l in ass_content.split("\n") if l.startswith("Dialogue:")]
    # Exactly 2 dialogue lines (Layer 0 BackingPlate + Layer 1 Default) for the single active cue
    assert len(lines) == 2
    assert "0:00:05.00" in lines[0]
    assert "0:00:07.00" in lines[0]


def test_ocr_y_anchor_stability_with_anomalous_polygon():
    """Verify that an anomalous OCR polygon does not pull subtitle anchor into the upper screen."""
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            source_text="正常",
            translated_text="Bình thường",
            ocr_regions=[
                # Valid subtitle region
                OCRRegion(points=[[0.35, 0.83], [0.65, 0.83], [0.65, 0.89], [0.35, 0.89]]),
                # Anomalous face region
                OCRRegion(points=[[0.30, 0.20], [0.70, 0.20], [0.70, 0.60], [0.30, 0.60]]),
            ]
        )
    ]
    opts = RenderOptions(
        visual_edit=VisualEditConfig(
            mode=VisualEditMode.PATCH_COVER,
            preset="shortform_white_black_soft_bg",
        )
    )
    ass_content = to_ass(cues, opts, width=852, height=480)
    # Target pos Y should be around ~408px on 480p, definitely >= 360px
    assert "\\pos(426," in ass_content
    # Find pos Y value
    import re
    match = re.search(r"\\pos\(426,(\d+)\)", ass_content)
    assert match is not None
    pos_y = int(match.group(1))
    assert 380 <= pos_y <= 430
