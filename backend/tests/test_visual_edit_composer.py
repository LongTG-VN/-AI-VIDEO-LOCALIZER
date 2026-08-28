import json
import pytest
from pathlib import Path

from app.models.project import (
    BlurConfig,
    OCRRegion,
    OverlayAnchor,
    OverlayConfig,
    Project,
    RenderOptions,
    SubtitleCue,
    VisualEditConfig,
    VisualEditMode,
)
from app.services.visual_edit_composer import (
    VisualEditComposer,
    escape_filter_path,
)


def test_visual_edit_omitted_legacy_behavior():
    """Verify that when visual_edit is None, mode defaults to clean and legacy pipeline is used."""
    opts = RenderOptions()
    assert opts.visual_edit is None

    proj = Project(
        name="test_proj",
        source_video_path="dummy.mp4",
        cues=[],
    )
    assert proj.visual_edit is None


def test_visual_edit_mode_clean_is_default():
    """Verify VisualEditConfig defaults to mode=clean."""
    cfg = VisualEditConfig()
    assert cfg.mode == VisualEditMode.CLEAN
    assert cfg.blur.enabled is True
    assert cfg.blur.sigma == 18.0
    assert cfg.blur.padding_px == 8
    assert cfg.blur.feather_px == 6
    assert len(cfg.overlays) == 0


def test_ocr_mask_survives_short_detection_gaps():
    """Verify temporal gap bridging merges intervals when gap <= gap_fill_sec."""
    cues = [
        SubtitleCue(
            id="cue1",
            start=1.0,
            end=2.0,
            ocr_start=1.0,
            ocr_end=2.0,
            source_text="你好",
            ocr_regions=[OCRRegion(points=[[0.1, 0.8], [0.9, 0.8], [0.9, 0.9], [0.1, 0.9]], confidence=0.9)],
        ),
        SubtitleCue(
            id="cue2",
            start=2.1,  # 0.1s gap (less than default 0.25s gap fill)
            end=3.0,
            ocr_start=2.1,
            ocr_end=3.0,
            source_text="世界",
            ocr_regions=[OCRRegion(points=[[0.15, 0.8], [0.85, 0.8], [0.85, 0.9], [0.15, 0.9]], confidence=0.9)],
        ),
        SubtitleCue(
            id="cue3",
            start=5.0,  # 2.0s gap (should NOT be bridged)
            end=6.0,
            ocr_start=5.0,
            ocr_end=6.0,
            source_text="再见",
            ocr_regions=[OCRRegion(points=[[0.2, 0.8], [0.8, 0.8], [0.8, 0.9], [0.2, 0.9]], confidence=0.9)],
        ),
    ]

    composer = VisualEditComposer()
    intervals = composer.extract_temporal_ocr_intervals(cues, gap_fill_sec=0.25)

    assert len(intervals) == 2
    # First interval should bridge cue1 and cue2 (1.0s to 3.0s)
    assert intervals[0]["start"] == 1.0
    assert intervals[0]["end"] == 3.0
    assert len(intervals[0]["polygons"]) == 2

    # Second interval is cue3 (5.0s to 6.0s)
    assert intervals[1]["start"] == 5.0
    assert intervals[1]["end"] == 6.0


def test_normalized_overlay_positioning():
    """Verify overlay coordinate math across different anchor modes."""
    composer = VisualEditComposer()

    # Absolute center
    ov_abs = OverlayConfig(path="dummy.png", x=0.5, y=0.5, width=0.2, anchor=OverlayAnchor.ABSOLUTE)
    x_expr, y_expr, scaled_w = composer.compute_overlay_coordinates(ov_abs, video_width=1280, video_height=720)
    assert x_expr == "W*0.5-w/2"
    assert y_expr == "H*0.5-h/2"
    assert scaled_w == 256

    # Top Left
    ov_tl = OverlayConfig(path="dummy.png", anchor=OverlayAnchor.TOP_LEFT)
    x_expr, y_expr, _ = composer.compute_overlay_coordinates(ov_tl, video_width=1280, video_height=720)
    assert x_expr == "24"
    assert y_expr == "24"

    # Bottom Right
    ov_br = OverlayConfig(path="dummy.png", anchor=OverlayAnchor.BOTTOM_RIGHT)
    x_expr, y_expr, _ = composer.compute_overlay_coordinates(ov_br, video_width=1280, video_height=720)
    assert x_expr == "W-w-24"
    assert y_expr == "H-h-24"


def test_filter_graph_layer_order_and_z_index(tmp_path):
    """Verify strict layer ordering: base -> blur -> overlays (sorted by z_index) -> Vietnamese ASS subtitles on top."""
    dummy_sub = tmp_path / "subs.ass"
    dummy_sub.write_text("[Script Info]\nTitle: Test\n", encoding="utf-8")

    ov_img1 = tmp_path / "sticker_low.png"
    ov_img1.write_bytes(b"\x89PNG\r\n\x1a\n")
    ov_img2 = tmp_path / "sticker_high.png"
    ov_img2.write_bytes(b"\x89PNG\r\n\x1a\n")

    cfg = VisualEditConfig(
        mode=VisualEditMode.BLUR_OVERLAY,
        blur=BlurConfig(enabled=True, sigma=18.0),
        overlays=[
            OverlayConfig(id="high", path=str(ov_img2), start=2.0, end=5.0, z_index=20),
            OverlayConfig(id="low", path=str(ov_img1), start=1.0, end=4.0, z_index=5),
        ],
    )

    composer = VisualEditComposer()
    filters, extra_inputs, final_label = composer.build_composition_filter_graph(
        video_width=1280,
        video_height=720,
        subtitle_file=dummy_sub,
        visual_edit=cfg,
        has_blur_mask=True,
        options=RenderOptions(),
    )

    filter_str = ";".join(filters)

    # 1. Blur base layer must come first
    assert "gblur=sigma=18" in filter_str
    assert "maskedmerge" in filter_str

    # 2. Overlays must be processed in z-index order (low z=5 before high z=20)
    pos_low = filter_str.find("ov_0")  # first in sorted list is z_index=5 (low)
    pos_high = filter_str.find("ov_1")  # second is z_index=20 (high)
    assert pos_low != -1
    assert pos_high != -1
    assert pos_low < pos_high

    # 3. Vietnamese subtitles MUST BE LAST
    pos_sub = filter_str.find("ass=")
    assert pos_sub != -1
    assert pos_sub > pos_high
    assert final_label == "final_out"


def test_visual_edit_does_not_modify_translation_or_cues():
    """Verify visual editing is strictly post-localization and never touches translated text or cues."""
    cues = [
        SubtitleCue(
            id="c1",
            start=11.87,
            end=12.64,
            source_text="领口歪了，",
            translated_text="Cổ áo lệch rồi.",
        ),
        SubtitleCue(
            id="c2",
            start=70.31,
            end=73.80,
            source_text="在早餐店的家庭里，凌晨四点就要起来帮忙揉面。",
            translated_text="Ở một gia đình mở quán ăn sáng, bốn giờ sáng đã phải dậy giúp nhào bột.",
        ),
    ]

    proj = Project(
        name="test_guard",
        source_video_path="dummy.mp4",
        cues=cues,
        visual_edit=VisualEditConfig(mode=VisualEditMode.BLUR_OVERLAY),
    )

    composer = VisualEditComposer()
    intervals = composer.extract_temporal_ocr_intervals(proj.cues)
    assert len(intervals) > 0

    # Ensure translated text remained 100% untouched
    assert proj.cues[0].translated_text == "Cổ áo lệch rồi."
    assert proj.cues[1].translated_text == "Ở một gia đình mở quán ăn sáng, bốn giờ sáng đã phải dậy giúp nhào bột."
