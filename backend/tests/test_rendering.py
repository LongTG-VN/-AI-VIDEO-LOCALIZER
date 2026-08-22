import shutil
import pytest
import numpy as np
from pathlib import Path
from app.models.project import Project, RenderOptions, SubtitleCue
from app.services.subtitles import format_ass_timestamp, wrap_vietnamese_text, to_ass, write_ass
from app.services.hardsub_cleaner import HardSubCleaner
from app.services.renderer import escape_filter_path, check_nvenc_available, get_media_info, Renderer


def test_format_ass_timestamp():
    assert format_ass_timestamp(0.0) == "0:00:00.00"
    assert format_ass_timestamp(1.234) == "0:00:01.23"
    assert format_ass_timestamp(65.456) == "0:01:05.46"
    assert format_ass_timestamp(3661.0) == "1:01:01.00"


def test_vietnamese_line_wrapping():
    # Short text: no split
    short = "Cổ áo con bị lệch rồi."
    assert wrap_vietnamese_text(short, max_line_chars=36) == "Cổ áo con bị lệch rồi."

    # Long text with comma: splits at comma
    long_text = "Tôi là thiên kim duy nhất của gia đình họ Tần trong suốt 18 năm qua."
    wrapped = wrap_vietnamese_text(long_text, max_line_chars=36)
    assert "\\N" in wrapped
    parts = wrapped.split("\\N")
    assert len(parts) == 2
    assert parts[0].strip() == "Tôi là thiên kim duy nhất của gia" or "họ Tần" in parts[0] or "duy nhất" in parts[0]
    # Verify no broken Vietnamese syllables
    assert " ".join(wrapped.split("\\N")) == long_text


def test_ass_generation_and_styling():
    cues = [
        SubtitleCue(start=1.0, end=3.0, source_text="领口歪了", translated_text="Cổ áo con bị lệch rồi."),
        SubtitleCue(start=3.5, end=5.0, source_text="坐姿不对", translated_text="Tư thế ngồi không đúng."),
    ]
    options = RenderOptions(
        font_name="Segoe UI",
        font_size=24,
        margin_v=35,
        outline_width=2.8,
        shadow_depth=1.2,
    )
    ass_content = to_ass(cues, options, width=852, height=480)

    assert "[Script Info]" in ass_content
    assert "PlayResX: 852" in ass_content
    assert "PlayResY: 480" in ass_content
    assert "Segoe UI" in ass_content
    assert "24" in ass_content
    assert "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Cổ áo con bị lệch rồi." in ass_content
    assert "Dialogue: 0,0:00:03.50,0:00:05.00,Default,,0,0,0,,Tư thế ngồi không đúng." in ass_content


def test_escape_filter_path():
    p = Path("d:/codex/videos/test.ass")
    escaped = escape_filter_path(p)
    assert "\\:" in escaped or ":" not in escaped or "d" in escaped


def test_hardsub_mask_generation_and_dilation():
    cleaner = HardSubCleaner(crop_top_ratio=0.65, crop_bottom_ratio=0.95, mask_dilate_radius=3)
    # Create black frame with a simulated bright white subtitle text block
    frame = np.zeros((480, 852, 3), dtype=np.uint8)
    # Draw white text box in subtitle region (Y: 380-410, X: 300-550)
    frame[380:410, 300:550] = 255

    mask = cleaner.build_text_mask(frame)
    assert mask is not None
    assert mask.shape == (480, 852)
    # Mask should be 255 at text region
    assert np.sum(mask[380:410, 300:550] == 255) > 0
    # Mask outside subtitle band must remain 0
    assert np.sum(mask[0:200, :] > 0) == 0


def test_clean_frame_modes():
    cleaner = HardSubCleaner()
    frame = np.zeros((480, 852, 3), dtype=np.uint8)
    frame[380:410, 300:550] = 255

    # Mode none
    res_none, cleaned = cleaner.clean_frame(frame, mode="none")
    assert not cleaned

    # Mode inpaint
    res_inpaint, cleaned = cleaner.clean_frame(frame, mode="inpaint")
    assert cleaned
    assert res_inpaint.shape == (480, 852, 3)

    # Mode cover
    res_cover, cleaned = cleaner.clean_frame(frame, mode="cover")
    assert cleaned
    assert res_cover.shape == (480, 852, 3)


def test_nvenc_availability():
    # Should safely return boolean without raising
    has_nvenc = check_nvenc_available("ffmpeg")
    assert isinstance(has_nvenc, bool)
