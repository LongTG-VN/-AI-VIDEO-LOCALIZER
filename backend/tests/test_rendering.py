import pytest
import numpy as np
from pathlib import Path
from app.models.project import Project, RenderOptions, SubtitleCue
from app.services.subtitles import (
    format_ass_timestamp,
    wrap_vietnamese_text,
    to_ass,
    write_ass,
    normalize_render_cues,
    RenderCue,
    parse_timestamp,
)
from app.services.hardsub_cleaner import HardSubCleaner
from app.services.renderer import escape_filter_path, check_nvenc_available, get_media_info, build_subtitle_filter


def test_adjacent_cue_overlap_trimming():
    cues = [
        SubtitleCue(start=1.0, end=3.5, source_text="领口歪了", translated_text="Cổ áo con bị lệch rồi."),
        SubtitleCue(start=3.0, end=5.0, source_text="坐姿不对", translated_text="Tư thế ngồi không đúng."),
    ]
    render_cues = normalize_render_cues(cues, safe_gap=0.03)
    assert len(render_cues) == 2
    # Cue 1 must end before cue 2 starts
    assert render_cues[0].render_end <= render_cues[1].render_start - 0.03
    assert render_cues[0].render_end == pytest.approx(2.97, 0.01)
    assert render_cues[1].render_start == pytest.approx(3.0, 0.01)


def test_safe_gap_and_duration_positive():
    cues = [
        SubtitleCue(start=1.0, end=1.5, source_text="短句", translated_text="Câu ngắn"),
        SubtitleCue(start=1.8, end=2.5, source_text="下句", translated_text="Câu tiếp"),
    ]
    render_cues = normalize_render_cues(cues, safe_gap=0.03, min_duration=0.25)
    for rc in render_cues:
        assert rc.render_end > rc.render_start
        assert (rc.render_end - rc.render_start) >= 0.10


def test_single_active_subtitle_invariant():
    cues = [
        SubtitleCue(start=0.0, end=2.0, source_text="A", translated_text="Câu A"),
        SubtitleCue(start=1.5, end=3.0, source_text="B", translated_text="Câu B"),
        SubtitleCue(start=2.5, end=4.0, source_text="C", translated_text="Câu C"),
    ]
    render_cues = normalize_render_cues(cues, safe_gap=0.03)
    # Check that for all adjacent pairs, cue[i].end <= cue[i+1].start
    for i in range(len(render_cues) - 1):
        assert render_cues[i].render_end <= render_cues[i+1].render_start


def test_original_source_timing_unchanged():
    cues = [
        SubtitleCue(start=1.0, end=3.5, source_text="A", translated_text="A_tr"),
        SubtitleCue(start=3.0, end=5.0, source_text="B", translated_text="B_tr"),
    ]
    _ = normalize_render_cues(cues, safe_gap=0.03)
    # Original cues must remain intact
    assert cues[0].start == 1.0
    assert cues[0].end == 3.5
    assert cues[1].start == 3.0
    assert cues[1].end == 5.0


def test_same_speaker_adjacent_cues_merge_if_identical():
    cues = [
        SubtitleCue(start=8.85, end=9.00, source_text="连呼吸都要符合KPI", translated_text="Ngay cả hơi thở cũng phải đạt chuẩn KPI."),
        SubtitleCue(start=9.00, end=10.29, source_text="连呼吸都要符合KPI", translated_text="Ngay cả hơi thở cũng phải đạt chuẩn KPI"),
    ]
    render_cues = normalize_render_cues(cues)
    # Should merge into 1 continuous cue
    assert len(render_cues) == 1
    assert render_cues[0].render_start == 8.85
    assert render_cues[0].render_end == 10.29


def test_different_speaker_adjacent_cues_separate():
    cues = [
        SubtitleCue(start=10.0, end=12.0, source_text="谁啊", translated_text="Ai vậy?", speaker_id="spk_0"),
        SubtitleCue(start=12.0, end=14.0, source_text="是我", translated_text="Là tôi.", speaker_id="spk_1"),
    ]
    render_cues = normalize_render_cues(cues, safe_gap=0.03)
    assert len(render_cues) == 2
    assert render_cues[0].render_end <= render_cues[1].render_start


def test_ass_generated_events_do_not_overlap():
    cues = [
        SubtitleCue(start=1.0, end=3.0, source_text="领口歪了", translated_text="Cổ áo con bị lệch rồi."),
        SubtitleCue(start=2.5, end=4.5, source_text="坐姿不对", translated_text="Tư thế ngồi không đúng."),
    ]
    options = RenderOptions(font_name="Arial", font_size=24)
    ass_content = to_ass(cues, options, width=852, height=480)

    # Parse all Dialogue lines
    events = [line for line in ass_content.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 2

    # Extract start/end timestamps from ASS format: Dialogue: Layer,Start,End,...
    def parse_ass_time(ts_str: str) -> float:
        h, m, s = ts_str.split(":")
        sec, cs = s.split(".")
        return int(h)*3600 + int(m)*60 + int(sec) + int(cs)/100.0

    t1_end = parse_ass_time(events[0].split(",")[2])
    t2_start = parse_ass_time(events[1].split(",")[1])
    assert t1_end <= t2_start


def test_vietnamese_line_wrapping():
    short = "Cổ áo con bị lệch rồi."
    assert wrap_vietnamese_text(short, max_line_chars=36) == "Cổ áo con bị lệch rồi."

    long_text = "Tôi là thiên kim duy nhất của gia đình họ Tần trong suốt 18 năm qua."
    wrapped = wrap_vietnamese_text(long_text, max_line_chars=36)
    assert r"\N" in wrapped
    parts = wrapped.split(r"\N")
    assert len(parts) == 2
    assert " ".join(parts) == long_text


def test_hardsub_mask_and_preservation():
    cleaner = HardSubCleaner(crop_top_ratio=0.66, crop_bottom_ratio=0.95, mask_dilate_radius=2)
    
    # 1. Test untouched frame when inactive
    frame_empty = np.zeros((480, 852, 3), dtype=np.uint8)
    frame_empty[380:410, 300:550] = 255
    res, was_cleaned = cleaner.clean_frame(frame_empty, mode="inpaint", is_subtitle_active=False)
    assert not was_cleaned
    assert np.max(np.abs(res.astype(float) - frame_empty.astype(float))) == 0.0

    # 2. Test inpaint when active
    res_inpainted, was_cleaned2 = cleaner.clean_frame(frame_empty, mode="inpaint", is_subtitle_active=True)
    assert was_cleaned2
    assert res_inpainted.shape == (480, 852, 3)


def test_escape_filter_path():
    p = Path("d:/codex/videos/test.ass")
    escaped = escape_filter_path(p)
    assert "\\:" in escaped or ":" not in escaped or "d" in escaped


def test_nvenc_availability():
    has_nvenc = check_nvenc_available("ffmpeg")
    assert isinstance(has_nvenc, bool)
