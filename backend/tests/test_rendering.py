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
    parse_timestamp,
)
from app.services.utterance_engine import (
    UtteranceEngine,
    RenderSubtitleCue,
    clean_vietnamese_typography,
    semantic_line_break,
)
from app.services.hardsub_cleaner import HardSubCleaner
from app.services.renderer import escape_filter_path, check_nvenc_available, get_media_info


def test_utterance_grouping():
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(id="c1", start=22.09, end=25.00, source_text="昨天的宏观经济笔记", translated_text="Quyển sổ ghi chép kinh tế vĩ mô hôm qua,", speaker_id="spk_dad"),
        SubtitleCue(id="c2", start=25.00, end=25.50, source_text="看完了吗", translated_text="con đã đọc xong chưa?", speaker_id="spk_dad"),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 1
    assert "con đọc xong chưa" in render_cues[0].render_text
    assert render_cues[0].source_cue_ids == ["c1", "c2"]
    assert render_cues[0].start == 22.09
    assert render_cues[0].end == 25.50


def test_incomplete_sentence_merge():
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(id="c1", start=35.23, end=36.00, source_text="你的存在", translated_text="Sự tồn tại của em,", speaker_id="spk_bro"),
        SubtitleCue(id="c2", start=36.00, end=37.50, source_text="拉低了秦家的执行效率", translated_text="đã kéo giảm hiệu suất làm việc của Gia đình họ Tần.", speaker_id="spk_bro"),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 1
    assert "làm giảm" in render_cues[0].render_text
    assert "nhà họ Tần" in render_cues[0].render_text


def test_speaker_change_prevents_merge():
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(id="c1", start=25.00, end=25.50, source_text="看完了吗", translated_text="Con đã đọc xong chưa?", speaker_id="spk_dad"),
        SubtitleCue(id="c2", start=25.80, end=26.60, source_text="看看了", translated_text="Con đã đọc rồi ạ.", speaker_id="spk_daughter"),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 2
    assert render_cues[0].speaker_id == "spk_dad"
    assert render_cues[1].speaker_id == "spk_daughter"


def test_duplicate_suffix_prefix_detection():
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(id="c1", start=14.81, end=15.75, source_text="你今天是去丢人还是去赴宴", translated_text="Hôm nay con đi làm mất mặt hay đi dự tiệc vậy?", speaker_id="spk_mom"),
        SubtitleCue(id="c2", start=15.75, end=16.51, source_text="还是去赴宴？", translated_text="Hay là đi dự tiệc?", speaker_id="spk_mom"),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 1
    assert metrics["suppressed_duplicates"] == 1
    assert render_cues[0].end == 16.51
    assert render_cues[0].source_cue_ids == ["c1", "c2"]


def test_source_cue_preservation():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=3.5, source_text="A", translated_text="A_tr"),
        SubtitleCue(id="c2", start=3.0, end=5.0, source_text="B", translated_text="B_tr"),
    ]
    engine = UtteranceEngine()
    _ = engine.process_cues(cues)
    # Original cues must remain intact
    assert cues[0].start == 1.0
    assert cues[0].end == 3.5
    assert cues[1].start == 3.0
    assert cues[1].end == 5.0


def test_render_cue_source_mapping():
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(id="uuid_1", start=10.0, end=11.0, source_text="我妈", translated_text="Mẹ tôi.", speaker_id="spk_1"),
        SubtitleCue(id="uuid_2", start=11.0, end=12.0, source_text="宋知雪", translated_text="Tống Tri Tuyết.", speaker_id="spk_1"),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 1
    assert render_cues[0].source_cue_ids == ["uuid_1", "uuid_2"]


def test_cps_calculation():
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=1.0, end=3.0, source_text="短句", translated_text="Một câu nói ngắn gọn hai giây."),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 1
    dur = render_cues[0].end - render_cues[0].start
    expected_cps = round(len("Một câu nói ngắn gọn hai giây.") / dur, 1)
    assert render_cues[0].cps == pytest.approx(expected_cps, 0.2)


def test_semantic_line_breaking():
    short = "Cổ áo con bị lệch rồi."
    assert semantic_line_break(short, max_line_chars=36) == "Cổ áo con bị lệch rồi."

    long_text = "Tôi là thiên kim duy nhất của nhà họ Tần trong suốt 18 năm qua."
    wrapped = semantic_line_break(long_text, max_line_chars=36)
    assert r"\N" in wrapped
    parts = wrapped.split(r"\N")
    assert len(parts) == 2
    assert " ".join(parts) == long_text


def test_vietnamese_punctuation_cleanup():
    raw = "Sự tồn tại của em, đã kéo giảm hiệu suất làm việc của Gia đình họ Tần."
    cleaned = clean_vietnamese_typography(raw)
    assert "nhà họ Tần" in cleaned
    assert "làm giảm" in cleaned


def test_single_active_timeline_after_grouping():
    cues = [
        SubtitleCue(start=0.0, end=2.0, source_text="A", translated_text="Câu A"),
        SubtitleCue(start=1.5, end=3.0, source_text="B", translated_text="Câu B"),
        SubtitleCue(start=2.5, end=4.0, source_text="C", translated_text="Câu C"),
    ]
    engine = UtteranceEngine(safe_gap=0.03)
    render_cues, _ = engine.process_cues(cues)
    for i in range(len(render_cues) - 1):
        assert render_cues[i].end <= render_cues[i+1].start


def test_monologue_grouping():
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=70.31, end=71.50, source_text="而我", translated_text="Còn tôi thì...", speaker_id="spk_narrator"),
        SubtitleCue(start=71.50, end=73.50, source_text="在一个开早餐店的家庭里", translated_text="lớn lên trong một gia đình bán đồ ăn sáng.", speaker_id="spk_narrator"),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 1
    assert "Còn tôi lớn lên" in render_cues[0].render_text
    assert "bán đồ ăn sáng" in render_cues[0].render_text


def test_question_answer_not_merged():
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=60.18, end=61.28, source_text="你还吃的下去", translated_text="Cô còn ăn nổi sao?", speaker_id="spk_rival"),
        SubtitleCue(start=61.42, end=62.16, source_text="有点凉了", translated_text="Hơi nguội rồi.", speaker_id="spk_heroine"),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 2
    assert "ăn nổi" in render_cues[0].render_text
    assert "nguội" in render_cues[1].render_text


def test_ass_generated_events_do_not_overlap():
    cues = [
        SubtitleCue(start=1.0, end=3.0, source_text="领口歪了", translated_text="Cổ áo con bị lệch rồi."),
        SubtitleCue(start=2.5, end=4.5, source_text="坐姿不对", translated_text="Tư thế ngồi không đúng."),
    ]
    options = RenderOptions(font_name="Arial", font_size=26, margin_v=38)
    ass_content = to_ass(cues, options, width=852, height=480)

    events = [line for line in ass_content.splitlines() if line.startswith("Dialogue:")]
    assert len(events) >= 1

    def parse_ass_time(ts_str: str) -> float:
        h, m, s = ts_str.split(":")
        sec, cs = s.split(".")
        return int(h)*3600 + int(m)*60 + int(sec) + int(cs)/100.0

    for i in range(len(events) - 1):
        t_end = parse_ass_time(events[i].split(",")[2])
        t_next_start = parse_ass_time(events[i+1].split(",")[1])
        assert t_end <= t_next_start


def test_hardsub_mask_and_preservation():
    cleaner = HardSubCleaner(crop_top_ratio=0.66, crop_bottom_ratio=0.95, mask_dilate_radius=2)
    frame_empty = np.zeros((480, 852, 3), dtype=np.uint8)
    frame_empty[380:410, 300:550] = 255
    res, was_cleaned = cleaner.clean_frame(frame_empty, mode="inpaint", is_subtitle_active=False)
    assert not was_cleaned
    assert np.max(np.abs(res.astype(float) - frame_empty.astype(float))) == 0.0
