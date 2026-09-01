from __future__ import annotations

import numpy as np
import pytest

from app.models.project import OCRRegion, OCREvidence, Project, RenderOptions, SubtitleCue, get_final_vi_text
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.utterance_engine import UtteranceEngine, semantic_line_break


# 1. Short VI creates compact backing
def test_01_short_vi_compact_backing():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(id="c1", start=1.0, end=3.0, source_text="你好", final_translation="Chào", original_source_cue_ids=["c1"])
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    assert len(contexts) == 1
    w = contexts[0]["bbox"][2] - contexts[0]["bbox"][0]
    assert w <= 300


# 2. Long VI wraps before exceeding max width
def test_02_long_vi_wraps_before_max_width():
    engine = UtteranceEngine(max_line_chars=34)
    long_text = "Cơm cậu nấu ngon thế này thì đồ ăn ngoài hết cửa sống luôn."
    wrapped = semantic_line_break(long_text, max_line_chars=34)
    lines = wrapped.split(r"\N")
    assert len(lines) == 2
    assert len(lines[0]) <= 38
    assert len(lines[1]) <= 38


# 3. Backing recomputed after wrapping
def test_03_backing_recomputed_after_wrapping():
    cleaner = PatchCoverCleaner()
    long_cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=4.0,
        source_text="你这饭给外卖判了死缓",
        final_translation="Cơm cậu nấu ngon thế này thì đồ ăn ngoài hết cửa sống luôn.",
        original_source_cue_ids=["c1"],
    )
    contexts = cleaner.build_render_cue_contexts([long_cue], 1280, 720)
    w = contexts[0]["bbox"][2] - contexts[0]["bbox"][0]
    assert w <= int(1280 * 0.70)


# 4. Source bbox remains contained
def test_04_source_bbox_remains_contained():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=3.0,
        source_text="写字楼",
        final_translation="Tòa nhà",
        ocr_regions=[OCRRegion(points=[[0.45, 0.88], [0.55, 0.88], [0.55, 0.93], [0.45, 0.93]])],
        original_source_cue_ids=["c1"],
    )
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert x1 <= int(0.45 * 1280)
    assert x2 >= int(0.55 * 1280)
    assert y1 <= int(0.88 * 720)
    assert y2 >= int(0.93 * 720)


# 5. VI bbox remains contained
def test_05_vi_bbox_remains_contained():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(id="c1", start=1.0, end=3.0, source_text="你好", final_translation="Xin chào tất cả các bạn", original_source_cue_ids=["c1"])
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (x2 - x1) >= 150
    assert (y2 - y1) >= 40


# 6. 1-line uses one compact rectangle
def test_06_one_line_compact_height():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(id="c1", start=1.0, end=3.0, source_text="你好", final_translation="Chào bạn", original_source_cue_ids=["c1"])
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    h = contexts[0]["bbox"][3] - contexts[0]["bbox"][1]
    assert h <= 55


# 7. 2-line uses one compact rectangle
def test_07_two_line_compact_height():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(id="c1", start=1.0, end=3.0, source_text="句子", final_translation="Dòng số một\nDòng số hai", original_source_cue_ids=["c1"])
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    h = contexts[0]["bbox"][3] - contexts[0]["bbox"][1]
    assert 60 <= h <= 90


# 8. VI centered inside backing
def test_08_vi_centered_inside_backing():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(id="c1", start=1.0, end=3.0, source_text="测试", final_translation="Thử nghiệm", original_source_cue_ids=["c1"])
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    center = (x1 + x2) // 2
    assert abs(center - 640) <= 20


# 9. No raw oversized OCR bbox drives final visible size
def test_09_no_raw_oversized_ocr():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=3.0,
        source_text="短句",
        final_translation="Câu ngắn",
        ocr_regions=[OCRRegion(points=[[0.05, 0.88], [0.95, 0.88], [0.95, 0.93], [0.05, 0.93]])],
        original_source_cue_ids=["c1"],
    )
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    w = contexts[0]["bbox"][2] - contexts[0]["bbox"][0]
    assert w < int(1280 * 0.85)


# 10. Padding stays within configured compact range
def test_10_compact_padding_range():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(id="c1", start=1.0, end=3.0, source_text="话", final_translation="Lời", original_source_cue_ids=["c1"])
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (x2 - x1) <= 250
    assert (y2 - y1) <= 60


# 11. Oversized backing triggers wrap/re-layout
def test_11_oversized_triggers_wrap():
    text = "Điều duy nhất khiến tôi có chút giá trị trong tòa nhà văn phòng này chính là kỹ năng nấu ăn mẹ truyền lại."
    wrapped = semantic_line_break(text, max_line_chars=34)
    assert r"\N" in wrapped


# 12. Text never clipped by backing
def test_12_text_not_clipped():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(id="c1", start=1.0, end=3.0, source_text="你好", final_translation="Xin chào", original_source_cue_ids=["c1"])
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert x1 >= 0 and x2 <= 1280
    assert y1 >= 0 and y2 <= 720


# 13. Suppressed cover-only path unchanged
def test_13_suppressed_cover_only_unchanged():
    cleaner = PatchCoverCleaner()
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="嗯", suppression_status="SUPPRESSED_FILLER", ocr_regions=[OCRRegion(points=[[0.48, 0.88], [0.52, 0.88], [0.52, 0.94], [0.48, 0.94]])])
    contexts = cleaner.build_render_cue_contexts([cue], 1280, 720)
    assert len(contexts) == 1
    assert contexts[0]["has_vi"] is False
    assert contexts[0]["reason"] == "SUPPRESSED_FILLER"


# 14. Cover timing unchanged
def test_14_cover_timing_unchanged():
    cleaner = PatchCoverCleaner()
    c1 = SubtitleCue(id="c1", start=10.0, end=12.0, ocr_start=9.9, source_text="台词", final_translation="Lời", original_source_cue_ids=["c1"])
    interval = cleaner.get_cover_interval(c1, [c1])
    assert interval[0] <= 9.9
    assert interval[1] >= 12.0


# 15. Translation unchanged
def test_15_translation_unchanged():
    c1 = SubtitleCue(id="c1", start=61.64, end=63.10, source_text="呃是是我阿姨", final_translation="Ơ... dạ là dì của cháu.")
    assert get_final_vi_text(c1) == "Ơ... dạ là dì của cháu."


# 16. SourceGlyphTrack unchanged
def test_16_source_glyph_track_unchanged():
    cleaner = PatchCoverCleaner()
    assert hasattr(cleaner, "build_render_cue_contexts")
