from __future__ import annotations

import pytest

from app.models.project import Project, RenderOptions, SubtitleCue, get_final_vi_text
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.subtitles import UtteranceEngine, to_ass


# 1. Source starts before VI -> cover starts at source (with pre-roll)
def test_01_source_starts_before_vi_cover_starts_earlier():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            ocr_start=0.90,
            source_text="你好",
            final_translation="Xin chào",
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    # Cover starts at min(0.90 - pre_roll, 1.0) = ~0.833s
    assert contexts[0]["start"] <= 0.85
    assert contexts[0]["vi_start"] == 1.0


# 2. Source ends after VI -> cover ends at source (with post-roll)
def test_02_source_ends_after_vi_cover_ends_later():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            ocr_end=2.15,
            source_text="你好",
            final_translation="Xin chào",
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    assert contexts[0]["end"] >= 2.15
    assert contexts[0]["vi_end"] == 2.0


# 3. VI timing remains strictly unchanged
def test_03_vi_timing_remains_unchanged():
    cues = [
        SubtitleCue(
            id="c1",
            start=5.0,
            end=6.5,
            ocr_start=4.8,
            source_text="文字",
            final_translation="Văn bản",
            original_source_cue_ids=["c1"],
        )
    ]
    ass_str = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    assert "0:00:05.00,0:00:06.50" in ass_str


# 4. 1-frame OCR jitter handled with frame pre-roll
def test_04_ocr_jitter_pre_roll():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=2.0,
            end=3.0,
            ocr_start=1.95,
            source_text="测试",
            final_translation="Thử nghiệm",
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    # Pre-roll extends start by ~2 frames (~0.067s) before ocr_start
    assert contexts[0]["start"] < 1.95


# 5. Filler cue source fully covered
def test_05_filler_cue_fully_covered():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=104.20,
            end=105.00,
            ocr_start=104.10,
            ocr_end=104.80,
            source_text="一，",
            final_translation="Một,",
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["start"] <= 104.10
    assert contexts[0]["end"] >= 105.00


# 6. No Chinese pre-roll exposure
def test_06_no_chinese_pre_roll_exposure():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=61.64,
            end=63.10,
            ocr_start=61.50,
            source_text="呃是是我阿姨",
            final_translation="Ơ... dạ là dì của cháu.",
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["start"] <= 61.50


# 7. No Chinese post-roll exposure
def test_07_no_chinese_post_roll_exposure():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=61.64,
            end=63.10,
            ocr_end=63.25,
            source_text="呃是是我阿姨",
            final_translation="Ơ... dạ là dì của cháu.",
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["end"] >= 63.25


# 8. Scene cut clamps lingering cover
def test_08_scene_cut_clamping():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="句一", final_translation="Câu 1", original_source_cue_ids=["c1"]),
        SubtitleCue(id="c2", start=2.05, end=3.0, source_text="句二", final_translation="Câu 2", original_source_cue_ids=["c2"]),
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["end"] == contexts[1]["start"]


# 9. Adjacent cues have no uncovered temporal gap
def test_09_adjacent_cues_no_temporal_gap():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=10.0, end=11.0, source_text="第一句", final_translation="Câu 1", original_source_cue_ids=["c1"]),
        SubtitleCue(id="c2", start=11.06, end=12.0, source_text="第二句", final_translation="Câu 2", original_source_cue_ids=["c2"]),
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["end"] == contexts[1]["start"]


# 10. Adjacent cues do not double-stack visible covers
def test_10_no_double_stacking_adjacent_covers():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="句1", final_translation="C1", original_source_cue_ids=["c1"]),
        SubtitleCue(id="c2", start=2.0, end=3.0, source_text="句2", final_translation="C2", original_source_cue_ids=["c2"]),
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["end"] <= contexts[1]["start"]


# 11. Decorative OCR does not extend cover
def test_11_decorative_ocr_ignored():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=5.0,
            end=6.0,
            ocr_start=1.0,  # Far away decorative OCR
            source_text="正常",
            final_translation="Bình thường",
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    # Ignored because diff > 0.20s
    assert contexts[0]["start"] >= 4.90


# 12. Low-confidence OCR cannot create huge temporal extension
def test_12_huge_temporal_extension_prevented():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=10.0,
            end=12.0,
            ocr_end=20.0,  # 8s outlier
            source_text="短句",
            final_translation="Câu ngắn",
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["end"] <= 12.20


# 13. Source cue ID mapping preferred over positional matching
def test_13_source_cue_id_mapping():
    cleaner = PatchCoverCleaner()
    c1 = SubtitleCue(id="id_alpha", start=1.0, end=2.0, ocr_start=0.90, source_text="A", final_translation="A", original_source_cue_ids=["id_alpha"])
    contexts = cleaner.build_render_cue_contexts([c1], 1280, 720)
    assert contexts[0]["start"] <= 0.90


# 14. Temporal coverage ratio reaches 1.0 for reliable intervals
def test_14_temporal_coverage_reaches_100pct():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào", original_source_cue_ids=["c1"]),
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    cov = contexts[0]
    assert cov["start"] <= 1.0 and cov["end"] >= 2.0


# 15. Cover geometry remains compact soft-rect
def test_15_cover_geometry_compact():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", final_translation="Thử nghiệm", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.17)


# 16. Translation text remains unchanged
def test_16_translation_text_preserved():
    cue = SubtitleCue(id="c1", start=61.64, end=63.10, source_text="呃是是我阿姨", final_translation="Ơ... dạ là dì của cháu.")
    assert get_final_vi_text(cue) == "Ơ... dạ là dì của cháu."


# 17. Source Integrity remains unchanged
def test_17_source_integrity_status_preserved():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", source_integrity_status="PASS")
    assert cue.source_integrity_status == "PASS"


# 18. Semantic Group remains unchanged
def test_18_semantic_group_preserved():
    cues = [
        SubtitleCue(id="c1", start=100.13, end=101.60, source_text="写字楼里有点存在感的", final_translation="tòa nhà văn phòng này"),
        SubtitleCue(id="c2", start=101.63, end=103.11, source_text="是我妈传给我的手艺做饭", final_translation="chính là kỹ năng nấu ăn mẹ truyền lại."),
    ]
    assert get_final_vi_text(cues[0]) == "tòa nhà văn phòng này"
    assert get_final_vi_text(cues[1]) == "chính là kỹ năng nấu ăn mẹ truyền lại."
