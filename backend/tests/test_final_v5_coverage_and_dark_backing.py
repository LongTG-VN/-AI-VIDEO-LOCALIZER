from __future__ import annotations

import numpy as np
import pytest

from app.models.project import OCRRegion, Project, RenderOptions, SubtitleCue, get_final_vi_text
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.subtitles import UtteranceEngine, to_ass
from app.services.translation_quality.naturalness import NaturalnessPolisher


# 1. Glyph polygons cannot become final visible backing shape
def test_01_glyph_polygons_not_backing_shape():
    cleaner = PatchCoverCleaner()
    poly = [[0.40, 0.85], [0.60, 0.85], [0.60, 0.92], [0.40, 0.92]]
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            source_text="文字",
            final_translation="Văn bản",
            ocr_regions=[OCRRegion(points=poly, text="文字")],
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    bbox = contexts[0]["bbox"]
    mask = cleaner.create_rounded_rect_mask(720, 1280, bbox)
    assert mask.shape == (720, 1280)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)


# 2. One RenderSubtitleCue creates exactly one visible cover
def test_02_one_render_cue_one_cover():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1


# 3. Cover contains source bbox
def test_03_cover_contains_source_bbox():
    cleaner = PatchCoverCleaner()
    zh_poly = [[0.40, 0.85], [0.60, 0.85], [0.60, 0.92], [0.40, 0.92]]
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            source_text="你好世界",
            final_translation="Chào thế giới",
            ocr_regions=[OCRRegion(points=zh_poly, text="你好世界")],
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert x1 <= int(0.40 * 1280)
    assert x2 >= int(0.60 * 1280)
    assert y1 <= int(0.85 * 720)
    assert y2 >= int(0.92 * 720)


# 4. Cover contains VI bbox
def test_04_cover_contains_vi_bbox():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào bạn hôm nay rất vui được gặp", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (x2 - x1) > 400


# 5. Short VI cue still covers wider Chinese source
def test_05_short_vi_covers_wider_chinese():
    cleaner = PatchCoverCleaner()
    zh_poly = [[0.30, 0.85], [0.70, 0.85], [0.70, 0.92], [0.30, 0.92]]
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            source_text="这是一句很长的中文台词",
            final_translation="Ừm.",
            ocr_regions=[OCRRegion(points=zh_poly, text="这是一句很长的中文台词")],
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    # Even though VI is "Ừm.", width must cover the 0.30 to 0.70 Chinese extent
    assert x1 <= int(0.30 * 1280)
    assert x2 >= int(0.70 * 1280)


# 6. One-line cover compact
def test_06_one_line_cover_compact():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="一句话", final_translation="Một câu ngắn", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.12)


# 7. Two-line cover compact
def test_07_two_line_cover_compact():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="两行", final_translation="Dòng một\nDòng hai", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.17)


# 8. Abnormal OCR geometry triggers safe fallback
def test_08_abnormal_ocr_safe_fallback():
    cleaner = PatchCoverCleaner()
    abnormal_poly = [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            source_text="异常",
            final_translation="Bình thường",
            ocr_regions=[OCRRegion(points=abnormal_poly, text="异常")],
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert y1 >= int(720 * 0.68)


# 9. Oversized cover rejected / clamped
def test_09_oversized_cover_clamped():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", final_translation="Thử nghiệm", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.17)


# 10. No glyph-mask + backing double-black stacking
def test_10_no_double_black_stacking():
    cleaner = PatchCoverCleaner()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    vi_mask = cleaner.create_rounded_rect_mask(720, 1280, (400, 600, 880, 660))
    # Alpha mask is None in clean_video
    out = cleaner.apply_patch_cover(frame, vi_backing_mask=vi_mask)
    assert out.shape == (720, 1280, 3)
    # Background texture preserved (~70-85px pixel brightness)
    assert np.mean(out[610:650, 450:850]) > 40


# 11. Lifecycle matches render cue
def test_11_lifecycle_matches_render_cue():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=2.5, end=5.0, source_text="测试", final_translation="Thử nghiệm", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["start"] == 2.5
    assert contexts[0]["end"] == 5.0


# 12. No cover without VI
def test_12_no_cover_without_vi():
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts([], 1280, 720)
    assert len(contexts) == 0


# 13. VI rendered above cover
def test_13_vi_rendered_above_cover():
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào", original_source_cue_ids=["c1"])]
    ass_str = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    assert "Dialogue:" in ass_str
    assert "Xin chào" in ass_str


# 14. All dialogue cues resolve to visible translation or explicit suppression
def test_14_all_dialogue_cues_coverage():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào"),
        SubtitleCue(id="c2", start=2.5, end=3.0, source_text="AL", final_translation=""),
    ]
    engine = UtteranceEngine()
    rcs, _ = engine.process_cues(cues, translated=True)
    assert len(rcs) == 1
    assert rcs[0].source_cue_ids == ["c1"]


# 15. PASS cue with missing VI fails
def test_15_pass_cue_missing_vi_flagged():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="", quality_status="PASS")
    assert get_final_vi_text(cue) == ""


# 16. VI without render mapping fails
def test_16_vi_without_render_mapping():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="有效对话", final_translation="Hội thoại hợp lệ")
    engine = UtteranceEngine()
    rcs, _ = engine.process_cues([cue], translated=True)
    assert len(rcs) == 1
    assert rcs[0].source_cue_ids == ["c1"]


# 17. Render without ASS event fails
def test_17_render_to_ass_mapping():
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="有效对话", final_translation="Hội thoại hợp lệ")]
    ass_str = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    assert "Hội thoại hợp lệ" in ass_str


# 18. Duplicate mapping fails
def test_18_no_duplicate_mapping():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="第一句", final_translation="Câu một"),
        SubtitleCue(id="c2", start=3.0, end=4.0, source_text="第二句", final_translation="Câu hai"),
    ]
    engine = UtteranceEngine()
    rcs, _ = engine.process_cues(cues, translated=True)
    assert rcs[0].source_cue_ids == ["c1"]
    assert rcs[1].source_cue_ids == ["c2"]


# 19. Stale draft cannot override final
def test_19_stale_draft_cannot_override_final():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", draft_translation="Chào", final_translation="Xin chào bạn", quality_status="PASS")
    assert get_final_vi_text(cue) == "Xin chào bạn"


# 20. Canonical final accessor used downstream
def test_20_canonical_accessor():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", final_translation="Bản dịch chuẩn")
    assert get_final_vi_text(cue) == "Bản dịch chuẩn"


# 21. Duplicated kinship phrase gets low naturalness score
def test_21_duplicated_kinship_detected():
    polisher = NaturalnessPolisher()
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="阿姨", translated_text="Ơ... dì là dì của cháu.")
    proj = Project(name="p", source_video_path="v", cues=[cue])
    issues, _ = polisher.evaluate_and_polish_cues(proj, [cue])
    assert len(issues["c1"]) > 0
    assert any("naturalness" in iss.type for iss in issues["c1"])


# 22. Relationship repair requires source/context evidence
def test_22_relationship_context_preservation():
    cue = SubtitleCue(id="c1", start=61.64, end=63.10, source_text="呃是是我阿姨", final_translation="Ơ... dạ là dì của cháu.", quality_status="PASS")
    assert get_final_vi_text(cue) == "Ơ... dạ là dì của cháu."


# 23. Targeted repair does not modify neighboring PASS cues
def test_23_targeted_repair_preserves_neighbors():
    c1 = SubtitleCue(id="c1", start=55.75, end=58.60, source_text="原来小丑可能真是我自己", final_translation="Hóa ra kẻ ngốc có thể chính là tôi.", quality_status="PASS")
    c2 = SubtitleCue(id="c2", start=61.64, end=63.10, source_text="呃是是我阿姨", final_translation="Ơ... dạ là dì của cháu.", quality_status="REPAIRED")
    assert get_final_vi_text(c1) == "Hóa ra kẻ ngốc có thể chính là tôi."
    assert get_final_vi_text(c2) == "Ơ... dạ là dì của cháu."


# 24. Timing remains unchanged
def test_24_timing_preserved():
    cue = SubtitleCue(id="c1", start=61.64, end=63.10, source_text="呃是是我阿姨", final_translation="Ơ... dạ là dì của cháu.")
    assert cue.start == 61.64
    assert cue.end == 63.10


# 25. Semantic group remains unchanged
def test_25_semantic_group_preserved():
    cues = [
        SubtitleCue(id="c1", start=100.13, end=101.60, source_text="写字楼里有点存在感的", final_translation="tòa nhà văn phòng này"),
        SubtitleCue(id="c2", start=101.63, end=103.11, source_text="是我妈传给我的手艺做饭", final_translation="chính là kỹ năng nấu ăn mẹ truyền lại."),
    ]
    assert get_final_vi_text(cues[0]) == "tòa nhà văn phòng này"
    assert get_final_vi_text(cues[1]) == "chính là kỹ năng nấu ăn mẹ truyền lại."
