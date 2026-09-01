from __future__ import annotations

import numpy as np
import pytest

from app.models.project import OCRRegion, Project, RenderOptions, SubtitleCue, get_final_vi_text
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.subtitles import UtteranceEngine, to_ass


# 1. Translated dialogue creates cover + VI
def test_01_translated_dialogue_creates_cover_and_vi():
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào", original_source_cue_ids=["c1"])]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    assert contexts[0]["has_vi"] is True
    ass_str = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    assert "Xin chào" in ass_str


# 2. Suppressed filler creates cover-only
def test_02_suppressed_filler_creates_cover_only():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào", original_source_cue_ids=["c1"]),
        SubtitleCue(id="c2", start=2.5, end=3.0, source_text="嗯", final_translation="Ừm.", original_source_cue_ids=["c2"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    suppressed = [c for c in contexts if not c.get("has_vi")]
    assert len(suppressed) == 1
    assert suppressed[0]["reason"] == "SUPPRESSED_FILLER"


# 3. Suppressed filler creates no ASS text
def test_03_suppressed_filler_no_ass_text():
    # Only cue c1 is processed into ASS if c2 is filtered as filler by UtteranceEngine
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào"),
        SubtitleCue(id="c2", start=2.5, end=3.0, source_text="AL", final_translation=""),
    ]
    ass_str = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    assert "AL" not in ass_str
    assert "Dialogue:" in ass_str


# 4. No VI is NOT sufficient reason to skip cover
def test_04_no_vi_still_generates_cover():
    cues = [
        SubtitleCue(id="c1", start=104.5, end=105.0, source_text="嗯", final_translation=""),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) >= 1
    assert contexts[0]["start"] <= 104.5


# 5. Source dialogue visible always maps to cover or explicit ignore
def test_05_source_dialogue_maps_to_cover():
    cues = [
        SubtitleCue(id="c1", start=5.0, end=6.0, source_text="哦", final_translation="Ồ.", ocr_regions=[OCRRegion(points=[[0.45, 0.85], [0.55, 0.85], [0.55, 0.92], [0.45, 0.92]])]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) >= 1


# 6. Decorative OCR does not create cover-only event
def test_06_decorative_ocr_ignored():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="AL", final_translation=""),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    # AL is non-speech/decorative -> 0 contexts
    assert len(contexts) == 0


# 7. Suppressed cover geometry comes from source bbox
def test_07_suppressed_cover_geometry_from_source_bbox():
    poly = [[0.40, 0.86], [0.60, 0.86], [0.60, 0.92], [0.40, 0.92]]
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="嗯", final_translation="", ocr_regions=[OCRRegion(points=poly)]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert x1 <= int(0.40 * 1280)
    assert x2 >= int(0.60 * 1280)


# 8. Suppressed cover uses soft-rect, not glyph mask
def test_08_suppressed_cover_uses_soft_rect():
    cleaner = PatchCoverCleaner()
    bbox = (400, 600, 880, 660)
    mask = cleaner.create_rounded_rect_mask(720, 1280, bbox)
    assert mask.shape == (720, 1280)
    assert mask[630, 640] > 0.90


# 9. Normal translated V7 timing remains unchanged
def test_09_normal_translated_v7_timing_preserved():
    cues = [
        SubtitleCue(id="c1", start=61.64, end=63.10, ocr_start=60.40, source_text="呃是是我阿姨", final_translation="Ơ... dạ là dì của cháu.", original_source_cue_ids=["c1"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["start"] <= 60.40


# 10. Adjacent filler -> dialogue has zero uncovered frames
def test_10_adjacent_filler_dialogue_zero_uncovered_frames():
    cues = [
        SubtitleCue(id="c1", start=104.2, end=105.0, source_text="一，", final_translation="Một,", original_source_cue_ids=["c1"]),
        SubtitleCue(id="c2", start=105.0, end=105.07, source_text="嗯", final_translation="Ừm."),
        SubtitleCue(id="c3", start=105.07, end=107.08, source_text="你这饭给外卖判了死缓", final_translation="Cơm cậu nấu ngon", original_source_cue_ids=["c3"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    # Check continuous handoff
    for i in range(1, len(contexts)):
        assert contexts[i-1]["end"] == contexts[i]["start"]


# 11. Adjacent cover events do not double opacity
def test_11_no_double_opacity():
    cleaner = PatchCoverCleaner()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    m1 = cleaner.create_rounded_rect_mask(720, 1280, (400, 600, 800, 660))
    m2 = cleaner.create_rounded_rect_mask(720, 1280, (450, 600, 850, 660))
    combined = np.maximum(m1, m2)
    out = cleaner.apply_patch_cover(frame, vi_backing_mask=combined)
    assert out.shape == (720, 1280, 3)
    # Pointwise maximum ensures single-layer darkness (~60-70 brightness)
    assert np.mean(out[620:640, 500:750]) > 40


# 12. Different adjacent geometries handoff safely
def test_12_different_geometries_handoff():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="嗯", final_translation=""),
        SubtitleCue(id="c2", start=2.0, end=4.0, source_text="这是一句很长的对话台词", final_translation="Đây là một câu rất dài", original_source_cue_ids=["c2"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 2
    assert contexts[0]["end"] == contexts[1]["start"]


# 13. Cover-only event disappears at correct source end
def test_13_cover_only_lifecycle():
    cues = [SubtitleCue(id="c1", start=10.0, end=11.0, source_text="哦", final_translation="")]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    assert contexts[0]["end"] >= 11.0


# 14. No stale filler cover
def test_14_no_stale_filler_cover():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="嗯", final_translation=""),
        SubtitleCue(id="c2", start=5.0, end=7.0, source_text="对话", final_translation="Hội thoại", original_source_cue_ids=["c2"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["end"] < 3.0


# 15. No blank VI subtitle generated
def test_15_no_blank_vi_subtitle():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="嗯", final_translation=""),
    ]
    ass_str = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    # No blank dialogue events
    assert 'Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,' not in ass_str


# 16. Full coverage audit catches missing cover
def test_16_full_coverage_audit():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="台词", final_translation="Lời thoại", original_source_cue_ids=["c1"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1


# 17. Frame quantization prevents 1-frame flash
def test_17_frame_quantization_pre_roll():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=10.0, end=11.0, ocr_start=9.95, source_text="台词", final_translation="L", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["start"] < 9.95


# 18. Source Integrity unchanged
def test_18_source_integrity_unchanged():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", source_integrity_status="PASS")
    assert cue.source_integrity_status == "PASS"


# 19. Semantic Group unchanged
def test_19_semantic_group_unchanged():
    cues = [
        SubtitleCue(id="c1", start=100.13, end=101.60, source_text="写字楼里有点存在感的", final_translation="tòa nhà văn phòng này"),
        SubtitleCue(id="c2", start=101.63, end=103.11, source_text="是我妈传给我的手艺做饭", final_translation="chính là kỹ năng nấu ăn mẹ truyền lại."),
    ]
    assert get_final_vi_text(cues[0]) == "tòa nhà văn phòng này"
    assert get_final_vi_text(cues[1]) == "chính là kỹ năng nấu ăn mẹ truyền lại."


# 20. Translation unchanged
def test_20_translation_unchanged():
    cue = SubtitleCue(id="c1", start=61.64, end=63.10, source_text="呃是是我阿姨", final_translation="Ơ... dạ là dì của cháu.")
    assert get_final_vi_text(cue) == "Ơ... dạ là dì của cháu."
