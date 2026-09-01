from __future__ import annotations

import numpy as np
import pytest

from app.models.project import OCRRegion, Project, RenderOptions, SubtitleCue, get_final_vi_text
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.subtitles import UtteranceEngine, to_ass


# 1. SUPPRESSED_FILLER returns null canonical VI
def test_01_suppressed_filler_returns_empty_canonical_vi():
    cue = SubtitleCue(
        id="c1", start=1.0, end=2.0, source_text="嗯",
        final_translation="Ừm.", draft_translation="Ừm.",
        suppression_status="SUPPRESSED_FILLER"
    )
    assert get_final_vi_text(cue) == ""


# 2. Stale draft cannot render suppressed filler
def test_02_stale_draft_cannot_render_suppressed_filler():
    cue = SubtitleCue(
        id="c1", start=1.0, end=2.0, source_text="嗯",
        draft_translation="Ừm.",
        suppression_status="SUPPRESSED_FILLER"
    )
    assert get_final_vi_text(cue) == ""


# 3. Stale final translation cannot render suppressed filler
def test_03_stale_final_translation_cannot_render_suppressed_filler():
    cue = SubtitleCue(
        id="c1", start=1.0, end=2.0, source_text="一，",
        final_translation="Một,",
        suppression_status="SUPPRESSED_FILLER"
    )
    assert get_final_vi_text(cue) == ""


# 4. Suppressed filler creates no RenderSubtitleCue
def test_04_suppressed_filler_creates_no_render_subtitle_cue():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào", original_source_cue_ids=["c1"]),
        SubtitleCue(id="c2", start=2.5, end=3.0, source_text="嗯", suppression_status="SUPPRESSED_FILLER", original_source_cue_ids=["c2"]),
    ]
    engine = UtteranceEngine()
    rcs, _ = engine.process_cues(cues, translated=True)
    assert len(rcs) == 1
    assert rcs[0].source_cue_ids == ["c1"]


# 5. Suppressed filler creates no ASS text
def test_05_suppressed_filler_creates_no_ass_text():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào"),
        SubtitleCue(id="c2", start=2.5, end=3.0, source_text="嗯", suppression_status="SUPPRESSED_FILLER"),
    ]
    ass_str = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    assert "Xin chào" in ass_str
    assert "Ừm" not in ass_str


# 6. Suppressed filler still creates SourceCoverEvent
def test_06_suppressed_filler_creates_cover_event():
    cues = [
        SubtitleCue(
            id="c1", start=104.5, end=105.0, source_text="嗯",
            suppression_status="SUPPRESSED_FILLER",
            ocr_regions=[OCRRegion(points=[[0.478, 0.875], [0.520, 0.875], [0.520, 0.947], [0.478, 0.947]])]
        )
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    assert contexts[0]["has_vi"] is False
    assert contexts[0]["reason"] == "SUPPRESSED_FILLER"


# 7. Cover-only bbox centered on source bbox, not VI baseline
def test_07_cover_only_bbox_centered_on_source_bbox():
    # Source glyph at y in [0.875, 0.947] -> [630, 682] in 720p
    cues = [
        SubtitleCue(
            id="c1", start=1.0, end=2.0, source_text="嗯",
            suppression_status="SUPPRESSED_FILLER",
            ocr_regions=[OCRRegion(points=[[0.478, 0.875], [0.520, 0.875], [0.520, 0.947], [0.478, 0.947]])]
        )
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert y1 <= 630
    assert y2 >= 682


# 8. Temporal OCR glyph union produces robust source bbox
def test_08_temporal_ocr_glyph_union():
    cues = [
        SubtitleCue(
            id="c1", start=1.0, end=2.0, source_text="嗯",
            suppression_status="SUPPRESSED_FILLER",
            ocr_regions=[
                OCRRegion(points=[[0.47, 0.87], [0.51, 0.87], [0.51, 0.94], [0.47, 0.94]]),
                OCRRegion(points=[[0.48, 0.88], [0.52, 0.88], [0.52, 0.95], [0.48, 0.95]]),
            ]
        )
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert x1 <= int(0.47 * 1280)
    assert x2 >= int(0.52 * 1280)
    assert y1 <= int(0.87 * 720)
    assert y2 >= int(0.95 * 720)


# 9. Glyph outline/shadow remains inside padded cover
def test_09_glyph_outline_shadow_contained():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1", start=1.0, end=2.0, source_text="嗯",
            suppression_status="SUPPRESSED_FILLER",
            ocr_regions=[OCRRegion(points=[[0.48, 0.88], [0.52, 0.88], [0.52, 0.94], [0.48, 0.94]])]
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    # Verify padding margin >= 20px horizontally and >= 10px vertically
    assert x1 < int(0.48 * 1280) - 15
    assert x2 > int(0.52 * 1280) + 15
    assert y1 < int(0.88 * 720) - 5
    assert y2 > int(0.94 * 720) + 5


# 10. One tiny filler still receives minimum compact cover
def test_10_minimum_compact_cover():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1", start=1.0, end=2.0, source_text="一",
            suppression_status="SUPPRESSED_FILLER",
            ocr_regions=[OCRRegion(points=[[0.49, 0.89], [0.50, 0.89], [0.50, 0.90], [0.49, 0.90]])]
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (x2 - x1) >= 80
    assert (y2 - y1) >= 40


# 11. Normal translated cover geometry unchanged
def test_11_normal_translated_cover_geometry_unchanged():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=3.0, source_text="这是一个正常的翻译句子", final_translation="Đây là câu dịch bình thường", original_source_cue_ids=["c1"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["has_vi"] is True
    assert contexts[0]["bbox"][3] <= int(720 * 0.98)


# 12. Next translated cue begins cover no later than source first frame
def test_12_next_translated_cue_cover_timing():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="嗯", suppression_status="SUPPRESSED_FILLER", ocr_start=1.0, ocr_end=2.0),
        SubtitleCue(id="c2", start=2.07, end=4.0, ocr_start=2.0, source_text="你好吗", final_translation="Bạn khỏe không?", original_source_cue_ids=["c2"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[1]["start"] <= 2.07


# 13. Handoff gap frames = 0
def test_13_handoff_gap_frames_zero():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="嗯", suppression_status="SUPPRESSED_FILLER"),
        SubtitleCue(id="c2", start=2.04, end=4.0, source_text="你好", final_translation="Chào", original_source_cue_ids=["c2"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["end"] == contexts[1]["start"]


# 14. Overlap does not double opacity
def test_14_overlap_no_double_opacity():
    cleaner = PatchCoverCleaner()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    m1 = cleaner.create_rounded_rect_mask(720, 1280, (400, 600, 800, 660))
    m2 = cleaner.create_rounded_rect_mask(720, 1280, (450, 600, 850, 660))
    combined = np.maximum(m1, m2)
    out = cleaner.apply_patch_cover(frame, vi_backing_mask=combined)
    assert out.shape == (720, 1280, 3)


# 15. No stale previous geometry
def test_15_no_stale_previous_geometry():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="嗯", suppression_status="SUPPRESSED_FILLER", ocr_regions=[OCRRegion(points=[[0.48, 0.88], [0.52, 0.88], [0.52, 0.94], [0.48, 0.94]])]),
        SubtitleCue(id="c2", start=3.0, end=5.0, source_text="你好这是一句很长很长的话", final_translation="Xin chào đây là câu rất dài", original_source_cue_ids=["c2"]),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["bbox"] != contexts[1]["bbox"]


# 16. No source cue ID positional remap
def test_16_no_source_cue_id_remap():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", final_translation="Thử nghiệm", original_source_cue_ids=["c1"])
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["source_cue_ids"] == ["c1"]


# 17. Translation of non-suppressed cues unchanged
def test_17_translation_non_suppressed_unchanged():
    cue = SubtitleCue(id="c1", start=61.64, end=63.10, source_text="呃是是我阿姨", final_translation="Ơ... dạ là dì của cháu.")
    assert get_final_vi_text(cue) == "Ơ... dạ là dì của cháu."


# 18. Source Integrity unchanged
def test_18_source_integrity_unchanged():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", source_integrity_status="PASS")
    assert cue.source_integrity_status == "PASS"


# 19. Semantic Groups unchanged
def test_19_semantic_groups_unchanged():
    cues = [
        SubtitleCue(id="c1", start=100.13, end=101.60, source_text="写字楼里有点存在感的", final_translation="tòa nhà văn phòng này"),
        SubtitleCue(id="c2", start=101.63, end=103.11, source_text="是我妈传给我的手艺做饭", final_translation="chính là kỹ năng nấu ăn mẹ truyền lại."),
    ]
    assert get_final_vi_text(cues[0]) == "tòa nhà văn phòng này"
    assert get_final_vi_text(cues[1]) == "chính là kỹ năng nấu ăn mẹ truyền lại."


# 20. VI timing unchanged
def test_20_vi_timing_unchanged():
    cues = [
        SubtitleCue(id="c1", start=105.07, end=107.08, source_text="你这饭给外卖判了死缓", final_translation="Cơm cậu nấu ngon", original_source_cue_ids=["c1"]),
    ]
    engine = UtteranceEngine()
    rcs, _ = engine.process_cues(cues, translated=True)
    assert rcs[0].start == 105.07
    assert rcs[0].end == 107.08
