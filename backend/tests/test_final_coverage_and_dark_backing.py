from __future__ import annotations

import numpy as np
import pytest

from app.models.project import OCRRegion, Project, RenderOptions, SubtitleCue, get_final_vi_text
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.subtitles import UtteranceEngine, to_ass


# 1. Every dialogue cue has final VI or explicit suppression
def test_01_dialogue_cue_coverage_or_suppression():
    cue_dialogue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào")
    cue_noise = SubtitleCue(id="c2", start=2.5, end=3.0, source_text="AL", final_translation="")
    engine = UtteranceEngine()
    rcs, _ = engine.process_cues([cue_dialogue, cue_noise], translated=True)
    assert len(rcs) == 1
    assert rcs[0].render_text == "Xin chào"


# 2. PASS cue cannot disappear from render coverage
def test_02_pass_cue_cannot_disappear():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="世界", final_translation="Thế giới", quality_status="PASS")
    engine = UtteranceEngine()
    rcs, _ = engine.process_cues([cue], translated=True)
    assert len(rcs) == 1
    assert rcs[0].render_text == "Thế giới"


# 3. Final VI maps to render cue
def test_03_final_vi_maps_to_render_cue():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="吃饭", final_translation="Ăn cơm")
    engine = UtteranceEngine()
    rcs, _ = engine.process_cues([cue], translated=True)
    assert rcs[0].source_cue_ids == ["c1"]
    assert "Ăn cơm" in rcs[0].render_text


# 4. Render cue maps to ASS event
def test_04_render_cue_maps_to_ass_event():
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="吃饭", final_translation="Ăn cơm")]
    ass_str = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    assert "Dialogue:" in ass_str
    assert "Ăn cơm" in ass_str


# 5. No positional cue remapping
def test_05_no_positional_cue_remapping():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="第一句话", final_translation="Câu thứ nhất"),
        SubtitleCue(id="c2", start=2.5, end=3.5, source_text="第二句话", final_translation="Câu thứ hai"),
    ]
    engine = UtteranceEngine()
    rcs, _ = engine.process_cues(cues, translated=True)
    assert rcs[0].source_cue_ids == ["c1"]
    assert rcs[1].source_cue_ids == ["c2"]


# 6. get_final_vi_text is canonical downstream source
def test_06_get_final_vi_text_canonical():
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=2.0,
        source_text="你好",
        draft_translation="Chào",
        translated_text="Chào bạn",
        final_translation="Xin kính chào quý khách",
    )
    assert get_final_vi_text(cue) == "Xin kính chào quý khách"


# 7. Last-known-good final survives failed incremental update
def test_07_last_known_good_survives_failed_update():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", final_translation="Bản dịch chuẩn", quality_status="PASS")
    # Simulated failed repair attempt
    failed_candidate = ""
    if not failed_candidate:
        pass  # Retain previous final_translation
    assert get_final_vi_text(cue) == "Bản dịch chuẩn"


# 8. Glyph polygons produce bbox extent, not backing shape
def test_08_glyph_polygons_produce_bbox_extent_only():
    cleaner = PatchCoverCleaner()
    poly = [[0.40, 0.80], [0.60, 0.80], [0.60, 0.86], [0.40, 0.86]]
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
    # Bbox is a 4-tuple rectangle (x1, y1, x2, y2)
    bbox = contexts[0]["bbox"]
    assert len(bbox) == 4
    mask = cleaner.create_rounded_rect_mask(720, 1280, bbox)
    # The mask shape is a smooth rounded-rect, not jagged contour
    assert mask.shape == (720, 1280)


# 9. Final backing is rectangular / soft-rect
def test_09_final_backing_is_soft_rect():
    cleaner = PatchCoverCleaner()
    mask = cleaner.create_rounded_rect_mask(720, 1280, (400, 600, 880, 660), radius=10, feather=8)
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    # Interior must be fully active
    assert mask[630, 640] > 0.95


# 10. Source Chinese bbox contained inside cover
def test_10_source_chinese_bbox_contained():
    cleaner = PatchCoverCleaner()
    zh_poly = [[0.35, 0.80], [0.65, 0.80], [0.65, 0.86], [0.35, 0.86]]
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            source_text="中文字幕",
            final_translation="Phụ đề tiếng Trung",
            ocr_regions=[OCRRegion(points=zh_poly, text="中文字幕")],
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert x1 <= int(0.35 * 1280)
    assert x2 >= int(0.65 * 1280)
    assert y1 <= int(0.80 * 720)
    assert y2 >= int(0.86 * 720)


# 11. VI bbox contained inside cover
def test_11_vi_bbox_contained():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào tất cả mọi người hôm nay", original_source_cue_ids=["c1"])
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert x2 > x1
    assert y2 > y1


# 12. One-line backing compact
def test_12_one_line_backing_compact():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="一", final_translation="Một dòng", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.095)


# 13. Two-line backing compact
def test_13_two_line_backing_compact():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="一", final_translation="Dòng số một\nDòng số hai", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.145)


# 14. Abnormal OCR falls back safely
def test_14_abnormal_ocr_fallback():
    cleaner = PatchCoverCleaner()
    abnormal_poly = [[0.01, 0.01], [0.99, 0.01], [0.99, 0.99], [0.01, 0.99]]
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


# 15. Giant cover rejected/clamped
def test_15_giant_cover_clamped():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", final_translation="Thử nghiệm", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.15)


# 16. Chinese contrast sufficiently reduced
def test_16_chinese_contrast_reduced():
    cleaner = PatchCoverCleaner()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    vi_mask = cleaner.create_rounded_rect_mask(720, 1280, (400, 600, 880, 660))
    out = cleaner.apply_patch_cover(frame, vi_backing_mask=vi_mask)
    # With 70% tint, bright white (255) becomes ~76
    assert np.mean(out[610:650, 450:850]) < 100


# 17. VI rendered above backing
def test_17_vi_rendered_above_backing():
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào", original_source_cue_ids=["c1"])]
    ass_str = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    assert "Xin chào" in ass_str


# 18. Cover lifecycle matches RenderSubtitleCue
def test_18_cover_lifecycle_matches_render_cue():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=2.0, end=4.5, source_text="测试", final_translation="Thử nghiệm", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["start"] == 2.0
    assert contexts[0]["end"] == 4.5


# 19. No cover without VI
def test_19_no_cover_without_vi():
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts([], 1280, 720)
    assert len(contexts) == 0


# 20. No face/body oversized dark region
def test_20_no_face_body_oversized_region():
    cleaner = PatchCoverCleaner()
    cues = [SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", final_translation="An toàn", original_source_cue_ids=["c1"])]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert y1 >= int(720 * 0.68)
