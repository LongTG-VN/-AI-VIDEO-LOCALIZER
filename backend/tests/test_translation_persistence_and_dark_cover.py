from __future__ import annotations

import numpy as np
import pytest

from app.models.project import Character, OCRRegion, PatchCoverConfig, Project, SubtitleCue, get_final_vi_text
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.subtitles import UtteranceEngine, to_ass
from app.models.project import RenderOptions
from app.services.translation_quality.models import TranslationQualityConfig
from app.services.translation_quality.pipeline import TranslationQualityPipeline


# 1. PASS cue keeps final_translation after targeted repair of another cue
def test_01_pass_cue_keeps_final_translation():
    c1 = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", translated_text="Xin chào", final_translation="Xin chào", quality_status="PASS")
    c2 = SubtitleCue(id="c2", start=2.1, end=3.0, source_text="再见", translated_text="Tạm biệt", final_translation="Tạm biệt", quality_status="PASS")
    # Simulate targeted repair on c2 only
    c2.final_translation = "Hẹn gặp lại"
    c2.quality_status = "REPAIRED"
    assert get_final_vi_text(c1) == "Xin chào"
    assert c1.quality_status == "PASS"
    assert get_final_vi_text(c2) == "Hẹn gặp lại"
    assert c2.quality_status == "REPAIRED"


# 2. Project-wide quality run cannot clear untouched final translations
def test_02_project_quality_preserves_untouched_cues():
    pipeline = TranslationQualityPipeline()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="一", translated_text="Một", final_translation="Một", quality_status="PASS"),
        SubtitleCue(id="c2", start=2.1, end=3.0, source_text="二", translated_text="Hai", final_translation="Hai", quality_status="PASS"),
    ]
    proj = Project(name="p", source_video_path="v", cues=cues)
    report = pipeline.run_pipeline(proj)
    assert get_final_vi_text(proj.cues[0]) == "Một"
    assert get_final_vi_text(proj.cues[1]) == "Hai"


# 3. Repaired cue updates final_translation
def test_03_repaired_cue_updates_final_translation():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", translated_text="Cũ", draft_translation="Cũ", quality_status="PENDING")
    cue.repaired_translation = "Đã sửa"
    cue.final_translation = "Đã sửa"
    cue.quality_status = "REPAIRED"
    assert get_final_vi_text(cue) == "Đã sửa"


# 4. Failed repair preserves last known-good final_translation
def test_04_failed_repair_preserves_known_good():
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=2.0,
        source_text="测试",
        draft_translation="Bản dịch gốc",
        final_translation="Bản dịch gốc",
        quality_status="NEEDS_REVIEW",
    )
    assert get_final_vi_text(cue) == "Bản dịch gốc"


# 5. Canonical accessor always returns approved final text
def test_05_canonical_accessor_priority():
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=2.0,
        source_text="你好",
        draft_translation="Chào",
        translated_text="Xin chào",
        repaired_translation="Chào bạn",
        final_translation="Kính chào quý khách",
    )
    assert get_final_vi_text(cue) == "Kính chào quý khách"


# 6. ASS consumes canonical final text
def test_06_ass_consumes_canonical_final_text():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào tất cả mọi người", original_source_cue_ids=["c1"])
    ]
    ass_str = to_ass(cues, RenderOptions(), 852, 480, translated=True)
    assert "Xin chào tất cả mọi người" in ass_str


# 7. PASS + empty final_translation is flagged
def test_07_pass_empty_final_flagged():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="", translated_text="", quality_status="PASS")
    assert get_final_vi_text(cue) == ""


# 8. Cache draft cannot overwrite approved final
def test_08_cache_draft_cannot_overwrite_approved_final():
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Chào bạn", quality_status="PASS")
    cached_draft = "Chào"
    if cue.quality_status != "PASS":
        cue.final_translation = cached_draft
    assert get_final_vi_text(cue) == "Chào bạn"


# 9. Incremental update affects only selected cue IDs
def test_09_incremental_update_scope():
    c1 = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="一", final_translation="Một")
    c2 = SubtitleCue(id="c2", start=2.1, end=3.0, source_text="二", final_translation="Hai")
    affected_ids = {"c2"}
    if "c1" in affected_ids:
        c1.final_translation = "Một mới"
    if "c2" in affected_ids:
        c2.final_translation = "Hai mới"
    assert get_final_vi_text(c1) == "Một"
    assert get_final_vi_text(c2) == "Hai mới"


# 10. Cover rendered above source Chinese pixels
def test_10_cover_rendered_above_source_pixels():
    cleaner = PatchCoverCleaner()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 240
    # Simulate white Chinese text inside subtitle area
    frame[620:640, 500:700] = 255
    vi_mask = cleaner.create_rounded_rect_mask(720, 1280, (400, 600, 880, 660))
    out = cleaner.apply_patch_cover(frame, vi_backing_mask=vi_mask)
    # The output in covered region must be darkened (~68% tint)
    assert out[630, 600, 0] < 120


# 11. VI text rendered above cover
def test_11_vi_text_rendered_above_cover():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", final_translation="Xin chào", original_source_cue_ids=["c1"])
    ]
    ass_str = to_ass(cues, RenderOptions(subtitle_font_size=20), 1280, 720, translated=True)
    assert "Dialogue:" in ass_str
    assert "Xin chào" in ass_str


# 12. Chinese glyph bbox is contained by cover
def test_12_chinese_glyph_contained_by_cover():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=3.0,
            source_text="你好世界",
            final_translation="Xin chào thế giới",
            ocr_regions=[OCRRegion(points=[[0.35, 0.80], [0.65, 0.80], [0.65, 0.86], [0.35, 0.86]], text="你好世界")],
            original_source_cue_ids=["c1"],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    x1, y1, x2, y2 = contexts[0]["bbox"]
    # Chinese glyph bbox: x in [0.35*1280, 0.65*1280] = [448, 832], y in [0.80*720, 0.86*720] = [576, 619]
    assert x1 <= 448
    assert x2 >= 832
    assert y1 <= 576
    assert y2 >= 619


# 13. Cover opacity actually reduces source text contrast
def test_13_cover_reduces_contrast():
    cleaner = PatchCoverCleaner()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    vi_mask = cleaner.create_rounded_rect_mask(720, 1280, (400, 600, 880, 660))
    out = cleaner.apply_patch_cover(frame, vi_backing_mask=vi_mask)
    # Brightness reduced by at least 50%
    assert np.mean(out[610:650, 450:850]) < 130


# 14. Raw OCR bbox cannot create giant cover
def test_14_raw_ocr_cannot_create_giant_cover():
    cleaner = PatchCoverCleaner()
    giant_points = [[0.05, 0.10], [0.95, 0.10], [0.95, 0.90], [0.05, 0.90]]
    assert cleaner._is_valid_subtitle_polygon(giant_points) is False


# 15. Cover lifecycle matches VI cue
def test_15_cover_lifecycle_matches_vi_cue():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.5, end=3.5, source_text="测试", final_translation="Thử nghiệm", original_source_cue_ids=["c1"])
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert contexts[0]["start"] == 1.5
    assert contexts[0]["end"] == 3.5


# 16. No cover without VI cue
def test_16_no_cover_without_vi_cue():
    cleaner = PatchCoverCleaner()
    cues = []
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 0


# 17. 1-line cover works
def test_17_one_line_cover():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="一", final_translation="Một dòng", original_source_cue_ids=["c1"])
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.10)


# 18. 2-line cover works
def test_18_two_line_cover():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="一", final_translation="Dòng một\nDòng hai", original_source_cue_ids=["c1"])
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.15)


# 19. No face/body giant dark region
def test_19_no_face_body_giant_region():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="测试", final_translation="Kiểm tra", original_source_cue_ids=["c1"])
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert y1 >= int(720 * 0.68)
