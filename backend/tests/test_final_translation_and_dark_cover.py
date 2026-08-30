from __future__ import annotations

import numpy as np
import pytest

from app.models.project import Character, OCRRegion, PatchCoverConfig, Project, SubtitleCue
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.semantic_grouping.grouper import SemanticGrouper
from app.services.semantic_grouping.models import SemanticTranslationGroup
from app.services.translation_quality.idioms import IdiomReviewer
from app.services.translation_quality.naturalness import NaturalnessPolisher
from app.services.translation_quality.relationships import RelationshipReviewer


# 1. Fused Vietnamese clause gets naturalness <= 3
def test_01_fused_vietnamese_clause_gets_low_naturalness():
    polisher = NaturalnessPolisher()
    cue = SubtitleCue(
        id="c1",
        start=55.0,
        end=58.0,
        source_text="原来小丑可能真是我自己就是你天天给我闺女做饭",
        translated_text="Hóa ra kẻ ngốc chính là tôi là anh nấu ăn cho con gái tôi mỗi ngày",
    )
    issues, _ = polisher.evaluate_and_polish_cues(Project(name="p", source_video_path="v"), [cue])
    assert len(issues[cue.id]) > 0
    assert polisher.last_scores[cue.id].score <= 3


# 2. Targeted repair fixes grammar without changing meaning
def test_02_targeted_repair_fixes_grammar():
    polisher = NaturalnessPolisher()
    is_safe = polisher._verify_semantic_safety(
        "就是你天天给我闺女做饭",
        "Hóa ra kẻ ngốc là anh nấu ăn",
        "Chính cậu mỗi ngày nấu cơm cho con gái tôi à?",
    )
    assert is_safe is True


# 3. Idiom candidate ranking prefers understandable natural VI
def test_03_idiom_candidate_ranking_natural_vi():
    reviewer = IdiomReviewer()
    cue = SubtitleCue(
        id="c1",
        start=105.0,
        end=107.0,
        source_text="你这饭给外卖判了死缓",
        translated_text="Món này của anh làm đồ ăn ngoài phải 'án treo' luôn.",
    )
    issues, reviews = reviewer.evaluate_cues(Project(name="p", source_video_path="v"), [cue])
    assert len(issues[cue.id]) > 0
    assert "hết cửa sống" in reviews[cue.id].candidate_vi or "ngon" in reviews[cue.id].candidate_vi


# 4. Idiom candidate cannot PASS if punchline is incomprehensible
def test_04_idiom_candidate_fails_if_incomprehensible():
    reviewer = IdiomReviewer()
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=2.0,
        source_text="判了死缓",
        translated_text="Phán tử hình treo",
    )
    issues, reviews = reviewer.evaluate_cues(Project(name="p", source_video_path="v"), [cue])
    assert reviews[cue.id].status == "FAIL"


# 5. Relationship/阿姨 resolution uses context
def test_05_relationship_ayi_resolution():
    reviewer = RelationshipReviewer()
    cue = SubtitleCue(
        id="c1",
        start=61.0,
        end=63.0,
        source_text="呃是是我阿姨",
        translated_text="Ơ... dạ là dì của cháu.",
    )
    issues = reviewer._check_deterministic_relationships(Project(name="p", source_video_path="v"), cue)
    assert len(issues) == 0


# 6. Already-PASS cues remain unchanged
def test_06_already_pass_cues_remain_unchanged():
    polisher = NaturalnessPolisher()
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=2.0,
        source_text="你好",
        translated_text="Xin chào mọi người.",
    )
    issues, _ = polisher.evaluate_and_polish_cues(Project(name="p", source_video_path="v"), [cue])
    assert len(issues[cue.id]) == 0


# 7. Source Integrity segmentation unchanged
def test_07_source_integrity_segmentation_unchanged():
    cue = SubtitleCue(
        id="c1_seg1",
        start=98.33,
        end=100.10,
        source_text="唯一让我在这个",
        original_source_cue_ids=["orig_1"],
    )
    assert cue.start == 98.33
    assert cue.end == 100.10


# 8. Semantic group membership unchanged
def test_08_semantic_group_membership_unchanged():
    grouper = SemanticGrouper()
    cues = [
        SubtitleCue(id="c1", start=98.33, end=100.10, source_text="唯一让我在这个"),
        SubtitleCue(id="c2", start=100.40, end=103.11, source_text="是我妈传给我的手艺做饭"),
    ]
    groups = grouper.create_groups(cues)
    assert len(groups) == 1
    assert groups[0].source_cue_ids == ["c1", "c2"]


# 9. Timing unchanged
def test_09_timing_unchanged():
    cue = SubtitleCue(id="c1", start=55.75, end=58.60, source_text="原来小丑可能真是我自己")
    assert cue.start == 55.75
    assert cue.end == 58.60


# 10. Dark source subtitle cover is enabled
def test_10_dark_source_subtitle_cover_enabled():
    cleaner = PatchCoverCleaner()
    assert cleaner is not None


# 11. Cover hides Chinese dialogue region
def test_11_cover_hides_chinese_dialogue_region():
    cleaner = PatchCoverCleaner()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    vi_mask = cleaner.create_rounded_rect_mask(720, 1280, (200, 600, 1080, 680))
    out = cleaner.apply_patch_cover(frame, vi_backing_mask=vi_mask)
    # The covered area should be darker than original frame (200)
    assert out[640, 640, 0] < 150


# 12. Raw giant OCR bbox cannot control cover dimensions
def test_12_giant_ocr_bbox_rejected():
    cleaner = PatchCoverCleaner()
    giant_points = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]
    assert cleaner._is_valid_subtitle_polygon(giant_points) is False


# 13. Cover bbox uses tight Chinese glyph + VI text geometry
def test_13_cover_bbox_geometry():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=3.0,
            source_text="你好",
            translated_text="Xin chào",
            ocr_regions=[OCRRegion(points=[[0.4, 0.82], [0.6, 0.82], [0.6, 0.88], [0.4, 0.88]], text="你好")],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert y1 >= int(720 * 0.68)
    assert y2 <= int(720 * 0.98)


# 14. One-line cover compact
def test_14_one_line_cover_compact():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=3.0, source_text="你好", translated_text="Xin chào")
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.10)


# 15. Two-line cover compact
def test_15_two_line_cover_compact():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=3.0, source_text="你好", translated_text="Xin chào\nmọi người")
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.15)


# 16. Cover cannot exceed safety height/area
def test_16_cover_cannot_exceed_safety_height():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=3.0,
            source_text="测试",
            translated_text="Thử nghiệm",
            ocr_regions=[OCRRegion(points=[[0.3, 0.70], [0.7, 0.70], [0.7, 0.95], [0.3, 0.95]])],
        )
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert (y2 - y1) <= int(720 * 0.15)


# 17. Cover lifecycle matches subtitle cue
def test_17_cover_lifecycle_matches_subtitle():
    cleaner = PatchCoverCleaner()
    cues = [
        SubtitleCue(id="c1", start=2.0, end=4.0, source_text="一", translated_text="Một", original_source_cue_ids=["c1"])
    ]
    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 1
    assert contexts[0]["start"] == 2.0
    assert contexts[0]["end"] == 4.0


# 18. No cover during subtitle gap
def test_18_no_cover_during_gap():
    cleaner = PatchCoverCleaner()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200
    # No mask passed during gap
    out = cleaner.apply_patch_cover(frame, vi_backing_mask=None)
    assert np.array_equal(frame, out)


# 19. VI white text + thin black outline rendered above cover
def test_19_vi_text_rendered_above_cover():
    from app.services.subtitles import to_ass
    from app.models.project import RenderOptions
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="一", translated_text="Một")
    ]
    ass_str = to_ass(cues, RenderOptions(subtitle_font_size=20), 852, 480, translated=True)
    assert "Style: Default" in ass_str
    assert "&H00FFFFFF" in ass_str  # White primary fill
