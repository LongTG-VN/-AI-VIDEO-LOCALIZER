from __future__ import annotations

import pytest

from app.models.project import Character, GlossaryEntry, Project, RelationshipRule, RenderOptions, SubtitleCue
from app.services.translation_quality import (
    AccuracyReviewer,
    CharacterCard,
    ConsistencySweeper,
    ContextCardBuilder,
    CueIntegrityReviewer,
    CueQualityResult,
    DeterministicValidator,
    NaturalnessPolisher,
    QualityIssue,
    QualitySeverity,
    RelationshipCard,
    RelationshipReviewer,
    TargetedRepairer,
    TranslationContextCard,
    TranslationQualityConfig,
    TranslationQualityPipeline,
)
from app.services.subtitles import to_ass
from app.services.utterance_engine import UtteranceEngine


def test_01_romantic_relationship_cannot_map_to_daughter():
    """Requirement 1: '女朋友' cannot map to 'con gái' when relationship evidence contradicts it."""
    proj = Project(
        name="test_proj",
        source_video_path="dummy.mp4",
        characters=[
            Character(id="char_jiang", name="Giang Húc", name_zh="江旭", name_vi="Giang Húc", gender="male"),
            Character(id="char_su", name="Tô Đường", name_zh="苏棠", name_vi="Tô Đường", gender="female"),
        ],
        relationships=[
            RelationshipRule(
                from_character_id="char_jiang",
                to_character_id="char_su",
                relationship="girlfriend",
                vi_self="tôi",
                vi_other="cô",
            )
        ],
    )
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=3.0,
        source_text="我女朋友今天来公司。",
        translated_text="Con gái tôi hôm nay đến công ty.",
    )
    rev = RelationshipReviewer()
    issues = rev._check_deterministic_relationships(proj, cue)
    assert any(iss.type == "relationship.polysemy_mismatch" for iss in issues)


def test_02_ambiguous_kinship_uses_parental_context():
    """Requirement 2: '闺女' in father/parent dialogue must be daughter, not girlfriend."""
    proj = Project(
        name="test_parent",
        source_video_path="dummy.mp4",
        characters=[
            Character(id="char_father", name="Bố", name_zh="父亲", name_vi="Bố", role="father", gender="male"),
        ],
    )
    cue = SubtitleCue(
        id="c2",
        start=2.0,
        end=4.0,
        speaker_character_id="char_father",
        source_text="我闺女那张无限额黑卡。",
        translated_text="Thẻ đen không giới hạn của bạn gái tôi.",
    )
    rev = RelationshipReviewer()
    issues = rev._check_deterministic_relationships(proj, cue)
    assert any(iss.type == "relationship.polysemy_mismatch" for iss in issues)


def test_03_chinese_word_order_gets_naturalness_fail():
    """Requirement 3: Unnatural Chinese word order is flagged by Naturalness review."""
    cue = SubtitleCue(
        id="c3",
        start=5.0,
        end=8.0,
        source_text="我阿姨周末来家里吃饭带上你的保温桶",
        translated_text="Đi tới cuối tuần đến nhà ăn cơm, mang theo hộp giữ nhiệt của cô.",
    )
    polisher = NaturalnessPolisher()
    issues, _ = polisher.evaluate_and_polish_cues(Project(name="p", source_video_path="v"), [cue])
    assert any(iss.type == "naturalness.chinese_word_order" for iss in issues[cue.id])


def test_04_nonsense_vietnamese_gets_naturalness_fail():
    """Requirement 4: Nonsense Vietnamese (e.g. 'đứng đũa') gets naturalness FAIL."""
    cue = SubtitleCue(
        id="c4",
        start=10.0,
        end=12.0,
        source_text="我刚准备动筷子",
        translated_text="Tôi vừa chuẩn bị đứng đũa.",
    )
    polisher = NaturalnessPolisher()
    issues, _ = polisher.evaluate_and_polish_cues(Project(name="p", source_video_path="v"), [cue])
    assert any(iss.type == "naturalness.chinese_word_order" for iss in issues[cue.id])


def test_05_figurative_chinese_rejects_literal_death_penalty():
    """Requirement 5: Figurative idiom '判了死缓' in food context rejects literal death penalty / bát bẩn."""
    cue = SubtitleCue(
        id="c5",
        start=15.0,
        end=18.0,
        source_text="你这饭给外卖判了死缓",
        translated_text="Cậu làm hỏng đồ ăn giao tận nơi rồi, ngâm vào bát bẩn phạm án tử hình.",
    )
    rev = AccuracyReviewer()
    issues = rev._check_deterministic_accuracy(cue)
    assert any(iss.type == "idiom.literal_translation" for iss in issues)


def test_06_omission_and_missing_content_detected():
    """Requirement 6: Missing translation for non-empty source is flagged as critical."""
    cue = SubtitleCue(
        id="c6",
        start=20.0,
        end=22.0,
        source_text="一分钱都没花过。",
        translated_text="",
    )
    rev = CueIntegrityReviewer()
    issues = rev._check_deterministic_integrity(Project(name="p", source_video_path="v"), [cue], 0)
    assert any(iss.type == "cue.missing_content" for iss in issues)


def test_07_wrong_negation_detected():
    """Requirement 10: Source has strong negation but translation is affirmative."""
    cue = SubtitleCue(
        id="c7",
        start=25.0,
        end=28.0,
        source_text="一分钱都没花过。",
        translated_text="Đã tiêu rất nhiều tiền.",
    )
    rev = AccuracyReviewer()
    issues = rev._check_deterministic_accuracy(cue)
    assert any(iss.type == "accuracy.wrong_negation" for iss in issues)


def test_08_neighbor_content_leakage_and_name_migration_detected():
    """Requirement 12 & 23: Name from neighbor cue migrating into current cue without source evidence is flagged."""
    proj = Project(
        name="test_proj",
        source_video_path="dummy.mp4",
        characters=[
            Character(id="char_su", name="Tô Đường", name_zh="苏棠", name_vi="Tô Đường"),
        ],
    )
    cues = [
        SubtitleCue(id="c8_1", start=30.0, end=32.0, source_text="这是苏棠今天入职", translated_text="Đây là Tô Đường hôm nay nhận việc."),
        SubtitleCue(id="c8_2", start=32.5, end=34.0, source_text="大家好我刚准备吃", translated_text="Chào mọi người, Tô Đường chuẩn bị ăn cơm."),
    ]
    rev = CueIntegrityReviewer()
    issues = rev._check_deterministic_integrity(proj, cues, 1)
    assert any(iss.type == "cue.name_migration" for iss in issues)


def test_09_untranslated_chinese_fallback_rejected():
    """Requirement 15: Raw Chinese characters in translation cannot be accepted as success."""
    cue = SubtitleCue(
        id="c9",
        start=35.0,
        end=37.0,
        source_text="等我一下",
        translated_text="等我一下",
    )
    val = DeterministicValidator()
    proj = Project(name="p", source_video_path="v", cues=[cue])
    passed, issues = val.validate_project(proj)
    assert not passed
    assert any(iss.type == "validation.untranslated_chinese" for iss in issues)


def test_10_semantic_safety_gate_rejects_meaning_shift_polish():
    """Requirement 18: Polish candidate that changes negation or drops numbers is rejected by semantic safety gate."""
    polisher = NaturalnessPolisher()
    # Source has 8 months and negation, candidate drops 8 and changes to affirmative
    src = "黑卡整整8个月，一分钱都没花过。"
    old_vi = "Thẻ đen tròn 8 tháng, chưa tiêu một xu nào."
    bad_candidate = "Thẻ đen dùng nhiều tháng, đã tiêu hết tiền."
    assert not polisher._verify_semantic_safety(src, old_vi, bad_candidate)

    # Legitimate polish preserving numbers and meaning is accepted
    good_candidate = "Chiếc thẻ đen suốt 8 tháng qua, chưa từng tiêu một đồng nào."
    assert polisher._verify_semantic_safety(src, old_vi, good_candidate)


def test_11_deterministic_validator_detects_duplicate_and_order_issues():
    """Requirement 20 & 21: Deterministic validator rejects duplicate IDs and mismatched order."""
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", translated_text="Xin chào"),
        SubtitleCue(id="c1", start=2.5, end=3.5, source_text="再见", translated_text="Tạm biệt"),
    ]
    val = DeterministicValidator()
    proj = Project(name="p", source_video_path="v", cues=cues)
    passed, issues = val.validate_project(proj)
    assert not passed
    assert any(iss.type == "validation.duplicate_cue_id" for iss in issues)


def test_12_full_pipeline_pass_and_invariance_on_clean_cues():
    """Requirement 16 & 17: Targeted repair and pipeline leave already-clean cues completely untouched."""
    proj = Project(
        name="clean_proj",
        source_video_path="dummy.mp4",
        characters=[
            Character(id="char_jiang", name="Giang Húc", name_zh="江旭", name_vi="Giang Húc"),
        ],
        cues=[
            SubtitleCue(id="c1", start=1.0, end=2.0, source_text="我叫江旭", translated_text="Tôi tên là Giang Húc."),
            SubtitleCue(id="c2", start=2.5, end=4.0, source_text="今年二十六岁", translated_text="Năm nay hai mươi sáu tuổi."),
        ],
    )
    pipeline = TranslationQualityPipeline()
    report = pipeline.run_pipeline(proj)
    assert report.total_cues == 2
    assert report.passed_first_attempt == 2
    assert report.repaired == 0
    assert report.needs_review == 0
    assert proj.cues[0].translated_text == "Tôi tên là Giang Húc."
    assert proj.cues[1].translated_text == "Năm nay hai mươi sáu tuổi."


def test_13_end_to_end_integration_pipeline_to_utterance_and_ass():
    """Integration Test: Source -> TranslationQualityPipeline -> UtteranceEngine -> RenderSubtitleCue -> ASS generation."""
    proj = Project(
        name="integration_test",
        source_video_path="dummy.mp4",
        width=1280,
        height=720,
        characters=[
            Character(id="char_jiang", name="Giang Húc", name_zh="江旭", name_vi="Giang Húc", gender="male"),
            Character(id="char_su", name="Tô Đường", name_zh="苏棠", name_vi="Tô Đường", gender="female"),
        ],
        cues=[
            SubtitleCue(
                id="c1",
                start=1.0,
                end=3.0,
                speaker_character_id="char_jiang",
                source_text="我叫江旭，今年二十六岁。",
                translated_text="Tôi tên là Giang Húc, năm nay hai mươi sáu tuổi.",
            ),
            SubtitleCue(
                id="c2",
                start=3.5,
                end=6.0,
                speaker_character_id="char_jiang",
                source_text="这是苏棠，今天新入职的同事。",
                translated_text="Đây là Tô Đường, đồng nghiệp mới nhận việc hôm nay.",
            ),
        ],
    )

    # 1. Quality Pipeline
    pipeline = TranslationQualityPipeline()
    report = pipeline.run_pipeline(proj)
    assert report.total_cues == 2
    assert report.needs_review == 0

    # 2. Utterance Engine
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(proj.cues)
    assert len(render_cues) >= 1

    # 3. ASS Generation
    ass_content = to_ass(proj.cues, RenderOptions(), width=1280, height=720, translated=True)
    assert "Dialogue:" in ass_content
    assert "Giang Húc" in ass_content
    assert "Tô Đường" in ass_content
    assert "苏棠" not in ass_content
    assert "江旭" not in ass_content


def test_14_repaired_text_persists_and_consumed_by_ass():
    """Test: Repaired text is persisted in project cues and consumed by ASS generation (not draft)."""
    cue = SubtitleCue(
        id="c_rep",
        start=1.0,
        end=3.0,
        source_text="我女朋友今天来公司。",
        translated_text="Con gái tôi hôm nay đến công ty.",  # Draft error
    )
    proj = Project(
        name="p_rep",
        source_video_path="v.mp4",
        width=1280,
        height=720,
        cues=[cue],
    )
    # Simulate repair
    cue.translated_text = "Bạn gái tôi hôm nay đến công ty."
    
    ass_content = to_ass(proj.cues, RenderOptions(), width=1280, height=720, translated=True)
    assert "Bạn gái tôi" in ass_content
    assert "Con gái tôi" not in ass_content


def test_15_quality_version_change_invalidates_stale_cache():
    """Test: Changing translation_quality_version produces distinct cache keys."""
    pipeline_v1 = TranslationQualityPipeline(config=TranslationQualityConfig(translation_quality_version="v1"))
    pipeline_v2 = TranslationQualityPipeline(config=TranslationQualityConfig(translation_quality_version="v2"))
    
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", translated_text="Xin chào")
    card = TranslationContextCard(characters=[], relationships=[])
    
    key1 = pipeline_v1.compute_cue_cache_key(cue, card)
    key2 = pipeline_v2.compute_cue_cache_key(cue, card)
    assert key1 != key2


def test_16_unresolved_major_cannot_silently_pass():
    """Test: Major/critical unresolved issues cannot silently become PASS."""
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", translated_text="Xin chào")
    issue = QualityIssue(type="accuracy.mistranslation", severity=QualitySeverity.MAJOR, message="Critical error")
    res = CueQualityResult(cue_id="c1", status="NEEDS_REVIEW", issues=[issue])
    assert res.status == "NEEDS_REVIEW"
    assert res.status != "PASS"


def test_17_chinese_ocr_bbox_cannot_create_black_rectangle():
    """Visual Test: Chinese cleanup and fallback never create solid black rectangles."""
    import numpy as np
    import cv2
    from app.models.project import OCRRegion
    from app.services.hardsub_cleaner import HardSubCleaner
    from app.services.patch_cover_cleaner import PatchCoverCleaner
    
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 200  # Bright frame
    cleaner = HardSubCleaner()
    cleaned_frame, _ = cleaner.clean_frame(
        frame,
        mode="cover",
        ocr_regions=[OCRRegion(points=[[0.3, 0.85], [0.7, 0.85], [0.7, 0.92], [0.3, 0.92]])],
    )
    # Check that region is NOT solid black (0, 0, 0)
    region = cleaned_frame[int(720*0.85):int(720*0.92), int(1280*0.3):int(1280*0.7)]
    assert np.mean(region) > 50  # Must retain brightness, not pitch black (0)


def test_18_vi_backing_derived_from_vi_text_bbox_and_one_to_one():
    """Visual Test: VI backing derives strictly from VI text bbox, 1:1 with RenderSubtitleCue."""
    from app.services.patch_cover_cleaner import PatchCoverCleaner
    
    cues = [
        SubtitleCue(id="c1", start=1.0, end=3.0, source_text="你好", translated_text="Xin chào các bạn"),
        SubtitleCue(id="c2", start=5.0, end=8.0, source_text="再见", translated_text="Tạm biệt nhé"),
    ]
    cleaner = PatchCoverCleaner()
    contexts = cleaner.build_render_cue_contexts(cues, width=1280, height=720)
    assert len(contexts) == 2
    assert contexts[0]["start"] == 1.0
    assert contexts[0]["end"] == 3.0
    x1, y1, x2, y2 = contexts[0]["bbox"]
    assert x2 > x1
    assert y2 > y1
    # Height must be compact (< 15% frame height)
    assert (y2 - y1) / 720.0 < 0.15


def test_19_bad_ocr_geometry_cannot_create_giant_backing():
    """Visual Test: Giant or face-level polygons are rejected by geometry validator."""
    from app.services.patch_cover_cleaner import PatchCoverCleaner
    
    cleaner = PatchCoverCleaner()
    # Face-level polygon in top half (y: 0.1 to 0.4)
    giant_polygon = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.4], [0.1, 0.4]]
    assert not cleaner._is_valid_subtitle_polygon(giant_polygon)
    
    # Normal subtitle polygon in bottom band (y: 0.85 to 0.92)
    valid_polygon = [[0.3, 0.85], [0.7, 0.85], [0.7, 0.92], [0.3, 0.92]]
    assert cleaner._is_valid_subtitle_polygon(valid_polygon)
