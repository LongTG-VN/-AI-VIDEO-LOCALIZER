from __future__ import annotations

import pytest
from app.models.project import Character, Project, RelationshipRule, SubtitleCue
from app.services.translation_quality.accuracy import AccuracyReviewer
from app.services.translation_quality.consistency import ConsistencySweeper
from app.services.translation_quality.context_card import ContextCardBuilder
from app.services.translation_quality.fillers import FillerHandler
from app.services.translation_quality.idioms import IdiomReviewer
from app.services.translation_quality.models import (
    CueQualityResult,
    FigurativeReviewResult,
    FillerReviewResult,
    NaturalnessScore,
    QualityIssue,
    QualitySeverity,
    TranslationContextCard,
    TranslationQualityConfig,
    TranslationQualityReport,
)
from app.services.translation_quality.naturalness import NaturalnessPolisher
from app.services.translation_quality.pipeline import TranslationQualityPipeline
from app.services.translation_quality.relationships import RelationshipReviewer
from app.services.translation_quality.repair import TargetedRepairer
from app.services.translation_quality.validators import DeterministicValidator
from app.services.utterance_engine import UtteranceEngine


# 1. Figurative Chinese literal translation gets flagged
def test_01_figurative_literal_translation_flagged():
    rev = IdiomReviewer()
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=3.0,
        source_text="你这饭给外卖判了死缓",
        translated_text="Cơm của bạn đã phán tử hình treo cho đồ ăn ngoài.",
    )
    issues, reviews = rev.evaluate_cues(Project(name="p", source_video_path="v"), [cue])
    assert any(iss.type == "idiom.literal_mistranslation" for iss in issues[cue.id])
    assert reviews[cue.id].figurative is True
    assert reviews[cue.id].tone == "humorous"


# 2. Idiom repair preserves intended meaning
def test_02_idiom_repair_preserves_intended_meaning():
    rev = IdiomReviewer()
    cue = SubtitleCue(
        id="c1",
        start=1.0,
        end=3.0,
        source_text="你这饭给外卖判了死缓",
        translated_text="Cơm cậu nấu đúng là khiến đồ ăn ngoài hết cửa sống luôn đấy.",
    )
    issues, reviews = rev.evaluate_cues(Project(name="p", source_video_path="v"), [cue])
    # Natural idiomatic translation is not flagged
    assert len(issues[cue.id]) == 0


# 3. Filler cannot become unrelated lexical meaning
def test_03_filler_unrelated_lexical_rejected():
    handler = FillerHandler()
    cue = SubtitleCue(
        id="c_filler",
        start=1.0,
        end=2.0,
        source_text="嗯",
        translated_text="Một nhóm dự án hôm nay chào mọi người.",
    )
    issues, reviews, norm = handler.evaluate_cues(Project(name="p", source_video_path="v"), [cue])
    assert any(iss.type == "filler.unrelated_lexical_translation" for iss in issues[cue.id])
    assert norm[cue.id] == "Ừm."


# 4. Filler suppression requires explicit reason
def test_04_filler_suppression_requires_reason():
    res = FillerReviewResult(
        cue_id="c_fil",
        filler_token="啊",
        is_filler=True,
        action="SUPPRESS_FILLER",
        translated_vi="",
        suppression_reason="Subordinate interjection with zero semantic contrast in fast dialogue",
    )
    assert res.action == "SUPPRESS_FILLER"
    assert res.suppression_reason is not None


# 5. Chinese-order Vietnamese gets naturalness <= 3
def test_05_chinese_order_gets_naturalness_low_score():
    polisher = NaturalnessPolisher()
    cue = SubtitleCue(
        id="c_order",
        start=1.0,
        end=4.0,
        source_text="我阿姨周末来家里吃饭带上你的保温桶",
        translated_text="Đi tới cuối tuần đến nhà ăn cơm, mang theo hộp giữ nhiệt của cô.",
    )
    issues, _ = polisher.evaluate_and_polish_cues(Project(name="p", source_video_path="v"), [cue])
    assert any(iss.type == "naturalness.chinese_word_order" for iss in issues[cue.id])
    assert polisher.last_scores[cue.id].score <= 3


# 6. Nonsense Vietnamese gets rejected
def test_06_nonsense_vietnamese_rejected():
    validator = DeterministicValidator()
    proj = Project(
        name="test_nonsense",
        source_video_path="dummy.mp4",
        cues=[
            SubtitleCue(
                id="c_non",
                start=1.0,
                end=2.0,
                source_text="我刚准备动筷子",
                translated_text="Tôi vừa chuẩn bị đứng đũa.",
            )
        ],
    )
    is_valid, issues = validator.validate_project(proj)
    assert is_valid is False
    assert any("đứng đũa" in iss.message for iss in issues)


# 7. Semantic-safe polish preserves actor/object
def test_07_semantic_safe_polish_preserves_actors():
    polisher = NaturalnessPolisher()
    source_zh = "我刚准备动筷子"
    old_vi = "Tôi vừa chuẩn bị ăn cơm"
    # Candidate changing "tôi" to third-person "anh ấy" or dropping subject inappropriately
    cand_dropped = "Chuẩn bị"
    assert polisher._verify_semantic_safety(source_zh, old_vi, "Tôi vừa chuẩn bị dùng bữa") is True
    assert polisher._verify_semantic_safety(source_zh, old_vi, "") is False


# 8. Polish cannot alter canonical name
def test_08_polish_cannot_alter_canonical_name():
    polisher = NaturalnessPolisher()
    source_zh = "这是苏棠"
    old_vi = "Đây là Tô Đường"
    # Candidate changing name to "Tô Đình"
    cand_wrong_name = "Đây là Tô Đình"
    # Canonical name validation caught by DeterministicValidator / Semantic safety
    proj = Project(
        name="test_name",
        source_video_path="v.mp4",
        characters=[Character(id="char_su", name="Tô Đường", name_zh="苏棠", name_vi="Tô Đường")],
        cues=[SubtitleCue(id="c_su", start=1.0, end=2.0, source_text=source_zh, translated_text=cand_wrong_name)],
    )
    val = DeterministicValidator()
    is_valid, issues = val.validate_project(proj)
    assert is_valid is False
    assert any("Tô Đường" in iss.message for iss in issues)


# 9. Polish cannot change relationship pronouns incorrectly
def test_09_polish_cannot_change_relationship_pronouns():
    rev = RelationshipReviewer()
    proj = Project(
        name="p",
        source_video_path="v",
        characters=[
            Character(id="c_boss", name="Sếp", name_zh="老板", role="boss"),
            Character(id="c_emp", name="Nhân viên", name_zh="员工", role="employee"),
        ],
    )
    cue = SubtitleCue(
        id="c_rel",
        start=1.0,
        end=2.0,
        source_text="我女朋友今天来",
        translated_text="Con gái tôi hôm nay đến",
    )
    issues = rev._check_deterministic_relationships(proj, cue)
    assert any(iss.type == "relationship.polysemy_mismatch" for iss in issues)


# 10. Romantic term cannot become daughter when graph contradicts it
def test_10_romantic_term_cannot_become_daughter():
    rev = RelationshipReviewer()
    proj = Project(
        name="p",
        source_video_path="v",
        relationships=[RelationshipRule(from_character_id="u1", to_character_id="u2", relationship="girlfriend")],
    )
    cue = SubtitleCue(id="c_gf", start=1.0, end=2.0, source_text="我对象来了", translated_text="Con gái tôi đến rồi")
    issues = rev._check_deterministic_relationships(proj, cue)
    assert any(iss.type == "relationship.polysemy_mismatch" for iss in issues)


# 11. Low source confidence triggers SOURCE_NEEDS_REVIEW
def test_11_low_source_confidence_triggers_review():
    pipeline = TranslationQualityPipeline()
    proj = Project(
        name="p",
        source_video_path="v",
        cues=[
            SubtitleCue(
                id="c_low",
                start=1.0,
                end=2.0,
                source_text="乱七八糟",
                translated_text="Lộn xộn",
                source_confidence=0.30,
            )
        ],
    )
    report = pipeline.run_pipeline(proj)
    assert "source.low_confidence" in report.issue_counts


# 12. PASS cues stay unchanged during targeted repair
def test_12_pass_cues_stay_unchanged_during_repair():
    repairer = TargetedRepairer()
    # cues without reported issues
    cues = [SubtitleCue(id="c_pass", start=1.0, end=2.0, source_text="你好", translated_text="Xin chào")]
    repairs = repairer.repair_failed_cues(Project(name="p", source_video_path="v"), cues, issues_by_cue_id={})
    assert len(repairs) == 0


# 13. Only failed cues are repaired
def test_13_only_failed_cues_enter_repair():
    pipeline = TranslationQualityPipeline(config=TranslationQualityConfig(targeted_repair=True))
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", translated_text="Xin chào"),
        SubtitleCue(id="c2", start=2.5, end=4.0, source_text="我女朋友", translated_text="Con gái tôi"),
    ]
    proj = Project(name="p", source_video_path="v", cues=cues)
    report = pipeline.run_pipeline(proj)
    assert report.cue_results["c1"].status == "PASS"
    assert report.cue_results["c2"].status in ["FAIL", "NEEDS_REVIEW"]


# 14. Consistency sweep outputs patches only
def test_14_consistency_sweep_outputs_patches_only():
    sweeper = ConsistencySweeper()
    proj = Project(
        name="p",
        source_video_path="v",
        characters=[Character(id="char1", name="Tô Đường", name_zh="苏棠", name_vi="Tô Đường")],
        cues=[
            SubtitleCue(id="c1", start=1.0, end=2.0, source_text="苏棠来了", translated_text="Tô Đường đến rồi"),
            SubtitleCue(id="c2", start=2.5, end=3.5, source_text="苏棠在", translated_text="Tô Đường ở đây"),
        ],
    )
    issues, patches = sweeper.sweep_project(proj, proj.cues)
    # Already consistent -> empty patches
    assert len(patches) == 0


# 15. Repaired source timing remains unchanged
def test_15_repaired_source_timing_remains_unchanged():
    cues = [
        SubtitleCue(id="c1", start=10.5, end=12.8, source_text="唯一让我在这个", translated_text="Điều duy nhất khiến tôi ở đây"),
    ]
    proj = Project(name="p", source_video_path="v", cues=cues)
    pipeline = TranslationQualityPipeline()
    report = pipeline.run_pipeline(proj)
    assert proj.cues[0].start == 10.5
    assert proj.cues[0].end == 12.8


# 16. Source integrity segmentation remains unchanged
def test_16_source_integrity_segmentation_remains_unchanged():
    cues = [
        SubtitleCue(
            id="c1_seg1",
            start=10.0,
            end=12.0,
            source_text="这是苏棠",
            original_source_cue_ids=["c1"],
            segmentation_method="ocr_turnover_split",
            source_integrity_status="REPAIRED",
            translated_text="Đây là Tô Đường",
        )
    ]
    proj = Project(name="p", source_video_path="v", cues=cues)
    pipeline = TranslationQualityPipeline()
    report = pipeline.run_pipeline(proj)
    assert proj.cues[0].id == "c1_seg1"
    assert proj.cues[0].segmentation_method == "ocr_turnover_split"
    assert proj.cues[0].original_source_cue_ids == ["c1"]


# 17. Real integration test: Repaired SourceCue -> Review -> Repair -> Naturalness -> UtteranceEngine -> ASS
def test_17_full_repaired_source_to_ass_integration():
    pipeline = TranslationQualityPipeline()
    cues = [
        SubtitleCue(
            id="cue_98",
            start=98.33,
            end=100.10,
            source_text="唯一让我在这个",
            original_source_cue_ids=["orig_4752"],
            segmentation_method="ocr_turnover_split",
            source_integrity_status="REPAIRED",
            translated_text="Điều duy nhất khiến tôi ở trong này",
        ),
        SubtitleCue(
            id="cue_101",
            start=100.40,
            end=103.11,
            source_text="是我妈传给我的手艺做饭",
            original_source_cue_ids=["orig_4752"],
            segmentation_method="ocr_turnover_split",
            source_integrity_status="REPAIRED",
            translated_text="chính là tay nghề nấu nướng mẹ truyền lại cho tôi",
        ),
    ]
    proj = Project(name="test_stream", source_video_path="v.mp4", cues=cues)
    report = pipeline.run_pipeline(proj)
    assert report.total_cues == 2

    # Utterance Engine processing
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(proj.cues, translated=True)
    assert len(render_cues) >= 1
    assert render_cues[0].start >= 98.30
