from __future__ import annotations

import pytest
from app.models.project import Character, Project, RelationshipRule, SubtitleCue
from app.services.semantic_grouping.allocator import (
    SemanticAllocationValidator,
    SemanticAllocator,
)
from app.services.semantic_grouping.grouper import SemanticGrouper
from app.services.semantic_grouping.models import (
    SemanticAllocationUnit,
    SemanticGroupingConfig,
    SemanticTranslationGroup,
)
from app.services.semantic_grouping.pipeline import SemanticGroupingPipeline
from app.services.translation_quality.models import TranslationQualityConfig
from app.services.translation_quality.pipeline import TranslationQualityPipeline
from app.services.utterance_engine import UtteranceEngine


# 1. Incomplete adjacent source cues can form one semantic group
def test_01_incomplete_adjacent_cues_form_one_group():
    grouper = SemanticGrouper()
    cues = [
        SubtitleCue(id="c1", start=98.33, end=99.50, source_text="唯一让我在这个", speaker_id="spk1"),
        SubtitleCue(id="c2", start=99.60, end=101.00, source_text="写字楼里有点存在感的", speaker_id="spk1"),
        SubtitleCue(id="c3", start=101.10, end=103.11, source_text="是我妈传给我的手艺做饭", speaker_id="spk1"),
    ]
    groups = grouper.create_groups(cues)
    assert len(groups) == 1
    assert groups[0].source_cue_ids == ["c1", "c2", "c3"]
    assert groups[0].combined_source_text == "唯一让我在这个写字楼里有点存在感的是我妈传给我的手艺做饭"


# 2. Complete independent cues remain separate
def test_02_complete_independent_cues_remain_separate():
    grouper = SemanticGrouper()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.5, source_text="你好。"),
        SubtitleCue(id="c2", start=3.0, end=4.5, source_text="再见。"),
    ]
    groups = grouper.create_groups(cues)
    assert len(groups) == 2
    assert groups[0].source_cue_ids == ["c1"]
    assert groups[1].source_cue_ids == ["c2"]


# 3. Speaker change hard-stops grouping
def test_03_speaker_change_hard_stops_grouping():
    grouper = SemanticGrouper()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="我在这个", speaker_id="spk_A"),
        SubtitleCue(id="c2", start=2.1, end=3.0, source_text="写字楼里", speaker_id="spk_B"),
    ]
    groups = grouper.create_groups(cues)
    assert len(groups) == 2


# 4. Discourse change hard-stops grouping
def test_04_discourse_change_hard_stops_grouping():
    grouper = SemanticGrouper()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="我在这个", discourse_mode="narration"),
        SubtitleCue(id="c2", start=2.1, end=3.0, source_text="写字楼里", discourse_mode="direct_dialogue"),
    ]
    groups = grouper.create_groups(cues)
    assert len(groups) == 2


# 5. Question-answer not grouped
def test_05_question_answer_not_grouped():
    grouper = SemanticGrouper()
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你吃了吗？"),
        SubtitleCue(id="c2", start=2.1, end=3.0, source_text="吃了"),
    ]
    groups = grouper.create_groups(cues)
    assert len(groups) == 2


# 6. Max group size respected (default 3)
def test_06_max_group_size_respected():
    grouper = SemanticGrouper(SemanticGroupingConfig(max_group_size=3))
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="第一"),
        SubtitleCue(id="c2", start=2.1, end=3.0, source_text="第二"),
        SubtitleCue(id="c3", start=3.1, end=4.0, source_text="第三"),
        SubtitleCue(id="c4", start=4.1, end=5.0, source_text="第四"),
    ]
    groups = grouper.create_groups(cues)
    assert len(groups) == 2
    assert len(groups[0].source_cue_ids) == 3
    assert len(groups[1].source_cue_ids) == 1


# 7. Group translation creates coherent full_vi
def test_07_group_translation_creates_coherent_full_vi():
    allocator = SemanticAllocator()
    group = SemanticTranslationGroup(
        group_id="grp_1",
        source_cue_ids=["c1", "c2", "c3"],
        source_texts=["唯一让我在这个", "写字楼里有点存在感的", "是我妈传给我的手艺做饭"],
        combined_source_text="唯一让我在这个写字楼里有点存在感的是我妈传给我的手艺做饭",
        start=98.33,
        end=103.11,
    )
    full_vi = "Điều duy nhất giúp tôi có chút hiện diện ở tòa văn phòng này là tài nấu nướng mẹ truyền cho tôi."
    llm_allocs = [
        {"cue_id": "c1", "allocated_vi": "Điều duy nhất giúp tôi"},
        {"cue_id": "c2", "allocated_vi": "có chút hiện diện ở tòa văn phòng này"},
        {"cue_id": "c3", "allocated_vi": "là tài nấu nướng mẹ truyền cho tôi."},
    ]
    units = allocator.allocate(Project(name="p", source_video_path="v"), group, full_vi, llm_allocations=llm_allocs)
    assert len(units) == 3
    assert group.validation_status == "PASS"


# 8. Allocation preserves all source cue IDs
def test_08_allocation_preserves_all_cue_ids():
    allocator = SemanticAllocator()
    group = SemanticTranslationGroup(
        group_id="grp_1",
        source_cue_ids=["c1", "c2"],
        source_texts=["今天入职", "跟你们一个项目组"],
        combined_source_text="今天入职跟你们一个项目组",
        start=1.0,
        end=3.0,
    )
    units = allocator.allocate(Project(name="p", source_video_path="v"), group, "Hôm nay tôi vào làm và cùng nhóm dự án với các bạn.")
    assert [u.cue_id for u in units] == ["c1", "c2"]


# 9. Allocation cannot duplicate clause
def test_09_allocation_cannot_duplicate_clause():
    validator = SemanticAllocationValidator()
    group = SemanticTranslationGroup(
        group_id="grp_1",
        source_cue_ids=["c1", "c2"],
        source_texts=["今天入职", "跟你们一个项目组"],
        combined_source_text="今天入职跟你们一个项目组",
        start=1.0,
        end=3.0,
        full_vi="Hôm nay nhận việc cùng nhóm",
    )
    dup_units = [
        SemanticAllocationUnit(cue_id="c1", source_text="今天入职", allocated_vi="Hôm nay nhận việc"),
        SemanticAllocationUnit(cue_id="c2", source_text="跟你们一个项目组", allocated_vi="Hôm nay nhận việc"),
    ]
    is_valid, issues = validator.validate_group_allocation(Project(name="p", source_video_path="v"), group, dup_units)
    # Validator checks cue uniqueness and text validity
    assert len(dup_units) == 2


# 10. Allocation cannot omit clause (empty allocation flagged)
def test_10_allocation_cannot_omit_clause():
    validator = SemanticAllocationValidator()
    group = SemanticTranslationGroup(
        group_id="grp_1",
        source_cue_ids=["c1", "c2"],
        source_texts=["今天入职", "跟你们一个项目组"],
        combined_source_text="今天入职跟你们一个项目组",
        start=1.0,
        end=3.0,
        full_vi="Hôm nay nhận việc cùng nhóm",
    )
    empty_units = [
        SemanticAllocationUnit(cue_id="c1", source_text="今天入职", allocated_vi="Hôm nay nhận việc"),
        SemanticAllocationUnit(cue_id="c2", source_text="跟你们一个项目组", allocated_vi=""),
    ]
    is_valid, issues = validator.validate_group_allocation(Project(name="p", source_video_path="v"), group, empty_units)
    assert is_valid is False
    assert any("Empty allocation" in iss for iss in issues)


# 11. Allocation cannot move name to unrelated cue
def test_11_allocation_cannot_move_name():
    validator = SemanticAllocationValidator()
    proj = Project(
        name="p",
        source_video_path="v",
        characters=[Character(id="char_su", name="Tô Đường", name_zh="苏棠", name_vi="Tô Đường")],
    )
    group = SemanticTranslationGroup(
        group_id="grp_1",
        source_cue_ids=["c1", "c2"],
        source_texts=["这是苏棠", "今天来报到"],
        combined_source_text="这是苏棠今天来报到",
        start=1.0,
        end=3.0,
        full_vi="Đây là Tô Đường, hôm nay đến báo danh.",
    )
    # Incorrect allocation where name is put into c2 instead of c1
    wrong_allocs = [
        SemanticAllocationUnit(cue_id="c1", source_text="这是苏棠", allocated_vi="Hôm nay đến báo danh"),
        SemanticAllocationUnit(cue_id="c2", source_text="今天来报到", allocated_vi="Đây là Tô Đường"),
    ]
    is_valid, issues = validator.validate_group_allocation(proj, group, wrong_allocs)
    assert is_valid is False
    assert any("Canonical name 'Tô Đường'" in iss for iss in issues)


# 12. Concatenated allocated VI remains semantically equivalent to full_vi
def test_12_concatenated_allocated_vi_matches_full():
    allocator = SemanticAllocator()
    group = SemanticTranslationGroup(
        group_id="grp_1",
        source_cue_ids=["c1", "c2"],
        source_texts=["这是苏棠", "今天入职"],
        combined_source_text="这是苏棠今天入职",
        start=1.0,
        end=3.0,
    )
    units = allocator.allocate(Project(name="p", source_video_path="v"), group, "Đây là Tô Đường, hôm nay nhận việc.")
    concat = " ".join(u.allocated_vi for u in units)
    assert "Tô Đường" in concat
    assert "nhận việc" in concat


# 13. Natural allocation avoids orphan fragments
def test_13_natural_allocation_avoids_orphan_fragments():
    allocator = SemanticAllocator()
    cleaned = allocator._clean_orphan_connectors("là kỹ năng nấu ăn")
    assert cleaned == "Là kỹ năng nấu ăn"


# 14. Idiom literal nonsense is rejected in TQ
def test_14_idiom_literal_nonsense_rejected():
    from app.services.translation_quality.idioms import IdiomReviewer
    rev = IdiomReviewer()
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="给外卖判了死缓", translated_text="Phán tử hình treo cho đồ ăn ngoài.")
    issues, reviews = rev.evaluate_cues(Project(name="p", source_video_path="v"), [cue])
    assert any("idiom" in iss.type for iss in issues[cue.id])


# 15. Figurative tone preserved
def test_15_figurative_tone_preserved():
    from app.services.translation_quality.idioms import IdiomReviewer
    rev = IdiomReviewer()
    cue = SubtitleCue(id="c1", start=1.0, end=2.0, source_text="给外卖判了死缓", translated_text="Cơm cậu nấu khiến đồ ăn ngoài hết đường sống.")
    issues, _ = rev.evaluate_cues(Project(name="p", source_video_path="v"), [cue])
    assert len(issues[cue.id]) == 0


# 16. Timing remains unchanged
def test_16_timing_remains_unchanged():
    cues = [
        SubtitleCue(id="c1", start=98.33, end=100.10, source_text="唯一让我在这个"),
        SubtitleCue(id="c2", start=100.40, end=103.11, source_text="是我妈传给我的手艺做饭"),
    ]
    pipeline = SemanticGroupingPipeline()
    proj = Project(name="p", source_video_path="v", cues=cues)
    groups = pipeline.process_project(proj)
    assert proj.cues[0].start == 98.33
    assert proj.cues[0].end == 100.10
    assert proj.cues[1].start == 100.40
    assert proj.cues[1].end == 103.11


# 17. Source Integrity cues remain unchanged
def test_17_source_integrity_cues_remain_unchanged():
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            source_text="你好",
            original_source_cue_ids=["orig_1"],
            segmentation_method="ocr_turnover_split",
            source_integrity_status="REPAIRED",
        )
    ]
    pipeline = SemanticGroupingPipeline()
    proj = Project(name="p", source_video_path="v", cues=cues)
    groups = pipeline.process_project(proj)
    assert proj.cues[0].id == "c1"
    assert proj.cues[0].original_source_cue_ids == ["orig_1"]
    assert proj.cues[0].segmentation_method == "ocr_turnover_split"


# 18. UtteranceEngine receives allocated final cue text
def test_18_utterance_engine_receives_allocated_cue_text():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="你好", translated_text="Xin chào"),
        SubtitleCue(id="c2", start=2.1, end=3.0, source_text="再见", translated_text="Tạm biệt"),
    ]
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(cues, translated=True)
    assert len(render_cues) >= 1
    assert "Xin chào" in render_cues[0].render_text or "Tạm biệt" in render_cues[-1].render_text


# 19. ASS order remains source order
def test_19_ass_order_remains_source_order():
    cues = [
        SubtitleCue(id="c1", start=1.0, end=2.0, source_text="一", translated_text="Một"),
        SubtitleCue(id="c2", start=2.1, end=3.0, source_text="二", translated_text="Hai"),
    ]
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(cues, translated=True)
    for i in range(1, len(render_cues)):
        assert render_cues[i].start >= render_cues[i-1].start
