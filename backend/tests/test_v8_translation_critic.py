from __future__ import annotations

import pytest
from app.models.project import Character, GlossaryEntry, Project, RelationshipRule, SubtitleCue
from app.services.critic import CriticIssueEnum, TranslationCritic, deterministic_validate_cue


@pytest.fixture
def sample_project() -> Project:
    p = Project(
        id="test-proj-v9",
        name="Test Project V9",
        source_video_path="dummy.mp4",
        source_language="zh",
        target_language="vi",
        characters=[
            Character(
                id="char_heroine",
                name="Tần Phù Chi",
                name_zh="秦扶栀",
                name_vi="Tần Phù Chi",
                aliases=["秦福之"],
                gender="female",
                role="nữ chính",
            ),
            Character(
                id="char_mother",
                name="Tống Tri Tuyết",
                name_zh="宋知雪",
                name_vi="Tống Tri Tuyết",
                gender="female",
                role="mẹ",
            ),
            Character(
                id="char_brother",
                name="Anh trai",
                name_zh="大哥",
                name_vi="Anh trai",
                gender="male",
                role="anh trai",
            ),
        ],
        relationships=[
            RelationshipRule(
                from_character_id="char_brother",
                to_character_id="char_heroine",
                relationship="older_brother_to_younger_sister",
                relationship_type="older_brother_to_younger_sister",
                vi_self="anh",
                vi_other="em",
                vi_self_pronoun="anh",
                vi_target_pronoun="em",
            ),
            RelationshipRule(
                from_character_id="char_mother",
                to_character_id="char_heroine",
                relationship="mother_to_daughter",
                relationship_type="mother_to_daughter",
                vi_self="mẹ",
                vi_other="con",
                vi_self_pronoun="mẹ",
                vi_target_pronoun="con",
            ),
        ],
        glossary=[
            GlossaryEntry(source="秦扶栀", target="Tần Phù Chi"),
            GlossaryEntry(source="秦福之", target="Tần Phù Chi"),
            GlossaryEntry(source="KPI", target="KPI"),
        ],
    )
    return p


def test_female_daughter_meaning_preservation(sample_project: Project):
    """Test A: Female daughter meaning preservation vs 'không có con người'."""
    ctx_wrong = {
        "chinese_source": "她的眼里没有女儿 只有一件需要时刻打磨的商品",
        "vietnamese_translation": "Trong mắt bà không có con người, chỉ có một món hàng cần được mài giũa bất cứ lúc nào.",
        "characters": [{"name_zh": "宋知雪", "name_vi": "Tống Tri Tuyết"}],
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_wrong)
    assert not is_pass
    assert CriticIssueEnum.MEANING_SHIFT.value in issues or CriticIssueEnum.DROPPED_CLAUSE.value in issues

    ctx_correct = {
        "chinese_source": "她的眼里没有女儿 只有一件需要时刻打磨的商品",
        "vietnamese_translation": "Trong mắt bà ấy không xem tôi là con gái, mà chỉ là một món hàng cần mài giũa bất cứ lúc nào.",
        "characters": [{"name_zh": "宋知雪", "name_vi": "Tống Tri Tuyết"}],
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_correct)
    assert is_pass_c
    assert len(issues_c) == 0


def test_brother_to_sister_pronoun(sample_project: Project):
    """Test B: Brother -> Sister pronoun validation (must use 'em', not 'cô' or 'mày')."""
    ctx_wrong = {
        "chinese_source": "你的存在拉低了秦家的执行效率",
        "vietnamese_translation": "Sự tồn tại của cô đang kéo giảm hiệu suất điều hành của nhà họ Tần.",
        "expected_vi_self": "anh",
        "expected_vi_target": "em",
        "relationship": "older_brother_to_younger_sister",
        "speaker_role": "anh trai",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_wrong)
    assert not is_pass
    assert CriticIssueEnum.PRONOUN_MISMATCH.value in issues

    ctx_correct = {
        "chinese_source": "你的存在拉低了秦家的执行效率",
        "vietnamese_translation": "Sự tồn tại của em đang kéo giảm hiệu suất thực thi của nhà họ Tần.",
        "expected_vi_self": "anh",
        "expected_vi_target": "em",
        "relationship": "older_brother_to_younger_sister",
        "speaker_role": "anh trai",
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_correct)
    assert is_pass_c
    assert len(issues_c) == 0


def test_explicit_character_name_preservation(sample_project: Project):
    """Test C: Explicit character name preservation."""
    ctx_missing_name = {
        "chinese_source": "看清楚 秦扶栀",
        "vietnamese_translation": "Nhìn cho rõ vào,",
        "characters": [{"name_zh": "秦扶栀", "name_vi": "Tần Phù Chi", "aliases": ["秦福之"]}],
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_missing_name)
    assert not is_pass
    assert CriticIssueEnum.NAME_MISMATCH.value in issues

    ctx_with_name = {
        "chinese_source": "看清楚 秦扶栀",
        "vietnamese_translation": "Nhìn cho rõ vào, Tần Phù Chi.",
        "characters": [{"name_zh": "秦扶栀", "name_vi": "Tần Phù Chi", "aliases": ["秦福之"]}],
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_with_name)
    assert is_pass_c
    assert len(issues_c) == 0


def test_unsupported_addition_hallucination(sample_project: Project):
    """Test D: Unsupported addition / hallucination."""
    ctx_hallucinated = {
        "chinese_source": "看清楚",
        "vietnamese_translation": "Cố lên, nhìn cho rõ vào.",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_hallucinated)
    assert not is_pass
    assert CriticIssueEnum.HALLUCINATION.value in issues


def test_dropped_clause(sample_project: Project):
    """Test E: Dropped clause in compound statement."""
    ctx_dropped = {
        "chinese_source": "她的眼里没有女儿 只有一件需要时刻打磨的商品",
        "vietnamese_translation": "Chỉ là một món hàng cần mài giũa bất cứ lúc nào.",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_dropped)
    assert not is_pass
    assert CriticIssueEnum.DROPPED_CLAUSE.value in issues


def test_action_verb_eat_chicken_leg(sample_project: Project):
    """Test F: Action verb '啃完' (eat/gnaw/finish eating) vs mistranslated 'giấu'."""
    ctx_wrong = {
        "chinese_source": "但我现在只想把这只偷偷藏起来的鸡腿啃完",
        "vietnamese_translation": "Nhưng bây giờ tôi chỉ muốn giấu chiếc đùi gà này.",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_wrong)
    assert not is_pass
    assert CriticIssueEnum.ACTION_ERROR.value in issues

    ctx_correct = {
        "chinese_source": "但我现在只想把这只偷偷藏起来的鸡腿啃完",
        "vietnamese_translation": "Nhưng bây giờ tôi chỉ muốn gặm cho xong chiếc đùi gà đã lén giấu này.",
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_correct)
    assert is_pass_c
    assert len(issues_c) == 0


def test_action_verb_recite_chapter_conclusion(sample_project: Project):
    """Test G: Action verb '背一下' (recite from memory) vs unsupported hurry modifier."""
    ctx_wrong = {
        "chinese_source": "背一下第三章的结论",
        "vietnamese_translation": "Nhanh lên nào, hãy nhớ lại kết luận của chương ba.",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_wrong)
    assert not is_pass
    assert CriticIssueEnum.HALLUCINATION.value in issues or CriticIssueEnum.ACTION_ERROR.value in issues

    ctx_correct = {
        "chinese_source": "背一下第三章的结论",
        "vietnamese_translation": "Đọc thuộc lòng kết luận của chương ba xem nào.",
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_correct)
    assert is_pass_c
    assert len(issues_c) == 0


def test_name_lock_and_phonetic_invention(sample_project: Project):
    """Test H: Banned phonetic invention 'Ken Văn' when source has canonical character."""
    ctx_invented = {
        "chinese_source": "看清楚 秦扶栀",
        "vietnamese_translation": "Ken Văn, nhìn kỹ.",
        "characters": [{"name_zh": "秦扶栀", "name_vi": "Tần Phù Chi", "aliases": ["秦福之"]}],
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_invented)
    assert not is_pass
    assert CriticIssueEnum.NAME_MISMATCH.value in issues


def test_synthetic_characters_actions_generalization():
    """Test I: Generalization test with synthetic character and action names."""
    ctx_synthetic = {
        "chinese_source": "背诵第四节并且把苹果啃完",
        "vietnamese_translation": "Học thuộc tiết bốn và ăn hết quả táo.",
        "characters": [{"name_zh": "李明", "name_vi": "Lý Minh", "aliases": []}],
    }
    is_pass, issues, _ = deterministic_validate_cue(ctx_synthetic)
    assert is_pass
    assert len(issues) == 0


def test_mother_daughter_referent_and_commodity():
    """Test J: Mother referent resolution in '她的眼里没有女儿 只有一件需要时刻打磨的商品'."""
    ctx_wrong_referent = {
        "chinese_source": "她的眼里没有女儿 只有一件需要时刻打磨的商品",
        "vietnamese_translation": "Trong mắt cô ấy, không có con gái mà chỉ là một món hàng cần được rèn giũa liên tục.",
        "speaker_role": "narrator",
        "expected_vi_self": "tôi",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_wrong_referent)
    assert not is_pass
    assert CriticIssueEnum.REFERENT_ERROR.value in issues

    ctx_correct_referent = {
        "chinese_source": "她的眼里没有女儿 只有一件需要时刻打磨的商品",
        "vietnamese_translation": "Trong mắt mẹ, không xem tôi là con gái mà chỉ có một món hàng cần được rèn giũa liên tục.",
        "speaker_role": "narrator",
        "expected_vi_self": "tôi",
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_correct_referent)
    assert is_pass_c
    assert len(issues_c) == 0


def test_speaker_name_metadata_not_prepended():
    """Test K: Speaker metadata name must not be prepended to dialogue text."""
    ctx_prepended = {
        "chinese_source": "看清楚 秦扶栀",
        "vietnamese_translation": "Mạnh Kinh Xuân Nhìn cho rõ vào, Tần Phù Chi.",
        "speaker_name_vi": "Mạnh Kinh Xuân",
        "speaker_name_zh": "孟惊春",
        "characters": [{"name_zh": "秦扶栀", "name_vi": "Tần Phù Chi", "aliases": []}],
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_prepended)
    assert not is_pass
    assert CriticIssueEnum.NAME_MISMATCH.value in issues

    ctx_clean = {
        "chinese_source": "看清楚 秦扶栀",
        "vietnamese_translation": "Nhìn cho rõ vào, Tần Phù Chi.",
        "speaker_name_vi": "Mạnh Kinh Xuân",
        "speaker_name_zh": "孟惊春",
        "characters": [{"name_zh": "秦扶栀", "name_vi": "Tần Phù Chi", "aliases": []}],
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_clean)
    assert is_pass_c
    assert len(issues_c) == 0


def test_relational_negation_vs_literal_existence():
    """Test L: Relational negation '没有女儿' must express 'không xem/coi tôi là con gái', not literal absence 'không có con gái'."""
    ctx_literal = {
        "chinese_source": "她的眼里没有女儿 只有一件需要时刻打磨的商品",
        "vietnamese_translation": "Trong mắt mẹ, không có con gái mà chỉ là một món hàng cần được mài giũa liên tục.",
        "speaker_role": "narrator",
        "expected_vi_self": "tôi",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_literal)
    assert not is_pass
    assert CriticIssueEnum.MEANING_SHIFT.value in issues

    ctx_relational = {
        "chinese_source": "她的眼里没有女儿 只有一件需要时刻打磨的商品",
        "vietnamese_translation": "Trong mắt mẹ, không xem tôi là con gái mà chỉ là một món hàng cần được mài giũa liên tục.",
        "speaker_role": "narrator",
        "expected_vi_self": "tôi",
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_relational)
    assert is_pass_c
    assert len(issues_c) == 0


def test_vocative_not_converted_to_possessor():
    """Test M: Vocative character name must not be turned into possessor ('của <Name>')."""
    ctx_possessor = {
        "chinese_source": "秦扶栀昨天的宏观经济笔记看完吗？",
        "vietnamese_translation": "Con đã đọc xong bài ghi chú kinh tế vĩ mô của Tần Phù Chi hôm qua chưa?",
        "speaker": "char_father",
        "addressee": "char_heroine",
        "characters": [{"name_zh": "秦扶栀", "name_vi": "Tần Phù Chi", "aliases": []}],
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_possessor)
    assert not is_pass
    assert CriticIssueEnum.GRAMMATICAL_ERROR.value in issues

    ctx_vocative = {
        "chinese_source": "秦扶栀昨天的宏观经济笔记看完吗？",
        "vietnamese_translation": "Tần Phù Chi, con đã đọc xong bài ghi chú kinh tế vĩ mô hôm qua chưa?",
        "speaker": "char_father",
        "addressee": "char_heroine",
        "characters": [{"name_zh": "秦扶栀", "name_vi": "Tần Phù Chi", "aliases": []}],
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_vocative)
    assert is_pass_c
    assert len(issues_c) == 0


def test_dangling_fragment_detection_and_resolution():
    """Test N: Detects dangling trailing tokens and commas in Vietnamese subtitles."""
    from app.services.critic import has_dangling_fragment
    assert has_dangling_fragment("Tần Phù Chi, cô,")
    assert has_dangling_fragment("Nhìn rõ, Tần Phù Chi, cô")
    assert has_dangling_fragment("Nhưng bây giờ tôi chỉ muốn,")
    assert not has_dangling_fragment("Nhìn cho rõ đây!")
    assert not has_dangling_fragment("Tần Phù Chi, cô còn ăn được nữa à?")


