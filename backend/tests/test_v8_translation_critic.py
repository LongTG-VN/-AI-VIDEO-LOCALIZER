import pytest
from app.models.project import Character, GlossaryEntry, Project, RelationshipRule, SubtitleCue
from app.services.critic import CriticIssueEnum, deterministic_validate_cue, build_critic_context


def create_test_project() -> Project:
    project = Project(
        id="test-v8",
        name="Test V8 Project",
        source_video_path="test_video.mp4",
        source_language="zh",
        target_language="vi",
        characters=[
            Character(id="char_mom", name="宋知雪", name_zh="宋知雪", name_vi="Tống Tri Tuyết", role="Mẹ nuôi", gender="female"),
            Character(id="char_daughter", name="秦扶栀", name_zh="秦扶栀", name_vi="Tần Phù Chi", role="Nữ chính", gender="female"),
            Character(id="char_brother", name="秦砚川", name_zh="秦砚川", name_vi="Tần Nghiễn Xuyên", role="Anh trai", gender="male"),
        ],
        relationships=[
            RelationshipRule(from_character_id="char_mom", to_character_id="char_daughter", relationship="mother_daughter", vi_self_pronoun="mẹ", vi_target_pronoun="con"),
            RelationshipRule(from_character_id="char_brother", to_character_id="char_daughter", relationship="sibling", vi_self_pronoun="anh", vi_target_pronoun="em"),
        ],
        glossary=[
            GlossaryEntry(source="秦扶栀", target="Tần Phù Chi"),
            GlossaryEntry(source="宋知雪", target="Tống Tri Tuyết"),
            GlossaryEntry(source="秦砚川", target="Tần Nghiễn Xuyên"),
        ],
        cues=[],
    )
    return project


def test_mother_daughter_pronoun_flagged():
    """Test 1: Mother-daughter dialogue containing 'mày' is flagged as pronoun mismatch."""
    project = create_test_project()
    cue = SubtitleCue(
        id="cue-1",
        start=10.0,
        end=12.0,
        speaker_character_id="char_mom",
        addressee_character_id="char_daughter",
        source_text="你今天怎么回事？",
        translated_text="Mày hôm nay bị làm sao thế?",
    )
    ctx = build_critic_context(project, cue)
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.PRONOUN_MISMATCH.value in issues


def test_brother_sister_pronoun_flagged():
    """Test 2: Sibling dialogue containing 'mày' is flagged as pronoun mismatch."""
    project = create_test_project()
    cue = SubtitleCue(
        id="cue-2",
        start=35.0,
        end=37.0,
        speaker_character_id="char_brother",
        addressee_character_id="char_daughter",
        source_text="你的存在拉低了秦家的执行效率",
        translated_text="Sự tồn tại của mày đã kéo giảm hiệu suất làm việc của nhà họ Tần.",
    )
    ctx = build_critic_context(project, cue)
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.PRONOUN_MISMATCH.value in issues


def test_monologue_pronoun_flagged():
    """Test 3: Monologue speaking as 'con...' to audience is flagged as pronoun mismatch."""
    project = create_test_project()
    cue = SubtitleCue(
        id="cue-3",
        start=54.0,
        end=56.0,
        speaker_character_id="char_daughter",
        addressee_character_id=None,
        source_text="但我现在只想把这只偷偷藏起来的鸡腿",
        translated_text="Con đang muốn ăn cái đùi gà này",
    )
    ctx = build_critic_context(project, cue)
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.PRONOUN_MISMATCH.value in issues


def test_gender_reference_flagged():
    """Test 4: Female referent 她 turned into 'ông ta' is flagged as gender mismatch."""
    project = create_test_project()
    cue = SubtitleCue(
        id="cue-4",
        start=18.0,
        end=21.0,
        speaker_character_id="char_daughter",
        addressee_character_id=None,
        source_text="她的眼里没有女儿",
        translated_text="Trong mắt ông ta không có đứa con gái nào",
    )
    ctx = build_critic_context(project, cue)
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.GENDER_MISMATCH.value in issues


def test_dropped_clause_flagged():
    """Test 5: Contrast clause with missing second part is flagged as dropped clause."""
    project = create_test_project()
    cue = SubtitleCue(
        id="cue-5",
        start=18.0,
        end=21.0,
        speaker_character_id="char_daughter",
        addressee_character_id=None,
        source_text="她的眼里没有女儿只有一件需要时刻打磨的商品",
        translated_text="Trong mắt mẹ chỉ có tôi",
    )
    ctx = build_critic_context(project, cue)
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.DROPPED_CLAUSE.value in issues


def test_hallucination_flagged():
    """Test 6: Hallucinated phrase 'cố lên' on '看清楚' is flagged."""
    project = create_test_project()
    cue = SubtitleCue(
        id="cue-6",
        start=56.0,
        end=58.0,
        speaker_character_id="char_mom",
        addressee_character_id="char_daughter",
        source_text="看清楚 秦扶栀",
        translated_text="Cố lên nhìn cho rõ vào, Tần Phù Chi",
    )
    ctx = build_critic_context(project, cue)
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.HALLUCINATION.value in issues


def test_name_mismatch_flagged():
    """Test 7: Glossary name translated incorrectly is flagged."""
    project = create_test_project()
    cue = SubtitleCue(
        id="cue-7",
        start=56.0,
        end=58.0,
        speaker_character_id="char_mom",
        addressee_character_id="char_daughter",
        source_text="看清楚 秦扶栀",
        translated_text="Nhìn cho rõ vào, Tiểu Lan",
    )
    ctx = build_critic_context(project, cue)
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.NAME_MISMATCH.value in issues


def test_valid_natural_translation_passes():
    """Test 8: Valid natural translation adhering to pronouns and glossary passes 100%."""
    project = create_test_project()
    cue = SubtitleCue(
        id="cue-8",
        start=35.0,
        end=37.0,
        speaker_character_id="char_brother",
        addressee_character_id="char_daughter",
        source_text="你的存在拉低了秦家的执行效率",
        translated_text="Sự tồn tại của em đã kéo giảm hiệu suất làm việc của nhà họ Tần.",
    )
    ctx = build_critic_context(project, cue)
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert is_pass
    assert len(issues) == 0
