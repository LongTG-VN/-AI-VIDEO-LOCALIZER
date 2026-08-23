from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from app.models.project import Character, Project, RenderOptions, SubtitleCue
from app.services.critic import CriticIssueEnum, deterministic_validate_cue
from app.services.subtitles import write_ass
from app.services.utterance_engine import (
    DiscourseMode,
    RenderSubtitleCue,
    UtteranceEngine,
    clean_vietnamese_typography,
    merge_vietnamese_clauses,
    validate_render_cue_entity_ownership,
)


def _make_cue(
    cue_id: str,
    start: float,
    end: float,
    source: str,
    translated: str | None = None,
    speaker: str = "speaker_a",
    addressee: str | None = None,
    mode: str = "direct_dialogue",
) -> SubtitleCue:
    return SubtitleCue(
        id=cue_id,
        start=start,
        end=end,
        source_text=source,
        translated_text=translated,
        speaker_id=speaker,
        addressee_id=addressee,
        discourse_mode=mode,
    )


def test_explicit_vocative_stays_with_originating_cue():
    """1. Explicit vocative stays with originating source cue end-to-end."""
    chars = [Character(id="c1", name="Tần Phù Chi", name_zh="秦扶栀", name_vi="Tần Phù Chi")]
    cues = [
        _make_cue("cue1", 55.85, 57.59, "看清楚，", "Nhìn cho rõ", mode="direct_dialogue"),
        _make_cue("cue2", 57.59, 59.84, "秦扶栀你，", "Tần Phù Chi, cô", mode="direct_dialogue"),
        _make_cue("cue3", 60.18, 61.28, "还吃得下去，", "còn ăn được sao?", mode="direct_dialogue"),
    ]
    project = Project(name="vocative", source_video_path="v.mp4", characters=chars, cues=cues)
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(project.cues, translated=True)

    # Cue 1 has no name
    rc1 = next(rc for rc in render_cues if "cue1" in rc.source_cue_ids)
    assert "Tần Phù Chi" not in rc1.render_text

    # Cue 2+3 has Tần Phù Chi
    rc2 = next(rc for rc in render_cues if "cue2" in rc.source_cue_ids)
    assert "Tần Phù Chi" in rc2.render_text
    assert "Nhìn cho rõ" not in rc2.render_text


def test_name_cannot_migrate_during_grouping():
    """2. Name cannot migrate during grouping to unrelated cues."""
    chars = [{"zh": "秦扶栀", "vi": "Tần Phù Chi"}]
    cues_by_id = {
        "c1": _make_cue("c1", 10.0, 11.0, "看清楚"),
        "c2": _make_cue("c2", 12.0, 13.0, "秦扶栀你"),
    }
    bad_rc = [
        RenderSubtitleCue(
            render_id="sub_001",
            source_cue_ids=["c1"],
            start=10.0,
            end=11.0,
            source_text="看清楚",
            translated_text="Nhìn cho rõ Tần Phù Chi",
            render_text="Nhìn cho rõ Tần Phù Chi",
        )
    ]
    violations = validate_render_cue_entity_ownership(bad_rc, cues_by_id, chars)
    assert len(violations) == 1
    assert "Entity ownership violation" in violations[0]


def test_speaker_metadata_cannot_become_subtitle_text():
    """3. Speaker/addressee metadata cannot become subtitle text."""
    ctx = {
        "chinese_source": "看清楚",
        "vietnamese_translation": "Mạnh Kinh Xuân Nhìn cho rõ",
        "speaker_name_vi": "Mạnh Kinh Xuân",
        "speaker_name_zh": "孟惊春",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.NAME_MISMATCH.value in issues


def test_duplicate_vietnamese_pronoun_is_detected_and_cleaned():
    """4. Duplicate Vietnamese pronoun is detected and cleaned."""
    ctx = {
        "chinese_source": "你还吃得下去吗？",
        "vietnamese_translation": "Cô còn ăn được không, cô?",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.GRAMMATICAL_ERROR.value in issues

    cleaned = clean_vietnamese_typography("Cô còn ăn được không, cô?")
    assert cleaned == "Cô còn ăn được không?"


def test_mid_sentence_accidental_capitalization_is_fixed():
    """5. Mid-sentence accidental capitalization is detected and fixed."""
    ctx = {
        "chinese_source": "秦扶栀你还吃得下去",
        "vietnamese_translation": "Tần Phù Chi, cô Còn ăn được",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.GRAMMATICAL_ERROR.value in issues

    merged = merge_vietnamese_clauses("Tần Phù Chi, cô", "Còn ăn được sao?")
    assert "còn ăn được" in merged
    assert "Còn" not in merged


def test_neckline_clothing_context_not_translated_as_necklace():
    """6. 领口 in clothing context is not translated as necklace ('vòng cổ')."""
    ctx_wrong = {
        "chinese_source": "领口歪了",
        "vietnamese_translation": "Vòng cổ bị lệch",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_wrong)
    assert not is_pass
    assert CriticIssueEnum.MEANING_SHIFT.value in issues

    ctx_correct = {
        "chinese_source": "领口歪了",
        "vietnamese_translation": "Cổ áo bị lệch",
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_correct)
    assert is_pass_c


def test_relational_negation_vietnamese_must_be_grammatical():
    """7. Relational-negation Vietnamese must be grammatical (subject explicit)."""
    ctx_subjectless = {
        "chinese_source": "她的眼里没有女儿 只有一件需要时刻打磨的商品",
        "vietnamese_translation": "Trong mắt mẹ, không xem tôi là con gái mà chỉ có một món hàng cần liên tục mài giũa.",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_subjectless)
    assert not is_pass
    assert CriticIssueEnum.GRAMMATICAL_ERROR.value in issues

    ctx_grammatical = {
        "chinese_source": "她的眼里没有女儿 只有一件需要时刻打磨的商品",
        "vietnamese_translation": "Mẹ không xem tôi là con gái mà chỉ coi tôi như một món hàng cần liên tục mài giũa.",
    }
    is_pass_g, issues_g, _ = deterministic_validate_cue(ctx_grammatical)
    assert is_pass_g


def test_narration_direct_dialogue_hard_boundary():
    """8. Narration/direct-dialogue hard boundary remains PASS."""
    cues = [
        _make_cue("n1", 70.31, 73.77, "四点就得起床帮着揉面。", "Bốn giờ sáng đã phải dậy giúp nhào bột.", mode="narration"),
        _make_cue("d1", 73.80, 76.64, "秦扶栀，你抢了我十八年！", "Tần Phù Chi, cô đã cướp mất mười tám năm của tôi!", mode="direct_dialogue"),
    ]
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(cues, translated=True)
    assert len(render_cues) == 2
    assert render_cues[0].discourse_mode == DiscourseMode.NARRATION.value
    assert render_cues[1].discourse_mode == DiscourseMode.DIRECT_DIALOGUE.value


def test_three_short_critique_cues_remain_separate():
    """9. Three short critique cues remain separate."""
    cues = [
        _make_cue("c1", 11.87, 12.64, "领口歪了", "Cổ áo lệch rồi.", mode="direct_dialogue"),
        _make_cue("c2", 12.67, 13.47, "坐姿不对", "Tư thế ngồi không đúng.", mode="direct_dialogue"),
        _make_cue("c3", 13.69, 14.49, "笑得太假", "Cười giả tạo quá.", mode="direct_dialogue"),
    ]
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(cues, translated=True)
    assert len(render_cues) == 3


def test_end_to_end_ass_generation_and_no_chinese_fallback():
    """10. End-to-end pipeline produces valid ASS subtitle events with no Chinese fallback."""
    cues = [
        _make_cue("c1", 1.0, 2.0, "请立即起床。", "Hãy dậy ngay.", mode="monologue"),
        _make_cue("c2", 2.5, 4.0, "我叫秦福，", "Tôi tên là Tần Phù Chi.", mode="monologue"),
    ]
    project = Project(name="ass_test", source_video_path="v.mp4", cues=cues)
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(project.cues, translated=True)

    with TemporaryDirectory() as tmp_dir:
        ass_path = Path(tmp_dir) / "output.ass"
        write_ass(ass_path, render_cues, RenderOptions())
        assert ass_path.exists()
        content = ass_path.read_text(encoding="utf-8")
        assert "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hãy dậy ngay." in content
        assert "Tôi tên là Tần Phù Chi." in content
        for line in content.splitlines():
            if line.startswith("Dialogue:"):
                assert not re.search(r"[\u4e00-\u9fff]", line)
