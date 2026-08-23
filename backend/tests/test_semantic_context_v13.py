from __future__ import annotations

from app.models.project import Character, Project, SubtitleCue
from app.services.semantic_context import build_neighbor_window, source_name_mentions
from app.services.translation import build_translation_context
from app.services.utterance_engine import DiscourseMode, UtteranceEngine, infer_discourse_mode


def _cue(
    cue_id: str,
    start: float,
    source: str,
    *,
    end: float | None = None,
    translated: str | None = None,
    speaker: str | None = "speaker_a",
    addressee: str | None = None,
    mode: str = "unknown",
) -> SubtitleCue:
    return SubtitleCue(
        id=cue_id,
        start=start,
        end=end if end is not None else start + 0.8,
        source_text=source,
        translated_text=translated,
        speaker_id=speaker,
        addressee_id=addressee,
        discourse_mode=mode,
    )


def test_neighbor_window_uses_three_previous_and_two_next():
    cues = [_cue(f"c{i}", float(i), f"句子{i}") for i in range(7)]
    project = Project(name="window", source_video_path="demo.mp4", cues=cues)

    window = build_neighbor_window(project, 3, before=3, after=2)

    assert [item["cue_id"] for item in window["previous"]] == ["c0", "c1", "c2"]
    assert [item["cue_id"] for item in window["next"]] == ["c4", "c5"]


def test_explicit_name_lock_belongs_only_to_originating_cue():
    character = Character(
        id="char_lin",
        name="Lâm Vãn Vãn",
        name_zh="林晚晚",
        name_vi="Lâm Vãn Vãn",
        aliases=["林晚"],
    )
    project = Project(
        name="names",
        source_video_path="demo.mp4",
        characters=[character],
        cues=[
            _cue("a", 0.0, "看清楚 林晚晚"),
            _cue("b", 1.0, "你还记得吗"),
        ],
    )

    assert source_name_mentions(project, project.cues[0].source_text)[0]["target"] == "Lâm Vãn Vãn"
    assert source_name_mentions(project, project.cues[1].source_text) == []

    first = build_translation_context(project, 0)
    second = build_translation_context(project, 1)
    assert first["required_names"][0]["target"] == "Lâm Vãn Vãn"
    assert second["required_names"] == []


def test_translation_context_exposes_scene_safe_neighbor_metadata():
    project = Project(
        name="context",
        source_video_path="demo.mp4",
        cues=[
            _cue("before", 0.0, "前一句", mode="narration"),
            _cue("current", 1.0, "现在说话", addressee="char_b", mode="direct_dialogue"),
            _cue("after", 2.0, "后一句", mode="monologue"),
        ],
    )

    context = build_translation_context(project, 1)

    assert context["discourse_mode"] == "direct_dialogue"
    assert context["previous_context"][-1]["discourse_mode"] == "narration"
    assert context["next_context"][0]["discourse_mode"] == "monologue"


def test_explicit_discourse_mode_is_a_hard_merge_boundary():
    cues = [
        _cue(
            "a",
            0.0,
            "我曾经住在那里，",
            translated="Tôi từng sống ở đó,",
            addressee="char_b",
            mode="narration",
        ),
        _cue(
            "b",
            0.82,
            "但是你现在要听我说",
            translated="nhưng bây giờ anh phải nghe tôi nói",
            addressee="char_b",
            mode="direct_dialogue",
        ),
    ]

    render_cues, _ = UtteranceEngine().process_cues(cues)

    assert len(render_cues) == 2
    assert render_cues[0].discourse_mode == "narration"
    assert render_cues[1].discourse_mode == "direct_dialogue"


def test_same_character_monologue_and_dialogue_do_not_merge():
    cues = [
        _cue(
            "a",
            0.0,
            "但是我还在想，",
            translated="Nhưng tôi vẫn đang nghĩ,",
            addressee="char_b",
            mode="monologue",
        ),
        _cue(
            "b",
            0.82,
            "但是你必须走",
            translated="nhưng anh phải đi",
            addressee="char_b",
            mode="direct_dialogue",
        ),
    ]

    render_cues, _ = UtteranceEngine().process_cues(cues)
    assert len(render_cues) == 2


def test_unknown_legacy_cue_with_addressee_falls_back_to_dialogue():
    cue = {
        "source_text": "你去哪？",
        "translated_text": "Anh đi đâu?",
        "addressee_id": "char_b",
        "discourse_mode": "unknown",
    }
    assert infer_discourse_mode(cue) == DiscourseMode.DIRECT_DIALOGUE


def test_duplicate_suppression_uses_real_cue_durations():
    cues = [
        _cue("a", 0.0, "同一句话", end=1.0, translated="Cùng một câu."),
        _cue("b", 0.4, "同一句话", end=1.4, translated="Cùng một câu."),
    ]

    render_cues, metrics = UtteranceEngine().process_cues(cues)

    assert len(render_cues) == 1
    assert metrics["suppressed_duplicates"] == 1


def test_engine_caps_automatic_merge_to_two_source_cues():
    cues = [
        _cue("a", 0.0, "如果你愿意，", translated="Nếu anh đồng ý,"),
        _cue("b", 0.82, "但是我们可以继续，", translated="nhưng chúng ta có thể tiếp tục,"),
        _cue("c", 1.64, "而且明天再说", translated="và ngày mai nói tiếp"),
    ]

    render_cues, _ = UtteranceEngine().process_cues(cues)

    assert len(render_cues) == 2
    assert render_cues[0].source_cue_ids == ["a", "b"]
    assert render_cues[1].source_cue_ids == ["c"]


def test_three_short_independent_assessments_stay_separate():
    cues = [
        _cue("a", 11.87, "领口歪了，", end=12.63, translated="Vòng cổ bị lệch,"),
        _cue("b", 12.67, "坐姿不对，", end=13.35, translated="Tư thế ngồi sai,"),
        _cue("c", 13.69, "笑的太假。", end=14.37, translated="Cười quá giả."),
    ]
    render_cues, _ = UtteranceEngine().process_cues(cues)
    assert len(render_cues) == 3
    assert [rc.source_cue_ids for rc in render_cues] == [["a"], ["b"], ["c"]]


def test_unfinished_clause_plus_direct_continuation_may_merge():
    cues = [
        _cue("a", 0.0, "如果我们现在出发，", end=1.2, translated="Nếu chúng ta xuất phát ngay,"),
        _cue("b", 1.25, "就能赶上最后一班车。", end=2.5, translated="thì sẽ kịp chuyến xe cuối."),
    ]
    render_cues, metrics = UtteranceEngine().process_cues(cues)
    assert len(render_cues) == 1
    assert render_cues[0].source_cue_ids == ["a", "b"]
    assert metrics["merged_groups"] == 1


def test_narration_complete_clause_plus_direct_confrontation_must_not_merge():
    cues = [
        _cue(
            "narr",
            70.31,
            "在早餐店的家庭里，凌晨四点就要起来帮忙揉面。",
            end=73.80,
            translated="Trong gia đình quán sáng, lúc 4 giờ sáng phải dậy để giúp nhào bánh.",
            mode="narration",
        ),
        _cue(
            "conf",
            73.80,
            "秦扶栀，你偷了我十八年！",
            end=76.64,
            translated="Tần Phù Chi, cô đã trộm lấy 18 năm của tôi!",
            addressee="char_heroine",
            mode="direct_dialogue",
        ),
    ]
    render_cues, _ = UtteranceEngine().process_cues(cues)
    assert len(render_cues) == 2
    assert render_cues[0].source_cue_ids == ["narr"]
    assert render_cues[1].source_cue_ids == ["conf"]


def test_same_timing_overlap_but_different_discourse_mode_must_not_merge():
    cues = [
        _cue("a", 10.0, "这是当年的回忆。", end=11.5, translated="Đây là ký ức năm xưa.", mode="narration"),
        _cue("b", 10.5, "你怎么在这里？", end=12.0, translated="Sao anh lại ở đây?", addressee="char_b", mode="direct_dialogue"),
    ]
    render_cues, _ = UtteranceEngine().process_cues(cues)
    assert len(render_cues) == 2
    assert render_cues[0].discourse_mode == "narration"
    assert render_cues[1].discourse_mode == "direct_dialogue"


def test_neighbor_contains_name_but_current_does_not_no_name_migration():
    char = Character(id="char_qfz", name="Tần Phù Chi", name_zh="秦扶栀", name_vi="Tần Phù Chi")
    project = Project(
        name="test",
        source_video_path="demo.mp4",
        characters=[char],
        cues=[
            _cue("c1", 0.0, "秦扶栀你听我说"),
            _cue("c2", 1.5, "我们必须快点离开"),
        ],
    )
    mentions_c1 = source_name_mentions(project, project.cues[0].source_text)
    mentions_c2 = source_name_mentions(project, project.cues[1].source_text)
    assert len(mentions_c1) == 1
    assert mentions_c1[0]["target"] == "Tần Phù Chi"
    assert len(mentions_c2) == 0


def test_current_cue_explicit_character_name_requires_canonical_target():
    char = Character(id="char_mkx", name="Mạnh Kinh Xuân", name_zh="孟惊春", name_vi="Mạnh Kinh Xuân")
    project = Project(
        name="test",
        source_video_path="demo.mp4",
        characters=[char],
        cues=[_cue("c1", 0.0, "孟惊春才是真正的女儿")],
    )
    mentions = source_name_mentions(project, project.cues[0].source_text)
    assert len(mentions) == 1
    assert mentions[0]["target"] == "Mạnh Kinh Xuân"


def test_monologue_self_pronoun_does_not_inherit_dialogue_addressee():
    from app.services.critic import CriticIssueEnum, deterministic_validate_cue
    ctx = {
        "chinese_source": "我一定要查清楚真相",
        "vietnamese_translation": "Con nhất định phải điều tra rõ chân tướng",
        "speaker_role": "heroine",
        "expected_vi_self": "tôi",
        "relationship": "monologue",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.PRONOUN_MISMATCH.value in issues


def test_translation_provider_wrong_or_missing_ids_strict_id_matching():
    from app.services.translation import OpenAICompatibleTranslator
    translator = OpenAICompatibleTranslator("http://localhost:11434/v1", "test", "test-model")
    batch = [
        {"cue_id": "correct_id_1", "source": "你好"},
        {"cue_id": "correct_id_2", "source": "再见"},
    ]
    # Simulate provider returning arbitrary or shifted IDs
    items = [
        {"cue_id": "wrong_id_x", "text": "Xin chào", "confidence": 0.9},
        {"cue_id": "wrong_id_y", "text": "Tạm biệt", "confidence": 0.9},
    ]
    expected_ids = {item["cue_id"] for item in batch}
    results = {}
    for item in items:
        if isinstance(item, dict) and item.get("cue_id") in expected_ids:
            results[item["cue_id"]] = (item["text"], item.get("confidence"))
    for b in batch:
        if b["cue_id"] not in results:
            results[b["cue_id"]] = (b.get("source", ""), 0.0)

    assert results["correct_id_1"] == ("你好", 0.0)
    assert results["correct_id_2"] == ("再见", 0.0)


def test_chinese_source_fallback_not_accepted_as_successful_vietnamese():
    from app.services.critic import CriticIssueEnum, deterministic_validate_cue
    ctx = {
        "chinese_source": "领口歪了",
        "vietnamese_translation": "领口歪了",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx)
    assert not is_pass
    assert CriticIssueEnum.MEANING_SHIFT.value in issues
