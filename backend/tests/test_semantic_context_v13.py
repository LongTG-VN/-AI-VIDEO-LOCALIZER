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
