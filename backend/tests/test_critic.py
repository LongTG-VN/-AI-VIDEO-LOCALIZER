from app.models.project import Character, Project, RelationshipRule, SubtitleCue
from app.services.critic import build_critic_context


def test_build_critic_context_contains_expected_pronouns():
    project = Project(
        name="demo",
        source_video_path="demo.mp4",
        characters=[
            Character(id="c1", name="Tần Phù Chi", name_vi="Tần Phù Chi", name_zh="秦扶栀", speaker_ids=["speaker_1"]),
            Character(id="c2", name="Tống Tri Tuyết", name_vi="Tống Tri Tuyết", name_zh="宋知雪", speaker_ids=["speaker_3"]),
        ],
        relationships=[
            RelationshipRule(
                from_character_id="c1",
                to_character_id="c2",
                relationship="daughter_to_mother",
                vi_self="con",
                vi_other="mẹ",
            )
        ],
        cues=[
            SubtitleCue(
                id="cue_01",
                start=0.0,
                end=2.0,
                speaker_id="speaker_1",
                addressee_id="speaker_3",
                source_text="我妈宋知雪",
                translated_text="Mẹ con là Tống Tri Tuyết",
            )
        ],
    )

    ctx = build_critic_context(project, project.cues[0])
    assert ctx["cue_id"] == "cue_01"
    assert ctx["expected_vi_self"] == "con"
    assert ctx["expected_vi_target"] == "mẹ"
    assert ctx["chinese_source"] == "我妈宋知雪"
    assert ctx["vietnamese_translation"] == "Mẹ con là Tống Tri Tuyết"
