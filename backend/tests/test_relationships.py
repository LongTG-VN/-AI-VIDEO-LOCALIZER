from app.models.project import Character, Project, RelationshipRule, Scene
from app.services.relationships import (
    active_relationship,
    active_scene,
    character_name,
    find_character,
    resolve_pronouns,
)


def test_directional_relationship_resolution():
    char_daughter = Character(
        id="char_qin_fuzhi",
        name="Tần Phù Chi",
        name_zh="秦扶栀",
        name_vi="Tần Phù Chi",
        role="daughter",
        speaker_ids=["speaker_1", "speaker_7"],
    )
    char_mother = Character(
        id="char_song_zhixue",
        name="Tống Tri Tuyết",
        name_zh="宋知雪",
        name_vi="Tống Tri Tuyết",
        role="mother",
        speaker_ids=["speaker_3"],
    )

    rule_d2m = RelationshipRule(
        from_character_id="char_qin_fuzhi",
        to_character_id="char_song_zhixue",
        relationship="daughter_to_mother",
        vi_self="con",
        vi_other="mẹ",
        confidence=0.98,
    )
    rule_m2d = RelationshipRule(
        from_character_id="char_song_zhixue",
        to_character_id="char_qin_fuzhi",
        relationship="mother_to_daughter",
        vi_self="mẹ",
        vi_other="con",
        confidence=0.98,
    )

    project = Project(
        name="demo",
        source_video_path="demo.mp4",
        characters=[char_daughter, char_mother],
        relationships=[rule_d2m, rule_m2d],
    )

    # Daughter to mother
    self_p, target_p, rel_type, conf = resolve_pronouns(
        project, "speaker_1", "speaker_3", timestamp=10.0
    )
    assert self_p == "con"
    assert target_p == "mẹ"
    assert rel_type == "daughter_to_mother"

    # Mother to daughter
    self_p_m, target_p_m, rel_type_m, _ = resolve_pronouns(
        project, "speaker_3", "speaker_7", timestamp=10.0
    )
    assert self_p_m == "mẹ"
    assert target_p_m == "con"
    assert rel_type_m == "mother_to_daughter"


def test_speaker_to_character_mapping():
    char = Character(
        id="char_01",
        name="Tần Phù Chi",
        name_zh="秦扶栀",
        name_vi="Tần Phù Chi",
        speaker_ids=["speaker_1", "speaker_7"],
    )
    project = Project(name="demo", source_video_path="demo.mp4", characters=[char])

    assert find_character(project, "char_01") == char
    assert find_character(project, "speaker_1") == char
    assert find_character(project, "speaker_7") == char
    assert find_character(project, "秦扶栀") == char
    assert character_name(project, "speaker_1") == "Tần Phù Chi (秦扶栀)"


def test_temporal_relationship_selection():
    a = Character(id="a", name="A")
    b = Character(id="b", name="B")
    rule1 = RelationshipRule(
        from_character_id="a",
        to_character_id="b",
        relationship="stranger",
        valid_from=0,
        valid_until=100,
        vi_self="tôi",
        vi_other="cô",
    )
    rule2 = RelationshipRule(
        from_character_id="a",
        to_character_id="b",
        relationship="romantic",
        valid_from=100,
        vi_self="anh",
        vi_other="em",
    )
    project = Project(
        name="demo",
        source_video_path="demo.mp4",
        characters=[a, b],
        relationships=[rule1, rule2],
    )

    assert active_relationship(project, "a", "b", 50).vi_other == "cô"
    assert active_relationship(project, "a", "b", 120).vi_other == "em"


def test_missing_or_uncertain_relationship_fallback():
    project = Project(name="demo", source_video_path="demo.mp4")
    self_p, target_p, rel_type, conf = resolve_pronouns(
        project, "unknown_speaker", "another_unknown", timestamp=10.0
    )
    assert self_p is None
    assert target_p is None
    assert rel_type is None
    assert conf is None


def test_active_scene_lookup():
    scene1 = Scene(id="s1", start=0.0, end=20.0, summary="Scene 1 intro")
    scene2 = Scene(id="s2", start=20.0, end=50.0, summary="Scene 2 confrontation")
    project = Project(name="demo", source_video_path="demo.mp4", scenes=[scene1, scene2])

    assert active_scene(project, 10.0).summary == "Scene 1 intro"
    assert active_scene(project, 35.0).summary == "Scene 2 confrontation"
    assert active_scene(project, 99.0) is None

