import pytest
from app.models.project import Character, Project, RelationshipRule, SubtitleCue
from app.services.relationships import resolve_pronouns, character_name, find_character


@pytest.fixture
def sample_project():
    daughter = Character(
        id="char_qin_fuzhi",
        name="Tần Phù Chi",
        name_zh="秦扶栀",
        name_vi="Tần Phù Chi",
        role="daughter",
        speaker_ids=["speaker_1", "speaker_5"],
    )
    mother = Character(
        id="char_song_zhixue",
        name="Tống Tri Tuyết",
        name_zh="宋知雪",
        name_vi="Tống Tri Tuyết",
        role="mother",
        speaker_ids=["speaker_3", "speaker_7"],
    )
    father = Character(
        id="char_qin_yanchuan",
        name="Tần Nghiên Xuyên",
        name_zh="秦砚川",
        name_vi="Tần Nghiên Xuyên",
        role="father",
        speaker_ids=["speaker_4"],
    )
    brother = Character(
        id="char_qin_yize",
        name="Tần Diệc Trạch",
        name_zh="秦亦泽",
        name_vi="Tần Diệc Trạch",
        role="older brother",
        speaker_ids=["speaker_6"],
    )
    rival = Character(
        id="char_meng_jingchun",
        name="Mạnh Kinh Xuân",
        name_zh="孟惊春",
        name_vi="Mạnh Kinh Xuân",
        role="antagonist",
        speaker_ids=["speaker_8"],
    )

    relationships = [
        RelationshipRule(from_character_id="char_qin_fuzhi", to_character_id="char_song_zhixue", relationship="daughter_to_mother", vi_self="con", vi_other="mẹ"),
        RelationshipRule(from_character_id="char_song_zhixue", to_character_id="char_qin_fuzhi", relationship="mother_to_daughter", vi_self="mẹ", vi_other="con"),
        RelationshipRule(from_character_id="char_qin_fuzhi", to_character_id="char_qin_yanchuan", relationship="daughter_to_father", vi_self="con", vi_other="bố"),
        RelationshipRule(from_character_id="char_qin_yanchuan", to_character_id="char_qin_fuzhi", relationship="father_to_daughter", vi_self="bố", vi_other="con"),
        RelationshipRule(from_character_id="char_qin_fuzhi", to_character_id="char_qin_yize", relationship="sister_to_brother", vi_self="em", vi_other="anh"),
        RelationshipRule(from_character_id="char_qin_yize", to_character_id="char_qin_fuzhi", relationship="brother_to_sister", vi_self="anh", vi_other="em"),
        RelationshipRule(from_character_id="char_meng_jingchun", to_character_id="char_qin_fuzhi", relationship="rival_to_rival", vi_self="tôi", vi_other="cô"),
        RelationshipRule(from_character_id="char_qin_fuzhi", to_character_id="char_meng_jingchun", relationship="rival_to_rival", vi_self="tôi", vi_other="cô"),
    ]

    return Project(
        name="Regression Demo",
        source_video_path="demo.mp4",
        characters=[daughter, mother, father, brother, rival],
        relationships=relationships,
    )


REGRESSION_CASES = [
    # Monologues (no direct addressee -> self=None, fallback to "tôi")
    ("300多个宾客看着我", "char_qin_fuzhi", None, None, None),
    ("但我现在", "char_qin_fuzhi", None, None, None),
    ("我叫秦扶栀", "char_qin_fuzhi", None, None, None),
    ("今天是我十八岁成人礼", "char_qin_fuzhi", None, None, None),
    
    # Mother -> Daughter
    ("领口歪了", "char_song_zhixue", "char_qin_fuzhi", "mẹ", "con"),
    ("坐姿不对", "char_song_zhixue", "char_qin_fuzhi", "mẹ", "con"),
    ("笑的太假", "char_song_zhixue", "char_qin_fuzhi", "mẹ", "con"),
    ("你今天是去丢人还是去赴宴", "char_song_zhixue", "char_qin_fuzhi", "mẹ", "con"),

    # Father -> Daughter
    ("看完了吗", "char_qin_yanchuan", "char_qin_fuzhi", "bố", "con"),
    ("背一下第三章的结论", "char_qin_yanchuan", "char_qin_fuzhi", "bố", "con"),

    # Daughter -> Father
    ("看看了", "char_qin_fuzhi", "char_qin_yanchuan", "con", "bố"),

    # Brother -> Sister
    ("你的存在拉低了秦家的执行效率", "char_qin_yize", "char_qin_fuzhi", "anh", "em"),

    # Rival -> Heroine
    ("看清楚 秦扶栀", "char_meng_jingchun", "char_qin_fuzhi", "tôi", "cô"),
    ("你还吃的下去", "char_meng_jingchun", "char_qin_fuzhi", "tôi", "cô"),
    ("你享受了本该属于我的人生", "char_meng_jingchun", "char_qin_fuzhi", "tôi", "cô"),
]


@pytest.mark.parametrize("source_text,speaker_id,addressee_id,expected_self,expected_target", REGRESSION_CASES)
def test_pronoun_regression(sample_project, source_text, speaker_id, addressee_id, expected_self, expected_target):
    self_p, target_p, _, _ = resolve_pronouns(sample_project, speaker_id, addressee_id, timestamp=10.0)
    assert self_p == expected_self, f"Failed self pronoun for {source_text}: expected {expected_self}, got {self_p}"
    assert target_p == expected_target, f"Failed target pronoun for {source_text}: expected {expected_target}, got {target_p}"
