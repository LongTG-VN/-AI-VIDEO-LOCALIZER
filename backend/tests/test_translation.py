from app.models.project import Character, Project, RelationshipRule, SubtitleCue
from app.services.translation import build_translation_context


def test_translation_context_contains_pronouns_and_neighbors():
    project = Project(name="demo", source_video_path="demo.mp4", characters=[Character(id="a", name="林晚晚"), Character(id="b", name="顾霆琛")], relationships=[RelationshipRule(from_character_id="a", to_character_id="b", relationship="dating", vi_self="em", vi_other="anh")], cues=[SubtitleCue(start=0,end=1,source_text="前一句"), SubtitleCue(start=2,end=3,source_text="关你什么事？",speaker_id="a",addressee_id="b"), SubtitleCue(start=4,end=5,source_text="后一句")])
    context = build_translation_context(project, 1)
    assert context["speaker"] == "林晚晚"
    assert context["addressee"] == "顾霆琛"
    assert context["preferred_vi_self"] == "em"
    assert context["preferred_vi_other"] == "anh"
    assert context["previous_source"] == "前一句"
    assert context["next_source"] == "后一句"
