from app.models.project import Character, Project, RelationshipRule
from app.services.relationships import active_relationship


def test_relationship_changes_over_time():
    a = Character(id="a", name="林晚晚")
    b = Character(id="b", name="顾霆琛")
    project = Project(name="demo", source_video_path="demo.mp4", characters=[a,b], relationships=[RelationshipRule(from_character_id="a", to_character_id="b", relationship="employee_to_boss", valid_from=0, valid_until=100, vi_self="em", vi_other="sếp"), RelationshipRule(from_character_id="a", to_character_id="b", relationship="dating", valid_from=100, vi_self="em", vi_other="anh")])
    assert active_relationship(project, "a", "b", 50).vi_other == "sếp"
    assert active_relationship(project, "a", "b", 120).vi_other == "anh"
