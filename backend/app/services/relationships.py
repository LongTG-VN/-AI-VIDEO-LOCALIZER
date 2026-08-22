from app.models.project import Project, RelationshipRule


def active_relationship(
    project: Project,
    from_character_id: str | None,
    to_character_id: str | None,
    timestamp: float,
) -> RelationshipRule | None:
    if not from_character_id or not to_character_id:
        return None

    candidates = [
        rule
        for rule in project.relationships
        if rule.from_character_id == from_character_id
        and rule.to_character_id == to_character_id
        and rule.valid_from <= timestamp
        and (rule.valid_until is None or timestamp < rule.valid_until)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda rule: rule.valid_from)


def character_name(project: Project, character_id: str | None) -> str | None:
    if not character_id:
        return None
    character = next((item for item in project.characters if item.id == character_id), None)
    return character.name if character else character_id
