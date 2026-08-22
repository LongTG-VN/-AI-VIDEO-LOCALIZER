from __future__ import annotations

from app.models.project import Character, Project, RelationshipRule, Scene


def find_character(project: Project, identifier: str | None) -> Character | None:
    if not identifier:
        return None
    # Check direct ID match first
    for char in project.characters:
        if char.id == identifier:
            return char
    # Check speaker_ids list
    for char in project.characters:
        if identifier in char.speaker_ids:
            return char
    # Check aliases
    for char in project.characters:
        if identifier in char.aliases:
            return char
    # Check names
    for char in project.characters:
        if identifier in (char.name, char.name_zh, char.name_vi):
            return char
    return None


def character_name(project: Project, identifier: str | None) -> str | None:
    if not identifier:
        return None
    char = find_character(project, identifier)
    if char:
        if char.name_vi and char.name_zh:
            return f"{char.name_vi} ({char.name_zh})"
        return char.name_vi or char.name_zh or char.name
    return identifier


def active_relationship(
    project: Project,
    from_id: str | None,
    to_id: str | None,
    timestamp: float,
) -> RelationshipRule | None:
    if not from_id or not to_id:
        return None

    # Canonicalize IDs if possible
    from_char = find_character(project, from_id)
    to_char = find_character(project, to_id)
    from_key = from_char.id if from_char else from_id
    to_key = to_char.id if to_char else to_id

    candidates = [
        rule
        for rule in project.relationships
        if (rule.from_character_id in (from_key, from_id) or (from_char and rule.from_character_id in from_char.speaker_ids))
        and (rule.to_character_id in (to_key, to_id) or (to_char and rule.to_character_id in to_char.speaker_ids))
        and rule.valid_from <= timestamp
        and (rule.valid_until is None or timestamp < rule.valid_until)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda rule: rule.valid_from)


def resolve_pronouns(
    project: Project,
    from_id: str | None,
    to_id: str | None,
    timestamp: float,
) -> tuple[str | None, str | None, str | None, float | None]:
    rule = active_relationship(project, from_id, to_id, timestamp)
    if not rule:
        return None, None, None, None
    self_pronoun = rule.vi_self_pronoun or rule.vi_self
    target_pronoun = rule.vi_target_pronoun or rule.vi_other
    rel_type = rule.relationship_type or rule.relationship
    return self_pronoun, target_pronoun, rel_type, rule.confidence


def active_scene(project: Project, timestamp: float) -> Scene | None:
    for scene in project.scenes:
        if scene.start <= timestamp < scene.end:
            return scene
    return None

