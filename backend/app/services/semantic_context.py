from __future__ import annotations

from typing import Any

from app.models.project import Project, SubtitleCue


VALID_DISCOURSE_MODES = {
    "direct_dialogue",
    "monologue",
    "narration",
    "system",
    "unknown",
}


def normalize_discourse_mode(value: str | None) -> str:
    """Return a stable discourse-mode value while remaining backward compatible."""
    normalized = (value or "unknown").strip().lower()
    return normalized if normalized in VALID_DISCOURSE_MODES else "unknown"


def cue_snapshot(cue: SubtitleCue) -> dict[str, Any]:
    """Small context-only representation of a cue for neighboring-window prompts."""
    return {
        "cue_id": cue.id,
        "start": float(cue.start),
        "end": float(cue.end),
        "source": cue.source_text,
        "speaker_id": cue.speaker_id,
        "speaker_character_id": cue.speaker_character_id,
        "addressee_id": cue.addressee_id,
        "addressee_character_id": cue.addressee_character_id,
        "discourse_mode": normalize_discourse_mode(getattr(cue, "discourse_mode", None)),
    }


def build_neighbor_window(
    project: Project,
    cue_index: int,
    *,
    before: int = 3,
    after: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Build a VideoLingo-style neighboring context window without changing cue ownership.

    Neighbor text is context only. The translator must still emit one result for the current
    cue and must not move entities or clauses between cue IDs.
    """
    start = max(0, cue_index - max(0, before))
    end = min(len(project.cues), cue_index + max(0, after) + 1)
    previous = [cue_snapshot(project.cues[i]) for i in range(start, cue_index)]
    following = [cue_snapshot(project.cues[i]) for i in range(cue_index + 1, end)]
    return {"previous": previous, "next": following}


def source_name_mentions(project: Project, source_text: str) -> list[dict[str, Any]]:
    """Return canonical character-name locks explicitly owned by this source cue.

    Only names actually present in the current source cue are returned. This prevents an
    explicit vocative from migrating into a neighboring subtitle during contextual translation.
    """
    source = source_text or ""
    mentions: list[dict[str, Any]] = []
    seen_targets: set[str] = set()

    for char in project.characters:
        source_forms = [char.name_zh, *char.aliases]
        matched_forms = sorted(
            {
                form
                for form in source_forms
                if form and len(form.strip()) >= 2 and form in source
            },
            key=len,
            reverse=True,
        )
        target = (char.name_vi or char.name or "").strip()
        if not matched_forms or not target or target in seen_targets:
            continue
        mentions.append(
            {
                "character_id": char.id,
                "source_forms": matched_forms,
                "target": target,
            }
        )
        seen_targets.add(target)

    # A name glossary entry may exist before/without a fully resolved Character object.
    for entry in project.glossary:
        category = (entry.category or "").strip().lower()
        if category not in {"name", "character", "person", "proper_name"}:
            continue
        if not entry.source or entry.source not in source or not entry.target:
            continue
        target = entry.target.strip()
        if target in seen_targets:
            continue
        mentions.append(
            {
                "character_id": None,
                "source_forms": [entry.source],
                "target": target,
            }
        )
        seen_targets.add(target)

    return mentions


def project_name_locks(project: Project) -> list[dict[str, Any]]:
    """Generate dynamic project-level name locks; no title-specific names live in code."""
    locks: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for char in project.characters:
        target = (char.name_vi or char.name or "").strip()
        if not target:
            continue
        for source in [char.name_zh, *char.aliases]:
            if not source or len(source.strip()) < 2:
                continue
            key = (source, target)
            if key in seen:
                continue
            locks.append({"source": source, "target": target, "character_id": char.id})
            seen.add(key)

    for entry in project.glossary:
        if not entry.source or not entry.target:
            continue
        category = (entry.category or "").strip().lower()
        if category not in {"name", "character", "person", "proper_name"}:
            continue
        key = (entry.source, entry.target)
        if key in seen:
            continue
        locks.append({"source": entry.source, "target": entry.target, "character_id": None})
        seen.add(key)

    return locks
