from __future__ import annotations

import json
from typing import Any

import httpx

from app.models.project import Character, Project, RelationshipRule


class ContextAnalysisError(RuntimeError):
    pass


def build_context_analysis_payload(project: Project) -> dict[str, Any]:
    speakers = sorted({cue.speaker_id for cue in project.cues if cue.speaker_id})
    cues = [
        {
            "cue_id": cue.id,
            "start": cue.start,
            "end": cue.end,
            "speaker_id": cue.speaker_id,
            "text": cue.source_text,
        }
        for cue in project.cues
    ]
    return {
        "source_language": project.source_language,
        "target_language": project.target_language,
        "known_speaker_ids": speakers,
        "cues": cues,
    }


_CONTEXT_SYSTEM = """You analyze Chinese drama/video dialogue for localization.
Use ONLY speaker IDs supplied in known_speaker_ids. Do not invent speaker IDs.
Infer character names only when supported by dialogue; otherwise use a neutral display name such as Speaker 1.
Infer dialogue addressee when reasonably clear from turn-taking/context.
Create directional relationship rules used for Vietnamese forms of address. A->B and B->A may differ.
Do not infer sensitive personal attributes. Gender can be null when unclear.
Return JSON only with this exact top-level shape:
{
  "characters": [{"id":"speaker_0","name":"...","aliases":[],"gender":null,"role":null,"notes":null}],
  "relationships": [{"from_character_id":"speaker_0","to_character_id":"speaker_1","relationship":"...","valid_from":0,"valid_until":null,"vi_self":"...","vi_other":"...","en_register":"neutral","notes":null}],
  "addressees": [{"cue_id":"...","addressee_id":"speaker_1"}]
}
For Vietnamese, choose natural pronouns/forms of address based on status, age cues and relationship. If uncertain, use conservative neutral choices and explain uncertainty in notes.
"""


class ContextAnalyzer:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def analyze(self, project: Project) -> Project:
        if not self.base_url or not self.model:
            raise ContextAnalysisError(
                "LLM_BASE_URL and LLM_MODEL must be configured before context analysis."
            )
        if not project.cues:
            raise ContextAnalysisError("Project has no cues to analyze.")
        payload = build_context_analysis_payload(project)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _CONTEXT_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=180,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            result = json.loads(raw)
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            raise ContextAnalysisError(f"Context provider error: {exc}") from exc

        allowed = set(payload["known_speaker_ids"])
        characters: list[Character] = []
        for item in result.get("characters", []):
            if item.get("id") not in allowed:
                continue
            characters.append(Character.model_validate(item))

        existing = {character.id for character in characters}
        for index, speaker_id in enumerate(payload["known_speaker_ids"], start=1):
            if speaker_id not in existing:
                characters.append(Character(id=speaker_id, name=f"Speaker {index}"))

        relationships: list[RelationshipRule] = []
        for item in result.get("relationships", []):
            if item.get("from_character_id") not in allowed or item.get("to_character_id") not in allowed:
                continue
            if item.get("from_character_id") == item.get("to_character_id"):
                continue
            relationships.append(RelationshipRule.model_validate(item))

        addressee_map = {
            item.get("cue_id"): item.get("addressee_id")
            for item in result.get("addressees", [])
            if item.get("addressee_id") in allowed
        }
        for cue in project.cues:
            if cue.id in addressee_map:
                cue.addressee_id = addressee_map[cue.id]

        project.characters = characters
        project.relationships = relationships
        return project
