from __future__ import annotations

import json

import httpx

from app.models.project import Project, SubtitleCue
from app.services.relationships import active_relationship, character_name


class TranslationError(RuntimeError):
    pass


def build_translation_context(project: Project, cue_index: int) -> dict:
    cue = project.cues[cue_index]
    relationship = active_relationship(project, cue.speaker_id, cue.addressee_id, cue.start)
    previous_text = project.cues[cue_index - 1].source_text if cue_index > 0 else None
    next_text = project.cues[cue_index + 1].source_text if cue_index + 1 < len(project.cues) else None
    return {
        "cue_id": cue.id,
        "speaker": character_name(project, cue.speaker_id),
        "addressee": character_name(project, cue.addressee_id),
        "relationship": relationship.relationship if relationship else None,
        "preferred_vi_self": relationship.vi_self if relationship else None,
        "preferred_vi_other": relationship.vi_other if relationship else None,
        "english_register": relationship.en_register if relationship else None,
        "previous_source": previous_text,
        "source": cue.source_text,
        "next_source": next_text,
        "glossary": [entry.model_dump() for entry in project.glossary],
    }


def build_system_prompt(target_language: str) -> str:
    target = "Vietnamese" if target_language == "vi" else "English"
    return f"""You are a professional Chinese audiovisual translator.
Translate each subtitle into natural {target}, preserving meaning, tone, names, status, and relationships.
Never merge or split cue IDs. Never invent dialogue.
For Vietnamese, pronouns and forms of address are semantically important: obey preferred_vi_self and preferred_vi_other when provided, while still producing natural dialogue.
For English, use the requested social register when provided.
Return JSON only in the exact shape: {{\"translations\":[{{\"cue_id\":\"...\",\"text\":\"...\",\"confidence\":0.0}}]}}.
Confidence must be between 0 and 1 and should be lower when source/context is ambiguous."""


class OpenAICompatibleTranslator:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _validate_config(self) -> None:
        if not self.base_url or not self.model:
            raise TranslationError("LLM_BASE_URL and LLM_MODEL must be configured before translation.")

    def translate_project(self, project: Project, batch_size: int = 20) -> list[SubtitleCue]:
        self._validate_config()
        translated_by_id: dict[str, tuple[str, float | None]] = {}
        contexts = [build_translation_context(project, index) for index in range(len(project.cues))]
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            payload = {"model": self.model, "temperature": 0.2, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": build_system_prompt(project.target_language)}, {"role": "user", "content": json.dumps({"cues": batch}, ensure_ascii=False)}]}
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            try:
                response = httpx.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                raise TranslationError(f"Translation provider error: {exc}") from exc
            items = parsed.get("translations", [])
            expected_ids = {item["cue_id"] for item in batch}
            returned_ids = {item.get("cue_id") for item in items}
            if expected_ids != returned_ids:
                raise TranslationError("Translation provider changed/dropped cue IDs; batch rejected to protect subtitle timing.")
            for item in items:
                translated_by_id[item["cue_id"]] = (str(item["text"]).strip(), float(item["confidence"]) if item.get("confidence") is not None else None)
        for cue in project.cues:
            if cue.id in translated_by_id:
                cue.translated_text, provider_confidence = translated_by_id[cue.id]
                if provider_confidence is not None:
                    cue.confidence = provider_confidence
        return project.cues
