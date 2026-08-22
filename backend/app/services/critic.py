from __future__ import annotations

import json
from typing import Any

import httpx

from app.models.project import Project, SubtitleCue
from app.services.relationships import (
    active_relationship,
    active_scene,
    character_name,
    find_character,
    resolve_pronouns,
)


class CriticError(RuntimeError):
    pass


def build_critic_context(project: Project, cue: SubtitleCue) -> dict[str, Any]:
    scene = active_scene(project, cue.start)
    speaker_char = find_character(project, cue.speaker_character_id or cue.speaker_id)
    addressee_char = find_character(project, cue.addressee_character_id or cue.addressee_id)
    self_pronoun, target_pronoun, rel_type, rel_conf = resolve_pronouns(
        project,
        cue.speaker_character_id or cue.speaker_id,
        cue.addressee_character_id or cue.addressee_id,
        cue.start,
    )
    return {
        "cue_id": cue.id,
        "start": cue.start,
        "end": cue.end,
        "speaker": character_name(project, cue.speaker_character_id or cue.speaker_id),
        "addressee": character_name(project, cue.addressee_character_id or cue.addressee_id),
        "relationship": rel_type,
        "expected_vi_self": self_pronoun,
        "expected_vi_target": target_pronoun,
        "scene_summary": scene.summary if scene else None,
        "chinese_source": cue.source_text,
        "vietnamese_translation": cue.translated_text,
    }


_CRITIC_SYSTEM = """You are a rigorous Vietnamese subtitle quality critic and validation engine.
Evaluate whether the translated Vietnamese subtitle faithfully and naturally translates the Chinese source within the drama context and adheres strictly to the expected Vietnamese pronouns.

For each cue, verify:
1. meaning: 'pass' if accurate, 'fail' if distorted.
2. name_consistency: 'pass' if character/proper names match glossary/context, 'fail' otherwise.
3. pronoun_consistency: 'pass' if the Vietnamese addressing matches expected_vi_self and expected_vi_target, 'fail' if wrong pronouns used (e.g. using tôi/cô when mother-daughter con/mẹ is expected).
4. relationship_consistency: 'pass' or 'fail'.
5. tone: 'pass' or 'fail'.
6. hallucination: 'pass' (no made up info) or 'fail'.
7. missing_information: 'pass' (no dropped key dialogue) or 'fail'.
8. naturalness_score: float 0.0 to 1.0.
9. needs_retry: true if meaning, name_consistency, or pronoun_consistency fails; false otherwise.
10. critique: brief feedback explaining what needs fixing if failed, or empty if pass.
11. suggested_fix: improved Vietnamese subtitle if failed, or null if pass.

Return JSON ONLY in this exact shape:
{
  "evaluations": [
    {
      "cue_id": "...",
      "meaning": "pass",
      "name_consistency": "pass",
      "pronoun_consistency": "pass",
      "relationship_consistency": "pass",
      "tone": "pass",
      "hallucination": "pass",
      "missing_information": "pass",
      "naturalness_score": 0.95,
      "needs_retry": false,
      "critique": "",
      "suggested_fix": null
    }
  ]
}
"""


class TranslationCritic:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def evaluate_cues(self, project: Project, cues: list[SubtitleCue], batch_size: int = 25) -> list[dict[str, Any]]:
        if not self.base_url or not self.model:
            raise CriticError("LLM_BASE_URL and LLM_MODEL must be configured for critic.")

        evaluations: list[dict[str, Any]] = []
        contexts = [build_critic_context(project, cue) for cue in cues]

        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            payload = {
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": _CRITIC_SYSTEM},
                    {"role": "user", "content": json.dumps({"cues": batch}, ensure_ascii=False)},
                ],
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            parsed = None
            for attempt in range(8):
                try:
                    response = httpx.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )
                    if response.status_code == 429:
                        import time
                        wait = 6 * (attempt + 1)
                        print(f"Critic rate limited (429), sleeping {wait}s (attempt {attempt+1}/8)...")
                        time.sleep(wait)
                        continue
                    response.raise_for_status()
                    raw = response.json()["choices"][0]["message"]["content"].strip()
                    if raw.startswith("```"):
                        lines = raw.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        raw = "\n".join(lines).strip()
                    parsed = json.loads(raw)
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 429 and attempt < 7:
                        import time
                        wait = 6 * (attempt + 1)
                        print(f"Critic rate limited (429), sleeping {wait}s (attempt {attempt+1}/8)...")
                        time.sleep(wait)
                        continue
                    raise CriticError(f"Critic evaluation error: {exc}") from exc
                except Exception as exc:
                    if attempt < 7:
                        import time
                        time.sleep(3)
                        continue
                    raise CriticError(f"Critic evaluation error: {exc}") from exc
            if parsed is not None:
                items = parsed.get("evaluations", [])
                evaluations.extend(items)

        return evaluations
