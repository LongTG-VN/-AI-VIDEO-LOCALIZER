from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx

from app.models.project import Character, GlossaryEntry, Project, RelationshipRule, Scene
from app.services.semantic_context import normalize_discourse_mode


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
            "existing_discourse_mode": normalize_discourse_mode(getattr(cue, "discourse_mode", None)),
        }
        for cue in project.cues
    ]
    return {
        "source_language": project.source_language,
        "target_language": project.target_language,
        "known_speaker_ids": speakers,
        "cues": cues,
    }


_CONTEXT_SYSTEM = """You are an expert Chinese-to-Vietnamese audiovisual context & character analyst.
You analyze the Chinese dialogue transcript to extract characters, consolidate raw speaker IDs, infer directional social relationships with exact Vietnamese pronouns, segment into scenes, extract glossary terms, determine addressees, and classify discourse mode for each cue.

CRITICAL INSTRUCTIONS:
1. Speaker Consolidation & Character Extraction:
   - ASR diarization may produce multiple raw speaker IDs for the same actual character.
   - Group raw speaker_ids into canonical characters. Keep all raw speaker_ids listed under `speaker_ids`.
   - Provide `name_zh`, standard Vietnamese/Han-Viet `name_vi`, `gender`, `role`, and `description` when supported by evidence.
   - Do not invent a named character if the transcript only supports an unknown speaker.

2. Directional Relationship Graph:
   - Relationships MUST be directional because A->B and B->A can use different Vietnamese pronouns.
   - For Vietnamese (`target_language=vi`), assign:
     - `vi_self_pronoun`: how character A refers to themselves when speaking to B.
     - `vi_target_pronoun`: how character A addresses B.
   - Specify `relationship_type`, `valid_from`, and `valid_until`.

3. Scene Segmentation:
   - Partition dialogue into logical scenes with `start`, `end`, `summary`, `tone`, and `characters`.

4. Glossary:
   - Extract character names, organizations, recurring proper nouns, acronyms, and domain terms with stable target forms.

5. Addressee Inference:
   - For each cue, infer `addressee_character_id` when someone is directly addressed.
   - If spoken to self/audience, narration, or no concrete person, use null.
   - Set `needs_review: true` when speaker/addressee evidence is materially ambiguous.

6. Discourse Mode:
   - Classify every cue as exactly one of:
     `direct_dialogue`, `monologue`, `narration`, `system`, `unknown`.
   - `direct_dialogue`: spoken to another character.
   - `monologue`: inner thought / self-talk by a character.
   - `narration`: exposition, memories, backstory, or voice-over describing events rather than addressing a present listener.
   - `system`: alarms, UI/system voice, announcements not belonging to normal character conversation.
   - `unknown`: only when evidence is insufficient.
   - A single character may switch between dialogue, monologue, and narration; speaker identity alone must NOT collapse these modes.

Output JSON ONLY with this exact structure:
{
  "characters": [
    {
      "id": "char_a",
      "name_zh": "林晚晚",
      "name_vi": "Lâm Vãn Vãn",
      "aliases": [],
      "gender": "female",
      "role": "lead",
      "description": "...",
      "speaker_ids": ["speaker_1"],
      "confidence": 0.95
    }
  ],
  "relationships": [
    {
      "from_character_id": "char_a",
      "to_character_id": "char_b",
      "relationship": "younger_sister_to_older_brother",
      "relationship_type": "younger_sister_to_older_brother",
      "valid_from": 0,
      "valid_until": null,
      "vi_self_pronoun": "em",
      "vi_target_pronoun": "anh",
      "confidence": 0.98,
      "notes": "..."
    }
  ],
  "scenes": [
    {
      "scene_id": "scene_01",
      "start": 0.0,
      "end": 20.0,
      "summary": "...",
      "tone": "tense",
      "characters": ["char_a", "char_b"]
    }
  ],
  "glossary": [
    {
      "source": "林晚晚",
      "target": "Lâm Vãn Vãn",
      "category": "name",
      "confidence": 1.0,
      "note": "character name"
    }
  ],
  "cues": [
    {
      "cue_id": "...",
      "speaker_character_id": "char_a",
      "addressee_character_id": "char_b",
      "discourse_mode": "direct_dialogue",
      "needs_review": false
    }
  ]
}
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
            "messages": [
                {"role": "system", "content": _CONTEXT_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }

        result = None
        for attempt in range(8):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=180,
                )
                if response.status_code == 429:
                    import time
                    wait = 6 * (attempt + 1)
                    print(f"Rate limited (429), sleeping {wait}s (attempt {attempt+1}/8)...")
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
                result = json.loads(raw)
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < 7:
                    import time
                    wait = 6 * (attempt + 1)
                    print(f"Rate limited (429), sleeping {wait}s (attempt {attempt+1}/8)...")
                    time.sleep(wait)
                    continue
                raise ContextAnalysisError(f"Context provider error: {exc} -> {exc.response.text}") from exc
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                if attempt < 7:
                    import time
                    time.sleep(3)
                    continue
                raise ContextAnalysisError(f"Context provider error: {exc}") from exc
        if result is None:
            raise ContextAnalysisError("Context provider error: Failed after retries.")

        # 1. Process Characters
        characters: list[Character] = []
        for item in result.get("characters", []):
            cid = item.get("id") or str(uuid4())
            name_vi = item.get("name_vi")
            name_zh = item.get("name_zh")
            name = name_vi or name_zh or item.get("name") or cid
            char = Character(
                id=cid,
                name=name,
                name_zh=name_zh,
                name_vi=name_vi,
                aliases=item.get("aliases", []),
                gender=item.get("gender"),
                role=item.get("role"),
                description=item.get("description"),
                speaker_ids=item.get("speaker_ids", []),
                confidence=item.get("confidence", 0.9),
                notes=item.get("notes"),
            )
            characters.append(char)

        # Fallback for unmapped speaker IDs
        mapped_speakers = set()
        for char in characters:
            mapped_speakers.update(char.speaker_ids)
            mapped_speakers.add(char.id)

        for speaker_id in payload["known_speaker_ids"]:
            if speaker_id not in mapped_speakers:
                fallback_char = Character(
                    id=f"char_{speaker_id}",
                    name=f"Speaker {speaker_id}",
                    speaker_ids=[speaker_id],
                    role="speaker",
                    confidence=0.5,
                )
                characters.append(fallback_char)

        char_id_set = {c.id for c in characters}

        def resolve_cid(key: str | None) -> str | None:
            if not key:
                return None
            if key in char_id_set:
                return key
            for c in characters:
                if key in c.speaker_ids or key == c.name or key == c.name_zh:
                    return c.id
            return key

        # 2. Process Relationships
        relationships: list[RelationshipRule] = []
        for item in result.get("relationships", []):
            from_cid = resolve_cid(item.get("from_character_id"))
            to_cid = resolve_cid(item.get("to_character_id"))
            if not from_cid or not to_cid or from_cid == to_cid:
                continue
            rule = RelationshipRule(
                id=item.get("id") or str(uuid4()),
                from_character_id=from_cid,
                to_character_id=to_cid,
                relationship=item.get("relationship", "social"),
                relationship_type=item.get("relationship_type"),
                valid_from=float(item.get("valid_from", 0.0)),
                valid_until=float(item.get("valid_until")) if item.get("valid_until") is not None else None,
                vi_self=item.get("vi_self_pronoun") or item.get("vi_self"),
                vi_other=item.get("vi_target_pronoun") or item.get("vi_other"),
                vi_self_pronoun=item.get("vi_self_pronoun") or item.get("vi_self"),
                vi_target_pronoun=item.get("vi_target_pronoun") or item.get("vi_other"),
                en_register=item.get("en_register"),
                confidence=item.get("confidence", 0.9),
                notes=item.get("notes"),
            )
            relationships.append(rule)

        # 3. Process Scenes
        scenes: list[Scene] = []
        for item in result.get("scenes", []):
            scene = Scene(
                id=item.get("id") or str(uuid4()),
                scene_id=item.get("scene_id"),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", project.duration or 0.0)),
                summary=item.get("summary", ""),
                tone=item.get("tone"),
                characters=[resolve_cid(cid) or cid for cid in item.get("characters", [])],
            )
            scenes.append(scene)

        # 4. Process Glossary
        glossary: list[GlossaryEntry] = []
        for item in result.get("glossary", []):
            entry = GlossaryEntry(
                id=item.get("id") or str(uuid4()),
                source=item.get("source", ""),
                target=item.get("target", ""),
                category=item.get("category"),
                confidence=item.get("confidence", 1.0),
                note=item.get("note"),
            )
            glossary.append(entry)

        # 5. Process Cue Mappings
        cue_map = {item.get("cue_id"): item for item in result.get("cues", []) if item.get("cue_id")}
        for cue in project.cues:
            mapped = cue_map.get(cue.id)
            if mapped:
                speaker_cid = resolve_cid(mapped.get("speaker_character_id"))
                addressee_cid = resolve_cid(mapped.get("addressee_character_id"))
                cue.speaker_character_id = speaker_cid or resolve_cid(cue.speaker_id)
                cue.addressee_id = addressee_cid
                cue.addressee_character_id = addressee_cid
                cue.discourse_mode = normalize_discourse_mode(mapped.get("discourse_mode"))
                cue.needs_review = bool(mapped.get("needs_review", False))
            else:
                cue.speaker_character_id = resolve_cid(cue.speaker_id)
                cue.discourse_mode = normalize_discourse_mode(getattr(cue, "discourse_mode", None))

        project.characters = characters
        project.relationships = relationships
        project.scenes = scenes
        project.glossary = glossary
        return project
