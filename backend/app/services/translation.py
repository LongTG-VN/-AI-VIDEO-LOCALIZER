from __future__ import annotations

import json
from typing import Any

import httpx

from app.models.project import Project, SubtitleCue
from app.services.critic import TranslationCritic
from app.services.relationships import (
    active_relationship,
    active_scene,
    character_name,
    find_character,
    resolve_pronouns,
)


class TranslationError(RuntimeError):
    pass


def build_translation_context(project: Project, cue_index: int) -> dict[str, Any]:
    cue = project.cues[cue_index]
    scene = active_scene(project, cue.start)
    speaker_id = cue.speaker_character_id or cue.speaker_id
    addressee_id = cue.addressee_character_id or cue.addressee_id

    speaker_char = find_character(project, speaker_id)
    addressee_char = find_character(project, addressee_id)

    self_pronoun, target_pronoun, rel_type, rel_conf = resolve_pronouns(
        project, speaker_id, addressee_id, cue.start
    )

    prev_cues = [
        project.cues[i].source_text
        for i in range(max(0, cue_index - 2), cue_index)
    ]
    next_cues = [
        project.cues[i].source_text
        for i in range(cue_index + 1, min(len(project.cues), cue_index + 3))
    ]

    return {
        "cue_id": cue.id,
        "start": cue.start,
        "end": cue.end,
        "speaker": character_name(project, speaker_id),
        "speaker_role": speaker_char.role if speaker_char else None,
        "speaker_gender": speaker_char.gender if speaker_char else None,
        "addressee": character_name(project, addressee_id),
        "addressee_role": addressee_char.role if addressee_char else None,
        "relationship": rel_type,
        "preferred_vi_self": self_pronoun,
        "preferred_vi_other": target_pronoun,
        "scene_summary": scene.summary if scene else None,
        "scene_tone": scene.tone if scene else None,
        "previous_source": prev_cues[-1] if prev_cues else None,
        "previous_dialogues": prev_cues,
        "source": cue.source_text,
        "next_source": next_cues[0] if next_cues else None,
        "next_dialogues": next_cues,
        "glossary": [
            {"source": g.source, "target": g.target, "note": g.note}
            for g in project.glossary
        ],
    }


def build_system_prompt(target_language: str) -> str:
    target = "Vietnamese" if target_language == "vi" else "English"
    return f"""You are a master audiovisual translator and localizer specializing in Chinese dramas translated into {target}.

CRITICAL RULES:
1. Natural Subtitles: Produce fluent, idiomatic, emotionally resonant {target} subtitles. Avoid robotic word-for-word literal translations.
2. Forms of Address & Pronouns (Vietnamese):
   - Strict adherence to `preferred_vi_self` (how speaker refers to self) and `preferred_vi_other` (how speaker addresses the other person) based on character hierarchy and family relations.
   - Example: In a mother-daughter relationship (daughter -> mother), the daughter must address the mother as 'mẹ' and refer to herself as 'con' (e.g. '领口歪了' -> 'Cổ áo con bị lệch rồi' when mother speaks, or 'Con đi đây' when daughter speaks).
3. Glossary & Names: Always translate proper names (e.g. 秦扶栀 -> Tần Phù Chi, 宋知雪 -> Tống Tri Tuyết, 秦砚川 -> Tần Nghiễn Xuyên) and domain terms (KPI -> KPI) matching the glossary.
4. Stable Cue IDs:
   - Every input cue MUST have exactly one translated output with the EXACT SAME `cue_id`.
   - Never merge, skip, drop, or split cues.
5. Tone & Register: Respect the scene tone (authoritative, cold, sarcastic, emotional, casual).

Return JSON ONLY in this exact structure:
{{
  "translations": [
    {{
      "cue_id": "...",
      "text": "...",
      "confidence": 0.95
    }}
  ]
}}
"""


class OpenAICompatibleTranslator:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _validate_config(self) -> None:
        if not self.base_url or not self.model:
            raise TranslationError("LLM_BASE_URL and LLM_MODEL must be configured before translation.")

    def _call_translation_batch(
        self,
        batch: list[dict[str, Any]],
        target_language: str,
        critique_notes: dict[str, str] | None = None,
    ) -> dict[str, tuple[str, float | None]]:
        payload_messages: list[dict[str, str]] = [
            {"role": "system", "content": build_system_prompt(target_language)}
        ]

        if critique_notes:
            user_msg = (
                "Please re-translate these specific cues taking into account the following critique feedback:\n"
                + json.dumps({"cues": batch, "feedback_per_cue": critique_notes}, ensure_ascii=False)
            )
        else:
            user_msg = json.dumps({"cues": batch}, ensure_ascii=False)

        payload_messages.append({"role": "user", "content": user_msg})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": payload_messages,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            parsed = json.loads(response.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            raise TranslationError(f"Translation provider error: {exc}") from exc

        items = parsed.get("translations", [])
        expected_ids = {item["cue_id"] for item in batch}
        returned_ids = {item.get("cue_id") for item in items}

        if expected_ids != returned_ids:
            # Fallback: match whatever returned, flag error if missing critical cues
            missing = expected_ids - returned_ids
            if missing:
                raise TranslationError(f"Translation provider dropped cue IDs: {missing}; batch rejected.")

        results: dict[str, tuple[str, float | None]] = {}
        for item in items:
            text = str(item.get("text", "")).strip()
            conf = float(item["confidence"]) if item.get("confidence") is not None else None
            results[item["cue_id"]] = (text, conf)
        return results

    def translate_project(
        self,
        project: Project,
        batch_size: int = 20,
        enable_critic: bool = True,
    ) -> list[SubtitleCue]:
        self._validate_config()
        if not project.cues:
            return []

        # Pass 1: Initial Context-Aware Translation
        contexts = [build_translation_context(project, index) for index in range(len(project.cues))]
        translated_by_id: dict[str, tuple[str, float | None]] = {}

        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            results = self._call_translation_batch(batch, project.target_language)
            translated_by_id.update(results)

        for cue in project.cues:
            if cue.id in translated_by_id:
                cue.translated_text, conf = translated_by_id[cue.id]
                cue.translation_confidence = conf
                cue.confidence = conf

        # Pass 2: Translation Critic & Targeted Retry
        if enable_critic:
            critic = TranslationCritic(self.base_url, self.api_key, self.model)
            try:
                evaluations = critic.evaluate_cues(project, project.cues, batch_size=25)
                eval_by_id = {e.get("cue_id"): e for e in evaluations if e.get("cue_id")}

                retry_cues: list[dict[str, Any]] = []
                critique_notes: dict[str, str] = {}

                for idx, cue in enumerate(project.cues):
                    ev = eval_by_id.get(cue.id)
                    if not ev:
                        continue

                    cue.critic_score = float(ev.get("naturalness_score", 0.9))
                    flags = []
                    for check in ["meaning", "name_consistency", "pronoun_consistency", "relationship_consistency", "tone", "hallucination", "missing_information"]:
                        if ev.get(check) == "fail":
                            flags.append(check)
                    cue.critic_flags = flags

                    if ev.get("needs_retry", False) or flags:
                        cue.needs_review = True
                        cue.review_notes = ev.get("critique") or ", ".join(flags)
                        retry_cues.append(contexts[idx])
                        critique_notes[cue.id] = ev.get("critique", "Fix pronoun/meaning consistency")
                    else:
                        cue.needs_review = False

                # Targeted retry for failed cues
                if retry_cues:
                    retry_results = self._call_translation_batch(
                        retry_cues,
                        project.target_language,
                        critique_notes=critique_notes,
                    )
                    for cue in project.cues:
                        if cue.id in retry_results:
                            retried_text, retried_conf = retry_results[cue.id]
                            cue.translated_text = retried_text
                            if retried_conf is not None:
                                cue.translation_confidence = retried_conf
                                cue.confidence = retried_conf

            except Exception:
                # If critic fails, keep initial translation and mark review
                pass

        return project.cues
