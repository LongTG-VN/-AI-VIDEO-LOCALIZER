from __future__ import annotations

import json
import logging
import re
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


logger = logging.getLogger(__name__)


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
        "addressee_gender": addressee_char.gender if addressee_char else None,
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
   - Example: In family sibling dialogue (brother -> sister), use 'anh / em' (e.g. '你的存在...' -> 'Sự tồn tại của em...'). Never use 'mày / tao' unless explicitly hostile.
   - Example: In mother-daughter dialogue, use 'mẹ / con'.
   - In monologue / narration (when addressee is None), use 'tôi' or natural neutral phrasing.
3. Glossary & Names: Always translate proper names and domain terms matching the glossary.
4. Gender & Reference: Preserve female referent (她 -> cô ấy/mẹ) and male referent (他 -> anh ấy/bố). Never turn female referents into 'ông ta'.
5. Clause Completeness: Translate all meaningful clauses (e.g. contrast pairs 'không có... chỉ có...'). Never drop clauses.
6. Faithful & Hallucination-Free: Do not add unsupported content (e.g. do not add 'cố lên' to '看清楚').
7. Stable Cue IDs: Every input cue MUST have exactly one translated output with the EXACT SAME `cue_id`.

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
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model
        self.last_metrics: dict[str, Any] = {}

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
            feedback_str = json.dumps({"cues": batch, "feedback_per_cue": critique_notes}, ensure_ascii=False)
            user_msg = f"Please re-translate these specific cues taking into account the following critique feedback:\n{feedback_str}"
        else:
            user_msg = json.dumps({"cues": batch}, ensure_ascii=False)

        payload_messages.append({"role": "user", "content": user_msg})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": payload_messages,
        }

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
                    print(f"Translation rate limited (429), sleeping {wait}s (attempt {attempt+1}/8)...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                if "```" in content:
                    lines = content.splitlines()
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    content = "\n".join(lines).strip()
                parsed = None
                start_idx = content.find("{")
                while start_idx != -1:
                    depth = 0
                    end_idx = start_idx
                    for i, ch in enumerate(content[start_idx:], start_idx):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                end_idx = i
                                break
                    candidate = content[start_idx : end_idx + 1]
                    try:
                        parsed = json.loads(candidate)
                        break
                    except json.JSONDecodeError:
                        start_idx = content.find("{", start_idx + 1)
                if parsed is None:
                    parsed = json.loads(content)
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    logger.warning("Translation rate-limited (429) after all retries, skipping batch.")
                    parsed = {"translations": []}
                    break
                raise TranslationError(f"Translation provider error: {exc}") from exc
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                if attempt < 7:
                    import time
                    time.sleep(4)
                    continue
                logger.warning("Failed to parse translation batch: %s", exc)
                parsed = {"translations": []}
                break

        if parsed is None:
            parsed = {"translations": []}

        items = parsed.get("translations", [])
        expected_ids = {item["cue_id"] for item in batch}

        results: dict[str, tuple[str, float | None]] = {}
        for item in items:
            if isinstance(item, dict) and item.get("cue_id") in expected_ids:
                text = str(item.get("text", "")).strip()
                conf = float(item["confidence"]) if item.get("confidence") is not None else None
                results[item["cue_id"]] = (text, conf)

        if len(results) < len(batch) and len(items) == len(batch):
            for b_item, res_item in zip(batch, items):
                if b_item["cue_id"] not in results and isinstance(res_item, dict):
                    text = str(res_item.get("text", "")).strip()
                    conf = float(res_item.get("confidence", 0.8)) if res_item.get("confidence") is not None else None
                    results[b_item["cue_id"]] = (text, conf)

        for b_item in batch:
            if b_item["cue_id"] not in results:
                results[b_item["cue_id"]] = (b_item.get("source", ""), 0.5)

        return results

    def translate_project(
        self,
        project: Project,
        batch_size: int = 12,
        enable_critic: bool = True,
        max_retries: int = 2,
    ) -> list[SubtitleCue]:
        self._validate_config()
        if not project.cues:
            return []

        # 1. Initial Translation Pass
        untranslated_indices = [i for i, c in enumerate(project.cues) if not c.translated_text]
        contexts_by_id = {c.id: build_translation_context(project, i) for i, c in enumerate(project.cues)}
        untranslated_contexts = [contexts_by_id[project.cues[i].id] for i in untranslated_indices]

        for start in range(0, len(untranslated_contexts), batch_size):
            batch = untranslated_contexts[start : start + batch_size]
            results = self._call_translation_batch(batch, project.target_language)
            for cue in project.cues:
                if cue.id in results:
                    cue.translated_text, conf = results[cue.id]
                    cue.translation_confidence = conf
                    cue.confidence = conf
            import time
            time.sleep(3.0)

        # 2. Critic & Targeted Retry Pass (Up to max_retries)
        metrics = {
            "translated_cues": len(project.cues),
            "critic_pass_first_try": 0,
            "critic_retry_1_pass": 0,
            "critic_retry_2_pass": 0,
            "critic_final_fail": 0,
            "meaning_shift_failures": 0,
            "pronoun_mismatch_failures": 0,
            "relationship_mismatch_failures": 0,
            "gender_mismatch_failures": 0,
            "name_mismatch_failures": 0,
            "dropped_clause_failures": 0,
            "hallucination_failures": 0,
        }

        if enable_critic:
            critic = TranslationCritic(self.base_url, self.api_key, self.model)
            evaluations = critic.evaluate_cues(project, project.cues, batch_size=25)
            eval_by_id = {e.get("cue_id"): e for e in evaluations if e.get("cue_id")}

            first_pass_failed_ids: set[str] = set()
            for cue in project.cues:
                ev = eval_by_id.get(cue.id)
                if not ev:
                    continue
                issues = ev.get("issues", [])
                cue.critic_score = float(ev.get("naturalness_score", 0.9))
                cue.critic_flags = issues
                for iss in issues:
                    if f"{iss}_failures" in metrics:
                        metrics[f"{iss}_failures"] += 1
                if ev.get("needs_retry", False) or issues:
                    cue.needs_review = True
                    cue.review_notes = ev.get("critique") or ", ".join(issues)
                    first_pass_failed_ids.add(cue.id)
                else:
                    cue.needs_review = False
                    metrics["critic_pass_first_try"] += 1

            # Targeted Retries for failed cues
            current_failed_ids = set(first_pass_failed_ids)
            for retry_round in range(1, max_retries + 1):
                if not current_failed_ids:
                    break
                retry_cues = [contexts_by_id[cid] for cid in current_failed_ids if cid in contexts_by_id]
                critique_notes = {
                    cid: eval_by_id[cid].get("critique", "Fix pronoun and meaning consistency")
                    for cid in current_failed_ids if cid in eval_by_id
                }

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

                cues_to_reeval = [c for c in project.cues if c.id in current_failed_ids]
                new_evals = critic.evaluate_cues(project, cues_to_reeval, batch_size=25)
                new_eval_by_id = {e.get("cue_id"): e for e in new_evals if e.get("cue_id")}

                next_failed_ids: set[str] = set()
                for cue in cues_to_reeval:
                    ev = new_eval_by_id.get(cue.id)
                    if not ev:
                        continue
                    issues = ev.get("issues", [])
                    cue.critic_score = float(ev.get("naturalness_score", 0.9))
                    cue.critic_flags = issues
                    eval_by_id[cue.id] = ev
                    if ev.get("needs_retry", False) or issues:
                        cue.needs_review = True
                        cue.review_notes = ev.get("critique") or ", ".join(issues)
                        next_failed_ids.add(cue.id)
                    else:
                        cue.needs_review = False
                        if retry_round == 1:
                            metrics["critic_retry_1_pass"] += 1
                        elif retry_round == 2:
                            metrics["critic_retry_2_pass"] += 1

                current_failed_ids = next_failed_ids

            metrics["critic_final_fail"] = len(current_failed_ids)

        self.last_metrics = metrics
        logger.info("Translation & Critic completed with metrics: %s", metrics)
        return project.cues
