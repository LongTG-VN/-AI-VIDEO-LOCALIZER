from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.models.project import Project, SubtitleCue
from app.services.critic import TranslationCritic
from app.services.relationships import (
    active_scene,
    character_name,
    find_character,
    resolve_pronouns,
)
from app.services.semantic_context import (
    build_neighbor_window,
    normalize_discourse_mode,
    project_name_locks,
    source_name_mentions,
)

logger = logging.getLogger(__name__)


class TranslationError(RuntimeError):
    pass


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    start_idx = text.find("{")
    while start_idx != -1:
        depth = 0
        end_idx = start_idx
        for i, ch in enumerate(text[start_idx:], start_idx):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        candidate = text[start_idx : end_idx + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            start_idx = text.find("{", start_idx + 1)
    return None


def build_translation_context(project: Project, cue_index: int) -> dict[str, Any]:
    """Build context-rich but cue-owned translation input.

    Neighboring subtitles are supplied as read-only context. Content and named entities
    explicitly present in another cue must not be migrated into the current cue.
    """
    cue = project.cues[cue_index]
    scene = active_scene(project, cue.start)
    speaker_id = cue.speaker_character_id or cue.speaker_id
    addressee_id = cue.addressee_character_id or cue.addressee_id

    speaker_char = find_character(project, speaker_id)
    addressee_char = find_character(project, addressee_id)

    self_pronoun, target_pronoun, rel_type, rel_conf = resolve_pronouns(
        project,
        speaker_id,
        addressee_id,
        cue.start,
    )

    window = build_neighbor_window(project, cue_index, before=3, after=2)
    previous_source = project.cues[cue_index - 1].source_text if cue_index > 0 else None
    next_source = project.cues[cue_index + 1].source_text if cue_index + 1 < len(project.cues) else None

    return {
        "cue_id": cue.id,
        "start": cue.start,
        "end": cue.end,
        "speaker": character_name(project, speaker_id),
        "speaker_character_id": cue.speaker_character_id,
        "speaker_role": speaker_char.role if speaker_char else None,
        "speaker_gender": speaker_char.gender if speaker_char else None,
        "addressee": character_name(project, addressee_id),
        "addressee_character_id": cue.addressee_character_id,
        "addressee_role": addressee_char.role if addressee_char else None,
        "addressee_gender": addressee_char.gender if addressee_char else None,
        "relationship": rel_type,
        "relationship_confidence": rel_conf,
        "preferred_vi_self": self_pronoun,
        "preferred_vi_other": target_pronoun,
        "discourse_mode": normalize_discourse_mode(getattr(cue, "discourse_mode", None)),
        "scene_summary": scene.summary if scene else None,
        "scene_tone": scene.tone if scene else None,
        "source": cue.source_text,
        "previous_source": previous_source,
        "next_source": next_source,
        "previous_context": window["previous"],
        "next_context": window["next"],
        "required_names": source_name_mentions(project, cue.source_text),
    }


def build_system_prompt(target_language: str) -> str:
    target = "Vietnamese" if target_language == "vi" else "English"
    return f"""You are an audiovisual subtitle translator from Chinese into {target}.

Translate each input cue faithfully and naturally while using scene/neighbor context only to
resolve meaning, references, relationships, and register.

HARD RULES:
1. CUE OWNERSHIP
   - Return exactly one result for every input `cue_id`, with the exact same ID.
   - Translate ONLY the `source` owned by that cue.
   - `previous_context` and `next_context` are context only. Never move a clause, name, or
     vocative from a neighboring cue into the current cue.
   - Never prepend speaker metadata to subtitle text.

2. CONTEXT & MEANING
   - Use `scene_summary`, `scene_tone`, neighboring source cues, speaker/addressee metadata,
     and `discourse_mode` to resolve pronouns, ellipsis, referents, implied subjects, and
     relational meaning.
   - Clothing terminology: '领口' refers to clothing collar/neckline ('cổ áo' / 'phần cổ áo'), NEVER necklace ('vòng cổ').
   - Relational Negation vs Literal Existence: In dramatic monologue/dialogue, '眼里没有女儿/儿子' expresses relational disregard. Translate faithfully with complete grammatical subject: e.g. 'Mẹ không xem tôi là con gái mà chỉ coi tôi như một món hàng cần liên tục mài giũa' or 'Trong mắt mẹ, tôi không phải là con gái mà chỉ là một món hàng...'. NEVER produce subjectless 'Trong mắt mẹ, không xem tôi...'. NEVER 'không có con gái'.
   - Preserve the main action, all meaningful clauses, negation, contrast, gender, kinship,
     and who is doing what to whom.
   - Do not add unsupported commands, emotions, explanations, or actions.

3. VIETNAMESE RELATIONSHIP REGISTER
   - When target is Vietnamese, follow `preferred_vi_self` and `preferred_vi_other` whenever
     explicit self/target pronouns are needed.
   - Dialogue pronouns come from the directional relationship; narration/monologue should not
     blindly inherit the pronouns of a nearby conversation.

4. NAME & TERMINOLOGY LOCKS
   - `name_locks`, `characters`, `glossary`, and per-cue `required_names` are authoritative.
   - If a `required_names` item is explicitly present in this cue's source, preserve its
     canonical target form in THIS cue.
   - Never invent a phonetic alternative.
   - A name used as direct address (vocative) must remain a vocative; do not turn it into a
     possessor or other grammatical role without source evidence.
   - If source does NOT contain a character's name, do NOT migrate or borrow that name from neighbors.

5. SUBTITLE GRAMMAR & NATURALNESS
   - Produce a complete, idiomatic subtitle fragment appropriate to the source cue.
   - Avoid dangling endings such as a bare pronoun/conjunction followed by a comma when the
     source cue itself contains a complete unit.
   - Do not duplicate pronouns or names (e.g. avoid 'cô ... cô?').
   - Full-Clause Preservation: If source contains multiple clauses (e.g. background situation + specific action), all clauses must be translated and preserved in the subtitle without silently dropping any clause.
   - Dramatic & Narrative Naturalness: '本该属于我的人生' -> 'cuộc đời vốn dĩ thuộc về tôi'; '在早餐店的家庭里，凌晨四点就要起来帮忙揉面' -> 'ở một gia đình mở quán ăn sáng, bốn giờ sáng đã phải dậy giúp nhào bột'; '你抢了我十八年' -> 'cô đã cướp mất mười tám năm của tôi!'.

6. DISCOURSE
   - Respect `direct_dialogue`, `monologue`, `narration`, `system`, and `unknown`.
   - Do not rewrite narration as an imperative or direct dialogue.
   - Do not merge discourse modes in translation; grouping happens later.

Return JSON ONLY:
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
        characters: list[dict[str, Any]] | None = None,
        glossary: list[dict[str, Any]] | None = None,
        name_locks: list[dict[str, Any]] | None = None,
    ) -> dict[str, tuple[str, float | None]]:
        payload_messages: list[dict[str, str]] = [
            {"role": "system", "content": build_system_prompt(target_language)}
        ]

        payload_dict: dict[str, Any] = {"cues": batch}
        if characters:
            payload_dict["characters"] = characters
        if glossary:
            payload_dict["glossary"] = glossary
        if name_locks:
            payload_dict["name_locks"] = name_locks

        if critique_notes:
            payload_dict["feedback_per_cue"] = critique_notes
            user_msg = (
                "Re-translate only the failed cue IDs. Keep cue ownership and apply the "
                "feedback without copying content from neighboring cues:\n"
                + json.dumps(payload_dict, ensure_ascii=False)
            )
        else:
            user_msg = json.dumps(payload_dict, ensure_ascii=False)

        payload_messages.append({"role": "user", "content": user_msg})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "temperature": 0.15,
            "messages": payload_messages,
        }

        parsed = None
        for attempt in range(8):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=35.0,
                )
                if response.status_code == 429:
                    wait = 4 * (attempt + 1)
                    logger.info("Translation rate limited; sleeping %ss", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                raw_msg = response.json().get("choices", [{}])[0].get("message", {})
                content = (raw_msg.get("content") or "").strip()
                parsed = extract_json_object(content)
                if parsed is not None:
                    break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    time.sleep(4 * (attempt + 1))
                    continue
                raise TranslationError(f"Translation provider error: {exc}") from exc
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
                if attempt < 7:
                    time.sleep(2)
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

        # Strict ID matching only: do NOT silently accept cross-cue positional mapping
        for b_item in batch:
            if b_item["cue_id"] not in results:
                results[b_item["cue_id"]] = (b_item.get("source", ""), 0.0)

        return results

    def translate_project(
        self,
        project: Project,
        batch_size: int = 12,
        enable_critic: bool = True,
        max_retries: int = 2,
        use_quality_pipeline: bool = True,
        audit_output_dir: Any = None,
    ) -> list[SubtitleCue]:
        self._validate_config()
        if not project.cues:
            return []

        if use_quality_pipeline:
            from app.services.translation_quality import TranslationQualityPipeline
            pipeline = TranslationQualityPipeline(self.base_url, self.api_key, self.model)
            report = pipeline.run_pipeline(project, audit_output_dir=audit_output_dir)
            project.translation_quality = report.model_dump()
            self.last_metrics = report.metrics
            return project.cues

        char_list = [
            {
                "id": c.id,
                "name_zh": c.name_zh or c.name,
                "name_vi": c.name_vi or c.name,
                "aliases": c.aliases,
                "gender": c.gender,
                "role": c.role,
            }
            for c in project.characters
        ]
        glossary_list = [
            {
                "source": g.source,
                "target": g.target,
                "category": g.category,
            }
            for g in project.glossary
        ]
        name_locks = project_name_locks(project)

        untranslated_indices = [i for i, c in enumerate(project.cues) if not c.translated_text]
        contexts_by_id = {
            c.id: build_translation_context(project, i)
            for i, c in enumerate(project.cues)
        }
        untranslated_contexts = [
            contexts_by_id[project.cues[i].id]
            for i in untranslated_indices
        ]

        for start in range(0, len(untranslated_contexts), batch_size):
            batch = untranslated_contexts[start : start + batch_size]
            results = self._call_translation_batch(
                batch,
                project.target_language,
                characters=char_list,
                glossary=glossary_list,
                name_locks=name_locks,
            )
            for cue in project.cues:
                if cue.id in results:
                    cue.translated_text, conf = results[cue.id]
                    cue.translation_confidence = conf
                    cue.confidence = conf
            time.sleep(2.0)

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
            "action_error_failures": 0,
            "grammatical_error_failures": 0,
            "dangling_fragment_failures": 0,
            "vocative_error_failures": 0,
        }

        if enable_critic:
            critic = TranslationCritic(self.base_url, self.api_key, self.model)
            evaluations = critic.evaluate_cues(project, project.cues, batch_size=25)
            eval_by_id = {
                e.get("cue_id"): e
                for e in evaluations
                if e.get("cue_id")
            }

            first_pass_failed_ids: set[str] = set()
            for cue in project.cues:
                ev = eval_by_id.get(cue.id)
                if not ev:
                    continue
                issues = ev.get("issues", [])
                cue.critic_score = float(ev.get("naturalness_score", 0.9))
                cue.critic_flags = issues
                for issue in issues:
                    metric_key = f"{issue}_failures"
                    if metric_key in metrics:
                        metrics[metric_key] += 1
                if ev.get("needs_retry", False) or issues:
                    cue.needs_review = True
                    cue.review_notes = ev.get("critique") or ", ".join(issues)
                    first_pass_failed_ids.add(cue.id)
                else:
                    cue.needs_review = False
                    metrics["critic_pass_first_try"] += 1

            current_failed_ids = set(first_pass_failed_ids)
            for retry_round in range(1, max_retries + 1):
                if not current_failed_ids:
                    break

                retry_cues = [
                    contexts_by_id[cid]
                    for cid in current_failed_ids
                    if cid in contexts_by_id
                ]
                critique_notes = {
                    cid: eval_by_id[cid].get(
                        "critique",
                        "Preserve source meaning, cue-owned entities, relationships, and grammar.",
                    )
                    for cid in current_failed_ids
                    if cid in eval_by_id
                }

                retry_results = self._call_translation_batch(
                    retry_cues,
                    project.target_language,
                    critique_notes=critique_notes,
                    characters=char_list,
                    glossary=glossary_list,
                    name_locks=name_locks,
                )
                for cue in project.cues:
                    if cue.id in retry_results:
                        retried_text, retried_conf = retry_results[cue.id]
                        cue.translated_text = retried_text
                        if retried_conf is not None:
                            cue.translation_confidence = retried_conf
                            cue.confidence = retried_conf

                cues_to_reeval = [
                    c for c in project.cues if c.id in current_failed_ids
                ]
                new_evals = critic.evaluate_cues(project, cues_to_reeval, batch_size=25)
                new_eval_by_id = {
                    e.get("cue_id"): e
                    for e in new_evals
                    if e.get("cue_id")
                }

                next_failed_ids: set[str] = set()
                for cue in cues_to_reeval:
                    ev = new_eval_by_id.get(cue.id)
                    if not ev:
                        next_failed_ids.add(cue.id)
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
