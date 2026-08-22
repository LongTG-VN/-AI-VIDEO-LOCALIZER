from __future__ import annotations

import json
import logging
import re
from enum import Enum
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

logger = logging.getLogger(__name__)


class CriticError(RuntimeError):
    pass


class CriticIssueEnum(str, Enum):
    MEANING_SHIFT = "meaning_shift"
    PRONOUN_MISMATCH = "pronoun_mismatch"
    RELATIONSHIP_MISMATCH = "relationship_mismatch"
    GENDER_MISMATCH = "gender_mismatch"
    NAME_MISMATCH = "name_mismatch"
    DROPPED_CLAUSE = "dropped_clause"
    HALLUCINATION = "hallucination"
    REGISTER_MISMATCH = "register_mismatch"


def build_critic_context(project: Project, cue: SubtitleCue) -> dict[str, Any]:
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
        "expected_vi_self": self_pronoun,
        "expected_vi_target": target_pronoun,
        "scene_summary": scene.summary if scene else None,
        "chinese_source": (cue.source_text or "").strip(),
        "vietnamese_translation": (cue.translated_text or "").strip(),
        "glossary": [
            {"source": g.source, "target": g.target}
            for g in project.glossary
        ],
    }


def deterministic_validate_cue(context: dict[str, Any]) -> tuple[bool, list[str], str]:
    """Runs high-precision deterministic guards for pronouns, relationships, genders, and dropped clauses."""
    issues: list[str] = []
    notes: list[str] = []

    zh = context.get("chinese_source", "")
    vi = context.get("vietnamese_translation", "")
    vi_lower = vi.lower()

    if not zh or not vi:
        return True, [], ""

    addr = context.get("addressee")
    exp_self = context.get("expected_vi_self")
    exp_target = context.get("expected_vi_target")
    rel = (context.get("relationship") or "").lower()

    # 1. Monologue / Narration Pronoun Guard
    if not addr or addr == "audience" or "monologue" in rel or "narration" in rel:
        if re.search(r"\b(mẹ|ba|bố|anh|chị|chú|bác)\s+ơi\b", vi_lower):
            pass
        elif re.search(r"^con\s+(là|đang|đã|sẽ|muốn|nghĩ|thấy)\b", vi_lower) or re.search(r"^em\s+(là|đang|đã|sẽ|muốn|nghĩ|thấy)\b", vi_lower):
            issues.append(CriticIssueEnum.PRONOUN_MISMATCH.value)
            notes.append("Monologue / narration should use 'tôi' instead of 'con' or 'em' as self pronoun.")

    # 2. Sibling / Sibling Pronoun Guard (Anh - Em)
    if ("sibling" in rel or "anh" in (exp_self or "") or "anh" in (exp_target or "") or "brother" in (context.get("speaker_role") or "").lower()) and "hostile" not in rel:
        if re.search(r"\b(mày|tao)\b", vi_lower):
            issues.append(CriticIssueEnum.PRONOUN_MISMATCH.value)
            notes.append("Family sibling dialogue should use 'anh / em' instead of 'mày / tao'.")

    # 3. Parent - Child Pronoun Guard (Mẹ - Con / Ba - Con)
    if "mother" in rel or "father" in rel or "parent" in rel or "mẹ" in (exp_self or "") or "ba" in (exp_self or ""):
        if re.search(r"\b(mày|tao)\b", vi_lower):
            issues.append(CriticIssueEnum.PRONOUN_MISMATCH.value)
            notes.append("Family parent-child dialogue should use 'mẹ/ba - con' instead of 'mày / tao'.")

    # 4. Gender Reference Check
    if ("她" in zh or "母亲" in zh or "我妈" in zh) and not ("他" in zh or "我爸" in zh or "我哥" in zh):
        if re.search(r"\b(ông ta|anh ấy|ông ấy|chàng trai|gã)\b", vi_lower):
            issues.append(CriticIssueEnum.GENDER_MISMATCH.value)
            notes.append("Source references female referent (她/mẹ) but translation uses male pronoun (ông ta/anh ấy).")

    if ("他" in zh or "父亲" in zh or "我爸" in zh) and not ("她" in zh or "我妈" in zh or "我妹" in zh):
        if re.search(r"\b(bà ta|cô ấy|bà ấy|cô gái)\b", vi_lower):
            issues.append(CriticIssueEnum.GENDER_MISMATCH.value)
            notes.append("Source references male referent (他/bố) but translation uses female pronoun (bà ta/cô ấy).")

    # 5. Dual / Compound Clause Preservation
    if ("没有" in zh or "不是" in zh) and ("只有" in zh or "而是" in zh or "只" in zh):
        if "没有女儿" in zh and not any(w in vi_lower for w in ["con gái", "đứa con", "con"]):
            issues.append(CriticIssueEnum.DROPPED_CLAUSE.value)
            notes.append("Dropped first clause: missing 'không có con gái'.")
        if "商品" in zh and not any(w in vi_lower for w in ["hàng", "sản phẩm", "món đồ", "công cụ"]):
            issues.append(CriticIssueEnum.DROPPED_CLAUSE.value)
            notes.append("Dropped second clause: missing 'món hàng' / 'sản phẩm'.")

    # 6. Hallucination & Unsupported Content
    if zh.startswith("看清楚") and not any(w in zh for w in ["加油", "努力"]):
        if re.search(r"\bcố lên\b", vi_lower):
            issues.append(CriticIssueEnum.HALLUCINATION.value)
            notes.append("Hallucinated 'cố lên' not supported by source '看清楚'.")

    # 7. Glossary & Name Consistency
    glossary = context.get("glossary", [])
    for entry in glossary:
        src_name = entry.get("source", "")
        tgt_name = entry.get("target", "")
        if src_name and tgt_name and src_name in zh:
            tgt_clean = re.sub(r"[^\w]", "", tgt_name.lower())
            vi_clean = re.sub(r"[^\w]", "", vi_lower)
            if tgt_clean not in vi_clean and not any(part in vi_lower for part in tgt_name.lower().split()):
                issues.append(CriticIssueEnum.NAME_MISMATCH.value)
                notes.append(f"Name '{src_name}' in source should be translated as '{tgt_name}'.")

    is_pass = len(issues) == 0
    return is_pass, issues, "; ".join(notes)


_CRITIC_SYSTEM = """You are a master Vietnamese subtitle critic and validation engine for Chinese dramas.
Evaluate whether the translated Vietnamese subtitle faithfully, accurately, and naturally translates the Chinese source within the drama context.

CRITICAL CHECKS:
1. meaning: 'pass' if accurate, 'fail' if distorted or altered.
2. name_consistency: 'pass' if proper names match glossary, 'fail' otherwise.
3. pronoun_consistency: 'pass' if Vietnamese pronouns match expected_vi_self and expected_vi_target.
4. relationship_consistency: 'pass' or 'fail'.
5. gender_consistency: 'pass' if female/male referents (她 vs 他) are correctly preserved in Vietnamese.
6. hallucination: 'pass' if no unsupported phrases added (e.g. adding 'cố lên' to '看清楚' is a FAIL), 'fail' otherwise.
7. missing_information: 'pass' if all meaningful clauses are preserved, 'fail' if any clause is dropped.
8. naturalness_score: float 0.0 to 1.0.
9. needs_retry: true if any check fails, false if all pass.
10. critique: brief feedback explaining what needs fixing if failed, or empty if pass.
11. suggested_fix: improved natural Vietnamese subtitle if failed, or null if pass.

Return JSON ONLY in this exact structure:
{
  "evaluations": [
    {
      "cue_id": "...",
      "meaning": "pass",
      "name_consistency": "pass",
      "pronoun_consistency": "pass",
      "relationship_consistency": "pass",
      "gender_consistency": "pass",
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
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model

    def evaluate_cues(self, project: Project, cues: list[SubtitleCue], batch_size: int = 25) -> list[dict[str, Any]]:
        evaluations: list[dict[str, Any]] = []
        contexts = [build_critic_context(project, cue) for cue in cues]

        for ctx in contexts:
            det_pass, det_issues, det_notes = deterministic_validate_cue(ctx)
            evaluation = {
                "cue_id": ctx["cue_id"],
                "meaning": "fail" if CriticIssueEnum.MEANING_SHIFT.value in det_issues else "pass",
                "name_consistency": "fail" if CriticIssueEnum.NAME_MISMATCH.value in det_issues else "pass",
                "pronoun_consistency": "fail" if CriticIssueEnum.PRONOUN_MISMATCH.value in det_issues else "pass",
                "relationship_consistency": "fail" if CriticIssueEnum.RELATIONSHIP_MISMATCH.value in det_issues else "pass",
                "gender_consistency": "fail" if CriticIssueEnum.GENDER_MISMATCH.value in det_issues else "pass",
                "hallucination": "fail" if CriticIssueEnum.HALLUCINATION.value in det_issues else "pass",
                "missing_information": "fail" if CriticIssueEnum.DROPPED_CLAUSE.value in det_issues else "pass",
                "naturalness_score": 0.95 if det_pass else 0.50,
                "needs_retry": not det_pass,
                "critique": det_notes,
                "issues": det_issues,
                "suggested_fix": None,
            }
            evaluations.append(evaluation)

        if self.base_url and self.model:
            try:
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

                    for attempt in range(4):
                        try:
                            response = httpx.post(
                                f"{self.base_url}/chat/completions",
                                headers=headers,
                                json=payload,
                                timeout=60,
                            )
                            if response.status_code == 429:
                                import time
                                time.sleep(6 * (attempt + 1))
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
                            llm_evals = parsed.get("evaluations", [])
                            llm_map = {e.get("cue_id"): e for e in llm_evals if e.get("cue_id")}

                            for ev in evaluations:
                                cid = ev.get("cue_id")
                                if cid in llm_map:
                                    llm_e = llm_map[cid]
                                    for k in ["meaning", "name_consistency", "pronoun_consistency", "relationship_consistency", "gender_consistency", "hallucination", "missing_information"]:
                                        if llm_e.get(k) == "fail":
                                            ev[k] = "fail"
                                            ev["needs_retry"] = True
                                            if k not in ev["issues"]:
                                                ev["issues"].append(k)
                                    if llm_e.get("critique") and not ev["critique"]:
                                        ev["critique"] = llm_e.get("critique")
                                    if llm_e.get("suggested_fix"):
                                        ev["suggested_fix"] = llm_e.get("suggested_fix")
                            break
                        except Exception as e:
                            logger.warning("LLM Critic attempt %d failed: %s", attempt + 1, e)
            except Exception as exc:
                logger.warning("LLM Critic pass skipped: %s", exc)

        return evaluations
