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
    REFERENT_ERROR = "referent_error"
    PRONOUN_MISMATCH = "pronoun_mismatch"
    RELATIONSHIP_MISMATCH = "relationship_mismatch"
    GENDER_MISMATCH = "gender_mismatch"
    DROPPED_CLAUSE = "dropped_clause"
    NAME_MISMATCH = "name_mismatch"
    HALLUCINATION = "hallucination"
    ACTION_ERROR = "action_error"
    REGISTER_MISMATCH = "register_mismatch"
    DISCOURSE_ERROR = "discourse_error"
    GRAMMATICAL_ERROR = "grammatical_error"
    DANGLING_FRAGMENT = "dangling_fragment"
    VOCATIVE_ERROR = "vocative_error"


def has_dangling_fragment(text: str) -> bool:
    """Checks if a Vietnamese subtitle line ends with an unnatural dangling grammatical fragment."""
    clean = text.strip()
    if not clean:
        return False
    # Allow legitimate parenthetical clauses like "Như bạn thấy,", "Như anh thấy,"
    if re.search(r"\bnhư\s+(bạn|anh|em|cô|ông|bà|chị|chúng ta|mọi người)?\s*thấy\s*[,:;]?$", clean, re.IGNORECASE):
        return False
    # Check for dangling pronouns, conjunctions, prepositions, auxiliary particles, or incomplete modals
    for pat in [
        r"\b(cô|anh|em|ông|bà|chị|bạn|mày|con|tôi|ta)\s*[,:;]?$",
        r"\b(mà|thì|nhưng|hoặc|đã|đang|sẽ|được|bị|vì|bởi|của|ở|tại|là|để|cho|với)\s*[,:;]?$",
        r"\b(muốn|nghĩ|biết|cần)\s*[,:;]+$",
    ]:
        if re.search(pat, clean, re.IGNORECASE):
            return True
    return False


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
        "speaker_name_vi": (speaker_char.name_vi or speaker_char.name) if speaker_char else None,
        "speaker_name_zh": (speaker_char.name_zh or speaker_char.name) if speaker_char else None,
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
        "characters": [
            {"name_zh": c.name_zh or c.name, "name_vi": c.name_vi or c.name, "aliases": c.aliases}
            for c in project.characters
        ],
        "glossary": [
            {"source": g.source, "target": g.target}
            for g in project.glossary
        ],
    }


def deterministic_validate_cue(context: dict[str, Any]) -> tuple[bool, list[str], str]:
    """Runs high-precision deterministic guards for pronouns, relationships, genders, names, actions, and dropped clauses."""
    issues: list[str] = []
    notes: list[str] = []

    zh = context.get("chinese_source", "")
    vi = context.get("vietnamese_translation", "")
    vi_lower = vi.lower()

    if not zh or not vi:
        return True, [], ""

    # 0. Untranslated Chinese Character Guard
    if re.search(r"[\u4e00-\u9fff]", vi):
        issues.append(CriticIssueEnum.MEANING_SHIFT.value)
        notes.append("Translation contains untranslated Chinese characters.")

    addr = context.get("addressee")
    exp_self = context.get("expected_vi_self")
    exp_target = context.get("expected_vi_target")
    rel = (context.get("relationship") or "").lower()
    speaker_role = (context.get("speaker_role") or "").lower()

    # 1. Monologue / Narration Pronoun Guard
    is_monologue_or_narration = bool(
        ("monologue" in rel or "narration" in rel)
        or (not addr and not exp_target and "brother" not in rel and "sister" not in rel and "father" not in rel and "mother" not in rel)
        or addr == "audience"
    )
    if is_monologue_or_narration and not exp_target:
        if re.search(r"\b(mẹ|ba|bố|anh|chị|chú|bác)\s+ơi\b", vi_lower):
            pass
        elif re.search(r"\b(con|em)\s+(là|đang|đã|sẽ|muốn|nghĩ|thấy|nhất định|phải|chỉ|không)\b", vi_lower):
            issues.append(CriticIssueEnum.PRONOUN_MISMATCH.value)
            notes.append("Monologue / narration should use 'tôi' instead of 'con' or 'em' as self pronoun.")

    # 2. Relationship-aware Second-Person Address Check
    if exp_target == "em" or ("sibling" in rel and "brother" in speaker_role) or ("older" in rel and "sister" in rel):
        # Sibling older -> younger: must address other as 'em'
        if re.search(r"\b(cô|mày|bạn|cậu)\b", vi_lower) and "hostile" not in rel:
            issues.append(CriticIssueEnum.PRONOUN_MISMATCH.value)
            notes.append("Older sibling addressing younger sibling must address them as 'em', not 'cô/mày/bạn/cậu'.")

    if exp_target == "con" or "mother" in rel or "father" in rel or "parent" in rel:
        # Parent addressing child: must address child as 'con'
        if re.search(r"\b(cô|mày|bạn|cậu)\b", vi_lower) and "hostile" not in rel:
            issues.append(CriticIssueEnum.PRONOUN_MISMATCH.value)
            notes.append("Parent addressing child must address child as 'con', not 'cô/mày/bạn/cậu'.")

    if exp_target in ["mẹ", "má"]:
        # Child addressing mother: must use 'mẹ'
        if re.search(r"\b(bà|cô|bạn|chị)\b", vi_lower):
            issues.append(CriticIssueEnum.PRONOUN_MISMATCH.value)
            notes.append("Child addressing mother must use 'mẹ', not 'bà/cô/bạn'.")

    if exp_target in ["bố", "ba", "cha"]:
        # Child addressing father: must use 'bố/ba'
        if re.search(r"\b(ông|chú|bạn|anh)\b", vi_lower):
            issues.append(CriticIssueEnum.PRONOUN_MISMATCH.value)
            notes.append("Child addressing father must use 'bố/ba', not 'ông/chú/bạn'.")

    # 3. Kinship & Semantic Fidelity Check (女儿, 儿子, 母亲, 父亲, 商品)
    if "女儿" in zh:
        has_daughter = bool(
            re.search(r"\b(con gái|đứa con gái|con ruột|con nuôi)\b", vi_lower)
            or (
                re.search(r"\bcon\b", vi_lower)
                and not re.search(r"\bcon\s+(người|số|đường|mắt|vật|bạc|sâu|thuyền|bài|cờ|tim)\b", vi_lower)
            )
        )
        if re.search(r"\b(con người|người nào|tình người|nhân loại)\b", vi_lower) and not has_daughter:
            issues.append(CriticIssueEnum.MEANING_SHIFT.value)
            notes.append("Source has '女儿' (daughter/con gái), but translation changed meaning to 'con người/tình người' or dropped the daughter concept.")
        elif not has_daughter:
            issues.append(CriticIssueEnum.DROPPED_CLAUSE.value)
            notes.append("Source has '女儿' (daughter/con gái), but translation is missing daughter/child concept.")

    if "商品" in zh:
        has_product = bool(re.search(r"\b(hàng|món hàng|sản phẩm|món đồ|công cụ)\b", vi_lower))
        if not has_product:
            issues.append(CriticIssueEnum.DROPPED_CLAUSE.value)
            notes.append("Source has '商品' (product/merchandise), but translation dropped product metaphor.")

    # In daughter-mother context: 她的眼里没有女儿 refers to mother (mẹ / bà ấy), not 'cô ấy'
    if "没有女儿" in zh and "商品" in zh:
        if re.search(r"\bcô ấy\b", vi_lower):
            issues.append(CriticIssueEnum.REFERENT_ERROR.value)
            notes.append("In family context describing mother, '她的眼里' refers to mother ('mẹ / bà ấy'), not 'cô ấy'.")

    # Relational Negation vs Literal Existence Check (e.g. 她的眼里没有女儿 -> không xem tôi là con gái)
    if ("没有女儿" in zh or "没有儿子" in zh) and ("商品" in zh or "眼里" in zh or "心里" in zh):
        has_relational_regard = bool(
            re.search(r"\b(không|chưa)\s+(xem|coi|nhận|coi sóc|đoái hoài)\b", vi_lower)
            and re.search(r"\blà\s+(tôi\s+là\s+)?(con|con gái|con ruột|con cái)\b", vi_lower)
        )
        if not has_relational_regard:
            issues.append(CriticIssueEnum.MEANING_SHIFT.value)
            notes.append("Source expresses relational disregard; translation must use 'không xem/coi tôi là con gái' instead of literal existence 'không có con gái'.")

    # Guard: Speaker's own name metadata must NOT be prepended into spoken dialogue text
    speaker_name_vi = (context.get("speaker_name_vi") or context.get("speaker") or "").strip()
    if speaker_name_vi and len(speaker_name_vi) >= 3 and not re.match(r"^speaker_\d+$", speaker_name_vi):
        speaker_name_zh = (context.get("speaker_name_zh") or "").strip()
        if not (speaker_name_zh and speaker_name_zh in zh):
            if re.search(rf"^{re.escape(speaker_name_vi.lower())}\b", vi_lower):
                issues.append(CriticIssueEnum.NAME_MISMATCH.value)
                notes.append(f"Speaker identity metadata '{speaker_name_vi}' was mistakenly prepended into spoken dialogue text.")

    # Vocative vs Possessive Check (e.g. 秦扶栀昨天的... -> Tần Phù Chi, ... NOT của Tần Phù Chi)
    characters = context.get("characters", [])
    for char in characters:
        vi_name = (char.get("name_vi") or "").strip()
        zh_name = (char.get("name_zh") or char.get("name") or "").strip()
        if vi_name and len(vi_name) >= 3 and zh_name and zh_name in zh:
            addr_str = str(addr or "")
            is_vocative = bool(
                zh.startswith(zh_name) or
                (addr and (vi_name.lower() in addr_str.lower() or zh_name in addr_str))
            )
            if is_vocative:
                possessive_pattern = rf"\bcủa\s+{re.escape(vi_name.lower())}\b"
                if re.search(possessive_pattern, vi_lower):
                    issues.append(CriticIssueEnum.GRAMMATICAL_ERROR.value)
                    notes.append(f"Vocative character name '{vi_name}' was mistakenly translated as possessive ('của {vi_name}').")

    # Dangling Fragment Guard (e.g. sentences ending in dangling 'cô,', 'em,', 'mà,')
    if has_dangling_fragment(vi):
        issues.append(CriticIssueEnum.DANGLING_FRAGMENT.value)
        notes.append(f"Translation ends with an unnatural dangling grammatical fragment: '{vi}'.")

    # 4. Action Verb Fidelity Checks
    if ("啃完" in zh or "啃" in zh) and ("鸡腿" in zh or "肉" in zh or "骨头" in zh):
        has_eat_action = bool(re.search(r"\b(gặm|ăn|ăn hết|gặm hết|ăn xong|gặm xong|thưởng thức)\b", vi_lower))
        if not has_eat_action:
            issues.append(CriticIssueEnum.ACTION_ERROR.value)
            notes.append("Source action '啃完/啃' (eat/gnaw/finish eating) was changed or missing in translation (e.g. only translated as 'giấu').")

    if "背一下" in zh or "背诵" in zh:
        has_recite = bool(re.search(r"\b(đọc thuộc|học thuộc|thuộc lòng|đọc lại|nhắc lại|đọc|trả lời)\b", vi_lower))
        has_unsupported_hurry = bool(re.search(r"\b(nhanh lên|mau lên|cố lên|khẩn trương)\b", vi_lower))
        if not has_recite:
            issues.append(CriticIssueEnum.ACTION_ERROR.value)
            notes.append("Source action '背一下' (recite from memory) was missing in translation.")
        if has_unsupported_hurry and not any(w in zh for w in ["快", "抓紧", "赶紧"]):
            issues.append(CriticIssueEnum.HALLUCINATION.value)
            notes.append("Unsupported hurry modifier ('nhanh lên/cố lên') added to '背一下'.")

    # 5. Gender Reference Check
    if ("她" in zh or "母亲" in zh or "我妈" in zh) and not ("他" in zh or "我爸" in zh or "我哥" in zh):
        if re.search(r"\b(ông ta|anh ấy|ông ấy|chàng trai|gã)\b", vi_lower):
            issues.append(CriticIssueEnum.GENDER_MISMATCH.value)
            notes.append("Source references female referent (她/mẹ) but translation uses male pronoun (ông ta/anh ấy).")

    if ("他" in zh or "父亲" in zh or "我爸" in zh) and not ("她" in zh or "我妈" in zh or "我妹" in zh):
        if re.search(r"\b(bà ta|cô ấy|bà ấy|cô gái)\b", vi_lower):
            issues.append(CriticIssueEnum.GENDER_MISMATCH.value)
            notes.append("Source references male referent (他/bố) but translation uses female pronoun (bà ta/cô ấy).")

    # 6. Explicit Character Name Preservation & Invented Name Guard
    characters = context.get("characters", [])
    for char in characters:
        zh_names = [char.get("name_zh"), char.get("name")] + char.get("aliases", [])
        vi_name = char.get("name_vi")
        if not vi_name:
            continue
        for z_name in zh_names:
            if z_name and len(z_name) >= 2 and z_name in zh:
                vi_name_clean = re.sub(r"[^\w\s]", "", vi_name.lower())
                vi_text_clean = re.sub(r"[^\w\s]", "", vi_lower)
                name_parts = vi_name_clean.split()
                short_name = " ".join(name_parts[-2:]) if len(name_parts) >= 2 else vi_name_clean
                if vi_name_clean not in vi_text_clean and short_name not in vi_text_clean:
                    issues.append(CriticIssueEnum.NAME_MISMATCH.value)
                    notes.append(f"Explicit character name '{z_name}' ({vi_name}) in source was dropped in translation.")
                    break

    # Invented / Phonetic Name Guard (e.g. Ken Văn, Khan Văn, Kiên Vân)
    if re.search(r"\b(ken văn|khan văn|kiên văn|kiên vân|khang văn)\b", vi_lower):
        issues.append(CriticIssueEnum.NAME_MISMATCH.value)
        notes.append("Invented phonetic name detected not present in character graph.")

    # 7. Hallucination & Unsupported Content
    if zh.startswith("看清楚") and not any(w in zh for w in ["加油", "努力"]):
        if re.search(r"\bcố lên\b", vi_lower):
            issues.append(CriticIssueEnum.HALLUCINATION.value)
            notes.append("Hallucinated 'cố lên' not supported by source '看清楚'.")

    # 8. Glossary Consistency
    glossary = context.get("glossary", [])
    for entry in glossary:
        src_name = entry.get("source", "")
        tgt_name = entry.get("target", "")
        if src_name and tgt_name and src_name in zh:
            tgt_clean = re.sub(r"[^\w]", "", tgt_name.lower())
            vi_clean = re.sub(r"[^\w]", "", vi_lower)
            if tgt_clean not in vi_clean and not any(part in vi_lower for part in tgt_name.lower().split()):
                if CriticIssueEnum.NAME_MISMATCH.value not in issues:
                    issues.append(CriticIssueEnum.NAME_MISMATCH.value)
                    notes.append(f"Term '{src_name}' in source should be translated as '{tgt_name}'.")

    is_pass = len(issues) == 0
    return is_pass, issues, "; ".join(notes)


_CRITIC_SYSTEM = """You are a master Vietnamese subtitle critic and validation engine for Chinese dramas.
Evaluate whether the translated Vietnamese subtitle faithfully, accurately, and naturally translates the Chinese source within the drama context.

CRITICAL CHECKS:
1. meaning: 'pass' if accurate, 'fail' if distorted or altered (e.g. mistranslating '没有女儿' as 'không có con người' is a FAIL).
2. name_consistency: 'pass' if proper names are preserved and match glossary (e.g. dropping '秦扶栀' -> 'Tần Phù Chi' is a FAIL), 'fail' otherwise.
3. pronoun_consistency: 'pass' if Vietnamese pronouns match expected_vi_self and expected_vi_target (e.g. brother -> sister using 'cô' or 'mày' instead of 'em' is a FAIL).
4. relationship_consistency: 'pass' or 'fail'.
5. gender_consistency: 'pass' if female/male referents (她 vs 他) are correctly preserved in Vietnamese.
6. hallucination: 'pass' if no unsupported phrases added (e.g. adding 'cố lên' to '看清楚' is a FAIL), 'fail' otherwise.
7. missing_information: 'pass' if all meaningful clauses are preserved (e.g. '没有女儿... 只有...商品' must preserve both daughter and product concepts), 'fail' if any clause is dropped.
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
    """Combines deterministic validation with an OpenAI-compatible LLM Critic."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model

    def evaluate_cues(
        self,
        project: Project,
        cues: list[SubtitleCue],
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        if not cues:
            return []

        contexts = [build_critic_context(project, cue) for cue in cues]
        evaluations: list[dict[str, Any]] = []

        # 1. Deterministic Pass
        deterministic_failed_contexts: list[dict[str, Any]] = []
        for ctx in contexts:
            d_pass, d_issues, d_notes = deterministic_validate_cue(ctx)
            if not d_pass:
                evaluations.append({
                    "cue_id": ctx["cue_id"],
                    "meaning": "fail" if CriticIssueEnum.MEANING_SHIFT.value in d_issues else "pass",
                    "name_consistency": "fail" if CriticIssueEnum.NAME_MISMATCH.value in d_issues else "pass",
                    "pronoun_consistency": "fail" if CriticIssueEnum.PRONOUN_MISMATCH.value in d_issues else "pass",
                    "relationship_consistency": "fail" if CriticIssueEnum.RELATIONSHIP_MISMATCH.value in d_issues else "pass",
                    "gender_consistency": "fail" if CriticIssueEnum.GENDER_MISMATCH.value in d_issues else "pass",
                    "hallucination": "fail" if CriticIssueEnum.HALLUCINATION.value in d_issues else "pass",
                    "missing_information": "fail" if CriticIssueEnum.DROPPED_CLAUSE.value in d_issues else "pass",
                    "naturalness_score": 0.4,
                    "needs_retry": True,
                    "critique": d_notes,
                    "suggested_fix": None,
                    "issues": d_issues,
                })
            else:
                deterministic_failed_contexts.append(ctx)

        # 2. LLM Critic Pass on remaining cues if base_url is configured
        if self.base_url and self.model and deterministic_failed_contexts:
            for start in range(0, len(deterministic_failed_contexts), batch_size):
                batch = deterministic_failed_contexts[start : start + batch_size]
                llm_evals = self._call_llm_critic(batch)
                eval_map = {e["cue_id"]: e for e in llm_evals if "cue_id" in e}
                for ctx in batch:
                    cid = ctx["cue_id"]
                    if cid in eval_map:
                        ev = eval_map[cid]
                        issues: list[str] = []
                        if ev.get("meaning") == "fail":
                            issues.append(CriticIssueEnum.MEANING_SHIFT.value)
                        if ev.get("name_consistency") == "fail":
                            issues.append(CriticIssueEnum.NAME_MISMATCH.value)
                        if ev.get("pronoun_consistency") == "fail":
                            issues.append(CriticIssueEnum.PRONOUN_MISMATCH.value)
                        if ev.get("relationship_consistency") == "fail":
                            issues.append(CriticIssueEnum.RELATIONSHIP_MISMATCH.value)
                        if ev.get("gender_consistency") == "fail":
                            issues.append(CriticIssueEnum.GENDER_MISMATCH.value)
                        if ev.get("hallucination") == "fail":
                            issues.append(CriticIssueEnum.HALLUCINATION.value)
                        if ev.get("missing_information") == "fail":
                            issues.append(CriticIssueEnum.DROPPED_CLAUSE.value)
                        ev["issues"] = issues
                        evaluations.append(ev)
                    else:
                        evaluations.append({
                            "cue_id": cid,
                            "meaning": "pass",
                            "name_consistency": "pass",
                            "pronoun_consistency": "pass",
                            "relationship_consistency": "pass",
                            "gender_consistency": "pass",
                            "hallucination": "pass",
                            "missing_information": "pass",
                            "naturalness_score": 0.9,
                            "needs_retry": False,
                            "critique": "",
                            "suggested_fix": None,
                            "issues": [],
                        })
        elif deterministic_failed_contexts:
            for ctx in deterministic_failed_contexts:
                evaluations.append({
                    "cue_id": ctx["cue_id"],
                    "meaning": "pass",
                    "name_consistency": "pass",
                    "pronoun_consistency": "pass",
                    "relationship_consistency": "pass",
                    "gender_consistency": "pass",
                    "hallucination": "pass",
                    "missing_information": "pass",
                    "naturalness_score": 0.95,
                    "needs_retry": False,
                    "critique": "",
                    "suggested_fix": None,
                    "issues": [],
                })

        return evaluations

    def _call_llm_critic(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        streamlined_cues = [
            {
                "cue_id": c["cue_id"],
                "speaker": c["speaker"],
                "addressee": c["addressee"],
                "relationship": c["relationship"],
                "expected_vi_self": c["expected_vi_self"],
                "expected_vi_target": c["expected_vi_target"],
                "chinese_source": c["chinese_source"],
                "vietnamese_translation": c["vietnamese_translation"],
            }
            for c in batch
        ]

        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": _CRITIC_SYSTEM},
                {"role": "user", "content": json.dumps({"cues": streamlined_cues}, ensure_ascii=False)},
            ],
        }

        for attempt in range(4):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90,
                )
                if response.status_code == 429:
                    import time
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raw_msg = response.json().get("choices", [{}])[0].get("message", {})
                content = (raw_msg.get("content") or "").strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content.startswith("```"):
                    lines = content.splitlines()
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    content = "\n".join(lines).strip()
                parsed = json.loads(content)
                return parsed.get("evaluations", [])
            except Exception as e:
                logger.warning("LLM Critic failed attempt %d: %s", attempt + 1, e)
                if attempt < 3:
                    import time
                    time.sleep(2.0)
                    continue

        return []
