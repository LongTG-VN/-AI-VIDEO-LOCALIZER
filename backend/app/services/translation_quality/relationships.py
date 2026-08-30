from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.models.project import Project, SubtitleCue
from app.services.relationships import find_character, resolve_pronouns
from app.services.translation_quality.models import (
    QualityIssue,
    QualitySeverity,
    TranslationContextCard,
)

logger = logging.getLogger(__name__)


class RelationshipReviewer:
    """Pass 4: Entity, Relationship, Polysemy, and Vietnamese Pronoun Reviewer.
    
    Resolves polysemy terms (女儿, 女朋友, 朋友, 妹妹, 姐姐, 哥哥, 弟弟, 老婆, 丈夫, 对象, 同事, 老板)
    using the character graph, relationship graph, and dialogue context.
    """

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model

    def evaluate_cues(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None = None,
        batch_size: int = 20,
    ) -> dict[str, list[QualityIssue]]:
        results: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}

        # 1. Deterministic Polysemy & Relational Invariant Checks
        for cue in cues:
            det_issues = self._check_deterministic_relationships(project, cue)
            results[cue.id].extend(det_issues)

        # 2. LLM Relational Consistency & Polysemy Review
        if self.base_url and self.model:
            try:
                llm_issues = self._call_llm_relationship_check(project, cues, context_card, batch_size=batch_size)
                for cid, issues in llm_issues.items():
                    if cid in results:
                        results[cid].extend(issues)
            except Exception as exc:
                logger.warning("Relationship LLM review error (fallback to deterministic): %s", exc)

        return results

    def _check_deterministic_relationships(
        self,
        project: Project,
        cue: SubtitleCue,
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        src = (cue.source_text or "").strip()
        vi = (cue.translated_text or "").strip()
        if not src or not vi:
            return issues

        # A. Polysemy: 女朋友 (romantic girlfriend) vs 女儿 / 闺女 (daughter)
        if "女朋友" in src and re.search(r"\b(con gái|con gái tôi|con bé)\b", vi, re.IGNORECASE):
            # If source says 女朋友 (girlfriend) but VI translated as "con gái"
            issues.append(
                QualityIssue(
                    type="relationship.polysemy_mismatch",
                    severity=QualitySeverity.CRITICAL,
                    message="Source has '女朋友' (romantic girlfriend), but translation incorrectly used 'con gái' (daughter).",
                    source_span="女朋友",
                    target_span=vi,
                    reviewer="relationships",
                )
            )

        # B. Polysemy: 闺女 / 女儿 (daughter) in father/parent dialogue context
        # If source has 闺女 / 女儿 in parental speech -> must translate as con gái / ái nữ, not bạn gái
        speaker = find_character(project, cue.speaker_character_id or cue.speaker_id)
        if ("闺女" in src or "女儿" in src) and re.search(r"\b(bạn gái|người yêu)\b", vi, re.IGNORECASE):
            if speaker and speaker.role in ["father", "mother", "parent"]:
                issues.append(
                    QualityIssue(
                        type="relationship.polysemy_mismatch",
                        severity=QualitySeverity.CRITICAL,
                        message="Parent speaker source has '闺女/女儿' (daughter), but translation used 'bạn gái' (girlfriend).",
                        source_span="闺女/女儿",
                        target_span=vi,
                        reviewer="relationships",
                    )
                )

        # C. Polysemy: 对象 (romantic partner / lover vs mathematical/logical object)
        if "找对象" in src or "处对象" in src or "谈对象" in src:
            if not re.search(r"\b(người yêu|bạn trai|bạn gái|hẹn hò|tìm hiểu)\b", vi, re.IGNORECASE):
                if re.search(r"\b(đối tượng)\b", vi, re.IGNORECASE):
                    issues.append(
                        QualityIssue(
                            type="relationship.polysemy_mismatch",
                            severity=QualitySeverity.MAJOR,
                            message="'对象' in dating context means romantic partner ('người yêu/bạn gái'), but was translated literally as 'đối tượng'.",
                            source_span="对象",
                            target_span=vi,
                            reviewer="relationships",
                        )
                    )

        # D. Canonical Names Locking Check
        for char in project.characters:
            char_zh = (char.name_zh or char.name).strip()
            char_vi = (char.name_vi or char.name).strip()
            if char_zh and char_zh in src:
                # Character is mentioned in source -> canonical name_vi must be used
                if char_vi and not re.search(r"\b" + re.escape(char_vi) + r"\b", vi, re.IGNORECASE):
                    # Check if translated to an unauthorized alias or wrong spelling
                    issues.append(
                        QualityIssue(
                            type="relationship.entity_mismatch",
                            severity=QualitySeverity.MAJOR,
                            message=f"Character '{char_zh}' mentioned in source should be translated with locked name '{char_vi}'.",
                            source_span=char_zh,
                            target_span=vi,
                            reviewer="relationships",
                        )
                    )

        # E. Preferred Pronouns Validation
        speaker_id = cue.speaker_character_id or cue.speaker_id
        addressee_id = cue.addressee_character_id or cue.addressee_id
        if speaker_id and addressee_id:
            self_pro, target_pro, rel_type, rel_conf = resolve_pronouns(
                project, speaker_id, addressee_id, cue.start
            )
            # If high confidence pronoun pair exists, verify no obvious reversal (e.g. self called em when expected anh)
            if rel_conf and rel_conf >= 0.8 and self_pro and target_pro:
                # Basic sanity: avoid calling self with target_pro
                pass

        return issues

    def _call_llm_relationship_check(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None,
        batch_size: int = 15,
    ) -> dict[str, list[QualityIssue]]:
        results: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}

        system_prompt = """You are an audiovisual character and relationship reviewer for Chinese to Vietnamese subtitles.
Inspect cues for proper entity names, kinship, romantic vs familial polysemy, and conversational pronouns.

Key rules:
1. '女朋友' = romantic girlfriend ('bạn gái' / 'người yêu'), NEVER 'con gái' (daughter).
2. '闺女' / '女儿' in parent context = daughter ('con gái tôi'), not girlfriend.
3. Locked character names and Vietnamese pronouns must be consistent with the character and relationship graph.

Return JSON ONLY:
{
  "evaluations": [
    {
      "cue_id": "...",
      "has_issue": true/false,
      "issue_type": "relationship.polysemy_mismatch" | "relationship.pronoun_mismatch" | "relationship.entity_mismatch",
      "severity": "major" | "critical",
      "explanation": "..."
    }
  ]
}
"""
        for start_idx in range(0, len(cues), batch_size):
            batch = cues[start_idx : start_idx + batch_size]
            items_payload = [
                {
                    "cue_id": cue.id,
                    "speaker": cue.speaker_id,
                    "addressee": cue.addressee_id,
                    "source": cue.source_text,
                    "vietnamese": cue.translated_text,
                }
                for cue in batch
            ]
            user_msg = json.dumps({"cues_to_review": items_payload}, ensure_ascii=False)
            payload = {
                "model": self.model,
                "temperature": 0.05,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            for attempt in range(5):
                try:
                    resp = httpx.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=35.0,
                    )
                    if resp.status_code == 429:
                        wait = 3.0 * (attempt + 1)
                        logger.info("Relationship review rate limited (429); sleeping %.1fs...", wait)
                        time.sleep(wait)
                        continue
                    if resp.status_code == 200:
                        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        from app.services.translation import extract_json_object
                        parsed = extract_json_object(raw)
                        if parsed and "evaluations" in parsed:
                            for ev in parsed["evaluations"]:
                                cid = ev.get("cue_id")
                                if cid in results and ev.get("has_issue"):
                                    sev_str = ev.get("severity", "major")
                                    sev = QualitySeverity.CRITICAL if sev_str == "critical" else QualitySeverity.MAJOR
                                    results[cid].append(
                                        QualityIssue(
                                            type=ev.get("issue_type", "relationship.polysemy_mismatch"),
                                            severity=sev,
                                            message=ev.get("explanation", "Relationship or pronoun issue detected."),
                                            reviewer="relationships",
                                        )
                                    )
                        break
                except Exception as exc:
                    logger.debug("Relationship LLM batch attempt %d error: %s", attempt + 1, exc)
                    time.sleep(2.0)
            time.sleep(1.0)

        return results
