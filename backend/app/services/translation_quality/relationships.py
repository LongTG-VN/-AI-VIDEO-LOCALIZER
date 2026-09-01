from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.models.project import Character, Project, RelationshipRule, SubtitleCue
from app.services.translation_quality.models import (
    QualityIssue,
    QualitySeverity,
    TranslationContextCard,
)

logger = logging.getLogger(__name__)

# Key Chinese relationship & polysemy terms
RELATIONSHIP_TERMS = {
    "女儿": "con gái",
    "闺女": "con gái",
    "女朋友": "bạn gái",
    "对象": "người yêu / đối tượng",
    "朋友": "bạn bè",
    "妹妹": "em gái",
    "姐姐": "chị gái",
    "哥哥": "anh trai",
    "弟弟": "em trai",
    "老婆": "vợ",
    "丈夫": "chồng",
    "同事": "đồng nghiệp",
    "老板": "sếp / ông chủ",
    "阿姨": "dì / cô",
    "叔叔": "chú / bác",
}


class RelationshipReviewer:
    """Pass 3: Entity, Character, and Relationship Polysemy Reviewer.
    
    Verifies:
    - Character names match canonical project definitions
    - Pronoun consistency across conversations
    - Polysemous relationship terms (e.g. 女朋友/对象 vs 女儿) match Character Graph
    """

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model

    def _check_deterministic_relationships(self, project: Project, cue: SubtitleCue) -> list[QualityIssue]:
        """Helper for checking single cue relationship consistency."""
        issues: list[QualityIssue] = []
        src = (cue.source_text or "").strip()
        vi = (cue.translated_text or "").strip()

        # Check: romantic term (女朋友, 对象) rendered as "con gái" (daughter)
        if ("女朋友" in src or "对象" in src) and re.search(r"\bcon\s+gái\b", vi, re.IGNORECASE):
            issues.append(
                QualityIssue(
                    type="relationship.polysemy_mismatch",
                    severity=QualitySeverity.CRITICAL,
                    message="Romantic relationship term (女朋友/对象) mistranslated as daughter ('con gái').",
                    source_span=src,
                    target_span=vi,
                    reviewer="relationships",
                )
            )

        # Check: kinship term (闺女 / 女儿) rendered as girlfriend (bạn gái)
        if "闺女" in src and re.search(r"\bbạn\s+gái\b", vi, re.IGNORECASE):
            issues.append(
                QualityIssue(
                    type="relationship.polysemy_mismatch",
                    severity=QualitySeverity.CRITICAL,
                    message="Kinship term '闺女' (daughter) mistranslated as girlfriend ('bạn gái').",
                    source_span=src,
                    target_span=vi,
                    reviewer="relationships",
                )
            )

        return issues

    def evaluate_cues(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None = None,
        batch_size: int = 6,
    ) -> dict[str, list[QualityIssue]]:
        issues_by_id: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}

        # 1. Deterministic Polysemy & Relationship Checks
        for cue in cues:
            det_issues = self._check_deterministic_relationships(project, cue)
            issues_by_id[cue.id].extend(det_issues)

            # Check: canonical names in Character Card
            src = (cue.source_text or "").strip()
            vi = (cue.translated_text or "").strip()
            if context_card:
                for char in context_card.characters:
                    if char.name_zh and char.name_zh in src:
                        expected_vi = char.name_vi or char.canonical_name
                        if expected_vi and expected_vi.lower() not in vi.lower():
                            issues_by_id[cue.id].append(
                                QualityIssue(
                                    type="entity.canonical_name_mismatch",
                                    severity=QualitySeverity.MAJOR,
                                    message=f"Character '{char.name_zh}' expected name '{expected_vi}' is missing in translation.",
                                    source_span=src,
                                    target_span=vi,
                                    reviewer="relationships",
                                )
                            )

        # 2. LLM Relationship & Pragmatic Evaluation
        if self.base_url and self.model:
            try:
                llm_issues = self._call_llm_relationship_check(project, cues, context_card, batch_size=batch_size)
                for cid, issues in llm_issues.items():
                    if cid in issues_by_id:
                        issues_by_id[cid].extend(issues)
            except Exception as exc:
                logger.warning("Relationship LLM review error (fallback to deterministic): %s", exc)

        return issues_by_id

    def _call_llm_relationship_check(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None = None,
        batch_size: int = 6,
    ) -> dict[str, list[QualityIssue]]:
        issues_by_id: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}

        system_prompt = """You are an expert subtitle quality auditor specializing in Vietnamese honorifics, pronouns, and kinship/relationship polysemy.
Your job is to audit translated subtitles against the Character Graph and relationship context.

Check for:
1. Polysemy errors: romantic partner (女朋友, 对象) becoming kinship (con gái, em gái), or vice versa.
2. Inconsistent pronoun registers between characters.
3. Character canonical name errors.

Return JSON ONLY:
{
  "issues": [
    {
      "cue_id": "...",
      "type": "relationship.polysemy_mismatch|relationship.pronoun_mismatch|entity.name_error",
      "severity": "major|critical",
      "message": "...",
      "source_span": "...",
      "target_span": "..."
    }
  ]
}
"""

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for start_idx in range(0, len(cues), batch_size):
            batch = cues[start_idx : start_idx + batch_size]
            items = [
                {
                    "cue_id": c.id,
                    "source": c.source_text,
                    "vietnamese": c.translated_text or "",
                    "speaker": c.speaker_id,
                }
                for c in batch
            ]

            payload = {
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({"cues": items, "context": context_card.model_dump() if context_card else {}}, ensure_ascii=False)},
                ],
            }

            for attempt in range(5):
                try:
                    response = httpx.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=35.0)
                    if response.status_code == 429:
                        time.sleep(3.0 * (attempt + 1))
                        continue
                    if response.status_code == 413:
                        break
                    response.raise_for_status()
                    content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                    if m:
                        text = m.group(1)
                    parsed = json.loads(text)
                    for item in parsed.get("issues", []):
                        cid = item.get("cue_id")
                        if cid and cid in issues_by_id:
                            sev = QualitySeverity.CRITICAL if item.get("severity") == "critical" else QualitySeverity.MAJOR
                            issues_by_id[cid].append(
                                QualityIssue(
                                    type=item.get("type", "relationship.polysemy_mismatch"),
                                    severity=sev,
                                    message=item.get("message", "Relationship/entity issue detected"),
                                    source_span=item.get("source_span"),
                                    target_span=item.get("target_span"),
                                    reviewer="relationships",
                                )
                            )
                    break
                except Exception as exc:
                    if attempt < 4:
                        time.sleep(2.0)
                        continue
                    logger.debug("Relationship review batch failed: %s", exc)

        return issues_by_id
