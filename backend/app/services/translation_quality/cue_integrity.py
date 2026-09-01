from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.models.project import Project, SubtitleCue
from app.services.semantic_context import source_name_mentions
from app.services.translation_quality.models import (
    QualityIssue,
    QualitySeverity,
    TranslationContextCard,
)

logger = logging.getLogger(__name__)


class CueIntegrityReviewer:
    """Pass 2: Review cue ownership, neighbor-content leakage, and timeline integrity.
    
    This reviewer specifically checks that:
    1. Every translation belongs ONLY to its source cue.
    2. Names or clauses from previous/next cues did NOT migrate into the current cue.
    3. Content was not duplicated or dropped across boundaries.
    4. Discourse boundaries and cue ordering are preserved.
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
        """Review integrity for a batch of cues and return issues grouped by cue_id."""
        results: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}

        # 1. Deterministic Rule-Based Checks (Fast, Zero-Cost)
        for i, cue in enumerate(cues):
            cue_issues = self._check_deterministic_integrity(project, cues, i)
            results[cue.id].extend(cue_issues)

        # 2. LLM-Based Subtle Migration / Cross-Cue Shifting Check
        if self.base_url and self.model:
            try:
                llm_issues = self._call_llm_integrity_check(project, cues, context_card, batch_size=batch_size)
                for cid, issues in llm_issues.items():
                    if cid in results:
                        results[cid].extend(issues)
            except Exception as exc:
                logger.warning("Cue integrity LLM evaluation encountered error (fallback to deterministic): %s", exc)

        return results

    def _check_deterministic_integrity(
        self,
        project: Project,
        all_cues: list[SubtitleCue],
        cue_idx: int,
    ) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        cue = all_cues[cue_idx]
        src = (cue.source_text or "").strip()
        vi = (cue.translated_text or "").strip()

        # Check A: Missing / Empty translation for non-empty source
        if src and not vi:
            issues.append(
                QualityIssue(
                    type="cue.missing_content",
                    severity=QualitySeverity.CRITICAL,
                    message="Cue has non-empty source text but missing/empty Vietnamese translation.",
                    source_span=src,
                    reviewer="cue_integrity",
                )
            )

        # Check B: Untranslated source Chinese passed as Vietnamese
        if re.search(r"[\u4e00-\u9fff]", vi):
            issues.append(
                QualityIssue(
                    type="cue.untranslated_source",
                    severity=QualitySeverity.CRITICAL,
                    message="Vietnamese translation contains raw Chinese characters.",
                    target_span=vi,
                    reviewer="cue_integrity",
                )
            )

        # Check C: Duplicate content across adjacent cues with different source text
        if cue_idx > 0 and vi:
            prev_cue = all_cues[cue_idx - 1]
            prev_vi = (prev_cue.translated_text or "").strip()
            prev_src = (prev_cue.source_text or "").strip()
            if prev_vi and vi.lower() == prev_vi.lower() and src != prev_src:
                issues.append(
                    QualityIssue(
                        type="cue.duplicate_content",
                        severity=QualitySeverity.MAJOR,
                        message="Adjacent cues output identical translation text despite having different source text.",
                        target_span=vi,
                        reviewer="cue_integrity",
                    )
                )

        # Check D: Name Migration — character name present in VI but completely absent in source text
        # and present in neighbor cues
        required_names = source_name_mentions(project, src)
        for char in project.characters:
            char_vi = (char.name_vi or char.name).strip()
            char_zh = (char.name_zh or char.name).strip()
            if not char_vi or len(char_vi) < 2:
                continue
            # If name is in VI text
            if re.search(r"\b" + re.escape(char_vi) + r"\b", vi, re.IGNORECASE):
                # Is it present in source text?
                in_src = (char_zh in src) or any(alias in src for alias in char.aliases if alias)
                if not in_src:
                    # Check if it was leaked from previous/next cues
                    prev_src = all_cues[cue_idx - 1].source_text if cue_idx > 0 else ""
                    next_src = all_cues[cue_idx + 1].source_text if cue_idx + 1 < len(all_cues) else ""
                    if char_zh in prev_src or char_zh in next_src:
                        issues.append(
                            QualityIssue(
                                type="cue.name_migration",
                                severity=QualitySeverity.MAJOR,
                                message=f"Character name '{char_vi}' was migrated from neighboring cues into current cue without source evidence.",
                                source_span=src,
                                target_span=char_vi,
                                reviewer="cue_integrity",
                            )
                        )

        return issues

    def _call_llm_integrity_check(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None,
        batch_size: int = 15,
    ) -> dict[str, list[QualityIssue]]:
        results: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}

        system_prompt = """You are an audiovisual subtitle integrity and timeline reviewer.
Your job is to inspect source vs Vietnamese subtitle pairs and verify:
1. CONTENT OWNERSHIP: Each translation belongs ONLY to its source cue.
2. CONTENT MIGRATION: Content from previous or next cues did NOT leak into the current cue.
3. CLAUSE SHIFTING: Semantic units are rendered in the correct cue matching the Chinese dialogue.

Return JSON ONLY:
{
  "evaluations": [
    {
      "cue_id": "...",
      "has_issue": true/false,
      "issue_type": "cue.content_migration" | "cue.neighbor_leakage" | "cue.missing_content",
      "severity": "major" | "critical",
      "explanation": "..."
    }
  ]
}
"""
        for start_idx in range(0, len(cues), batch_size):
            batch = cues[start_idx : start_idx + batch_size]
            items_payload = []
            for j, cue in enumerate(batch):
                global_idx = start_idx + j
                prev_src = cues[global_idx - 1].source_text if global_idx > 0 else None
                next_src = cues[global_idx + 1].source_text if global_idx + 1 < len(cues) else None
                items_payload.append({
                    "cue_id": cue.id,
                    "source": cue.source_text,
                    "vietnamese": cue.translated_text,
                    "previous_source": prev_src,
                    "next_source": next_src,
                })

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
                        logger.info("Cue integrity review rate limited (429); sleeping %.1fs...", wait)
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
                                            type=ev.get("issue_type", "cue.content_migration"),
                                            severity=sev,
                                            message=ev.get("explanation", "Content migration detected."),
                                            reviewer="cue_integrity",
                                        )
                                    )
                        break
                except Exception as exc:
                    logger.debug("Integrity check batch attempt %d error: %s", attempt + 1, exc)
                    time.sleep(2.0)
            time.sleep(1.0)

        return results
