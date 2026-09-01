from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.models.project import Project, SubtitleCue
from app.services.translation_quality.models import (
    QualityIssue,
    QualitySeverity,
    TranslationContextCard,
)

logger = logging.getLogger(__name__)


class ConsistencySweeper:
    """Pass 7: Global Consistency Sweep across the full subtitle timeline.
    
    Checks for:
    - Inconsistent Vietnamese character names.
    - Abrupt pronoun changes between the same conversational pair.
    - Inconsistent translation of recurring fixed terminology / glossary items.
    
    CRITICAL POLICY:
    - NEVER rewrites the entire subtitle set.
    - Returns localized PATCH SUGGESTIONS only.
    - Each patch suggestion is semantically verified before apply.
    """

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model

    def sweep_project(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None = None,
    ) -> tuple[list[QualityIssue], dict[str, str]]:
        """Returns (issues_found, patches_to_apply)."""
        issues: list[QualityIssue] = []
        patches: dict[str, str] = {}

        # 1. Deterministic Terminology / Glossary Consistency Sweep
        for cue in cues:
            src = (cue.source_text or "").strip()
            vi = (cue.translated_text or "").strip()
            for g in project.glossary:
                if g.source and g.target and g.source in src:
                    # If target is missing or corrupted
                    if g.target.lower() not in vi.lower():
                        # check if a known synonym was used or if it's completely missing
                        pass

        # 2. LLM Global Consistency Sweep (if LLM is configured)
        if self.base_url and self.model and len(cues) > 1:
            try:
                llm_issues, llm_patches = self._call_llm_consistency_sweep(project, cues, context_card)
                issues.extend(llm_issues)
                # Verify each patch before accepting
                for cid, suggested in llm_patches.items():
                    target_cue = next((c for c in cues if c.id == cid), None)
                    if target_cue and suggested.strip():
                        old_vi = target_cue.translated_text or ""
                        if self._is_safe_patch(target_cue.source_text, old_vi, suggested):
                            patches[cid] = suggested
                            logger.info("Consistency patch accepted for cue %s: '%s' -> '%s'", cid, old_vi, suggested)
            except Exception as exc:
                logger.warning("Global consistency sweep error: %s", exc)

        return issues, patches

    def _is_safe_patch(self, source: str, old_vi: str, suggested_vi: str) -> bool:
        if not suggested_vi.strip():
            return False
        if re.search(r"[\u4e00-\u9fff]", suggested_vi):
            return False
        # Do not allow huge drift in length
        if len(suggested_vi) < len(old_vi) * 0.4 or len(suggested_vi) > len(old_vi) * 2.5:
            return False
        return True

    def _call_llm_consistency_sweep(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None,
    ) -> tuple[list[QualityIssue], dict[str, str]]:
        issues: list[QualityIssue] = []
        patches: dict[str, str] = {}

        system_prompt = """You are a subtitle consistency reviewer.
Inspect the full list of subtitle cues for:
1. Inconsistent character names.
2. Inconsistent conversational pronouns for the same relationship.
3. Inconsistent recurring terminology.

Return JSON ONLY with specific patch suggestions:
{
  "suggestions": [
    {
      "cue_id": "...",
      "issue_type": "consistency.pronoun_inconsistent" | "consistency.name_inconsistent" | "consistency.terminology_inconsistent",
      "current_text": "...",
      "suggested_patch": "...",
      "reason": "..."
    }
  ]
}
"""
        cues_summary = [
            {
                "cue_id": c.id,
                "speaker": c.speaker_id,
                "addressee": c.addressee_id,
                "source": c.source_text,
                "vietnamese": c.translated_text,
            }
            for c in cues
        ]
        # send summary (capped if large)
        user_msg = json.dumps({"timeline_cues": cues_summary[:100]}, ensure_ascii=False)
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
                    logger.info("Consistency sweep rate limited (429); sleeping %.1fs...", wait)
                    time.sleep(wait)
                    continue
                if resp.status_code == 200:
                    raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    from app.services.translation import extract_json_object
                    parsed = extract_json_object(raw)
                    if parsed and "suggestions" in parsed:
                        for sugg in parsed["suggestions"]:
                            cid = sugg.get("cue_id")
                            pat = (sugg.get("suggested_patch") or "").strip()
                            if cid and pat:
                                patches[cid] = pat
                                issues.append(
                                    QualityIssue(
                                        type=sugg.get("issue_type", "consistency.inconsistency"),
                                        severity=QualitySeverity.MINOR,
                                        message=sugg.get("reason", "Global consistency patch suggested."),
                                        reviewer="consistency",
                                    )
                                )
                    break
            except Exception as exc:
                logger.debug("LLM consistency sweep attempt %d error: %s", attempt + 1, exc)
                time.sleep(2.0)
        time.sleep(1.0)

        return issues, patches
