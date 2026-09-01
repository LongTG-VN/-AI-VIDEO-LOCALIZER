from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from app.models.project import Project, SubtitleCue
from app.services.semantic_context import build_neighbor_window, normalize_discourse_mode
from app.services.translation_quality.models import (
    CueQualityResult,
    QualityIssue,
    QualitySeverity,
    TranslationContextCard,
)

logger = logging.getLogger(__name__)


class TargetedRepairer:
    """Pass 5: Targeted Repair for Cues that Failed Quality Checks.
    
    CRITICAL POLICY:
    - Only cues with reported issues enter repair.
    - Preserves already correct portions of the translation.
    - Max 2 retries per failed cue.
    """

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model

    def repair_failed_cues(
        self,
        project: Project,
        cues_to_repair: list[SubtitleCue],
        issues_by_cue_id: dict[str, list[QualityIssue]],
        context_card: TranslationContextCard | None = None,
        batch_size: int = 4,
    ) -> dict[str, tuple[str, float | None]]:
        """Repairs only the provided failed cues and returns {cue_id: (repaired_text, confidence)}."""
        if not cues_to_repair or not self.base_url or not self.model:
            return {}

        results: dict[str, tuple[str, float | None]] = {}

        system_prompt = """You are a precision audiovisual subtitle repair engine for Chinese to Vietnamese localization.
Your job is to repair ONLY the specific quality issues identified for each cue.

STRICT REPAIR RULES:
1. CUE OWNERSHIP: Fix ONLY the translation for the exact cue_id. Do NOT borrow content or names from neighboring cues.
2. PRESERVATION: Keep all parts of the current Vietnamese translation that are already accurate and natural.
3. ISSUE FIXING: Correct the specific diagnosed issues (polysemy, wrong negation, missing clause, literal idiom, unnatural word order, or content migration).
4. NATURAL VIETNAMESE: Output idiomatic, complete Vietnamese appropriate for subtitles.

Return JSON ONLY:
{
  "repairs": [
    {
      "cue_id": "...",
      "repaired_vietnamese": "...",
      "confidence": 0.95,
      "repair_notes": "..."
    }
  ]
}
"""

        for start_idx in range(0, len(cues_to_repair), batch_size):
            batch = cues_to_repair[start_idx : start_idx + batch_size]
            items_payload = []
            for cue in batch:
                cue_issues = issues_by_cue_id.get(cue.id, [])
                issues_summary = [f"[{iss.type}] {iss.message}" for iss in cue_issues]

                # find global index in project for context window
                try:
                    p_idx = next(i for i, c in enumerate(project.cues) if c.id == cue.id)
                    window = build_neighbor_window(project, p_idx, before=2, after=1)
                except Exception:
                    window = {"previous": [], "next": []}

                items_payload.append({
                    "cue_id": cue.id,
                    "source": cue.source_text,
                    "current_vietnamese": cue.translated_text,
                    "diagnosed_issues": issues_summary,
                    "previous_context": window["previous"],
                    "next_context": window["next"],
                })

            user_msg = json.dumps({"cues_to_repair": items_payload}, ensure_ascii=False)
            payload = {
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            }
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            for attempt in range(6):
                try:
                    resp = httpx.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=35.0,
                    )
                    if resp.status_code == 429:
                        wait = 3.0 * (attempt + 1)
                        logger.info("Repair rate limited (429); sleeping %.1fs...", wait)
                        time.sleep(wait)
                        continue
                    if resp.status_code == 200:
                        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        from app.services.translation import extract_json_object
                        parsed = extract_json_object(raw)
                        if parsed and "repairs" in parsed:
                            for rep in parsed["repairs"]:
                                cid = rep.get("cue_id")
                                txt = str(rep.get("repaired_vietnamese", "")).strip()
                                conf = float(rep["confidence"]) if rep.get("confidence") is not None else None
                                if cid and txt:
                                    results[cid] = (txt, conf)
                            break
                except Exception as exc:
                    logger.warning("Targeted repair attempt %d failed: %s", attempt + 1, exc)
                    time.sleep(2.0)
            time.sleep(1.0)

        return results
