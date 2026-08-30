from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.models.project import Project, SubtitleCue
from app.services.translation_quality.models import (
    FigurativeReviewResult,
    QualityIssue,
    QualitySeverity,
    TranslationContextCard,
)

logger = logging.getLogger(__name__)


class IdiomReviewer:
    """Pass 4: Idiom & Figurative Expression Reviewer with Candidate Ranking V2.
    
    Identifies Chinese idioms, sarcasm, jokes, metaphors, hyperbole, and figurative speech.
    Ensures they are adapted into natural Vietnamese punchlines that a native viewer immediately understands
    without needing to know the Chinese source.
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
        batch_size: int = 6,
    ) -> tuple[dict[str, list[QualityIssue]], dict[str, FigurativeReviewResult]]:
        issues_by_id: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}
        figurative_reviews: dict[str, FigurativeReviewResult] = {}

        # 1. Deterministic heuristic checks for known awkward literal idiom translations
        for cue in cues:
            src = (cue.source_text or "").strip()
            vi = (cue.translated_text or "").strip()

            # Check culinary death penalty idiom (判了死缓)
            if "死缓" in src:
                is_natural = any(w in vi.lower() for w in ["hết đường sống", "hết cửa sống", "dẹp tiệm", "hết thời", "không còn đường sống"])
                if not is_natural:
                    issues_by_id[cue.id].append(
                        QualityIssue(
                            type="idiom.literal_mistranslation",
                            severity=QualitySeverity.MAJOR,
                            message="Literal or awkward legal metaphor used for culinary compliment (making takeaway food obsolete).",
                            source_span=src,
                            target_span=vi,
                            reviewer="idioms",
                        )
                    )
                    figurative_reviews[cue.id] = FigurativeReviewResult(
                        cue_id=cue.id,
                        figurative=True,
                        literal_meaning="Phán án tử hình treo cho đồ ăn ngoài",
                        intended_meaning="Món ăn tự nấu ngon vượt trội khiến đồ ăn ngoài không còn đường sống / hết thời",
                        tone="humorous",
                        speaker_intention="Khen ngợi tay nghề nấu nướng",
                        status="FAIL",
                        issues=["Literal or awkward translation of culinary humor"],
                        candidate_vi="Cơm cậu nấu ngon thế này thì đồ ăn ngoài hết cửa sống luôn.",
                    )

        # 2. LLM Figurative Review & Candidate Ranking V2
        if self.base_url and self.model:
            try:
                llm_issues, llm_figs = self._call_llm_idiom_check(project, cues, context_card, batch_size=batch_size)
                for cid, iss_list in llm_issues.items():
                    if cid in issues_by_id:
                        issues_by_id[cid].extend(iss_list)
                for cid, fig_res in llm_figs.items():
                    figurative_reviews[cid] = fig_res
            except Exception as exc:
                logger.warning("Idiom LLM review error (fallback to deterministic): %s", exc)

        return issues_by_id, figurative_reviews

    def _call_llm_idiom_check(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None = None,
        batch_size: int = 6,
    ) -> tuple[dict[str, list[QualityIssue]], dict[str, FigurativeReviewResult]]:
        issues_by_id: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}
        reviews_by_id: dict[str, FigurativeReviewResult] = {}

        system_prompt = """You are an expert Chinese-to-Vietnamese subtitle localization editor specializing in IDIOMS, HUMOR, SARCASM, and FIGURATIVE LANGUAGE.
Your job is to ensure that Chinese figurative expressions are NOT translated literally if they sound awkward in Vietnamese.

For each cue:
1. Determine if it contains figurative language / idiom / humor / sarcasm.
2. If figurative:
   - Identify literal meaning vs intended contextual meaning.
   - Generate 2-3 candidate Vietnamese adaptations ranked by naturalness and native Vietnamese comprehension.
   - Ask: "Would a native Vietnamese viewer immediately understand the punchline without knowing Chinese?"
   - If the current translation is literal/awkward, output status: FAIL with the top ranked natural candidate.

Return JSON ONLY:
{
  "reviews": [
    {
      "cue_id": "...",
      "figurative": true,
      "literal_meaning": "...",
      "intended_meaning": "...",
      "tone": "humorous|sarcastic|insulting|affectionate|dramatic|neutral",
      "status": "PASS|FAIL",
      "issue_message": "...",
      "ranked_candidates": ["...", "..."],
      "candidate_vi": "..."
    }
  ]
}
"""

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for start_idx in range(0, len(cues), batch_size):
            batch = cues[start_idx : start_idx + batch_size]
            items_payload = [
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
                    {"role": "user", "content": json.dumps({"cues": items_payload}, ensure_ascii=False)},
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

                    for item in parsed.get("reviews", []):
                        cid = item.get("cue_id")
                        if not cid or cid not in issues_by_id:
                            continue
                        is_fig = bool(item.get("figurative", False))
                        status = item.get("status", "PASS")
                        fig_res = FigurativeReviewResult(
                            cue_id=cid,
                            figurative=is_fig,
                            literal_meaning=item.get("literal_meaning", ""),
                            intended_meaning=item.get("intended_meaning", ""),
                            tone=item.get("tone", "neutral"),
                            status="FAIL" if status == "FAIL" else "PASS",
                            issues=[item.get("issue_message")] if item.get("issue_message") else [],
                            candidate_vi=item.get("candidate_vi", ""),
                        )
                        reviews_by_id[cid] = fig_res
                        if status == "FAIL":
                            issues_by_id[cid].append(
                                QualityIssue(
                                    type="idiom.literal_mistranslation",
                                    severity=QualitySeverity.MAJOR,
                                    message=item.get("issue_message", "Literal translation of Chinese figurative expression"),
                                    reviewer="idioms",
                                )
                            )
                    break
                except Exception as exc:
                    if attempt < 4:
                        time.sleep(2.0)
                        continue
                    logger.debug("Idiom review batch failed: %s", exc)

        return issues_by_id, reviews_by_id
