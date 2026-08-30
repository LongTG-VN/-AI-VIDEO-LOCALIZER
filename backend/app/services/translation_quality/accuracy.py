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


class AccuracyReviewer:
    """Pass 3: MQM-inspired semantic accuracy and idiom reviewer.
    
    Checks:
    - Omission / Addition of meaningful clauses
    - Mistranslation of key actions, subjects, objects, or negation
    - Literal translation of Chinese idioms and figurative expressions
    - Lost humor or pragmatics
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

        # 1. Deterministic Rule Checks (Negation, Quantities, Key Idioms)
        for cue in cues:
            det_issues = self._check_deterministic_accuracy(cue)
            results[cue.id].extend(det_issues)

        # 2. LLM MQM-Taxonomy Accuracy Evaluation
        if self.base_url and self.model:
            try:
                llm_issues = self._call_llm_accuracy_check(project, cues, context_card, batch_size=batch_size)
                for cid, issues in llm_issues.items():
                    if cid in results:
                        results[cid].extend(issues)
            except Exception as exc:
                logger.warning("Accuracy LLM review error (fallback to deterministic): %s", exc)

        return results

    def _check_deterministic_accuracy(self, cue: SubtitleCue) -> list[QualityIssue]:
        issues: list[QualityIssue] = []
        src = (cue.source_text or "").strip()
        vi = (cue.translated_text or "").strip()
        if not src or not vi:
            return issues

        # Check Negation consistency: e.g. source has 没/不/无/非 but Vietnamese lacks negation
        chinese_negation = bool(re.search(r"[没不无非别莫未]", src))
        vietnamese_negation = bool(re.search(r"\b(không|chưa|chẳng|đừng|chớ|vô|phi)\b", vi, re.IGNORECASE))
        # Note: certain rhetorical or affirmative expressions with 没 (like 没事) might be affirmative, but if strong contrast:
        if bool(re.search(r"(一分钱都没|什么都没有|根本不|绝不|没花过|没动过)", src)) and not vietnamese_negation:
            issues.append(
                QualityIssue(
                    type="accuracy.wrong_negation",
                    severity=QualitySeverity.CRITICAL,
                    message="Source clearly expresses strong negation but Vietnamese translation lacks negative marker.",
                    source_span=src,
                    target_span=vi,
                    reviewer="accuracy",
                )
            )

        # Check Numbers / Quantities (e.g. 8个月 -> 8 tháng)
        zh_digits = re.findall(r"\b\d+\b", src)
        for d in zh_digits:
            if d not in vi:
                # check spelled out Vietnamese (e.g. 8 -> tám)
                num_map = {"1": "một", "2": "hai", "3": "ba", "4": "bốn", "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín", "10": "mười", "18": "mười tám", "26": "hai mươi sáu"}
                if d in num_map and not re.search(r"\b" + re.escape(num_map[d]) + r"\b", vi, re.IGNORECASE):
                    issues.append(
                        QualityIssue(
                            type="accuracy.wrong_quantity",
                            severity=QualitySeverity.MAJOR,
                            message=f"Number '{d}' present in source text is missing in Vietnamese translation.",
                            source_span=src,
                            target_span=vi,
                            reviewer="accuracy",
                        )
                    )

        # Check known domain-critical mistranslations
        # e.g. 领口 (collar/neckline) translated as vòng cổ (necklace)
        if "领口" in src and re.search(r"\b(vòng cổ|dây chuyền)\b", vi, re.IGNORECASE):
            issues.append(
                QualityIssue(
                    type="accuracy.mistranslation",
                    severity=QualitySeverity.MAJOR,
                    message="'领口' means clothing neckline / collar ('cổ áo'), not necklace ('vòng cổ').",
                    source_span="领口",
                    target_span=vi,
                    reviewer="accuracy",
                )
            )

        # Check figurative idioms translated literally as word salad
        # e.g. 死缓 (stay of execution) in food context -> translated literally as legal death penalty or 'ngâm bát bẩn'
        if "死缓" in src and not re.search(r"\b(vượt trội|hơn hẳn|chết đứng|bỏ xa|đánh bại|chê bai)\b", vi, re.IGNORECASE):
            if re.search(r"\b(tử hình|tử hoãn|chết hoãn|bát bẩn|phạm án)\b", vi, re.IGNORECASE):
                issues.append(
                    QualityIssue(
                        type="idiom.literal_translation",
                        severity=QualitySeverity.MAJOR,
                        message="'给...判了死缓' is a humorous figurative expression (outclassing / making something obsolete), but was translated literally as legal death penalty.",
                        source_span="判了死缓",
                        target_span=vi,
                        reviewer="accuracy",
                    )
                )

        return issues

    def _call_llm_accuracy_check(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None,
        batch_size: int = 15,
    ) -> dict[str, list[QualityIssue]]:
        results: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}

        system_prompt = """You are an MQM-standard translation accuracy reviewer for Chinese to Vietnamese audiovisual subtitles.
Inspect each source vs Vietnamese pair for semantic fidelity.

Check ONLY:
1. ACCURACY:
   - omission (dropped clause or meaning)
   - addition (unsupported hallucinations)
   - mistranslation (wrong subject, action, object, negation, causal relationship)
2. IDIOM & FIGURATIVE SENSE:
   - literal word-for-word translation of idioms that produces nonsense or destroys humor.

Return JSON ONLY:
{
  "evaluations": [
    {
      "cue_id": "...",
      "has_issue": true/false,
      "issue_type": "accuracy.mistranslation" | "accuracy.omission" | "accuracy.addition" | "accuracy.wrong_negation" | "idiom.literal_translation",
      "severity": "minor" | "major" | "critical",
      "explanation": "..."
    }
  ]
}
"""
        for start_idx in range(0, len(cues), batch_size):
            batch = cues[start_idx : start_idx + batch_size]
            items_payload = [
                {"cue_id": cue.id, "source": cue.source_text, "vietnamese": cue.translated_text}
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
                        logger.info("Accuracy review rate limited (429); sleeping %.1fs...", wait)
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
                                            type=ev.get("issue_type", "accuracy.mistranslation"),
                                            severity=sev,
                                            message=ev.get("explanation", "Accuracy error detected."),
                                            reviewer="accuracy",
                                        )
                                    )
                        break
                except Exception as exc:
                    logger.debug("Accuracy LLM batch attempt %d error: %s", attempt + 1, exc)
                    time.sleep(2.0)
            time.sleep(1.0)

        return results
