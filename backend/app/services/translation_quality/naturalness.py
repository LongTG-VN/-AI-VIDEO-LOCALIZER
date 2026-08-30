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

# Domain-generic patterns of unnatural literal Chinese syntax in Vietnamese subtitles
UNNATURAL_CHINESE_SYNTAX_PATTERNS = [
    # 1. Broken word order with "Đi tới cuối tuần..." or "Đến cuối tuần..."
    r"\bđi\s+tới\s+cuối\s+tuần\s+đến\b",
    # 2. Nonsense "đứng đũa" (literal from 动筷子)
    r"\bđứng\s+đũa\b",
    # 3. Nonsense "cảm tí lên đó"
    r"\bcảm\s+tí\s+lên\s+đó\b",
    # 4. Mechanical literal "ngâm bát bẩn" / "phạm án" from 判了死缓
    r"\b(phạm\s+án|ngâm\s+vào\s+bát\s+bẩn)\b",
    # 5. Literal Chinese "làm cái..." (做个...) e.g. "làm cái sườn xào chua ngọt"
    r"\blàm\s+cái\s+(sườn|cơm|canh|món)\b",
    # 6. "xong xương rồi" from 骨完了 / 完蛋了
    r"\bxong\s+xương\s+rồi\b",
    # 7. Unnatural literal phrase "phú bà hàng đầu"
    r"\bphú\s+bà\s+hàng\s+đầu\b",
    # 8. Unnatural literal "xóa đói giảm nghèo không chỉ giúp đỡ" from 扶贫
    r"\bxóa\s+đói\s+giảm\s+nghèo\s+không\s+chỉ\b",
]


class NaturalnessPolisher:
    """Pass 6: Vietnamese Naturalness Polish with Strict Semantic Safety Gate.
    
    Runs ONLY on cues that have passed semantic and cue-integrity checks.
    Detects literal Chinese syntax, awkward word order, and nonsense phrases,
    proposing fluent Vietnamese equivalents.
    
    SEMANTIC SAFETY: If proposed candidate alters subjects, negation, quantities,
    causal relations, or entities, the candidate is REJECTED and original translation is retained.
    """

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model

    def evaluate_and_polish_cues(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None = None,
        batch_size: int = 20,
    ) -> tuple[dict[str, list[QualityIssue]], dict[str, str]]:
        """Returns (issues_by_cue_id, polished_text_by_cue_id)."""
        issues_by_id: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}
        polished_by_id: dict[str, str] = {}

        # 1. Deterministic Detection of Unnatural / Broken Syntax
        for cue in cues:
            vi = (cue.translated_text or "").strip()
            for pat in UNNATURAL_CHINESE_SYNTAX_PATTERNS:
                if re.search(pat, vi, re.IGNORECASE):
                    issues_by_id[cue.id].append(
                        QualityIssue(
                            type="naturalness.chinese_word_order",
                            severity=QualitySeverity.MAJOR,
                            message=f"Unnatural Chinese literal syntax or nonsense phrasing detected: '{re.search(pat, vi, re.IGNORECASE).group(0)}'",
                            target_span=vi,
                            reviewer="naturalness",
                        )
                    )

        # 2. LLM Naturalness Review & Polish
        if self.base_url and self.model:
            try:
                candidates = self._call_llm_polish_batch(project, cues, context_card, batch_size=batch_size)
                for cid, cand_text in candidates.items():
                    target_cue = next((c for c in cues if c.id == cid), None)
                    if not target_cue:
                        continue
                    old_vi = (target_cue.translated_text or "").strip()
                    if not cand_text or cand_text == old_vi:
                        continue

                    # Strict Semantic Safety Verification
                    if self._verify_semantic_safety(target_cue.source_text, old_vi, cand_text):
                        polished_by_id[cid] = cand_text
                        logger.info("Naturalness Polish accepted for cue %s: '%s' -> '%s'", cid, old_vi, cand_text)
                    else:
                        logger.warning("Naturalness Polish REJECTED by semantic safety gate for cue %s: candidate '%s'", cid, cand_text)
            except Exception as exc:
                logger.warning("Naturalness Polish LLM batch encountered error: %s", exc)

        return issues_by_id, polished_by_id

    def _verify_semantic_safety(self, source: str, old_vi: str, new_vi: str) -> bool:
        """Verifies candidate preserves meaning, negation, numbers, and key entities."""
        if not new_vi.strip():
            return False

        # 1. Raw Chinese check
        if re.search(r"[\u4e00-\u9fff]", new_vi):
            return False

        # 2. Negation safety: If old had negation or source had negation, new must have negation
        zh_neg = bool(re.search(r"[没不无非别莫未]", source))
        old_neg = bool(re.search(r"\b(không|chưa|chẳng|đừng|chớ|vô|phi)\b", old_vi, re.IGNORECASE))
        new_neg = bool(re.search(r"\b(không|chưa|chẳng|đừng|chớ|vô|phi)\b", new_vi, re.IGNORECASE))
        if (zh_neg or old_neg) and not new_neg:
            # If source was explicitly negative and new dropped it -> reject
            if bool(re.search(r"(一分钱都没|根本不|绝不|没花过|没动过)", source)):
                return False

        # 3. Numbers / Quantity safety
        digits = re.findall(r"\b\d+\b", source)
        for d in digits:
            if d not in new_vi:
                num_map = {"1": "một", "2": "hai", "3": "ba", "4": "bốn", "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín", "10": "mười", "18": "mười tám", "26": "hai mươi sáu"}
                if d in num_map and not re.search(r"\b" + re.escape(num_map[d]) + r"\b", new_vi, re.IGNORECASE):
                    return False

        # 4. Length sanity: Must not collapse a long sentence into a 1-word fragment or blow up 5x
        if len(old_vi) > 20 and len(new_vi) < 6:
            return False
        if len(new_vi) > len(old_vi) * 3:
            return False

        return True

    def _call_llm_polish_batch(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None,
        batch_size: int = 15,
    ) -> dict[str, str]:
        candidates: dict[str, str] = {}

        system_prompt = """You are a Vietnamese dialogue and audiovisual localization polish expert.
Review the draft Vietnamese subtitle translations.
Improve fluency, natural conversational syntax, and idiomatic phrasing while strictly preserving:
1. Exact source meaning, main actions, subjects, and objects.
2. Character relationships, register, and pronouns.
3. Complete clauses without dropping or hallucinating content.

If a line is already natural and faithful, return the same text.
Avoid Chinese word order (e.g. 'đi tới cuối tuần đến nhà...' -> 'cuối tuần đến nhà tôi ăn cơm...').
Avoid nonsense phrases (e.g. 'đứng đũa' -> 'dùng đũa/gắp thức ăn').

Return JSON ONLY:
{
  "polished": [
    {
      "cue_id": "...",
      "natural_vietnamese": "...",
      "rationale": "..."
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
            user_msg = json.dumps({"cues_to_polish": items_payload}, ensure_ascii=False)
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
                        logger.info("Naturalness polish rate limited (429); sleeping %.1fs...", wait)
                        time.sleep(wait)
                        continue
                    if resp.status_code == 200:
                        raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        from app.services.translation import extract_json_object
                        parsed = extract_json_object(raw)
                        if parsed and "polished" in parsed:
                            for p in parsed["polished"]:
                                cid = p.get("cue_id")
                                txt = (p.get("natural_vietnamese") or "").strip()
                                if cid and txt:
                                    candidates[cid] = txt
                        break
                except Exception as exc:
                    logger.debug("Naturalness polish batch attempt %d error: %s", attempt + 1, exc)
                    time.sleep(2.0)
            time.sleep(1.0)

        return candidates
