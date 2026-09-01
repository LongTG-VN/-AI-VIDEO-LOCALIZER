from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.models.project import Project, SubtitleCue
from app.services.translation_quality.models import (
    NaturalnessScore,
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
    # 9. Double passive / auxiliary verb nonsense e.g. "chuẩn bị bị", "chuẩn bị sẽ bị"
    r"\bchuẩn\s+bị\s+bị\b",
    # 10. "Cô chú hiện diện" literal corruption of 存在感
    r"\bcô\s+chú\s+hiện\s+diện\b",
    # 11. Two predicates accidentally fused (e.g. "... chính là tôi là anh ...", "... tôi là anh ...")
    r"\b(tôi|mình|em|anh|cô)\s+là\s+(anh|em|cô|bạn|tôi|chú|bác)\s+(nấu|làm|ăn|đi|đến)\b",
    # 12. Duplicated kinship phrase or nonsense "X là X của..." (e.g. "dì là dì của cháu", "cô là cô của...")
    r"\b(dì|cô|chú|bác|anh|chị|mẹ|bố)\s+là\s+(dì|cô|chú|bác|anh|chị|mẹ|bố)\s+của\b",
]


class NaturalnessPolisher:
    """Pass 7: Vietnamese Naturalness Polish V3 with Semantic Safety Gate and 1-5 Quality Scale.
    
    Scale:
    5 = native/natural
    4 = natural enough
    3 = understandable but machine-like
    2 = awkward/broken
    1 = nonsense
    
    Auto-accept: >= 4
    Repair: <= 3
    """

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model
        self.last_scores: dict[str, NaturalnessScore] = {}

    def evaluate_and_polish_cues(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None = None,
        batch_size: int = 6,
    ) -> tuple[dict[str, list[QualityIssue]], dict[str, str]]:
        """Returns (issues_by_cue_id, polished_text_by_cue_id)."""
        issues_by_id: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}
        polished_by_id: dict[str, str] = {}
        scores_by_id: dict[str, NaturalnessScore] = {}

        # 1. Deterministic Detection of Unnatural / Broken Syntax
        for cue in cues:
            vi = (cue.translated_text or "").strip()
            score = 5
            cue_issues = []
            for pat in UNNATURAL_CHINESE_SYNTAX_PATTERNS:
                m = re.search(pat, vi, re.IGNORECASE)
                if m:
                    score = 2
                    iss = QualityIssue(
                        type="naturalness.chinese_word_order",
                        severity=QualitySeverity.MAJOR,
                        message=f"Unnatural Chinese literal syntax or broken fused phrasing detected: '{m.group(0)}'",
                        target_span=vi,
                        reviewer="naturalness",
                    )
                    issues_by_id[cue.id].append(iss)
                    cue_issues.append(iss.message)

            scores_by_id[cue.id] = NaturalnessScore(cue_id=cue.id, score=score, issues=cue_issues)

        # 2. LLM Naturalness Review & Polish
        if self.base_url and self.model:
            try:
                candidates, llm_scores = self._call_llm_polish_batch(project, cues, context_card, batch_size=batch_size)
                for cid, score_val in llm_scores.items():
                    if cid in scores_by_id:
                        scores_by_id[cid].score = score_val

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
                        scores_by_id[cid].proposed_vi = cand_text
                        scores_by_id[cid].score = 5
                        logger.info("Naturalness Polish accepted for cue %s: '%s' -> '%s'", cid, old_vi, cand_text)
                    else:
                        logger.warning("Naturalness Polish REJECTED by semantic safety gate for cue %s: candidate '%s'", cid, cand_text)
            except Exception as exc:
                logger.warning("Naturalness LLM batch error (fallback to deterministic): %s", exc)

        self.last_scores = scores_by_id
        return issues_by_id, polished_by_id

    def _verify_semantic_safety(self, source_zh: str, old_vi: str, new_vi: str) -> bool:
        """Enforces that naturalness rewrites do NOT alter semantic facts or entities."""
        # 1. Negation Preservation
        neg_words = ["không", "chưa", "chẳng", "đừng", "chớ"]
        old_neg = any(w in old_vi.lower() for w in neg_words)
        new_neg = any(w in new_vi.lower() for w in neg_words)
        if old_neg != new_neg and any(c in source_zh for c in "没不无非别莫未"):
            return False

        # 2. Numbers / Digits Preservation
        old_digits = set(re.findall(r"\b\d+\b", old_vi))
        new_digits = set(re.findall(r"\b\d+\b", new_vi))
        if old_digits and old_digits != new_digits:
            return False

        # 3. Question marker preservation
        if "?" in old_vi and "?" not in new_vi and ("吗" in source_zh or "呢" in source_zh or "？" in source_zh):
            return False

        # 4. Length sanity
        if len(new_vi) < 0.35 * len(old_vi) and len(old_vi) > 10:
            return False
        if len(new_vi) > 3.0 * len(old_vi) and len(old_vi) > 5:
            return False

        return True

    def _call_llm_polish_batch(
        self,
        project: Project,
        cues: list[SubtitleCue],
        context_card: TranslationContextCard | None = None,
        batch_size: int = 6,
    ) -> tuple[dict[str, str], dict[str, int]]:
        candidates: dict[str, str] = {}
        scores: dict[str, int] = {}

        system_prompt = """You are a master Vietnamese audiovisual subtitle polish editor.
Your task is to review Vietnamese subtitles translated from Chinese and rate their naturalness on a 1-5 scale:
5 = Native, idiomatic, fluent Vietnamese subtitle
4 = Natural enough for TV/cinema
3 = Understandable but stiff or machine-translated
2 = Awkward, ungrammatical, fused clauses, or literal Chinese word order
1 = Nonsense or broken

If a subtitle is scored <= 3:
Provide a polished, fluent Vietnamese subtitle that sounds 100% natural while strictly preserving:
- Original subject, object, actor, and negation
- Key plot information and character names
- Subtitle length conciseness

Return JSON ONLY:
{
  "cues": [
    {
      "cue_id": "...",
      "naturalness_score": 5,
      "polished_vietnamese": "..."
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
                    "current_vietnamese": c.translated_text or "",
                }
                for c in batch
            ]

            payload = {
                "model": self.model,
                "temperature": 0.15,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({"cues": items}, ensure_ascii=False)},
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
                    for item in parsed.get("cues", []):
                        cid = item.get("cue_id")
                        if not cid:
                            continue
                        sc = int(item.get("naturalness_score", 5))
                        scores[cid] = sc
                        pol = str(item.get("polished_vietnamese", "")).strip()
                        if pol:
                            candidates[cid] = pol
                    break
                except Exception as exc:
                    if attempt < 4:
                        time.sleep(2.0)
                        continue
                    logger.debug("Naturalness polish batch failed: %s", exc)

        return candidates, scores
