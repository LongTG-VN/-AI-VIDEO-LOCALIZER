from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from app.models.project import Character, Project, SubtitleCue
from app.services.semantic_context import project_name_locks
from app.services.semantic_grouping.allocator import SemanticAllocator
from app.services.semantic_grouping.models import (
    SemanticAllocationUnit,
    SemanticTranslationGroup,
)

logger = logging.getLogger(__name__)


class GroupTranslator:
    """Translates SemanticTranslationGroups as complete thoughts and allocates Vietnamese text back to cues."""

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model
        self.allocator = SemanticAllocator()

    def translate_and_allocate_groups(
        self,
        project: Project,
        groups: list[SemanticTranslationGroup],
        batch_size: int = 4,
    ) -> list[SemanticTranslationGroup]:
        if not groups:
            return []

        if not self.base_url or not self.model:
            # Fallback allocation using existing or combined text
            for grp in groups:
                full_vi = " ".join(st for st in grp.source_texts)
                self.allocator.allocate(project, grp, full_vi)
            return groups

        system_prompt = """You are a master Chinese-to-Vietnamese audiovisual translation engine.
Your mission is to translate SEMANTIC TRANSLATION GROUPS into natural, coherent Vietnamese subtitles.

RULES:
1. COMPLETE THOUGHT: Translate the combined_source_text as ONE complete, cohesive sentence.
2. NATURAL IDIOM & TONE: Adapt Chinese idioms, humor, sarcasm, and figurative expressions into natural Vietnamese (do NOT translate word-for-word if awkward).
3. ALLOCATION: For multi-cue groups, distribute the full Vietnamese translation into natural, readable subtitle chunks for each constituent cue_id.
4. NO ORPHAN CONNECTORS: Subtitle chunks must NOT start or end with dangling connectors (mà, vì, nên, là, của, để).
5. TIMING COHESION: Keep canonical character names and key facts anchored to the respective source cue.

Return JSON ONLY:
{
  "groups": [
    {
      "group_id": "...",
      "full_vi": "...",
      "allocations": [
        {
          "cue_id": "...",
          "allocated_vi": "..."
        }
      ]
    }
  ]
}
"""

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        name_locks = project_name_locks(project)

        for start_idx in range(0, len(groups), batch_size):
            batch = groups[start_idx : start_idx + batch_size]
            items_payload = []
            for g in batch:
                items_payload.append({
                    "group_id": g.group_id,
                    "cues": [
                        {"cue_id": cid, "source_text": g.source_texts[i] if i < len(g.source_texts) else ""}
                        for i, cid in enumerate(g.source_cue_ids)
                    ],
                    "combined_source_text": g.combined_source_text,
                    "speaker": g.speaker_id,
                    "discourse_mode": g.discourse_mode,
                    "grouping_reason": g.grouping_reason,
                })

            payload = {
                "model": self.model,
                "temperature": 0.15,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"groups": items_payload, "name_locks": name_locks},
                            ensure_ascii=False,
                        ),
                    },
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

                    for g_item in parsed.get("groups", []):
                        gid = g_item.get("group_id")
                        matched_grp = next((g for g in batch if g.group_id == gid), None)
                        if not matched_grp:
                            continue
                        full_vi = g_item.get("full_vi", "")
                        allocs = g_item.get("allocations", [])
                        self.allocator.allocate(project, matched_grp, full_vi, llm_allocations=allocs)
                    break
                except Exception as exc:
                    if attempt < 4:
                        time.sleep(2.0)
                        continue
                    logger.warning("Group translation batch error (fallback to individual): %s", exc)
                    for matched_grp in batch:
                        if not matched_grp.allocations:
                            self.allocator.allocate(project, matched_grp, matched_grp.combined_source_text)

        return groups
