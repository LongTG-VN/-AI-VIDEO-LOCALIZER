from __future__ import annotations

import logging
from typing import Any

from app.models.project import Project
from app.services.translation_quality.models import (
    CharacterCard,
    RelationshipCard,
    TranslationContextCard,
)

logger = logging.getLogger(__name__)

# Common Chinese audiovisual polysemy and ambiguous terms that require relational / discourse context
COMMON_AMBIGUOUS_TERMS = [
    "女儿",  # daughter vs girlfriend (when misunderstood/slang)
    "女朋友",  # romantic girlfriend vs female friend
    "朋友",  # friend vs romantic partner / date
    "妹妹",  # biological younger sister vs younger female acquaintance/affectionate term
    "姐姐",  # biological elder sister vs older female acquaintance/respectful address
    "哥哥",  # biological elder brother vs boyfriend/older male
    "弟弟",  # biological younger brother vs younger male
    "老婆",  # wife vs informal affectionate address
    "丈夫",  # husband
    "对象",  # romantic partner / boyfriend / girlfriend vs target / object
    "同事",  # colleague / coworker
    "老板",  # boss / employer / shop owner
    "手艺",  # cooking craft / culinary skill vs handicraft
    "死缓",  # figurative death sentence (e.g. outclassing takeout food) vs literal legal stay of execution
    "包养",  # sugar dating / financial support
    "扶贫",  # poverty alleviation vs figurative helping out someone in financial need
    "领口",  # collar / neckline (NOT necklace / vòng cổ)
    "动筷子",  # start eating / begin the meal with chopsticks
    "拉丝",  # sparkling/lingering gaze (in eye context) vs cheese pulling
]


class ContextCardBuilder:
    """Pass 0: Build a centralized, read-only Video Translation Context Card.
    
    Reuses existing character graph, relationships, scenes, and glossary.
    Must NEVER inject spoken names into cues that lack them, and must NEVER move
    content between cues.
    """

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def build_context_card(self, project: Project) -> TranslationContextCard:
        # 1. Characters
        char_cards: list[CharacterCard] = []
        for c in project.characters:
            char_cards.append(
                CharacterCard(
                    character_id=c.id,
                    canonical_name=c.name_vi or c.name,
                    name_zh=c.name_zh or c.name,
                    name_vi=c.name_vi or c.name,
                    aliases=list(c.aliases),
                    gender_if_known=c.gender,
                    role=c.role,
                    description=c.description or c.notes,
                )
            )

        # 2. Relationships
        rel_cards: list[RelationshipCard] = []
        for r in project.relationships:
            pronouns: dict[str, str] = {}
            if r.vi_self or r.vi_self_pronoun:
                pronouns["speaker"] = r.vi_self or r.vi_self_pronoun or ""
            if r.vi_other or r.vi_target_pronoun:
                pronouns["listener"] = r.vi_other or r.vi_target_pronoun or ""
            rel_cards.append(
                RelationshipCard(
                    from_character_id=r.from_character_id,
                    to_character_id=r.to_character_id,
                    type=r.relationship or r.relationship_type or "unknown",
                    preferred_vi_pronouns=pronouns,
                    confidence=r.confidence,
                )
            )

        # 3. Terminology & Glossary
        terminology: dict[str, str] = {}
        for g in project.glossary:
            if g.source and g.target:
                terminology[g.source.strip()] = g.target.strip()

        # 4. Scenes summary / genre / tone
        scene_summaries = [s.summary for s in project.scenes if s.summary]
        story_summary = " | ".join(scene_summaries) if scene_summaries else None
        
        tones = [s.tone for s in project.scenes if s.tone]
        tone = ", ".join(set(tones)) if tones else "dramatic, natural audiovisual dialogue"

        style_rules = {
            "narration": "Translate as clear, natural Vietnamese narration. Preserve character perspectives and tone without injecting artificial dialogue vocatives.",
            "dialogue": "Use appropriate Vietnamese conversational pronouns (anh/em, tôi/cô, chú/cháu, etc.) based on directional relationships and relative social status. Avoid mechanical or literal word-for-word Chinese syntax.",
            "idioms": "Translate figurative and humorous expressions into idiomatic Vietnamese equivalents that match the intended pragmatic tone (humorous, sarcastic, affectionate, dramatic).",
        }

        # 5. Detect project-relevant ambiguous terms
        all_source_text = " ".join([c.source_text for c in project.cues])
        found_ambiguous = [term for term in COMMON_AMBIGUOUS_TERMS if term in all_source_text]

        card = TranslationContextCard(
            story_summary=story_summary,
            genre=None,
            tone=tone,
            characters=char_cards,
            relationships=rel_cards,
            terminology=terminology,
            ambiguous_terms=found_ambiguous,
            style_rules=style_rules,
        )
        logger.info(
            "Built TranslationContextCard: %d characters, %d relationships, %d terminology locks, %d ambiguous terms detected",
            len(card.characters),
            len(card.relationships),
            len(card.terminology),
            len(card.ambiguous_terms),
        )
        return card
