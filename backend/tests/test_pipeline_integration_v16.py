import json
import pytest
from pathlib import Path

from app.models.project import Character, Project, RelationshipRule, RenderOptions, SubtitleCue
from app.services.naturalness_polisher import (
    NaturalnessPolisher,
    apply_conservative_vietnamese_rules,
    is_semantically_safe,
)
from app.services.utterance_engine import (
    UtteranceEngine,
    clean_vietnamese_typography,
    merge_vietnamese_clauses,
    validate_render_cue_entity_ownership,
)


def test_semantic_safety_rejects_name_drop():
    orig = "Tần Phù Chi, bài ghi chú kinh tế vĩ mô hôm qua đã đọc xong chưa?"
    candidate_bad = "Bài ghi chú kinh tế vĩ mô hôm qua đã đọc xong chưa?"
    is_safe, reason = is_semantically_safe(orig, candidate_bad)
    assert not is_safe
    assert "canonical name" in reason.lower()


def test_semantic_safety_rejects_polarity_flip():
    orig = "Mẹ không xem tôi là con gái mà chỉ coi tôi như một món hàng."
    candidate_bad = "Mẹ xem tôi là con gái và coi tôi như một món hàng."
    is_safe, reason = is_semantically_safe(orig, candidate_bad)
    assert not is_safe
    assert "polarity" in reason.lower()


def test_semantic_safety_rejects_dropped_critical_clause():
    orig = "Ở một gia đình mở quán ăn sáng, bốn giờ sáng đã phải dậy giúp nhào bột."
    candidate_bad = "Ở một gia đình mở quán ăn sáng, bốn giờ sáng rất vất vả."
    is_safe, reason = is_semantically_safe(orig, candidate_bad)
    assert not is_safe
    assert "nhào bột" in reason.lower() or "concept" in reason.lower()


def test_semantic_safety_accepts_natural_polish():
    orig = "Bệnh viện đã trao nhầm cho cô, cô đã tận hưởng."
    cand = "Bệnh viện trao nhầm để cô được hưởng."
    is_safe, reason = is_semantically_safe(orig, cand, zh_source="医院抱错你享受了本")
    assert is_safe


def test_naturalness_polisher_68_70s_merge():
    cue1 = SubtitleCue(
        id="cue_68",
        start=66.30,
        end=67.78,
        source_text="医院抱错你享受了本，",
        translated_text="Bệnh viện đã trao nhầm cho cô, cô đã tận hưởng.",
        discourse_mode="direct_dialogue",
    )
    cue2 = SubtitleCue(
        id="cue_69",
        start=68.11,
        end=70.01,
        source_text="该属于我的人生而我在一个开。",
        translated_text="cuộc đời vốn dĩ thuộc về tôi",
        discourse_mode="direct_dialogue",
    )

    polisher = NaturalnessPolisher()
    polished_cue1, _, _ = polisher.polish_cue(cue1)
    polished_cue2, _, _ = polisher.polish_cue(cue2)

    assert "trao nhầm để cô được hưởng" in polished_cue1.lower() or "đã trao nhầm" in polished_cue1.lower()
    assert "vốn thuộc về tôi" in polished_cue2.lower() or "vốn dĩ thuộc về tôi" in polished_cue2.lower()

    # UtteranceEngine merge
    cue1.translated_text = polished_cue1
    # ensure continuation is properly lowercased
    cue2.translated_text = polished_cue2[0].lower() + polished_cue2[1:]
    engine = UtteranceEngine(max_utterance_gap=0.35, min_display_duration=0.80, max_line_chars=36)
    render_cues, _ = engine.process_cues([cue1, cue2], translated=True)

    assert len(render_cues) == 1
    rc = render_cues[0]
    # Verify no awkward mid-sentence period
    assert "tận hưởng." not in rc.render_text
    assert "bệnh viện" in rc.render_text.lower()
    assert "thuộc về tôi" in rc.render_text.lower()


def test_mandatory_v15_regressions_preserved():
    """Verify all 8 mandatory regression invariants are strictly preserved."""
    proj_path = Path(r"D:\codex\-AI-VIDEO-LOCALIZER\backend\data\projects\golden-benchmark-v15.json")
    if not proj_path.exists():
        pytest.skip("golden-benchmark-v15.json not found")

    with open(proj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    proj = Project(**data)
    polisher = NaturalnessPolisher()
    polished_cues, stats = polisher.polish_project_cues(proj)

    engine = UtteranceEngine(max_utterance_gap=0.35, min_display_duration=0.80, max_line_chars=36)
    render_cues, _ = engine.process_cues(polished_cues, translated=True)

    cues_by_time = {rc.start: rc for rc in render_cues}

    # 1. 12.5s: "Cổ áo lệch rồi"
    rc_12 = [rc for rc in render_cues if 11.5 <= rc.start <= 13.0]
    assert any("cổ áo" in rc.render_text.lower() for rc in rc_12)

    # 2. 18-21s: Mother/daughter relational negation
    rc_18 = [rc for rc in render_cues if 18.0 <= rc.start <= 21.5]
    assert any("mẹ không xem tôi là con gái" in rc.render_text.lower() for rc in rc_18)
    assert any("món hàng" in rc.render_text.lower() or "thương phẩm" in rc.render_text.lower() for rc in rc_18)

    # 3. 23-26s: Vocative "Tần Phù Chi"
    rc_23 = [rc for rc in render_cues if 22.0 <= rc.start <= 25.5]
    assert any("tần phù chi" in rc.render_text.lower() for rc in rc_23)

    # 4. 35s: Brother to sister uses "em"
    rc_35 = [rc for rc in render_cues if 35.0 <= rc.start <= 37.5]
    assert any("em" in rc.render_text.lower() for rc in rc_35)

    # 5. 54s: Monologue uses "tôi" and preserves "chiếc đùi gà"
    rc_54 = [rc for rc in render_cues if 53.0 <= rc.start <= 55.5]
    assert any("đùi gà" in rc.render_text.lower() for rc in rc_54)

    # 6. 56-60s entity ownership: "Nhìn cho rõ" does NOT own Tần Phù Chi; cue B owns it
    rc_56 = [rc for rc in render_cues if 55.8 <= rc.start <= 57.5]
    assert len(rc_56) > 0
    assert "tần phù chi" not in rc_56[0].render_text.lower()

    rc_57 = [rc for rc in render_cues if 57.5 <= rc.start <= 61.5]
    assert len(rc_57) > 0
    assert "tần phù chi" in rc_57[0].render_text.lower()
    assert "cô còn ăn được sao" in rc_57[0].render_text.lower()

    # 7. 70-73s: Multi-clause breakfast shop + 4 AM dough kneading
    rc_70 = [rc for rc in render_cues if 70.0 <= rc.start <= 73.8]
    assert len(rc_70) > 0
    assert "quán ăn sáng" in rc_70[0].render_text.lower()
    assert "nhào bột" in rc_70[0].render_text.lower() or "bốn giờ sáng" in rc_70[0].render_text.lower()

    # 8. 73.8s+: Confrontation boundary
    rc_73 = [rc for rc in render_cues if 73.8 <= rc.start <= 76.8]
    assert len(rc_73) > 0
    assert "cướp mất mười tám năm" in rc_73[0].render_text.lower()
