from __future__ import annotations

import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from app.models.project import Character, Project, RenderOptions, SubtitleCue
from app.services.critic import CriticIssueEnum, deterministic_validate_cue
from app.services.subtitles import write_ass
from app.services.utterance_engine import (
    DiscourseMode,
    RenderSubtitleCue,
    UtteranceEngine,
    clean_vietnamese_typography,
    semantic_line_break,
    validate_render_cue_entity_ownership,
)


def _make_cue(
    cue_id: str,
    start: float,
    end: float,
    source: str,
    translated: str | None = None,
    speaker: str = "speaker_a",
    addressee: str | None = None,
    mode: str = "narration",
) -> SubtitleCue:
    return SubtitleCue(
        id=cue_id,
        start=start,
        end=end,
        source_text=source,
        translated_text=translated,
        speaker_id=speaker,
        addressee_id=addressee,
        discourse_mode=mode,
    )


def test_multi_clause_source_cannot_lose_meaningful_clause():
    """1. Multi-clause source cannot lose a meaningful clause before ASS."""
    zh_source = "在早餐店的家庭里，凌晨四点就要起来帮忙揉面。"
    
    # Incomplete translation missing 4 AM kneading dough
    ctx_incomplete = {
        "chinese_source": zh_source,
        "vietnamese_translation": "Trong một quán ăn sáng,",
    }
    is_pass, issues, notes = deterministic_validate_cue(ctx_incomplete)
    assert not is_pass
    assert CriticIssueEnum.MEANING_SHIFT.value in issues

    # Complete translation preserving both breakfast shop context and 4 AM kneading dough
    ctx_complete = {
        "chinese_source": zh_source,
        "vietnamese_translation": "Ở một gia đình mở quán ăn sáng, bốn giờ sáng đã phải dậy giúp nhào bột.",
    }
    is_pass_c, issues_c, _ = deterministic_validate_cue(ctx_complete)
    assert is_pass_c
    assert len(issues_c) == 0

    # UtteranceEngine processes complete cue and preserves all clauses
    cue = _make_cue("cue_narr", 70.31, 73.80, zh_source, ctx_complete["vietnamese_translation"])
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues([cue], translated=True)
    assert len(render_cues) == 1
    rc_text = render_cues[0].render_text.lower()
    assert "quán ăn sáng" in rc_text or "quán sáng" in rc_text
    assert "nhào bột" in rc_text or "bốn giờ sáng" in rc_text


def test_semantic_line_breaking_cannot_delete_text():
    """2. Semantic line breaking cannot delete text or drop words."""
    test_sentences = [
        "Ở một gia đình mở quán ăn sáng, bốn giờ sáng đã phải dậy giúp nhào bột.",
        "Mẹ không xem tôi là con gái mà chỉ coi tôi như một món hàng cần liên tục mài giũa.",
        "Tần Phù Chi, cô đã cướp mất mười tám năm của tôi!",
        "Đây là một câu thoại rất dài nhằm kiểm tra thuật toán ngắt dòng thông minh mà không bị mất chữ.",
    ]
    for text in test_sentences:
        broken = semantic_line_break(text, max_line_chars=36)
        normalized_broken = " ".join(broken.replace(r"\N", " ").split())
        normalized_orig = " ".join(text.split())
        assert normalized_broken == normalized_orig, f"Text dropped during line break: orig={normalized_orig}, broken={normalized_broken}"


def test_generated_ass_contains_complete_render_text():
    """3. Generated ASS contains complete render_text without truncation."""
    cues = [
        _make_cue("c1", 70.31, 73.80, "在早餐店的家庭里，凌晨四点就要起来帮忙揉面。", "Ở một gia đình mở quán ăn sáng, bốn giờ sáng đã phải dậy giúp nhào bột.", mode="narration"),
        _make_cue("c2", 73.80, 76.64, "秦扶栀，你抢了我十八年！", "Tần Phù Chi, cô đã cướp mất mười tám năm của tôi!", mode="direct_dialogue"),
    ]
    project = Project(name="ass_complete", source_video_path="v.mp4", cues=cues)
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(project.cues, translated=True)

    with TemporaryDirectory() as tmp_dir:
        ass_path = Path(tmp_dir) / "output.ass"
        write_ass(ass_path, render_cues, RenderOptions())
        assert ass_path.exists()
        content = ass_path.read_text(encoding="utf-8")

        for rc in render_cues:
            expected_text = rc.render_text
            assert expected_text in content, f"Render cue {rc.render_id} missing from ASS: {expected_text}"


def test_stale_render_artifact_protection():
    """4. Content-addressed hashing detects stale artifacts vs fresh generation."""
    cues_v1 = [_make_cue("c1", 70.31, 73.80, "在早餐店的家庭里", "Trong một quán ăn sáng")]
    cues_v2 = [_make_cue("c1", 70.31, 73.80, "在早餐店的家庭里，凌晨四点就要起来帮忙揉面。", "Ở một gia đình mở quán ăn sáng, bốn giờ sáng đã phải dậy giúp nhào bột.")]

    engine = UtteranceEngine()
    rc1, _ = engine.process_cues(cues_v1, translated=True)
    rc2, _ = engine.process_cues(cues_v2, translated=True)

    with TemporaryDirectory() as tmp_dir:
        ass1 = Path(tmp_dir) / "v1.ass"
        ass2 = Path(tmp_dir) / "v2.ass"
        write_ass(ass1, rc1, RenderOptions())
        write_ass(ass2, rc2, RenderOptions())

        h1 = hashlib.sha256(ass1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(ass2.read_bytes()).hexdigest()

        assert h1 != h2, "Stale ASS and fresh ASS produced identical hashes!"


def test_narration_and_dialogue_boundary_preserved():
    """5. Narration / direct confrontation boundary remains intact."""
    cues = [
        _make_cue("n1", 70.31, 73.80, "在早餐店的家庭里，凌晨四点就要起来帮忙揉面。", "Ở một gia đình mở quán ăn sáng, bốn giờ sáng đã phải dậy giúp nhào bột.", mode="narration"),
        _make_cue("d1", 73.80, 76.64, "秦扶栀，你抢了我十八年！", "Tần Phù Chi, cô đã cướp mất mười tám năm của tôi!", mode="direct_dialogue"),
    ]
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(cues, translated=True)
    assert len(render_cues) == 2
    assert render_cues[0].discourse_mode == DiscourseMode.NARRATION.value
    assert render_cues[1].discourse_mode == DiscourseMode.DIRECT_DIALOGUE.value
    assert render_cues[0].end <= render_cues[1].start + 0.05
