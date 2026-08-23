from __future__ import annotations

import pytest
from app.models.project import SubtitleCue
from app.services.utterance_engine import UtteranceEngine, infer_discourse_mode, DiscourseMode


def test_narration_followed_by_confrontation_must_not_merge():
    """Test A: Narration followed by direct confrontation MUST NOT merge."""
    cues = [
        SubtitleCue(
            start=70.31,
            end=73.42,
            source_text="早餐店的家庭里凌晨四点就要",
            translated_text="Trong một gia đình mở tiệm ăn sáng, bốn giờ sáng đã phải",
            speaker_id="speaker_rival",
            addressee_id=None,
        ),
        SubtitleCue(
            start=73.44,
            end=75.56,
            source_text="起来帮忙揉面秦福之你偷了我十八年",
            translated_text="dậy phụ nhào bột. Tần Phù Chi, mày đã trộm của tao mười tám năm",
            speaker_id="speaker_rival",
            addressee_id="char_heroine",
        ),
    ]
    engine = UtteranceEngine(max_line_chars=36)
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 2
    assert "bốn giờ sáng" in render_cues[0].render_text
    assert "Tần Phù Chi" in render_cues[1].render_text


def test_direct_dialogue_followed_by_monologue_must_not_merge():
    """Test B: Direct dialogue followed by monologue MUST NOT merge."""
    cues = [
        SubtitleCue(
            start=51.0,
            end=53.0,
            source_text="你今天做得很棒",
            translated_text="Hôm nay em làm rất tốt.",
            speaker_id="speaker_brother",
            addressee_id="char_heroine",
        ),
        SubtitleCue(
            start=53.43,
            end=55.85,
            source_text="但我现在只想把这只偷偷藏起来的鸡腿",
            translated_text="Nhưng bây giờ tôi chỉ muốn gặm chiếc đùi gà lén giấu này.",
            speaker_id="speaker_heroine",
            addressee_id=None,
        ),
    ]
    engine = UtteranceEngine(max_line_chars=36)
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 2


def test_same_speaker_small_gap_only_must_not_merge():
    """Test C: Same speaker + small gap alone must NOT merge independent statements."""
    cues = [
        SubtitleCue(
            start=11.87,
            end=12.63,
            source_text="领口歪了，",
            translated_text="Cổ áo lệch kìa,",
            speaker_id="speaker_mother",
        ),
        SubtitleCue(
            start=12.67,
            end=13.35,
            source_text="坐姿不对，",
            translated_text="Dáng ngồi sai rồi,",
            speaker_id="speaker_mother",
        ),
        SubtitleCue(
            start=13.69,
            end=14.37,
            source_text="笑的太假。",
            translated_text="Cười giả tạo quá.",
            speaker_id="speaker_mother",
        ),
    ]
    engine = UtteranceEngine(max_line_chars=36)
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 3


def test_clearly_incomplete_source_continuation_may_merge():
    """Test D: Incomplete source continuation clause with strong conjunction may merge."""
    cues = [
        SubtitleCue(
            start=68.11,
            end=70.01,
            source_text="该属于我的人生，",
            translated_text="Cuộc đời vốn thuộc về tôi,",
            speaker_id="speaker_rival",
            addressee_id=None,
        ),
        SubtitleCue(
            start=70.31,
            end=73.42,
            source_text="而我在一个开早餐店的家庭里",
            translated_text="mà tôi lại phải sống ở một tiệm bán đồ ăn sáng",
            speaker_id="speaker_rival",
            addressee_id=None,
        ),
    ]
    engine = UtteranceEngine(max_line_chars=36)
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 1
    assert "thuộc về tôi" in render_cues[0].render_text
    assert "đồ ăn sáng" in render_cues[0].render_text


def test_question_answer_different_speakers_must_not_merge():
    """Test E: Question and answer across speakers must NOT merge."""
    cues = [
        SubtitleCue(
            start=23.0,
            end=24.5,
            source_text="准备好了吗？",
            translated_text="Con chuẩn bị xong chưa?",
            speaker_id="speaker_father",
            addressee_id="char_heroine",
        ),
        SubtitleCue(
            start=24.6,
            end=26.0,
            source_text="快好了。",
            translated_text="Sắp xong rồi ạ.",
            speaker_id="speaker_heroine",
            addressee_id="char_father",
        ),
    ]
    engine = UtteranceEngine(max_line_chars=36)
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 2
