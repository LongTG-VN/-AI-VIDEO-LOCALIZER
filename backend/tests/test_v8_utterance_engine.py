import pytest
from app.models.project import SubtitleCue
from app.services.utterance_engine import UtteranceEngine, is_short_imperative_or_assessment


def test_three_short_statements_stay_separate():
    """Test 1: Three short independent assessments must remain THREE separate render cues."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=11.87, end=12.63, source_text="领口歪了，", translated_text="Cổ áo bị lệch rồi,", speaker_id="speaker_1", addressee_id="char_2"),
        SubtitleCue(start=12.67, end=13.35, source_text="坐姿不对，", translated_text="Tư thế ngồi không đúng,", speaker_id="speaker_1", addressee_id="char_2"),
        SubtitleCue(start=13.69, end=14.37, source_text="笑的太假。", translated_text="Cười giả tạo quá.", speaker_id="speaker_1", addressee_id="char_2"),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 3
    assert "Cổ áo" in render_cues[0].translated_text
    assert "Tư thế" in render_cues[1].translated_text
    assert "Cười" in render_cues[2].translated_text
    assert metrics["merged_groups"] == 0


def test_same_speaker_alone_does_not_trigger_merge():
    """Test 2: Two distinct sentences from same speaker with no continuation word must not merge."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=10.0, end=11.5, source_text="今天天气很好", translated_text="Hôm nay thời tiết đẹp quá", speaker_id="spk_1"),
        SubtitleCue(start=11.7, end=13.0, source_text="我想去图书馆", translated_text="Tôi muốn đi thư viện", speaker_id="spk_1"),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 2
    assert metrics["merged_groups"] == 0


def test_timing_overlap_alone_does_not_trigger_merge():
    """Test 3: Temporal proximity or slight overlap alone does not merge independent cues."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=20.0, end=21.5, source_text="第一句话", translated_text="Câu đầu tiên", speaker_id="spk_1"),
        SubtitleCue(start=21.4, end=23.0, source_text="第二句话", translated_text="Câu thứ hai", speaker_id="spk_1"),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 2


def test_missing_punctuation_alone_does_not_trigger_merge():
    """Test 4: Absence of period in ASR transcript alone must never trigger merging."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=5.0, end=6.5, source_text="这只是一个测试", translated_text="Đây chỉ là một thử nghiệm", speaker_id="spk_1"),
        SubtitleCue(start=6.7, end=8.0, source_text="我们继续工作", translated_text="Chúng ta tiếp tục làm việc", speaker_id="spk_1"),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 2


def test_direct_dialogue_and_monologue_never_merge():
    """Test 5: Direct dialogue (with addressee) and monologue (no addressee) must never merge."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=53.4, end=55.8, source_text="但我现在只想把这只偷偷藏起来的鸡腿", translated_text="Nhưng bây giờ tôi chỉ muốn ăn cái đùi gà này", speaker_id="heroine", addressee_id=None),
        SubtitleCue(start=55.9, end=57.5, source_text="看清楚 秦扶栀", translated_text="Nhìn cho rõ vào, Tần Phù Chi", speaker_id="rival", addressee_id="heroine"),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 2
    assert metrics["wrong_speaker_merges"] == 0


def test_question_and_answer_never_merge():
    """Test 6: Question followed by answer must never merge even with zero gap."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=22.0, end=25.4, source_text="昨天的宏观经济笔记看完了吗？", translated_text="Vở ghi chép kinh tế vĩ mô hôm qua xem xong chưa?", speaker_id="spk_mom", addressee_id="spk_daughter"),
        SubtitleCue(start=25.5, end=26.6, source_text="快快了", translated_text="Sắp xong rồi ạ", speaker_id="spk_daughter", addressee_id="spk_mom"),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 2


def test_clearly_unfinished_source_can_merge():
    """Test 7: A sentence with explicit strong continuation word (e.g. 因为... 所以...) and comma can merge."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=30.0, end=31.2, source_text="虽然我很努力，", translated_text="Mặc dù tôi rất cố gắng,", speaker_id="spk_1", addressee_id=None),
        SubtitleCue(start=31.3, end=32.5, source_text="但是还是失败了。", translated_text="nhưng vẫn thất bại rồi.", speaker_id="spk_1", addressee_id=None),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 1
    assert "Mặc dù tôi rất cố gắng" in render_cues[0].translated_text
    assert "nhưng vẫn thất bại" in render_cues[0].translated_text
    assert metrics["merged_groups"] == 1


def test_long_merged_subtitle_rejected():
    """Test 8: Combining text that exceeds comfortable length limit (> 72 chars) must be rejected."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=1.0, end=3.0, source_text="这是一个非常长的一句话因为包含了许多复杂的内容，", translated_text="Đây là một câu rất dài bởi vì nó chứa đựng rất nhiều nội dung vô cùng phức tạp,", speaker_id="spk_1"),
        SubtitleCue(start=3.1, end=5.0, source_text="但是我们依然需要继续努力完成所有的目标。", translated_text="nhưng chúng ta vẫn cần phải tiếp tục nỗ lực hoàn thành tất cả mục tiêu đã đặt ra.", speaker_id="spk_1"),
    ]
    render_cues, metrics = engine.process_cues(cues)
    assert len(render_cues) == 2


def test_canonical_speaker_match_respected():
    """Test 9: Canonical speaker_character_id match is respected."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=1.0, end=2.0, source_text="我不去", translated_text="Tôi không đi", speaker_character_id="char_A"),
        SubtitleCue(start=2.1, end=3.0, source_text="你必须去", translated_text="Cô phải đi", speaker_character_id="char_B"),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 2


def test_addressee_mismatch_blocks_merge():
    """Test 10: Addressing different people blocks merging."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(start=1.0, end=2.0, source_text="你坐下", translated_text="Cậu ngồi xuống đi", speaker_id="spk_1", addressee_id="char_A"),
        SubtitleCue(start=2.1, end=3.0, source_text="你去倒水", translated_text="Còn cậu đi rót nước đi", speaker_id="spk_1", addressee_id="char_B"),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 2
