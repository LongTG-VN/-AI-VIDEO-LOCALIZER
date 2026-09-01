from app.models.project import SubtitleCue
from app.services.fusion import fuse_cues


def test_ocr_can_correct_aligned_asr_text():
    asr = SubtitleCue(
        id="a",
        start=1,
        end=3,
        source_text="我早知道了",
        asr_confidence=0.78,
        confidence=0.78,
        speaker_id="speaker_1",
    )
    ocr = SubtitleCue(
        id="o",
        start=1.1,
        end=3.1,
        source_text="我早就知道了",
        ocr_confidence=0.96,
        ocr_start=1.1,
        ocr_end=2.6,
        ocr_text="我早就知道了",
        confidence=0.96,
    )
    fused = fuse_cues([asr], [ocr])
    assert len(fused) == 1
    assert fused[0].source_text == "我早就知道了"
    assert fused[0].speaker_id == "speaker_1"
    assert fused[0].ocr_confidence == 0.96
    # Dialogue timing stays ASR-backed, visual timing stays OCR-backed.
    assert fused[0].start == 1
    assert fused[0].end == 3
    assert fused[0].ocr_start == 1.1
    assert fused[0].ocr_end == 2.6
    assert fused[0].ocr_text == "我早就知道了"


def test_unmatched_ocr_cue_gets_explicit_visual_timing():
    ocr = SubtitleCue(
        id="o2",
        start=4.0,
        end=4.5,
        source_text="字幕",
        ocr_confidence=0.91,
        confidence=0.91,
    )
    fused = fuse_cues([], [ocr])
    assert len(fused) == 1
    assert fused[0].ocr_start == 4.0
    assert fused[0].ocr_end == 4.5
    assert fused[0].ocr_text == "字幕"
