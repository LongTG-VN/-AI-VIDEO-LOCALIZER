from app.models.project import SubtitleCue
from app.services.fusion import fuse_cues


def test_ocr_can_correct_aligned_asr_text():
    asr = SubtitleCue(id="a", start=1, end=3, source_text="我早知道了", asr_confidence=0.78, confidence=0.78, speaker_id="speaker_1")
    ocr = SubtitleCue(id="o", start=1.1, end=3.1, source_text="我早就知道了", ocr_confidence=0.96, confidence=0.96)
    fused = fuse_cues([asr], [ocr])
    assert len(fused) == 1
    assert fused[0].source_text == "我早就知道了"
    assert fused[0].speaker_id == "speaker_1"
    assert fused[0].ocr_confidence == 0.96
