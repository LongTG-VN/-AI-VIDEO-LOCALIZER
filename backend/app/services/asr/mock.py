from pathlib import Path

from app.models.project import SubtitleCue
from app.services.asr.base import ASREngine


class MockASREngine(ASREngine):
    """Development engine so the complete pipeline can be exercised without model downloads."""

    def transcribe(self, audio_path: Path, language: str = "zh") -> list[SubtitleCue]:
        return [
            SubtitleCue(
                start=0.5,
                end=2.8,
                speaker_id="speaker_1",
                source_text="你怎么现在才回来？",
                asr_confidence=0.91,
                confidence=0.91,
            ),
            SubtitleCue(
                start=3.1,
                end=5.4,
                speaker_id="speaker_2",
                source_text="路上有点事，别生气。",
                asr_confidence=0.88,
                confidence=0.88,
            ),
        ]
