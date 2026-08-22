from __future__ import annotations

from pathlib import Path
from typing import Any

from app.models.project import SubtitleCue
from app.services.asr.base import ASREngine


class FunASREngine(ASREngine):
    """FunASR adapter using the current AutoModel API.

    Recommended Chinese setup: Paraformer + FSMN-VAD + CT punctuation + CAM++.
    For newer all-in-one models you can override model names through environment variables.
    """

    def __init__(
        self,
        model_name: str = "paraformer-zh",
        vad_model: str | None = "fsmn-vad",
        punc_model: str | None = "ct-punc",
        spk_model: str | None = "cam++",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.vad_model = vad_model or None
        self.punc_model = punc_model or None
        self.spk_model = spk_model or None
        self.device = device
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from funasr import AutoModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ASR_ENGINE=funasr but FunASR is not installed. Install backend/requirements-asr.txt."
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "device": self.device,
            "disable_update": True,
        }
        if self.vad_model:
            kwargs["vad_model"] = self.vad_model
            kwargs["vad_kwargs"] = {"max_single_segment_time": 60000}
        if self.punc_model:
            kwargs["punc_model"] = self.punc_model
        if self.spk_model:
            kwargs["spk_model"] = self.spk_model
        self._model = AutoModel(**kwargs)
        return self._model

    @staticmethod
    def _speaker_id(value: Any) -> str | None:
        if value is None:
            return None
        return f"speaker_{value}"

    def transcribe(self, audio_path: Path, language: str = "zh") -> list[SubtitleCue]:
        model = self._load()
        results = model.generate(
            input=str(audio_path),
            language=language,
            batch_size_s=300,
            sentence_timestamp=True,
            return_spk_res=True,
        )
        if not results:
            return []

        result = results[0]
        sentence_info = result.get("sentence_info") or []
        cues: list[SubtitleCue] = []
        for sentence in sentence_info:
            text = str(sentence.get("text") or "").strip()
            if not text:
                continue
            start = float(sentence.get("start", 0)) / 1000.0
            end = float(sentence.get("end", sentence.get("start", 0))) / 1000.0
            if end <= start:
                end = start + 0.5
            cues.append(
                SubtitleCue(
                    start=start,
                    end=end,
                    speaker_id=self._speaker_id(sentence.get("spk")),
                    source_text=text,
                )
            )
        if cues:
            return cues

        text = str(result.get("text") or "").strip()
        timestamps = result.get("timestamp") or []
        if not text:
            return []
        start = float(timestamps[0][0]) / 1000.0 if timestamps else 0.0
        end = float(timestamps[-1][1]) / 1000.0 if timestamps else max(0.5, start + 2.0)
        return [SubtitleCue(start=start, end=end, source_text=text)]
