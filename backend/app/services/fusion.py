from __future__ import annotations

from difflib import SequenceMatcher

from app.models.project import SubtitleCue


def temporal_overlap(a: SubtitleCue, b: SubtitleCue) -> float:
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shortest = max(0.001, min(a.end - a.start, b.end - b.start))
    return overlap / shortest


def text_similarity(a: str, b: str) -> float:
    left = "".join(a.split())
    right = "".join(b.split())
    return SequenceMatcher(None, left, right).ratio()


def _ocr_visual_start(cue: SubtitleCue) -> float:
    return float(cue.ocr_start if cue.ocr_start is not None else cue.start)


def _ocr_visual_end(cue: SubtitleCue) -> float:
    return float(cue.ocr_end if cue.ocr_end is not None else cue.end)


def fuse_cues(asr_cues: list[SubtitleCue], ocr_cues: list[SubtitleCue]) -> list[SubtitleCue]:
    """Fuse ASR and hard-subtitle OCR evidence without destroying either timeline.

    ASR remains the dialogue timing/speaker backbone. OCR can correct text when temporally
    aligned, while its own visual start/end are preserved independently for hard-sub cleanup.
    Unmatched OCR cues are retained so dialogue missed by ASR still reaches review.
    """
    fused: list[SubtitleCue] = []
    used_ocr: set[str] = set()

    for asr in asr_cues:
        asr_dur = max(0.01, asr.end - asr.start)
        candidates: list[tuple[float, SubtitleCue]] = []
        for ocr in ocr_cues:
            overlap = max(0.0, min(asr.end, ocr.end) - max(asr.start, ocr.start))
            if overlap <= 0.05:
                continue
            sim = text_similarity(asr.source_text, ocr.source_text)
            coverage = overlap / asr_dur
            if coverage >= 0.15 or overlap >= 0.20 or sim >= 0.35:
                score = coverage * 0.4 + sim * 0.4 + (ocr.ocr_confidence or 0.5) * 0.2
                candidates.append((score, ocr))

        if not candidates:
            fused.append(asr.model_copy(deep=True))
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]
        used_ocr.add(best.id)

        asr_conf = asr.asr_confidence or asr.confidence
        ocr_conf = best.ocr_confidence or best.confidence
        similarity = text_similarity(asr.source_text, best.source_text)
        choose_ocr = (ocr_conf or 0) > (asr_conf or 0) + 0.03 or similarity >= 0.85
        text = best.source_text if choose_ocr else asr.source_text
        evidence = [value for value in [asr_conf, ocr_conf] if value is not None]
        confidence = min(1.0, max(evidence, default=0.0) + (0.05 if similarity >= 0.85 else 0.0))
        fused.append(
            SubtitleCue(
                id=asr.id,
                start=asr.start,
                end=asr.end,
                speaker_id=asr.speaker_id,
                addressee_id=asr.addressee_id,
                source_text=text,
                translated_text=asr.translated_text,
                asr_confidence=asr_conf,
                ocr_confidence=ocr_conf,
                ocr_start=_ocr_visual_start(best),
                ocr_end=_ocr_visual_end(best),
                ocr_text=best.ocr_text or best.source_text,
                ocr_regions=list(best.ocr_regions),
                confidence=confidence if evidence else None,
            )
        )

    for cue in ocr_cues:
        if cue.id not in used_ocr:
            copy = cue.model_copy(deep=True)
            if copy.ocr_start is None:
                copy.ocr_start = copy.start
            if copy.ocr_end is None:
                copy.ocr_end = copy.end
            if copy.ocr_text is None:
                copy.ocr_text = copy.source_text
            fused.append(copy)
    return sorted(fused, key=lambda cue: (cue.start, cue.end))
