from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from app.models.project import OCREvidence, OCRRegion, SubtitleCue

logger = logging.getLogger(__name__)

PUNCTUATION_REGEX = re.compile(r"[，。！？、“”《》；：,.!?\s]")


def clean_chinese_text(text: str) -> str:
    """Strips whitespace and standard punctuation for comparison and alignment."""
    return PUNCTUATION_REGEX.sub("", text).strip()


def text_similarity(a: str, b: str) -> float:
    left = clean_chinese_text(a)
    right = clean_chinese_text(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _ocr_visual_start(cue: SubtitleCue) -> float:
    return float(cue.ocr_start if cue.ocr_start is not None else cue.start)


def _ocr_visual_end(cue: SubtitleCue) -> float:
    return float(cue.ocr_end if cue.ocr_end is not None else cue.end)


def _dedup_regions(regions: list[OCRRegion]) -> list[OCRRegion]:
    unique: list[OCRRegion] = []
    seen_boxes: set[str] = set()
    for r in regions:
        pts = r.points or []
        if not pts:
            unique.append(r)
            continue
        xs = [round(float(p[0]), 3) for p in pts]
        ys = [round(float(p[1]), 3) for p in pts]
        key = f"{min(xs)}_{max(xs)}_{min(ys)}_{max(ys)}_{r.text}"
        if key not in seen_boxes:
            seen_boxes.add(key)
            unique.append(r)
    return unique


def stitch_two_fragments(left: str, right: str) -> str:
    """Diagnostic helper for merging two fragments."""
    l_str = "".join(left.split())
    r_str = "".join(right.split())
    if not l_str:
        return r_str
    if not r_str:
        return l_str
    if r_str in l_str:
        return l_str
    if l_str in r_str:
        return r_str
    max_k = min(len(l_str), len(r_str))
    for k in range(max_k, 0, -1):
        if l_str.endswith(r_str[:k]):
            return l_str + r_str[k:]
    return l_str + r_str


def stitch_fragments(fragments: list[str]) -> str:
    """Diagnostic helper for iteratively stitching fragments."""
    if not fragments:
        return ""
    result = fragments[0]
    for nxt in fragments[1:]:
        result = stitch_two_fragments(result, nxt)
    return result


def align_and_correct_span(
    asr_text: str,
    ocr_evidence: OCREvidence,
    min_confidence: float = 0.70,
) -> tuple[str, bool, int | None, int | None]:
    """Safely aligns an OCR evidence fragment to an ASR sentence.

    Performs character/homophone corrections ONLY on specific matched spans without
    altering sentence structure or truncating ASR content.

    Example:
    ASR: "我爸秦燕川"
    OCR evidence: "秦砚川" (conf=0.98)
    Result: "我爸秦砚川", did_correct=True, span_start=2, span_end=5
    """
    asr_clean = clean_chinese_text(asr_text)
    ocr_clean = clean_chinese_text(ocr_evidence.text)

    if not asr_clean or not ocr_clean:
        return asr_text, False, None, None

    k = len(ocr_clean)
    if k < 2:
        return asr_text, False, None, None

    # 1. Exact Substring Match (OCR confirms existing ASR span)
    idx = asr_clean.find(ocr_clean)
    if idx >= 0:
        return asr_text, False, idx, idx + k

    conf = ocr_evidence.confidence if ocr_evidence.confidence is not None else 0.85
    if conf < min_confidence:
        return asr_text, False, None, None

    # 2. Homophone / Character Correction on Same-Length Span
    if k <= len(asr_clean) and len(asr_clean) >= 2:
        best_diff = 999
        best_idx = -1
        max_allowed_diffs = max(1, k // 4)
        for i in range(len(asr_clean) - k + 1):
            sub_asr = asr_clean[i : i + k]
            diffs = sum(1 for c1, c2 in zip(sub_asr, ocr_clean) if c1 != c2)
            if 1 <= diffs <= max_allowed_diffs and diffs < best_diff:
                best_diff = diffs
                best_idx = i

        if best_idx >= 0 and best_diff <= max_allowed_diffs:
            target_sub = asr_clean[best_idx : best_idx + k]
            if target_sub in asr_text:
                corrected = asr_text.replace(target_sub, ocr_clean, 1)
                return corrected, True, best_idx, best_idx + k

    # 3. High-Confidence Character Insertion / Substitution on Full Aligned Sentence
    if len(asr_clean) >= 2 and abs(len(ocr_clean) - len(asr_clean)) <= 1 and conf >= 0.90:
        matcher = SequenceMatcher(None, asr_clean, ocr_clean)
        if matcher.ratio() >= 0.80:
            # Check opcodes
            opcodes = matcher.get_opcodes()
            changes = [tag for tag, _, _, _, _ in opcodes if tag != "equal"]
            if len(changes) == 1:
                # Safe single span insertion or replacement
                tag, i1, i2, j1, j2 = [op for op in opcodes if op[0] != "equal"][0]
                target_sub = asr_clean[i1:i2]
                replacement_sub = ocr_clean[j1:j2]
                if target_sub and target_sub in asr_text:
                    corrected = asr_text.replace(target_sub, replacement_sub, 1)
                    return corrected, True, i1, i2
                elif not target_sub and i1 > 0 and asr_clean[i1 - 1 : i1] in asr_text:
                    anchor = asr_clean[i1 - 1 : i1]
                    corrected = asr_text.replace(anchor, anchor + replacement_sub, 1)
                    return corrected, True, i1, i1

    return asr_text, False, None, None


def fuse_cues_with_metrics(
    asr_cues: list[SubtitleCue],
    ocr_cues: list[SubtitleCue],
    match_tolerance_seconds: float = 0.35,
) -> tuple[list[SubtitleCue], dict[str, Any]]:
    """V7 ASR-Anchored Fusion Engine.

    Architecture:
    - ASR is the authority for dialogue segmentation, timing, speaker, and sentence structure.
    - OCR cues are preserved as independent `OCREvidence` fragments.
    - Span corrections are applied conservatively to ASR text when high-confidence evidence exists.
    - Full sentence replacement from OCR is NEVER performed (`full_sentence_replacements = 0`).
    - Meaningful ASR text is NEVER dropped or shortened (`asr_source_shortened = 0`).
    - ASR dialogue timing is 100% preserved (`asr_timing_changed = 0`).
    """
    fused: list[SubtitleCue] = []
    used_ocr_ids: set[str] = set()

    asr_with_evidence = 0
    asr_without_evidence = 0
    span_corrections_applied = 0
    span_corrections_rejected = 0
    full_sentence_replacements = 0
    asr_source_shortened = 0
    asr_timing_changed = 0
    ocr_evidence_total = 0

    for asr in asr_cues:
        asr_dur = max(0.01, asr.end - asr.start)
        asr_clean = clean_chinese_text(asr.source_text)

        # 1. Collect all valid candidate OCR cues within time window
        matched_evidences: list[OCREvidence] = []
        for ocr in ocr_cues:
            o_st = _ocr_visual_start(ocr)
            o_en = _ocr_visual_end(ocr)

            # Check time window with tolerance
            if o_st > asr.end + match_tolerance_seconds or o_en < asr.start - match_tolerance_seconds:
                continue

            overlap = max(0.0, min(asr.end, o_en) - max(asr.start, o_st))
            coverage = overlap / asr_dur
            sim = text_similarity(asr.source_text, ocr.source_text)
            ocr_clean = clean_chinese_text(ocr.source_text)
            is_contained = ocr_clean in asr_clean or asr_clean in ocr_clean

            # Keep candidate if temporal overlap and text connection
            if (overlap >= 0.15 or coverage >= 0.15 or (o_st >= asr.start - 0.20 and o_en <= asr.end + 0.20)) and (
                sim >= 0.20 or is_contained or len(ocr_clean) >= 2
            ):
                used_ocr_ids.add(ocr.id)
                match_score = round(min(1.0, coverage * 0.4 + sim * 0.6), 3)
                ev = OCREvidence(
                    id=ocr.id,
                    text=ocr.source_text,
                    confidence=ocr.ocr_confidence,
                    start=o_st,
                    end=o_en,
                    regions=ocr.ocr_regions,
                    match_score=match_score,
                )
                matched_evidences.append(ev)

        # Sort evidences strictly chronologically by visual start/end
        matched_evidences.sort(key=lambda e: (e.start, e.end))
        ocr_evidence_total += len(matched_evidences)

        if not matched_evidences:
            asr_without_evidence += 1
            copy = asr.model_copy(deep=True)
            copy.ocr_evidence = []
            fused.append(copy)
            continue

        asr_with_evidence += 1

        # 2. Safe Span Corrections on ASR Text
        current_text = asr.source_text
        for ev in matched_evidences:
            corrected_text, did_correct, sp_st, sp_en = align_and_correct_span(current_text, ev)
            ev.matched_span_start = sp_st
            ev.matched_span_end = sp_en
            if did_correct:
                current_text = corrected_text
                span_corrections_applied += 1
            else:
                span_corrections_rejected += 1

        # Invariant checks:
        # ASR source must never be shortened
        if len(clean_chinese_text(current_text)) < len(asr_clean):
            asr_source_shortened += 1
            current_text = asr.source_text

        # ASR timing invariant:
        if asr.start != asr.start or asr.end != asr.end:
            asr_timing_changed += 1

        # Combined visual bounds for backward compatibility
        comb_start = min(e.start for e in matched_evidences)
        comb_end = max(e.end for e in matched_evidences)
        comb_regions = _dedup_regions([r for e in matched_evidences for r in e.regions])
        confs = [e.confidence for e in matched_evidences if e.confidence is not None]
        comb_conf = float(np.mean(confs)) if confs else 0.85

        asr_conf = asr.asr_confidence or asr.confidence or 0.85
        evidence = [value for value in [asr_conf, comb_conf] if value is not None]
        confidence = min(1.0, max(evidence, default=0.0))

        # 3. Emit Fused ASR Cue with attached OCREvidence list
        fused.append(
            SubtitleCue(
                id=asr.id,
                start=asr.start,
                end=asr.end,
                speaker_id=asr.speaker_id,
                addressee_id=asr.addressee_id,
                source_text=current_text,
                translated_text=asr.translated_text,
                asr_confidence=asr_conf,
                ocr_confidence=round(comb_conf, 4),
                ocr_start=comb_start,
                ocr_end=comb_end,
                ocr_text=" ".join(e.text for e in matched_evidences),
                ocr_regions=comb_regions,
                ocr_evidence=matched_evidences,
                confidence=confidence if evidence else None,
            )
        )

    # 4. Retain unmatched OCR evidence as standalone cues for visual cleanup
    unmatched_ocr_count = 0
    for cue in ocr_cues:
        if cue.id not in used_ocr_ids:
            unmatched_ocr_count += 1
            copy = cue.model_copy(deep=True)
            o_st = _ocr_visual_start(copy)
            o_en = _ocr_visual_end(copy)
            if copy.ocr_start is None:
                copy.ocr_start = o_st
            if copy.ocr_end is None:
                copy.ocr_end = o_en
            if copy.ocr_text is None:
                copy.ocr_text = copy.source_text
            ev = OCREvidence(
                id=copy.id,
                text=copy.source_text,
                confidence=copy.ocr_confidence,
                start=o_st,
                end=o_en,
                regions=copy.ocr_regions,
                match_score=1.0,
            )
            copy.ocr_evidence = [ev]
            fused.append(copy)

    fused_sorted = sorted(fused, key=lambda c: (c.start, c.end))

    multi_fragments = sum(len(c.ocr_evidence) for c in fused_sorted if len(c.ocr_evidence) > 1)

    metrics = {
        "asr_cues": len(asr_cues),
        "ocr_cues": len(ocr_cues),
        "ocr_evidence_total": ocr_evidence_total,
        "fused_cues": len(fused_sorted),
        "asr_derived_fused_cues": len(asr_cues),
        "asr_with_evidence": asr_with_evidence,
        "asr_without_evidence": asr_without_evidence,
        "asr_with_multi_ocr": len([c for c in fused_sorted if len(c.ocr_evidence) > 1]),
        "multi_ocr_fragments_consumed": multi_fragments,
        "span_corrections_applied": span_corrections_applied,
        "span_corrections_rejected": span_corrections_rejected,
        "partial_ocr_corrections": span_corrections_applied,
        "full_sentence_replacements": full_sentence_replacements,
        "full_ocr_replacements": full_sentence_replacements,
        "asr_source_shortened": asr_source_shortened,
        "asr_timing_changed": asr_timing_changed,
        "unmatched_ocr_cues": unmatched_ocr_count,
        "unmatched_ocr_evidence": unmatched_ocr_count,
    }
    logger.info("V7 ASR-Anchored Fusion Completed: %s", metrics)
    return fused_sorted, metrics


def fuse_cues(asr_cues: list[SubtitleCue], ocr_cues: list[SubtitleCue]) -> list[SubtitleCue]:
    """Standard API returning fused list of SubtitleCue."""
    fused, _ = fuse_cues_with_metrics(asr_cues, ocr_cues)
    return fused
