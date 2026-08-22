import logging
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from app.models.project import OCRRegion, SubtitleCue

logger = logging.getLogger(__name__)

PUNCTUATION_REGEX = re.compile(r"[，。！？、“”《》；：,.!?]")


def clean_chinese_text(text: str) -> str:
    """Strips whitespace and standard punctuation for comparison and stitching."""
    return PUNCTUATION_REGEX.sub("", "".join(text.split())).strip()


def text_similarity(a: str, b: str) -> float:
    left = clean_chinese_text(a)
    right = clean_chinese_text(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def stitch_two_fragments(left: str, right: str) -> str:
    """Merges two overlapping or adjacent OCR fragments into a coherent string."""
    l_str = "".join(left.split())
    r_str = "".join(right.split())
    if not l_str:
        return r_str
    if not r_str:
        return l_str

    # 1. Substring inclusion
    if r_str in l_str:
        return l_str
    if l_str in r_str:
        return r_str

    # Clean punctuation for overlap checking
    l_clean = clean_chinese_text(l_str)
    r_clean = clean_chinese_text(r_str)

    if r_clean in l_clean:
        return l_str
    if l_clean in r_clean:
        return r_str

    # 2. Longest suffix-prefix overlap
    max_k = min(len(l_str), len(r_str))
    for k in range(max_k, 0, -1):
        if l_str.endswith(r_str[:k]):
            return l_str + r_str[k:]

    # Check clean suffix-prefix overlap
    max_kc = min(len(l_clean), len(r_clean))
    for k in range(max_kc, 0, -1):
        if l_clean.endswith(r_clean[:k]):
            # Suffix of clean matches prefix of clean
            return l_str + r_str[k:]

    # 3. Fuzzy suffix-prefix overlap
    for k in range(max_k, 2, -1):
        sub_l = l_str[-k:]
        sub_r = r_str[:k]
        if SequenceMatcher(None, sub_l, sub_r).ratio() >= 0.80:
            return l_str + r_str[k:]

    # 4. Fallback concatenation
    return l_str + r_str


def stitch_fragments(fragments: list[str]) -> str:
    """Iteratively stitches multiple OCR fragments in chronological order."""
    if not fragments:
        return ""
    result = fragments[0]
    for nxt in fragments[1:]:
        result = stitch_two_fragments(result, nxt)
    return result


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


def _apply_partial_ocr_correction(asr_text: str, ocr_text: str) -> tuple[str, bool]:
    """Applies high-confidence named-entity/token correction inside ASR sentence.

    Example: ASR='我爸秦燕川', OCR='秦砚川' -> '我爸秦砚川'
    """
    asr_clean = clean_chinese_text(asr_text)
    ocr_clean = clean_chinese_text(ocr_text)

    if not asr_clean or not ocr_clean or len(ocr_clean) < 2:
        return asr_text, False

    # If OCR is shorter than ASR, slide OCR window over ASR
    k = len(ocr_clean)
    if k <= len(asr_clean) and len(asr_clean) >= 3:
        best_diff = 999
        best_idx = -1
        for i in range(len(asr_clean) - k + 1):
            sub_asr = asr_clean[i : i + k]
            # Character differences
            diffs = sum(1 for c1, c2 in zip(sub_asr, ocr_clean) if c1 != c2)
            if diffs == 1 and diffs < best_diff:
                best_diff = diffs
                best_idx = i

        if best_idx >= 0 and best_diff == 1:
            # Replace target substring in ASR
            target_sub = asr_clean[best_idx : best_idx + k]
            if target_sub in asr_text:
                corrected = asr_text.replace(target_sub, ocr_clean, 1)
                return corrected, True

    return asr_text, False


def fuse_cues_with_metrics(
    asr_cues: list[SubtitleCue],
    ocr_cues: list[SubtitleCue],
    match_tolerance_seconds: float = 0.35,
) -> tuple[list[SubtitleCue], dict[str, Any]]:
    """Fuses ASR cues with multiple OCR fragments, using ASR timing as the dialogue backbone."""
    fused: list[SubtitleCue] = []
    used_ocr_ids: set[str] = set()

    asr_with_zero_ocr = 0
    asr_with_one_ocr = 0
    asr_with_multi_ocr = 0
    multi_ocr_fragments_consumed = 0
    partial_ocr_corrections = 0
    full_ocr_replacements = 0
    asr_full_text_preserved = 0

    for asr in asr_cues:
        asr_dur = max(0.01, asr.end - asr.start)
        asr_clean = clean_chinese_text(asr.source_text)

        # 1. Collect all matching candidate OCR cues within time window
        candidates: list[SubtitleCue] = []
        for ocr in ocr_cues:
            o_st = _ocr_visual_start(ocr)
            o_en = _ocr_visual_end(ocr)

            # Match window with tolerance
            if o_st > asr.end + match_tolerance_seconds or o_en < asr.start - match_tolerance_seconds:
                continue

            overlap = max(0.0, min(asr.end, o_en) - max(asr.start, o_st))
            coverage = overlap / asr_dur
            sim = text_similarity(asr.source_text, ocr.source_text)
            ocr_clean = clean_chinese_text(ocr.source_text)
            is_contained = ocr_clean in asr_clean or asr_clean in ocr_clean

            # Keep candidate if temporal overlap or strong text connection
            if overlap >= 0.15 or coverage >= 0.15 or sim >= 0.25 or is_contained or (o_st >= asr.start - 0.20 and o_en <= asr.end + 0.20):
                candidates.append(ocr)

        if not candidates:
            asr_with_zero_ocr += 1
            asr_full_text_preserved += 1
            copy = asr.model_copy(deep=True)
            fused.append(copy)
            continue

        if len(candidates) == 1:
            asr_with_one_ocr += 1
        else:
            asr_with_multi_ocr += 1
            multi_ocr_fragments_consumed += len(candidates)

        # Sort candidates strictly chronologically by visual time
        candidates.sort(key=lambda c: (_ocr_visual_start(c), _ocr_visual_end(c)))
        used_ocr_ids.update(c.id for c in candidates)

        # 2. Construct Combined OCR Hypothesis
        stitched_text = stitch_fragments([c.source_text for c in candidates])
        comb_start = min(_ocr_visual_start(c) for c in candidates)
        comb_end = max(_ocr_visual_end(c) for c in candidates)

        all_regions: list[OCRRegion] = []
        for c in candidates:
            all_regions.extend(c.ocr_regions)
        comb_regions = _dedup_regions(all_regions)

        confs = [c.ocr_confidence for c in candidates if c.ocr_confidence is not None]
        comb_conf = float(np.mean(confs)) if confs else 0.85

        asr_conf = asr.asr_confidence or asr.confidence or 0.85
        sim = text_similarity(asr.source_text, stitched_text)
        stitched_clean = clean_chinese_text(stitched_text)
        len_ratio = len(stitched_clean) / max(1, len(asr_clean))

        # 3. Full Sentence Selection with Completeness Guard
        chosen_text = asr.source_text

        # Check for partial homophone/name correction (e.g. 我爸秦燕川 -> 我爸秦砚川)
        corrected_text, did_correct = _apply_partial_ocr_correction(asr.source_text, stitched_text)
        if did_correct:
            chosen_text = corrected_text
            partial_ocr_corrections += 1
        elif len_ratio < 0.75 and stitched_clean in asr_clean:
            # Completeness Guard: OCR is only a partial fragment of ASR (e.g. '拉低了秦家的执行效率' vs '你的存在拉低了秦家的执行效率')
            # Keep full ASR sentence!
            chosen_text = asr.source_text
            asr_full_text_preserved += 1
        elif sim >= 0.85 or (comb_conf >= asr_conf and len(stitched_clean) >= len(asr_clean) and sim >= 0.60):
            chosen_text = stitched_text
            full_ocr_replacements += 1
        else:
            chosen_text = asr.source_text
            asr_full_text_preserved += 1

        evidence = [value for value in [asr_conf, comb_conf] if value is not None]
        confidence = min(1.0, max(evidence, default=0.0) + (0.05 if sim >= 0.85 else 0.0))

        # Invariant: ASR timing is backbone (start=asr.start, end=asr.end)
        fused.append(
            SubtitleCue(
                id=asr.id,
                start=asr.start,
                end=asr.end,
                speaker_id=asr.speaker_id,
                addressee_id=asr.addressee_id,
                source_text=chosen_text,
                translated_text=asr.translated_text,
                asr_confidence=asr_conf,
                ocr_confidence=round(comb_conf, 4),
                ocr_start=comb_start,
                ocr_end=comb_end,
                ocr_text=stitched_text,
                ocr_regions=comb_regions,
                confidence=confidence if evidence else None,
            )
        )

    # 4. Retain unmatched OCR cues
    unmatched_ocr_cues = 0
    for cue in ocr_cues:
        if cue.id not in used_ocr_ids:
            unmatched_ocr_cues += 1
            copy = cue.model_copy(deep=True)
            if copy.ocr_start is None:
                copy.ocr_start = copy.start
            if copy.ocr_end is None:
                copy.ocr_end = copy.end
            if copy.ocr_text is None:
                copy.ocr_text = copy.source_text
            fused.append(copy)

    fused_sorted = sorted(fused, key=lambda c: (c.start, c.end))

    metrics = {
        "asr_cues": len(asr_cues),
        "ocr_cues": len(ocr_cues),
        "fused_cues": len(fused_sorted),
        "asr_with_zero_ocr": asr_with_zero_ocr,
        "asr_with_one_ocr": asr_with_one_ocr,
        "asr_with_multi_ocr": asr_with_multi_ocr,
        "multi_ocr_fragments_consumed": multi_ocr_fragments_consumed,
        "partial_ocr_corrections": partial_ocr_corrections,
        "full_ocr_replacements": full_ocr_replacements,
        "asr_full_text_preserved": asr_full_text_preserved,
        "unmatched_ocr_cues": unmatched_ocr_cues,
    }
    logger.info("Multi-OCR Fusion Completed: %s", metrics)
    return fused_sorted, metrics


def fuse_cues(asr_cues: list[SubtitleCue], ocr_cues: list[SubtitleCue]) -> list[SubtitleCue]:
    """Standard API returning fused list of SubtitleCue."""
    fused, _ = fuse_cues_with_metrics(asr_cues, ocr_cues)
    return fused
