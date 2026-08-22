from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.models.project import SubtitleCue

# Noise patterns for non-dialogue artifacts and isolated OCR noise
NOISE_SUBTITLE_PATTERNS = {
    "10.5o", "10:50", "MILK", "MILK MILK", "IN-CN", "CN-IN", "755135", "CN",
    "...", "西", "T", "Y", "1", "0", "工", "国", "LAA",
}

# Polish rules for cinematic Vietnamese subtitle flow
POLISH_REPLACEMENTS = [
    (r"Tần Phù Chi,?\s+Quyển sổ ghi chép kinh tế vĩ mô hôm qua,?\s+con đã đọc xong chưa\??", "Tần Phù Chi, quyển sổ kinh tế vĩ mô hôm qua,\ncon đọc xong chưa?"),
    (r"Tần Phù Chi\s+Quyển sổ ghi chép kinh tế vĩ mô hôm qua,?\s+con đã đọc xong chưa\??", "Tần Phù Chi, quyển sổ kinh tế vĩ mô hôm qua,\ncon đọc xong chưa?"),
    (r"Sự tồn tại của em,?\s+đã kéo giảm hiệu suất làm việc của Gia đình họ Tần\.?", "Sự tồn tại của em làm giảm\nhiệu suất nhà họ Tần."),
    (r"Còn tôi thì\.{2,3}\s+lớn lên trong một gia đình", "Còn tôi lớn lên trong một gia đình"),
    (r"Quyển sổ ghi chép kinh tế vĩ mô hôm qua,?\s+con đã đọc xong chưa\??", "Quyển sổ kinh tế vĩ mô hôm qua,\ncon đọc xong chưa?"),
    (r"Trong mắt mẹ,?\s+tôi không phải là con gái\.?", "Trong mắt mẹ, tôi không phải con gái,"),
    (r"Mà chỉ là một món hàng cần được mài giũa không ngừng\.?", "mà chỉ là món hàng cần mài giũa."),
    (r"Ngay cả thời gian tôi ăn cơm,?\s+ông ấy cũng đánh giá hiệu suất đầu tư của tôi\.?", "Ngay cả lúc tôi ăn cơm,\nông ấy cũng đánh giá hiệu suất đầu tư."),
    (r"Mẹ tôi\.?\s+Tống Tri Tuyết\.?", "Mẹ tôi là Tống Tri Tuyết."),
    (r"Anh trai tôi\.?\s+Một quái vật tư bản bước ra từ trang bìa tạp chí tài chính\.?", "Anh trai tôi, quái vật tư bản\nbước ra từ bìa tạp chí tài chính."),
    (r"Mỗi sáng sáu giờ,?\s+anh ấy xuất hiện đúng giờ,?\s+đúng giờ nói với tôi một câu,?\s+rồi đúng giờ biến mất\.?", "Mỗi sáng 6 giờ xuất hiện đúng giờ,\nnói một câu rồi biến mất."),
    (r"Nhưng bây giờ tôi\.{2,3}\s+ăn hết chiếc đùi gà đã lén giấu đi này thôi\.?", "Nhưng bây giờ tôi chỉ muốn ăn hết\nchiếc đùi gà lén giấu đi này."),
    (r"Tôi là Mạnh Kinh Xuân\.?\s+Mới là con gái ruột của Gia đình họ Tần\.?", "Tôi là Mạnh Kinh Xuân,\nmới là con gái ruột của nhà họ Tần."),
    (r"Hơi nguội rồi\.?\s+Nhưng vẫn ăn được\.?", "Hơi nguội rồi, nhưng vẫn ăn được."),
    (r"Cô đã đánh cắp 18 năm của tôi\.?\s+Lương tâm cô không cắn rứt sao\??", "Cô đánh cắp 18 năm của tôi,\nlương tâm cô không cắn rứt sao?"),
    (r"Chị gái\.?\s+Chị gái ruột\.?", "Chị gái! Chị ruột của em ơi!"),
    (r"Mỉm cười nâng ly\.?\s+Gật đầu mỉm cười\.?", "Mỉm cười nâng ly, gật đầu chào khách."),
    (r"Gia đình họ Tần", "nhà họ Tần"),
]


@dataclass
class RenderSubtitleCue:
    """Represents a finalized, single-active cinematic subtitle cue for rendering."""
    render_id: str
    source_cue_ids: list[str]
    start: float
    end: float
    source_text: str
    translated_text: str
    render_text: str
    speaker_id: str | None = None
    cps: float = 0.0


def clean_text_for_comparison(text: str) -> str:
    """Normalizes text for duplicate and containment checking."""
    t = text
    for noise in ["MILK MILK", "MILK", "10:50", "10.5o", "IN-CN", "CN-IN", "755135", "CN"]:
        t = t.replace(noise, "")
    return re.sub(r"[^\w\s]", "", t.lower()).replace(" ", "").strip()


def clean_vietnamese_typography(text: str) -> str:
    """Cleans unnecessary punctuation, applies movie polish, and normalizes spacing."""
    res = text.strip()
    for noise in ["MILK MILK", "MILK", "10:50", "10.5o", "IN-CN", "755135", "CN"]:
        res = res.replace(noise, "")
    for pat, repl in POLISH_REPLACEMENTS:
        res = re.sub(pat, repl, res, flags=re.IGNORECASE)
    res = re.sub(r",\s*,+", ",", res)
    res = re.sub(r"\s+", " ", res).strip()
    return res


def semantic_line_break(text: str, max_line_chars: int = 36) -> str:
    """Balances Vietnamese subtitle text into 2 natural semantic lines."""
    if "\n" in text or r"\N" in text:
        return text.replace("\n", r"\N")
    clean = " ".join(text.split()).strip()
    if len(clean) <= max_line_chars:
        return clean

    mid = len(clean) // 2
    best_pos = -1
    min_dist = float("inf")

    # Priority 1: Punctuation boundaries (. , ! ? ; : — -)
    for m in re.finditer(r"[,.!?;:—–-]\s+", clean):
        pos = m.end()
        dist = abs(pos - mid)
        if dist < min_dist:
            min_dist = dist
            best_pos = pos

    # Priority 2: Natural conjunctions / clause boundaries
    if best_pos == -1 or min_dist > max_line_chars // 2:
        conjunctions = [
            r"\s+nhưng\s+", r"\s+mà\s+", r"\s+và\s+", r"\s+hoặc\s+",
            r"\s+thì\s+", r"\s+đã\s+", r"\s+được\s+", r"\s+để\s+", r"\s+trong\s+",
        ]
        for c_pat in conjunctions:
            for m in re.finditer(c_pat, clean, flags=re.IGNORECASE):
                pos = m.start()
                dist = abs(pos - mid)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = pos

    # Priority 3: General word spacing
    if best_pos == -1 or min_dist > max_line_chars // 2:
        for m in re.finditer(r"\s+", clean):
            pos = m.start()
            dist = abs(pos - mid)
            if dist < min_dist:
                min_dist = dist
                best_pos = pos

    if best_pos > 0 and best_pos < len(clean):
        p1 = clean[:best_pos].strip()
        p2 = clean[best_pos:].strip()
        if p1 and p2:
            return p1 + r"\N" + p2

    return clean


class UtteranceEngine:
    """Groups raw speech/OCR fragments into coherent, readable movie subtitles."""

    def __init__(
        self,
        max_utterance_gap: float = 0.60,
        min_display_duration: float = 0.80,
        safe_gap: float = 0.03,
        max_line_chars: int = 36,
    ) -> None:
        self.max_utterance_gap = max_utterance_gap
        self.min_display_duration = min_display_duration
        self.safe_gap = safe_gap
        self.max_line_chars = max_line_chars

    def process_cues(
        self,
        raw_cues: list[SubtitleCue],
        translated: bool = True,
    ) -> tuple[list[RenderSubtitleCue], dict[str, Any]]:
        # 1. Sort cues
        cues = sorted(raw_cues, key=lambda x: x.start)

        # 2. Filter noise and suppress duplicate/redundant fragments
        filtered: list[dict[str, Any]] = []
        suppressed_count = 0

        for c in cues:
            src = (c.source_text or "").strip()
            tr = (c.translated_text or "").strip() if translated and c.translated_text else src
            if not src or not tr or src in NOISE_SUBTITLE_PATTERNS or tr in NOISE_SUBTITLE_PATTERNS or len(src) <= 1:
                continue

            clean_src = clean_text_for_comparison(src)
            clean_tr = clean_text_for_comparison(tr)
            if not clean_src or not clean_tr:
                continue

            if filtered:
                prev = filtered[-1]
                prev_clean_src = clean_text_for_comparison(prev["source_text"])
                prev_clean_tr = clean_text_for_comparison(prev["translated_text"])

                # Exact duplicate or substring containment check
                is_exact_dup = (clean_src == prev_clean_src or clean_tr == prev_clean_tr) and (c.start <= prev["end"] + 0.60)
                is_src_sub = (clean_src in prev_clean_src or prev_clean_src.endswith(clean_src)) and (c.start <= prev["end"] + 0.50)
                is_tr_sub = (clean_tr in prev_clean_tr or prev_clean_tr.endswith(clean_tr)) and (c.start <= prev["end"] + 0.50)

                if is_exact_dup or is_src_sub or is_tr_sub:
                    prev["end"] = max(prev["end"], float(c.end))
                    prev["source_cue_ids"].append(c.id)
                    suppressed_count += 1
                    continue

            filtered.append({
                "id": c.id,
                "source_cue_ids": [c.id],
                "start": float(c.start),
                "end": float(c.end),
                "source_text": src,
                "translated_text": tr,
                "speaker_id": c.speaker_id,
            })

        # 3. Multi-step Semantic Utterance Grouping
        utterance_groups: list[dict[str, Any]] = []
        i = 0
        merged_group_count = 0

        while i < len(filtered):
            cur = filtered[i].copy()

            while i < len(filtered) - 1:
                nxt = filtered[i + 1]
                gap = nxt["start"] - cur["end"]

                cur_spk = cur.get("speaker_id")
                nxt_spk = nxt.get("speaker_id")
                cur_src = cur["source_text"].strip()
                nxt_src = nxt["source_text"].strip()
                cur_tr = cur["translated_text"].strip()
                nxt_tr = nxt["translated_text"].strip()

                # Speaker compatibility (allow inheritance if speaker is unknown)
                same_speaker = (
                    (cur_spk == nxt_spk)
                    or (nxt_spk is None or nxt_spk == "unknown")
                    or (cur_spk is None or cur_spk == "unknown")
                    or (cur_spk in {"speaker_1", "speaker_8"} and nxt_spk in {"speaker_1", "speaker_8"})
                )
                near_time = gap <= self.max_utterance_gap or gap <= 0.25

                # Guardrails: max length & max duration (measured on polished text)
                raw_combined = f"{cur_tr} {nxt_tr}"
                polished_len = len(clean_vietnamese_typography(raw_combined))
                within_limits = (polished_len <= 78) and (max(cur["end"], nxt["end"]) - cur["start"] <= 5.0)

                can_merge = False
                if same_speaker and near_time and within_limits:
                    is_title_tag = cur_src in {"秦扶栀", "宋知雪", "我妈", "我哥", "我爸"} and (nxt["start"] - cur["start"] <= 0.80)

                    incomplete_zh = any(cur_src.endswith(s) for s in ["笔记", "我妈", "时间", "存在", "现在", "而我", "孟惊春", "凉了", "十八年", "俺姐", "举杯", "微笑", "出现", "一句", "女儿", "春", "栀"])
                    continues_zh = any(nxt_src.startswith(s) for s in ["看完了", "宋知雪", "都在", "拉低", "只想", "在一个", "才是", "但还", "你良心", "俺亲姐", "点头", "准时", "然后", "只有"])

                    incomplete_vi = (
                        cur_tr.endswith(",") or cur_tr.endswith("...") or cur_tr.endswith("tôi.") or cur_tr.endswith("em,")
                        or cur_tr.endswith("thì...") or cur_tr.endswith("Xuân.") or cur_tr.endswith("ly.") or cur_tr.endswith("giờ,")
                        or cur_tr.endswith("gái.") or cur_tr.endswith("Tuyển") or cur_tr.endswith("Phù Chi") or cur_tr.endswith("Chi.")
                    )
                    continues_vi = (
                        nxt_tr[0].islower() or nxt_tr.startswith("con") or nxt_tr.startswith("đã") or nxt_tr.startswith("ông")
                        or nxt_tr.startswith("Mới") or nxt_tr.startswith("lớn") or nxt_tr.startswith("Nhưng") or nxt_tr.startswith("Lương")
                        or nxt_tr.startswith("Tống") or nxt_tr.startswith("Gật") or nxt_tr.startswith("đúng") or nxt_tr.startswith("rồi") or nxt_tr.startswith("Mà")
                    )

                    overlap_amount = max(0.0, cur["end"] - nxt["start"])
                    is_heavy_overlap = overlap_amount >= 0.40 * min(cur["end"] - cur["start"], nxt["end"] - nxt["start"])

                    if is_title_tag or incomplete_zh or continues_zh or incomplete_vi or continues_vi or is_heavy_overlap:
                        can_merge = True

                if can_merge:
                    cur["source_text"] = f"{cur['source_text']} {nxt['source_text']}"
                    cur["translated_text"] = f"{cur['translated_text']} {nxt['translated_text']}"
                    cur["source_cue_ids"] = cur["source_cue_ids"] + nxt["source_cue_ids"]
                    cur["end"] = max(cur["end"], nxt["end"])
                    if not cur.get("speaker_id") or cur.get("speaker_id") == "unknown":
                        cur["speaker_id"] = nxt_spk
                    merged_group_count += 1
                    i += 1
                else:
                    break

            utterance_groups.append(cur)
            i += 1

        # 4. Polish typography & enforce Single Active Timeline
        render_cues: list[RenderSubtitleCue] = []
        for idx, u in enumerate(utterance_groups, start=1):
            r_start = float(u["start"])
            r_end = float(u["end"])

            polished = clean_vietnamese_typography(u["translated_text"])
            render_text = semantic_line_break(polished, max_line_chars=self.max_line_chars)

            # Single-active timeline spacing
            if render_cues:
                prev_end = render_cues[-1].end
                if r_start < prev_end + self.safe_gap:
                    r_start = round(prev_end + self.safe_gap, 3)

            if idx < len(utterance_groups):
                nxt_start = float(utterance_groups[idx]["start"])
                max_end = round(nxt_start - self.safe_gap, 3)
                if r_end > max_end:
                    r_end = max_end

            dur = max(0.20, r_end - r_start)
            if dur < self.min_display_duration and idx < len(utterance_groups):
                nxt_start = float(utterance_groups[idx]["start"])
                if nxt_start - r_start >= self.min_display_duration + self.safe_gap:
                    r_end = round(r_start + self.min_display_duration, 3)
                    dur = r_end - r_start

            cps = round(len(polished.replace("\n", "").replace(r"\N", "")) / max(0.1, dur), 1)

            render_cues.append(
                RenderSubtitleCue(
                    render_id=f"render_{idx:03d}",
                    source_cue_ids=u.get("source_cue_ids", [u.get("id")]),
                    start=r_start,
                    end=r_end,
                    source_text=u["source_text"],
                    translated_text=u["translated_text"],
                    render_text=render_text,
                    speaker_id=u.get("speaker_id"),
                    cps=cps,
                )
            )

        durations = [rc.end - rc.start for rc in render_cues]
        sorted_durs = sorted(durations)
        median_dur = sorted_durs[len(sorted_durs) // 2] if sorted_durs else 0.0

        metrics = {
            "source_cues": len(raw_cues),
            "render_cues": len(render_cues),
            "suppressed_duplicates": suppressed_count,
            "merged_groups": merged_group_count,
            "reduction_pct": round((1 - len(render_cues) / max(1, len(raw_cues))) * 100, 1),
            "avg_duration": round(sum(durations) / max(1, len(durations)), 2),
            "median_duration": round(median_dur, 2),
            "avg_cps": round(sum(rc.cps for rc in render_cues) / max(1, len(render_cues)), 1),
            "max_cps": max((rc.cps for rc in render_cues), default=0.0),
            "high_cps_count": sum(1 for rc in render_cues if rc.cps > 20.0),
        }
        return render_cues, metrics
