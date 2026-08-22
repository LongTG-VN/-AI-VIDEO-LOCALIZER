from __future__ import annotations

import re

# Common number mappings for Vietnamese speech
NUMBERS_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín", "10": "mười"
}

ACRONYMS_MAP = {
    "KPI": "K P I",
    "CEO": "C E O",
    "AI": "A I",
    "VIP": "V I P",
    "DNA": "D N A",
    "GPS": "G P S",
}


def number_to_vietnamese_words(n_str: str) -> str:
    """Basic conversion of simple numbers (1-99) to Vietnamese words."""
    try:
        val = int(n_str)
    except ValueError:
        return n_str

    if 0 <= val <= 10:
        return NUMBERS_MAP.get(str(val), n_str)
    elif 11 <= val <= 19:
        unit = val % 10
        unit_str = "mốt" if unit == 1 else ("lăm" if unit == 5 else NUMBERS_MAP.get(str(unit), ""))
        return f"mười {unit_str}".strip()
    elif 20 <= val <= 99:
        ten = val // 10
        unit = val % 10
        ten_str = f"{NUMBERS_MAP.get(str(ten), '')} mươi"
        if unit == 0:
            return ten_str
        elif unit == 1:
            return f"{ten_str} mốt"
        elif unit == 4:
            return f"{ten_str} tư"
        elif unit == 5:
            return f"{ten_str} lăm"
        else:
            return f"{ten_str} {NUMBERS_MAP.get(str(unit), '')}"
    return n_str


def normalize_for_speech(text: str) -> str:
    """Normalizes subtitle/render text into clean, natural Vietnamese spoken text.

    Removes ASS tags, replaces line breaks, expands common numerals and acronyms,
    and strips typographical noise.
    """
    if not text:
        return ""

    # 1. Remove ASS override tags: {\...}
    s = re.sub(r"\{[^}]*\}", "", text)

    # 2. Replace ASS line breaks \N with space
    s = s.replace(r"\N", " ")

    # 3. Replace pipe separators (used in subtitle visual splitting)
    s = s.replace("|", " ")

    # 4. Normalize percentage: 0.3% -> không phẩy ba phần trăm, 100% -> một trăm phần trăm
    s = re.sub(r"(\d+)[.,](\d+)%", lambda m: f"{number_to_vietnamese_words(m.group(1))} phẩy {number_to_vietnamese_words(m.group(2))} phần trăm", s)
    s = re.sub(r"(\d+)%", lambda m: f"{number_to_vietnamese_words(m.group(1))} phần trăm", s)

    # 5. Acronyms expansion
    for acr, exp in ACRONYMS_MAP.items():
        s = re.sub(r"\b" + acr + r"\b", exp, s, flags=re.IGNORECASE)

    # 6. Normalize common temporal/quantifier patterns: e.g. "18 năm" -> "mười tám năm", "4 giờ" -> "bốn giờ"
    def _replace_num_phrase(match: re.Match) -> str:
        num_part = match.group(1)
        suffix = match.group(2)
        return f"{number_to_vietnamese_words(num_part)} {suffix}"

    s = re.sub(r"\b(\d{1,2})\s*(năm|tháng|ngày|giờ|tuổi|phút|giây|người|lần|cái|chiếc|con)\b", _replace_num_phrase, s, flags=re.IGNORECASE)

    # 7. Remove non-speech punctuation noise (preserve essential speech commas and periods)
    s = re.sub(r"[\"«»„“”_~*^#\[\](){}]", " ", s)

    # 8. Collapse multiple whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # 9. Ensure does not end with stray punctuation
    s = re.sub(r"[,;:-]+$", "", s).strip()

    return s
