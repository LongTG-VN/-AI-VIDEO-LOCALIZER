import cv2
import numpy as np
import pytest

from app.models.project import OCRRegion, SubtitleCue
from app.services.hardsub_cleaner import HardSubCleaner
from app.services.utterance_engine import (
    UtteranceEngine,
    clean_vietnamese_typography,
    semantic_line_break,
)


def test_semantic_ordering_case1_goods_vs_daughter():
    """Case 1: 'Trong mắt mẹ, tôi không phải con gái, mà chỉ là món hàng cần mài giũa.' must NOT be reversed."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(
            id="c1",
            start=18.00,
            end=19.50,
            source_text="她的眼里没有女儿",
            translated_text="Trong mắt mẹ, tôi không phải là con gái.",
            speaker_id="spk_narrator",
            speaker_character_id="char_qin_fuzhi",
        ),
        SubtitleCue(
            id="c2",
            start=19.60,
            end=21.50,
            source_text="只有一件需要时刻打磨的商品",
            translated_text="mà chỉ là món hàng cần mài giũa.",
            speaker_id="spk_narrator",
            speaker_character_id="char_qin_fuzhi",
        ),
    ]
    render_cues, _ = engine.process_cues(cues)
    if len(render_cues) == 1:
        text = render_cues[0].render_text.replace(r"\N", " ")
        pos1 = text.find("Trong mắt mẹ")
        pos2 = text.find("món hàng")
        assert pos1 != -1 and pos2 != -1
        assert pos1 < pos2, "Semantic order was reversed inside single cue!"
    else:
        assert "Trong mắt mẹ" in render_cues[0].render_text
        assert "món hàng" in render_cues[1].render_text
        assert render_cues[0].start < render_cues[1].start


def test_semantic_ordering_case2_chicken_leg_monologue():
    """Case 2: 'Nhưng bây giờ tôi... chỉ muốn ăn hết chiếc đùi gà...' must maintain chronological order."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(
            id="c1",
            start=53.43,
            end=54.50,
            source_text="但我现在",
            translated_text="Nhưng bây giờ tôi...",
            speaker_id="spk_narrator",
            speaker_character_id="char_qin_fuzhi",
        ),
        SubtitleCue(
            id="c2",
            start=54.50,
            end=56.50,
            source_text="只想把这只偷偷藏起来的鸡腿啃完",
            translated_text="chỉ muốn ăn hết chiếc đùi gà đã lén giấu đi này thôi.",
            speaker_id="spk_narrator",
            speaker_character_id="char_qin_fuzhi",
        ),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 1
    text = render_cues[0].render_text.replace(r"\N", " ")
    pos1 = text.find("Nhưng bây giờ tôi")
    pos2 = text.find("đùi gà")
    assert pos1 != -1 and pos2 != -1
    assert pos1 < pos2, "Chronological order was reversed!"


def test_semantic_ordering_case3_breakfast_family_monologue():
    """Case 3: 'Còn tôi thì... lớn lên trong một gia đình...' must keep order."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(
            id="c1",
            start=70.31,
            end=71.40,
            source_text="而我",
            translated_text="Còn tôi thì...",
            speaker_id="spk_narrator",
            speaker_character_id="char_meng_jingchun",
        ),
        SubtitleCue(
            id="c2",
            start=71.50,
            end=73.50,
            source_text="在一个开早餐店的家庭里",
            translated_text="lớn lên trong một gia đình bán đồ ăn sáng.",
            speaker_id="spk_narrator",
            speaker_character_id="char_meng_jingchun",
        ),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 1
    text = render_cues[0].render_text.replace(r"\N", " ")
    pos1 = text.find("Còn tôi")
    pos2 = text.find("gia đình bán đồ ăn sáng")
    assert pos1 != -1 and pos2 != -1
    assert pos1 < pos2, "Order was reversed!"


def test_speaker_dialogue_to_monologue_boundary_not_merged():
    """Case 4: Mother direct dialogue must NEVER merge with daughter monologue/intro."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(
            id="c1",
            start=14.81,
            end=16.50,
            source_text="你今天是去丢人还是去赴宴",
            translated_text="Hôm nay con đi làm mất mặt hay đi dự tiệc vậy?",
            speaker_id="spk_mom",
            speaker_character_id="char_song_zhixue",
            addressee_id="char_qin_fuzhi",
        ),
        SubtitleCue(
            id="c2",
            start=16.53,
            end=17.80,
            source_text="宋知雪",
            translated_text="Tống Tri Tuyết.",
            speaker_id=None,
            speaker_character_id=None,
            addressee_id=None,
        ),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) == 2, "Dialogue and monologue were wrongly merged across boundary!"
    assert "dự tiệc vậy" in render_cues[0].render_text
    assert "Tống Tri Tuyết" in render_cues[1].render_text


def test_no_semantic_rewriting_preserves_subject():
    """Case 5: '你的存在...' must preserve its own subject and not replace with name."""
    engine = UtteranceEngine()
    cues = [
        SubtitleCue(
            id="c1",
            start=34.00,
            end=34.50,
            source_text="秦扶栀",
            translated_text="Tần Phù Chi.",
            speaker_id="spk_bro",
            speaker_character_id="char_qin_yize",
            addressee_id="char_qin_fuzhi",
        ),
        SubtitleCue(
            id="c2",
            start=35.23,
            end=37.50,
            source_text="你的存在拉低了秦家的执行效率",
            translated_text="Sự tồn tại của em đã kéo giảm hiệu suất làm việc của nhà họ Tần.",
            speaker_id="spk_bro",
            speaker_character_id="char_qin_yize",
            addressee_id="char_qin_fuzhi",
        ),
    ]
    render_cues, _ = engine.process_cues(cues)
    assert len(render_cues) >= 1
    found_presence = any("Sự tồn tại của em" in rc.render_text for rc in render_cues)
    assert found_presence, "Subject was lost!"


def test_exact_outer_mask_preservation():
    """Section 22: cleaned[mask == 0] == original[mask == 0] bit-identical outside applied mask."""
    cleaner = HardSubCleaner()
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.putText(frame, "TEST", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    regions = [
        OCRRegion(text="TEST", confidence=0.99, points=[[0.2, 0.5], [0.8, 0.5], [0.8, 0.8], [0.2, 0.8]])
    ]
    cleaned, was_cleaned = cleaner.clean_frame(
        frame,
        mode="inpaint",
        is_subtitle_active=True,
        ocr_regions=regions,
    )
    assert was_cleaned

    # Unmodified outer frame borders must be exactly identical
    assert np.array_equal(cleaned[:30], frame[:30]), "Top background modified!"
    assert np.array_equal(cleaned[85:], frame[85:]), "Bottom background modified!"
    assert np.array_equal(cleaned[:, :10], frame[:, :10]), "Left background modified!"
    assert np.array_equal(cleaned[:, 90:], frame[:, 90:]), "Right background modified!"


def test_donor_local_color_reject():
    """Section 18: If donor local ring color differs significantly (>16.0), reject donor."""
    cleaner = HardSubCleaner()
    rng = np.random.RandomState(42)
    base = rng.randint(80, 200, (80, 200, 3), dtype=np.uint8)
    current = base.copy()
    # Darkened donor where local color difference is ~25 (between 16 and 35)
    dark_donor = np.clip(base.astype(int) - 22, 0, 255).astype(np.uint8)
    mask = np.zeros((80, 200), dtype=np.uint8)
    mask[30:50, 50:150] = 255

    aligned, g_score, l_score = cleaner._align_temporal_candidate(current, dark_donor, mask)
    assert aligned is None, "Dark mismatched donor should have been rejected by color check!"
    assert cleaner._metrics.get("donor_local_color_rejects", 0) >= 1


def test_black_blob_guard_fallback():
    """Section 19: If temporal result produces dark blob in mask area, fallback to Telea."""
    cleaner = HardSubCleaner()
    roi = np.full((80, 200, 3), 180, dtype=np.uint8)
    mask = np.zeros((80, 200), dtype=np.uint8)
    mask[30:50, 50:150] = 255

    # Donor that is pure black inside mask
    black_donor = np.full((80, 200, 3), 180, dtype=np.uint8)
    black_donor[25:55, 45:155] = 0

    result, used_temp, score = cleaner._quality_inpaint_roi(roi, mask, [black_donor])
    # Guard should trigger and reject dark blob
    cleaned_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    assert np.mean(cleaned_gray[mask > 0]) > 100, "Black blob was not prevented!"
