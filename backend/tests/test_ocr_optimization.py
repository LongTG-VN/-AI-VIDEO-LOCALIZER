import pytest
import numpy as np
from app.models.project import SubtitleCue
from app.services.ocr.paddle import (
    PaddleSubtitleOCREngine,
    _similar,
    _normalize,
    VALID_SINGLE_CHAR_CHINESE,
    KNOWN_NOISE_PATTERNS
)


def test_roi_crop_boundaries():
    engine = PaddleSubtitleOCREngine(crop_top_ratio=0.65, crop_bottom_ratio=0.95, crop_left_ratio=0.06, crop_right_ratio=0.94)
    assert engine.crop_top_ratio == 0.65
    assert engine.crop_bottom_ratio == 0.95
    assert engine.crop_left_ratio == 0.06
    assert engine.crop_right_ratio == 0.94

    # Clamping tests
    extreme_engine = PaddleSubtitleOCREngine(crop_top_ratio=-0.5, crop_bottom_ratio=1.5)
    assert extreme_engine.crop_top_ratio == 0.0
    assert extreme_engine.crop_bottom_ratio == 1.0


def test_multiline_merge_and_sort():
    engine = PaddleSubtitleOCREngine()
    # Mock PaddleOCR prediction with 2 vertically stacked lines
    class MockResult:
        json = {
            "res": {
                "rec_texts": ["只有一件需要时刻打磨的商品", "她的眼里没有女儿"],
                "rec_scores": [0.98, 0.96],
                "dt_polys": [
                    [[100, 420], [300, 420], [300, 440], [100, 440]],  # Line 2 (Y=420)
                    [[100, 390], [300, 390], [300, 410], [100, 410]],  # Line 1 (Y=390)
                ],
            }
        }

    text, score, noise_cnt, ml_cnt, regions = engine._filter_and_merge_predictions(
        [MockResult()], x1=50, y1=200, w=852, h=480
    )
    # Should sort top-to-bottom: "她的眼里没有女儿" then "只有一件需要时刻打磨的商品"
    assert text == "她的眼里没有女儿只有一件需要时刻打磨的商品"
    assert score == pytest.approx(0.97, 0.01)
    assert ml_cnt == 1
    assert noise_cnt == 0
    assert len(regions) == 2
    assert regions[0].text == "她的眼里没有女儿"
    assert regions[1].text == "只有一件需要时刻打磨的商品"
    # Points should be normalized to full frame [0, 1]
    assert 0.0 <= regions[0].points[0][0] <= 1.0
    assert 0.0 <= regions[0].points[0][1] <= 1.0


def test_noise_filtering():
    engine = PaddleSubtitleOCREngine()
    class MockResult:
        json = {
            "res": {
                "rec_texts": ["10.5o", "MILK", "T", "领口歪了", "10:50"],
                "rec_scores": [0.70, 0.65, 0.40, 0.99, 0.85],
                "dt_polys": [
                    [[10, 10], [50, 10], [50, 20], [10, 20]],
                    [[20, 20], [60, 20], [60, 30], [20, 30]],
                    [[30, 30], [40, 30], [40, 40], [30, 40]],
                    [[100, 400], [300, 400], [300, 430], [100, 430]],
                    [[10, 50], [50, 50], [50, 60], [10, 60]],
                ],
            }
        }

    text, score, noise_cnt, ml_cnt, regions = engine._filter_and_merge_predictions([MockResult()])
    assert text == "领口歪了"
    assert score == pytest.approx(0.99, 0.01)
    assert noise_cnt == 4
    assert len(regions) == 1
    assert regions[0].text == "领口歪了"


def test_preservation_of_valid_short_chinese():
    engine = PaddleSubtitleOCREngine()
    for valid_word in ["妈", "爸", "不", "好", "走", "对"]:
        class MockResult:
            json = {
                "res": {
                    "rec_texts": [valid_word],
                    "rec_scores": [0.92],
                    "dt_polys": [[[100, 400], [130, 400], [130, 430], [100, 430]]],
                }
            }
        text, score, noise_cnt, _, regions = engine._filter_and_merge_predictions([MockResult()])
        assert text == valid_word
        assert noise_cnt == 0
        assert len(regions) == 1


def test_temporal_normalization_and_confidence():
    engine = PaddleSubtitleOCREngine()
    cues = [
        SubtitleCue(start=1.0, end=1.5, source_text="领口歪了", ocr_confidence=0.95),
        SubtitleCue(start=1.5, end=2.0, source_text="领口歪了", ocr_confidence=0.98),
        SubtitleCue(start=2.5, end=3.0, source_text="坐姿不对", ocr_confidence=0.92),
    ]
    # Test similarity and normalization
    assert _similar(cues[0].source_text, cues[1].source_text) >= 0.85
    assert _similar(cues[1].source_text, cues[2].source_text) < 0.50
