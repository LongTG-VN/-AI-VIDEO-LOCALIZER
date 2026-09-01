from __future__ import annotations

import asyncio
import pytest
from pathlib import Path

from app.models.project import Character
from app.services.tts.factory import get_tts_provider
from app.services.tts.mock import MockTTSProvider
from app.services.tts.normalizer import normalize_for_speech, number_to_vietnamese_words
from app.services.tts.voice_mapper import (
    create_voice_profile_for_character,
    create_voice_profiles_for_project,
)


def test_vietnamese_number_normalization():
    assert number_to_vietnamese_words("0") == "không"
    assert number_to_vietnamese_words("5") == "năm"
    assert number_to_vietnamese_words("18") == "mười tám"
    assert number_to_vietnamese_words("24") == "hai mươi tư"
    assert number_to_vietnamese_words("35") == "ba mươi lăm"


def test_speech_text_normalization():
    raw_sub = r"Cổ áo lệch kìa,\N dáng ngồi sai rồi | KPI 0.3% và 18 năm!"
    clean = normalize_for_speech(raw_sub)
    assert r"\N" not in clean
    assert "|" not in clean
    assert "K P I" in clean
    assert "không phẩy ba phần trăm" in clean
    assert "mười tám năm" in clean


def test_character_voice_mapping_consistency():
    heroine = Character(
        id="char_heroine",
        name="Tần Phù Chi",
        name_vi="Tần Phù Chi",
        gender="female",
        role="nữ chính",
        description="tiểu thư nhà họ Tần",
    )
    mother = Character(
        id="char_mother",
        name="Tống Tri Tuyết",
        name_vi="Tống Tri Tuyết",
        gender="female",
        role="mẹ",
        description="mẹ của Tần Phù Chi",
    )
    father = Character(
        id="char_father",
        name="Tần Nghiễn Xuyên",
        name_vi="Tần Nghiễn Xuyên",
        gender="male",
        role="bố",
        description="bố của Tần Phù Chi",
    )
    brother = Character(
        id="char_brother",
        name="Anh trai",
        name_vi="Anh trai",
        gender="male",
        role="anh trai",
        description="anh trai tập đoàn tài chính",
    )
    rival = Character(
        id="char_rival",
        name="Mạnh Kim Xuân",
        name_vi="Mạnh Kim Xuân",
        gender="female",
        role="thiên kim thật, đối thủ",
        description="người đòi lại thân phận",
    )

    profiles = create_voice_profiles_for_project([heroine, mother, father, brother, rival])

    # 1. Voice Identity & Style Differentiation
    assert profiles["char_heroine"].voice_id == "vi-VN-HoaiMyNeural"
    assert profiles["char_heroine"].gender_style == "young_female"

    assert profiles["char_mother"].voice_id == "vi-VN-HoaiMyNeural"
    assert profiles["char_mother"].pitch_offset == "-25Hz"
    assert profiles["char_mother"].gender_style == "mature_female"

    assert profiles["char_father"].voice_id == "vi-VN-NamMinhNeural"
    assert profiles["char_father"].pitch_offset == "-25Hz"
    assert profiles["char_father"].gender_style == "mature_male"

    assert profiles["char_brother"].voice_id == "vi-VN-NamMinhNeural"
    assert profiles["char_brother"].pitch_offset == "+15Hz"
    assert profiles["char_brother"].gender_style == "young_male"

    assert profiles["char_rival"].gender_style == "young_female_rival"

    # 2. Invariant: Same Character -> Same Profile
    second_run_heroine = create_voice_profile_for_character(heroine)
    assert second_run_heroine.voice_id == profiles["char_heroine"].voice_id
    assert second_run_heroine.pitch_offset == profiles["char_heroine"].pitch_offset


def test_mock_tts_provider(tmp_path: Path):
    async def _run():
        provider = get_tts_provider("mock")
        assert isinstance(provider, MockTTSProvider)

        out = tmp_path / "test_mock.wav"
        res = await provider.synthesize(
            text="Xin chào Việt Nam",
            voice_id="vi-VN-HoaiMyNeural",
            output_path=out,
        )
        assert res.exists()
        assert res.stat().st_size > 0

    asyncio.run(_run())
