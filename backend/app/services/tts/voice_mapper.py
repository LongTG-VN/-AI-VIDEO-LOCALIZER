from __future__ import annotations

import re
from app.models.project import Character, CharacterVoiceProfile


def create_voice_profile_for_character(char: Character) -> CharacterVoiceProfile:
    """Generates a deterministic CharacterVoiceProfile based on character metadata.

    Standard Vietnamese Edge Neural Voices:
    - vi-VN-HoaiMyNeural (Female)
    - vi-VN-NamMinhNeural (Male)

    Pitch and Rate styles differentiate characters:
    - Young female heroine: HoaiMy, +15Hz, +0%
    - Mature female mother: HoaiMy, -25Hz, -5%
    - Rival female antagonist: HoaiMy, -10Hz, +5%
    - Young male brother: NamMinh, +15Hz, +0%
    - Mature male father: NamMinh, -25Hz, -5%
    - Neutral system/announcer: NamMinh, +0Hz, +10%
    """
    gender = (char.gender or "").lower()
    role = (char.role or "").lower()
    desc = (char.description or "").lower()
    name = (char.name_vi or char.name or "").lower()

    # 1. System / Announcer / Narration
    if any(k in role or k in name or k in desc for k in ["hệ thống", "system", "báo thức", "alarm", "nhắc nhở"]):
        return CharacterVoiceProfile(
            character_id=char.id,
            voice_id="vi-VN-NamMinhNeural",
            gender_style="neutral_system",
            base_rate="+10%",
            pitch_offset="+0Hz",
            volume="+0%",
        )

    # 2. Mother / Mature Female
    if any(k in role or k in desc or k in name for k in ["mẹ", "mother", "phu nhân", "bà"]):
        return CharacterVoiceProfile(
            character_id=char.id,
            voice_id="vi-VN-HoaiMyNeural",
            gender_style="mature_female",
            base_rate="-5%",
            pitch_offset="-25Hz",
            volume="+0%",
        )

    # 3. Father / Mature Male / Boss
    if any(k in role or k in desc or k in name for k in ["bố", "cha", "father", "ông", "tổng tài", "chủ tịch"]):
        return CharacterVoiceProfile(
            character_id=char.id,
            voice_id="vi-VN-NamMinhNeural",
            gender_style="mature_male",
            base_rate="-5%",
            pitch_offset="-25Hz",
            volume="+0%",
        )

    # 4. Brother / Young Male
    if any(k in role or k in desc or k in name for k in ["anh", "brother", "cậu", "nam", "thiếu gia"]):
        return CharacterVoiceProfile(
            character_id=char.id,
            voice_id="vi-VN-NamMinhNeural",
            gender_style="young_male",
            base_rate="+0%",
            pitch_offset="+15Hz",
            volume="+0%",
        )

    # 5. Rival / Antagonist Female (e.g. Mạnh Kim Xuân / Real daughter)
    if any(k in role or k in desc for k in ["đối thủ", "tình địch", "rival", "thật", "antagonist"]):
        return CharacterVoiceProfile(
            character_id=char.id,
            voice_id="vi-VN-HoaiMyNeural",
            gender_style="young_female_rival",
            base_rate="+5%",
            pitch_offset="-10Hz",
            volume="+0%",
        )

    # 6. Default Female / Heroine
    if gender == "female" or any(k in role or k in desc for k in ["nữ chính", "heroine", "con gái", "tiểu thư"]):
        return CharacterVoiceProfile(
            character_id=char.id,
            voice_id="vi-VN-HoaiMyNeural",
            gender_style="young_female",
            base_rate="+0%",
            pitch_offset="+15Hz",
            volume="+0%",
        )

    # 7. Default Male
    if gender == "male":
        return CharacterVoiceProfile(
            character_id=char.id,
            voice_id="vi-VN-NamMinhNeural",
            gender_style="young_male",
            base_rate="+0%",
            pitch_offset="+0Hz",
            volume="+0%",
        )

    # Fallback neutral
    return CharacterVoiceProfile(
        character_id=char.id,
        voice_id="vi-VN-HoaiMyNeural",
        gender_style="default_female",
        base_rate="+0%",
        pitch_offset="+0Hz",
        volume="+0%",
    )


def create_voice_profiles_for_project(characters: list[Character]) -> dict[str, CharacterVoiceProfile]:
    """Generates voice profiles mapping character_id -> CharacterVoiceProfile."""
    profiles: dict[str, CharacterVoiceProfile] = {}
    for char in characters:
        profiles[char.id] = create_voice_profile_for_character(char)
    return profiles
