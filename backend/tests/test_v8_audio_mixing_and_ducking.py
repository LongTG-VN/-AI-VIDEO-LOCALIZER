from __future__ import annotations

import math
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from app.models.project import DubbingOptions
from app.services.audio_mixer import AudioMixer, CueAudioSegment


def create_dummy_wav(path: Path, duration: float, sample_rate: int = 44100, amp: float = 0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    sig = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo = np.column_stack([sig, sig])
    sf.write(str(path), stereo, sample_rate, subtype="PCM_16")


def test_audio_mixer_duration_and_peak(tmp_path: Path):
    mixer = AudioMixer(sample_rate=44100)

    # 1. Create 5s background audio
    bg_file = tmp_path / "bg.wav"
    create_dummy_wav(bg_file, duration=5.0, amp=0.4)

    dur = mixer.get_audio_duration(bg_file)
    assert abs(dur - 5.0) < 0.01

    # 2. Create 1.5s TTS cue
    cue_file = tmp_path / "cue1.wav"
    create_dummy_wav(cue_file, duration=1.5, amp=0.5)

    # 3. Mix
    out_mix = tmp_path / "mixed.wav"
    options = DubbingOptions(
        ducking_music_db=-4.0,
        ducking_dialogue_db=-24.0,
        crossfade_ms=20,
    )
    cue_segments = [
        CueAudioSegment(start=1.0, end=2.5, audio_path=cue_file, volume_linear=1.0)
    ]

    res_path, peak_db = mixer.mix_dubbing_tracks(
        background_path=bg_file,
        vocal_path=None,
        cue_segments=cue_segments,
        output_mix_path=out_mix,
        options=options,
        target_duration=5.0,
    )

    assert res_path.exists()
    assert peak_db <= -0.90  # Target <= -1.0 dBFS with small floating tolerance

    # 4. Verify output duration
    out_dur = mixer.get_audio_duration(res_path)
    assert abs(out_dur - 5.0) < 0.05


def test_time_stretch_audio_bounds(tmp_path: Path):
    mixer = AudioMixer(sample_rate=44100)
    orig_wav = tmp_path / "orig.wav"
    create_dummy_wav(orig_wav, duration=2.0, amp=0.3)

    # Speed up by 1.25x
    stretched = tmp_path / "stretched.wav"
    out_path, new_dur = mixer.time_stretch_audio(orig_wav, stretched, speed=1.25)
    assert out_path.exists()
    assert abs(new_dur - (2.0 / 1.25)) < 0.15
