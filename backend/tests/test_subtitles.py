from app.models.project import SubtitleCue
from app.services.subtitles import format_timestamp, parse_srt, to_srt


def test_timestamp_format():
    assert format_timestamp(3723.045) == "01:02:03,045"


def test_parse_and_export_srt():
    source = """1
00:00:01,000 --> 00:00:03,500
你好

2
00:00:05,000 --> 00:00:07,000
回来了吗？
"""
    cues = parse_srt(source)
    assert len(cues) == 2
    assert cues[0].source_text == "你好"
    cues[0].translated_text = "Xin chào."
    cues[1].translated_text = "Em về rồi à?"
    rendered = to_srt(cues)
    assert "Xin chào." in rendered
    assert "00:00:05,000" in rendered


def test_source_fallback_when_untranslated():
    cue = SubtitleCue(start=0, end=1, source_text="测试")
    assert "测试" in to_srt([cue])
