from pathlib import Path
from app.models.project import RenderOptions
from app.services.renderer import build_subtitle_filter


def test_subtitle_filter_contains_style(tmp_path: Path):
    path = tmp_path / "captions.srt"
    path.write_text("", encoding="utf-8")
    value = build_subtitle_filter(path, RenderOptions(font_size=24, margin_v=40))
    assert "subtitles=" in value
    assert "FontSize=24" in value
    assert "MarginV=40" in value
