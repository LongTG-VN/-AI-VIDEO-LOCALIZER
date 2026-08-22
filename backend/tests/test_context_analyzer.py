from app.models.project import Project, SubtitleCue
from app.services.context_analyzer import build_context_analysis_payload


def test_context_payload_keeps_stable_speaker_and_cue_ids():
    project = Project(name="demo", source_video_path="demo.mp4", cues=[SubtitleCue(id="c1", start=0, end=1, speaker_id="speaker_0", source_text="你去哪？"), SubtitleCue(id="c2", start=1, end=2, speaker_id="speaker_1", source_text="关你什么事？")])
    payload = build_context_analysis_payload(project)
    assert payload["known_speaker_ids"] == ["speaker_0", "speaker_1"]
    assert payload["cues"][1]["cue_id"] == "c2"
