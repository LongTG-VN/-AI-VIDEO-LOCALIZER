from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from app.core.localization_policy import LocalizationPolicy, PipelineVersionManifest
from app.models.project import OCRRegion, Project, RenderOptions, SubtitleCue, get_final_source_text, get_final_vi_text
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.pipeline_orchestrator import LocalizationPipeline
from app.services.utterance_engine import UtteranceEngine
from app.services.subtitles import to_ass


def test_01_canonical_accessors():
    cue1 = SubtitleCue(
        id="c1",
        start=1.0,
        end=3.0,
        source_text="你好世界",
        repaired_source="你好世界，我是小明",
        final_translation="Chào thế giới, tôi là Tiểu Minh.",
    )
    assert get_final_source_text(cue1) == "你好世界，我是小明"
    assert get_final_vi_text(cue1) == "Chào thế giới, tôi là Tiểu Minh."

    suppressed = SubtitleCue(
        id="c2",
        start=3.5,
        end=4.0,
        source_text="嗯",
        suppression_status="SUPPRESSED_FILLER",
        final_translation="Ừm",
    )
    assert get_final_source_text(suppressed) == "嗯"
    assert get_final_vi_text(suppressed) == ""


def test_02_policy_and_manifest():
    policy = LocalizationPolicy(profile="stable")
    assert policy.version_manifest.pipeline_version == "stable-v1"
    assert policy.version_manifest.subtitle_layout_version == "v12"
    assert policy.version_manifest.source_cover_version == "v11"
    h = policy.compute_config_hash()
    assert len(h) == 16


def test_03_source_cover_and_layout_integration():
    cleaner = PatchCoverCleaner()
    engine = UtteranceEngine(max_line_chars=34)

    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=3.0,
            source_text="这是一家非常有名的餐厅",
            final_translation="Đây là một nhà hàng vô cùng nổi tiếng ở trung tâm thành phố.",
            ocr_regions=[OCRRegion(points=[[0.35, 0.85], [0.65, 0.85], [0.65, 0.92], [0.35, 0.92]])],
            original_source_cue_ids=["c1"],
        ),
        SubtitleCue(
            id="c2",
            start=3.5,
            end=4.2,
            source_text="嗯",
            suppression_status="SUPPRESSED_FILLER",
            ocr_regions=[OCRRegion(points=[[0.48, 0.88], [0.52, 0.88], [0.52, 0.93], [0.48, 0.93]])],
            original_source_cue_ids=["c2"],
        ),
    ]

    contexts = cleaner.build_render_cue_contexts(cues, 1280, 720)
    assert len(contexts) == 2
    assert contexts[0]["has_vi"] is True
    assert contexts[1]["has_vi"] is False

    render_cues, _ = engine.process_cues(cues, translated=True)
    assert len(render_cues) == 1  # Suppressed cue produces no VI render cue

    ass_text = to_ass(cues, RenderOptions(), 1280, 720, translated=True)
    assert "Đây là một nhà hàng" in ass_text
    assert "SUPPRESSED" not in ass_text


def test_04_pipeline_orchestrator_invariants():
    pipeline = LocalizationPipeline()
    cues = [
        SubtitleCue(
            id="c1",
            start=1.0,
            end=2.0,
            source_text="好的",
            final_translation="Được thôi.",
            original_source_cue_ids=["c1"],
        ),
        SubtitleCue(
            id="c2",
            start=2.5,
            end=3.0,
            source_text="啊",
            suppression_status="SUPPRESSED_FILLER",
            original_source_cue_ids=["c2"],
        ),
    ]
    contexts = pipeline.patch_cleaner.build_render_cue_contexts(cues, 1280, 720)
    render_cues, _ = pipeline.utterance_engine.process_cues(cues, translated=True)
    inv = pipeline._check_invariants(cues, contexts, render_cues)

    assert inv["critical_failures"] == 0
    assert inv["unexplained_missing_source_cues"] == 0
    assert inv["duplicate_source_mappings"] == 0
    assert inv["pass_cue_without_final_vi"] == 0
    assert inv["suppressed_cue_with_vi"] == 0
