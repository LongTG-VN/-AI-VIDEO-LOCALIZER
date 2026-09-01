from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.models.project import Project, RenderOptions, VisualEditConfig, SubtitleCue
from app.services.patch_cover_cleaner import PatchCoverCleaner
from app.services.renderer import Renderer
from app.services.subtitles import to_ass
from app.services.translation_quality import (
    TranslationQualityConfig,
    TranslationQualityPipeline,
)
from app.services.utterance_engine import UtteranceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Master LLM credentials (loaded dynamically from environment / master .env)
env_path = Path(r"D:\antigravity\Skill\.env")
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

LLM_BASE_URL = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def main():
    logger.info("=== STARTING TRANSLATION QUALITY PIPELINE V1 ON UNSEEN VIDEO ===")

    project_json = Path(r"D:\codex\-AI-VIDEO-LOCALIZER\backend\data\test_pipeline_40954759795\project_fused_translated.json")
    if not project_json.exists():
        logger.error("Project json not found: %s", project_json)
        sys.exit(1)

    with open(project_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    from app.models.project import Character, GlossaryEntry, RelationshipRule
    project = Project.model_validate(data)
    logger.info("Loaded project '%s' with %d cues", project.name, len(project.cues))

    # Populate canonical character, relationship, and glossary graph
    project.characters = [
        Character(id="char_father", name="Bố", name_zh="父亲", name_vi="Bố", role="father", gender="male", aliases=["爸", "父亲", "伯父"]),
        Character(id="char_jiang", name="Giang Húc", name_zh="江旭", name_vi="Giang Húc", role="protagonist", gender="male", aliases=["江总", "小江"]),
        Character(id="char_su", name="Tô Đường", name_zh="苏棠", name_vi="Tô Đường", role="daughter", gender="female", aliases=["苏小姐", "棠棠"]),
    ]
    project.relationships = [
        RelationshipRule(from_character_id="char_father", to_character_id="char_su", relationship="father_daughter", vi_self="bố", vi_other="con gái"),
        RelationshipRule(from_character_id="char_father", to_character_id="char_jiang", relationship="elder_younger", vi_self="bác", vi_other="cháu"),
        RelationshipRule(from_character_id="char_jiang", to_character_id="char_su", relationship="colleague_romantic", vi_self="tôi", vi_other="cô"),
    ]
    project.glossary = [
        GlossaryEntry(source="黑卡", target="thẻ đen", category="item"),
        GlossaryEntry(source="无限额", target="không giới hạn", category="finance"),
        GlossaryEntry(source="江旭", target="Giang Húc", category="name"),
        GlossaryEntry(source="苏棠", target="Tô Đường", category="name"),
    ]
    logger.info("Configured project with %d characters, %d relationships, %d glossary entries",
                len(project.characters), len(project.relationships), len(project.glossary))

    audit_dir = Path(r"D:\codex\-AI-VIDEO-LOCALIZER\backend\data\test_pipeline_40954759795\translation_quality")
    audit_dir.mkdir(parents=True, exist_ok=True)

    # 1. INITIAL AUDIT ONLY (Before Repair)
    logger.info("--- PHASE 1: INITIAL QUALITY AUDIT ONLY (NO REPAIR) ---")
    audit_config = TranslationQualityConfig(
        enabled=True,
        context_card=True,
        cue_integrity=True,
        accuracy=True,
        relationships=True,
        targeted_repair=False,  # Audit only first
        naturalness=False,
        consistency=False,
        deterministic_validation=True,
    )
    audit_pipeline = TranslationQualityPipeline(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        config=audit_config,
    )
    initial_report = audit_pipeline.run_pipeline(project)

    logger.info("Initial Audit Report: Total=%d, Passed=%d, Failed=%d",
                initial_report.total_cues, initial_report.passed_first_attempt, len(initial_report.cue_results) - initial_report.passed_first_attempt)
    logger.info("Initial Issue Counts: %s", json.dumps(initial_report.issue_counts, indent=2))

    with open(audit_dir / "initial_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(initial_report.model_dump(mode="json"), f, ensure_ascii=False, indent=2)

    # 2. FULL PIPELINE EXECUTION (Pass 0 to Pass 8)
    logger.info("--- PHASE 2: EXECUTING FULL TRANSLATION QUALITY PIPELINE V1 ---")
    full_config = TranslationQualityConfig(
        enabled=True,
        context_card=True,
        cue_integrity=True,
        accuracy=True,
        relationships=True,
        targeted_repair=True,
        max_retries=2,
        naturalness=True,
        consistency=True,
        deterministic_validation=True,
    )
    full_pipeline = TranslationQualityPipeline(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        config=full_config,
    )
    final_report = full_pipeline.run_pipeline(project, audit_output_dir=audit_dir)

    logger.info("Final Pipeline Report: Total=%d, Passed_1st=%d, Repaired=%d, Needs_Review=%d",
                final_report.total_cues, final_report.passed_first_attempt, final_report.repaired, final_report.needs_review)

    # Save repaired project json
    project_repaired_json = Path(r"D:\codex\-AI-VIDEO-LOCALIZER\backend\data\test_pipeline_40954759795\project_repaired_v1.json")
    with open(project_repaired_json, "w", encoding="utf-8") as f:
        json.dump(project.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    logger.info("Saved repaired project to %s", project_repaired_json)

    # 3. 98–110s DEEP LINEAGE TRACE
    logger.info("--- PHASE 3: 98–110s DEEP LINEAGE TRACE ---")
    engine = UtteranceEngine()
    render_cues, _ = engine.process_cues(project.cues, translated=True)

    trace_records = []
    for c in project.cues:
        if 90.0 <= c.start <= 118.0:
            matching_render = [rc for rc in render_cues if c.id in rc.source_cue_ids]
            rc_info = matching_render[0] if matching_render else None
            trace_records.append({
                "source_cue_id": c.id,
                "start": c.start,
                "end": c.end,
                "source_text": c.source_text,
                "speaker": c.speaker_id,
                "addressee": c.addressee_id,
                "discourse_mode": c.discourse_mode,
                "initial_draft_vi": initial_report.cue_results[c.id].original_translation if c.id in initial_report.cue_results else "",
                "final_repaired_vi": c.translated_text,
                "issues": [iss.model_dump() for iss in initial_report.cue_results[c.id].issues] if c.id in initial_report.cue_results else [],
                "render_cue_id": rc_info.render_id if rc_info else None,
                "render_text": rc_info.render_text if rc_info else None,
                "render_start": rc_info.start if rc_info else None,
                "render_end": rc_info.end if rc_info else None,
            })

    with open(audit_dir / "trace_98_110s.json", "w", encoding="utf-8") as f:
        json.dump(trace_records, f, ensure_ascii=False, indent=2)
    logger.info("Saved 98-110s lineage trace to %s", audit_dir / "trace_98_110s.json")

    # 4. RENDER OUTPUT VIDEO: UNSEEN_3MIN_TRANSLATION_QUALITY_V1.mp4
    logger.info("--- PHASE 4: RENDERING FINAL LOCALIZED VIDEO WITH LOCKED PATCHCOVER PRESET ---")
    output_mp4 = Path(r"D:\codex\UNSEEN_3MIN_TRANSLATION_QUALITY_V1.mp4")
    render_options = RenderOptions(
        visual_edit=VisualEditConfig(preset="shortform_white_black_soft_bg"),
        hardsub_removal_mode="auto",
        use_nvenc=True,
    )
    renderer = Renderer("ffmpeg", "ffprobe")
    render_metrics = renderer.render(project, output_mp4, render_options)
    logger.info("Render finished successfully: %s", render_metrics)

    logger.info("=== TRANSLATION QUALITY PIPELINE V1 COMPLETED SUCCESSFULLY! ===")
    logger.info("Output Video: %s", output_mp4)
    logger.info("File Size: %d bytes (%.2f MB)", output_mp4.stat().st_size, output_mp4.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
