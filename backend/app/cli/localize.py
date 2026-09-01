from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.core.config import Settings
from app.core.localization_policy import LocalizationPolicy
from app.models.project import Project
from app.services.pipeline_orchestrator import LocalizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("localize_cli")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Video Localizer — Canonical Localization Pipeline CLI",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video or existing project.json file",
    )
    parser.add_argument(
        "--target", "-t",
        default="vi",
        help="Target language code (default: vi)",
    )
    parser.add_argument(
        "--profile", "-p",
        default="stable",
        choices=["stable", "debug"],
        help="Localization profile to execute (default: stable)",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for finalized artifacts (default: output)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debugging and QA artifact retention",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or Initialize Project
    if input_path.suffix.lower() == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        project = Project(**data)
    else:
        project = Project(
            name=input_path.stem,
            source_video_path=str(input_path.resolve()),
            target_language=args.target,
        )

    # 2. Setup Policy
    policy = LocalizationPolicy(
        profile="debug" if args.debug or args.profile == "debug" else "stable",
    )

    pipeline = LocalizationPipeline(policy=policy)
    logger.info("Starting LocalizationPipeline with profile '%s'...", policy.profile)

    try:
        report = pipeline.run(project, out_dir)
        logger.info("Pipeline completed successfully! Summary:")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
