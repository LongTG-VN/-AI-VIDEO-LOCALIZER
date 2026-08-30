from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from app.models.project import Project, SubtitleCue
from app.services.translation_quality.accuracy import AccuracyReviewer
from app.services.translation_quality.consistency import ConsistencySweeper
from app.services.translation_quality.context_card import ContextCardBuilder
from app.services.translation_quality.cue_integrity import CueIntegrityReviewer
from app.services.translation_quality.fillers import FillerHandler
from app.services.translation_quality.idioms import IdiomReviewer
from app.services.translation_quality.models import (
    CueQualityResult,
    FigurativeReviewResult,
    FillerReviewResult,
    NaturalnessScore,
    QualityIssue,
    QualitySeverity,
    TranslationContextCard,
    TranslationQualityConfig,
    TranslationQualityReport,
)
from app.services.translation_quality.naturalness import NaturalnessPolisher
from app.services.translation_quality.relationships import RelationshipReviewer
from app.services.translation_quality.repair import TargetedRepairer
from app.services.translation_quality.validators import DeterministicValidator

logger = logging.getLogger(__name__)


class TranslationQualityPipeline:
    """End-to-End Orchestrator for Translation Quality Pipeline V2.
    
    Pass 0: GLOBAL CONTEXT CARD
    Pass 1: DRAFT TRANSLATION (on repaired source cues)
    Pass 2: CUE INTEGRITY REVIEW
    Pass 3: ACCURACY REVIEW
    Pass 4: RELATIONSHIP & ENTITY REVIEW
    Pass 5: IDIOM & FIGURATIVE REVIEW
    Pass 6: FILLER HANDLING & NORMALIZATION
    Pass 7: TARGETED REPAIR (only failed cues, max 2 retries)
    Pass 8: VIETNAMESE NATURALNESS POLISH V2 (1-5 scale, semantic safety gate)
    Pass 9: GLOBAL CONSISTENCY SWEEP (patch suggestions only)
    Pass 10: DETERMINISTIC FINAL VALIDATION (17 Hard Checks)
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        config: TranslationQualityConfig | None = None,
    ):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.api_key = api_key
        self.model = model
        self.config = config or TranslationQualityConfig()

        # Initialize modular reviewers & components
        self.context_card_builder = ContextCardBuilder(self.base_url, self.api_key, self.model)
        self.cue_integrity_reviewer = CueIntegrityReviewer(self.base_url, self.api_key, self.model)
        self.accuracy_reviewer = AccuracyReviewer(self.base_url, self.api_key, self.model)
        self.relationship_reviewer = RelationshipReviewer(self.base_url, self.api_key, self.model)
        self.idiom_reviewer = IdiomReviewer(self.base_url, self.api_key, self.model)
        self.filler_handler = FillerHandler()
        self.repairer = TargetedRepairer(self.base_url, self.api_key, self.model)
        self.naturalness_polisher = NaturalnessPolisher(self.base_url, self.api_key, self.model)
        self.consistency_sweeper = ConsistencySweeper(self.base_url, self.api_key, self.model)
        self.deterministic_validator = DeterministicValidator()

        self._translation_cache: dict[str, str] = {}
        self.last_report: TranslationQualityReport | None = None
        self.audit_artifacts: dict[str, Any] = {}

    def compute_cue_cache_key(
        self,
        cue: SubtitleCue,
        context_card: TranslationContextCard | None = None,
    ) -> str:
        """Cache identity including source_text, cue_id, context_hash, relationship_hash, prompt_version, quality_version."""
        ctx_hash = hashlib.sha256(
            json.dumps(context_card.model_dump() if context_card else {}, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        rel_str = f"{cue.speaker_character_id}:{cue.addressee_character_id}:{cue.discourse_mode}"
        rel_hash = hashlib.sha256(rel_str.encode("utf-8")).hexdigest()[:16]
        raw_key = f"{cue.id}:{cue.source_text}:{ctx_hash}:{rel_hash}:{self.config.translation_prompt_version}:{self.config.translation_quality_version}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def run_pipeline(
        self,
        project: Project,
        audit_output_dir: Path | str | None = None,
    ) -> TranslationQualityReport:
        t_start = time.time()
        cues = project.cues
        if not cues:
            return TranslationQualityReport()

        report = TranslationQualityReport(total_cues=len(cues))
        issue_counts: dict[str, int] = {}
        metrics: dict[str, Any] = {
            "draft_calls": 0,
            "review_calls": 0,
            "repair_calls": 0,
            "naturalness_calls": 0,
            "idiom_flagged": 0,
            "filler_handled": 0,
            "retry_distribution": {"0_retries": 0, "1_retry": 0, "2_retries": 0, "needs_review": 0},
            "cache_hits": 0,
        }

        # PASS 0: GLOBAL CONTEXT CARD
        context_card: TranslationContextCard | None = None
        if self.config.context_card:
            context_card = self.context_card_builder.build_context_card(project)
            self.audit_artifacts["context_card.json"] = context_card.model_dump()

        # PASS 1: DRAFT TRANSLATION (Uses SemanticTranslationGroups on Repaired Source Cues)
        untranslated = [c for c in cues if not c.translated_text]
        if untranslated and self.base_url and self.model:
            from app.services.semantic_grouping.pipeline import SemanticGroupingPipeline
            group_pipeline = SemanticGroupingPipeline(self.base_url, self.api_key, self.model)
            groups = group_pipeline.process_project(project, save_trace_dir=audit_output_dir)
            metrics["semantic_groups"] = len(groups)
            metrics["multi_cue_groups"] = sum(1 for g in groups if len(g.source_cue_ids) > 1)
            metrics["draft_calls"] += len(groups)
            self.audit_artifacts["semantic_translation_groups.json"] = [g.model_dump() for g in groups]

        draft_map = {c.id: c.translated_text or "" for c in cues}
        self.audit_artifacts["draft_translation.json"] = draft_map

        # PASS 2, 3, 4, 5, 6: MULTI-PASS REVIEW
        all_issues_by_cue: dict[str, list[QualityIssue]] = {c.id: [] for c in cues}

        # Spot check: Low source confidence
        for cue in cues:
            if cue.source_confidence is not None and cue.source_confidence < 0.50:
                all_issues_by_cue[cue.id].append(
                    QualityIssue(
                        type="source.low_confidence",
                        severity=QualitySeverity.MAJOR,
                        message=f"Source cue confidence is low ({cue.source_confidence:.2f}); verify Chinese source integrity.",
                        reviewer="source_spot_check",
                    )
                )

        # Pass 2: Cue Integrity
        if self.config.cue_integrity:
            integ_issues = self.cue_integrity_reviewer.evaluate_cues(project, cues, context_card)
            self.audit_artifacts["cue_integrity_review.json"] = {
                cid: [iss.model_dump() for iss in iss_list] for cid, iss_list in integ_issues.items() if iss_list
            }
            for cid, iss_list in integ_issues.items():
                all_issues_by_cue[cid].extend(iss_list)

        # Pass 3: Accuracy Review
        if self.config.accuracy:
            acc_issues = self.accuracy_reviewer.evaluate_cues(project, cues, context_card)
            self.audit_artifacts["accuracy_review.json"] = {
                cid: [iss.model_dump() for iss in iss_list] for cid, iss_list in acc_issues.items() if iss_list
            }
            for cid, iss_list in acc_issues.items():
                all_issues_by_cue[cid].extend(iss_list)

        # Pass 4: Entity & Relationship Review
        if self.config.relationships:
            rel_issues = self.relationship_reviewer.evaluate_cues(project, cues, context_card)
            self.audit_artifacts["relationship_review.json"] = {
                cid: [iss.model_dump() for iss in iss_list] for cid, iss_list in rel_issues.items() if iss_list
            }
            for cid, iss_list in rel_issues.items():
                all_issues_by_cue[cid].extend(iss_list)

        # Pass 5: Idiom & Figurative Review
        if self.config.idioms:
            idiom_issues, idiom_reviews = self.idiom_reviewer.evaluate_cues(project, cues, context_card)
            self.audit_artifacts["idiom_review.json"] = {
                cid: res.model_dump() for cid, res in idiom_reviews.items()
            }
            for cid, iss_list in idiom_issues.items():
                if iss_list:
                    metrics["idiom_flagged"] += len(iss_list)
                    all_issues_by_cue[cid].extend(iss_list)

        # Pass 6: Filler Handling & Normalization
        if self.config.fillers:
            fil_issues, fil_reviews, fil_normalized = self.filler_handler.evaluate_cues(project, cues, context_card)
            self.audit_artifacts["filler_review.json"] = {
                cid: res.model_dump() for cid, res in fil_reviews.items()
            }
            for cid, norm_text in fil_normalized.items():
                target_cue = next((c for c in cues if c.id == cid), None)
                if target_cue:
                    target_cue.translated_text = norm_text
                    metrics["filler_handled"] += 1
            for cid, iss_list in fil_issues.items():
                all_issues_by_cue[cid].extend(iss_list)

        # Classify first-pass status
        first_pass_failed_ids: set[str] = set()
        repair_log: list[dict[str, Any]] = []

        for cue in cues:
            cue_issues = all_issues_by_cue.get(cue.id, [])
            for iss in cue_issues:
                issue_counts[iss.type] = issue_counts.get(iss.type, 0) + 1

            has_major_crit = any(iss.severity in [QualitySeverity.MAJOR, QualitySeverity.CRITICAL] for iss in cue_issues)
            if has_major_crit:
                first_pass_failed_ids.add(cue.id)
                report.cue_results[cue.id] = CueQualityResult(
                    cue_id=cue.id,
                    status="FAIL",
                    issues=cue_issues,
                    original_translation=cue.translated_text or "",
                    final_translation=cue.translated_text or "",
                )
            else:
                report.passed_first_attempt += 1
                report.cue_results[cue.id] = CueQualityResult(
                    cue_id=cue.id,
                    status="PASS",
                    issues=cue_issues,
                    original_translation=cue.translated_text or "",
                    final_translation=cue.translated_text or "",
                )

        metrics["retry_distribution"]["0_retries"] = report.passed_first_attempt

        # PASS 7: TARGETED REPAIR (Only failed cues enter repair)
        current_failed_ids = set(first_pass_failed_ids)
        cues_repaired_count = 0

        if self.config.targeted_repair and current_failed_ids:
            for retry_round in range(1, self.config.max_retries + 1):
                if not current_failed_ids:
                    break

                failed_cues = [c for c in cues if c.id in current_failed_ids]
                logger.info("Starting Targeted Repair Round %d for %d failed cues...", retry_round, len(failed_cues))
                metrics["repair_calls"] += len(failed_cues)

                repairs = self.repairer.repair_failed_cues(project, failed_cues, all_issues_by_cue, context_card=context_card)

                for cue in failed_cues:
                    if cue.id in repairs:
                        repaired_text, rep_conf = repairs[cue.id]
                        repair_log.append({
                            "cue_id": cue.id,
                            "round": retry_round,
                            "before": cue.translated_text,
                            "after": repaired_text,
                            "issues_targeted": [iss.type for iss in all_issues_by_cue.get(cue.id, [])],
                        })
                        cue.translated_text = repaired_text
                        if rep_conf is not None:
                            cue.translation_confidence = rep_conf
                            cue.confidence = rep_conf

                # Re-evaluate only repaired cues with reviewers
                re_integ = self.cue_integrity_reviewer.evaluate_cues(project, failed_cues, context_card) if self.config.cue_integrity else {}
                re_acc = self.accuracy_reviewer.evaluate_cues(project, failed_cues, context_card) if self.config.accuracy else {}
                re_rel = self.relationship_reviewer.evaluate_cues(project, failed_cues, context_card) if self.config.relationships else {}
                re_idiom, _ = self.idiom_reviewer.evaluate_cues(project, failed_cues, context_card) if self.config.idioms else ({}, {})

                next_failed: set[str] = set()
                for cue in failed_cues:
                    new_issues = []
                    new_issues.extend(re_integ.get(cue.id, []))
                    new_issues.extend(re_acc.get(cue.id, []))
                    new_issues.extend(re_rel.get(cue.id, []))
                    new_issues.extend(re_idiom.get(cue.id, []))
                    all_issues_by_cue[cue.id] = new_issues

                    has_still_fail = any(iss.severity in [QualitySeverity.MAJOR, QualitySeverity.CRITICAL] for iss in new_issues)
                    if has_still_fail:
                        next_failed.add(cue.id)
                    else:
                        cues_repaired_count += 1
                        res = report.cue_results[cue.id]
                        res.status = "PASS"
                        res.attempts = retry_round
                        res.final_translation = cue.translated_text or ""
                        if retry_round == 1:
                            metrics["retry_distribution"]["1_retry"] += 1
                        elif retry_round == 2:
                            metrics["retry_distribution"]["2_retries"] += 1

                current_failed_ids = next_failed

            for cid in current_failed_ids:
                res = report.cue_results[cid]
                res.status = "NEEDS_REVIEW"
                res.attempts = self.config.max_retries
                target_cue = next((c for c in cues if c.id == cid), None)
                if target_cue:
                    target_cue.needs_review = True
                    target_cue.review_notes = "; ".join([iss.message for iss in all_issues_by_cue.get(cid, [])])
                    res.review_notes = target_cue.review_notes

            metrics["retry_distribution"]["needs_review"] = len(current_failed_ids)

        report.repaired = cues_repaired_count
        report.needs_review = len(current_failed_ids)
        self.audit_artifacts["repair_log.json"] = repair_log

        # PASS 8: VIETNAMESE NATURALNESS POLISH V2 (Semantic Safety Gate)
        if self.config.naturalness:
            passed_cues = [c for c in cues if c.id not in current_failed_ids]
            nat_issues, polished_map = self.naturalness_polisher.evaluate_and_polish_cues(
                project, passed_cues, context_card
            )
            nat_scores = self.naturalness_polisher.last_scores
            self.audit_artifacts["naturalness_review.json"] = {
                "scores": {cid: score.model_dump() for cid, score in nat_scores.items()},
                "issues": {cid: [iss.model_dump() for iss in iss_list] for cid, iss_list in nat_issues.items() if iss_list},
                "polished_applied": polished_map,
            }
            metrics["naturalness_calls"] += len(passed_cues)
            for cid, pol_text in polished_map.items():
                t_cue = next((c for c in cues if c.id == cid), None)
                if t_cue:
                    t_cue.translated_text = pol_text
                    if cid in report.cue_results:
                        report.cue_results[cid].final_translation = pol_text

            for cid, n_score in nat_scores.items():
                if cid in report.cue_results:
                    report.cue_results[cid].naturalness_score = n_score.score

        # PASS 9: GLOBAL CONSISTENCY SWEEP
        if self.config.consistency:
            cons_issues, patches_map = self.consistency_sweeper.sweep_project(project, cues, context_card)
            self.audit_artifacts["consistency_review.json"] = {
                "issues": [iss.model_dump() for iss in cons_issues],
                "patches_applied": patches_map,
            }
            for cid, pat_text in patches_map.items():
                t_cue = next((c for c in cues if c.id == cid), None)
                if t_cue:
                    t_cue.translated_text = pat_text
                    if cid in report.cue_results:
                        report.cue_results[cid].final_translation = pat_text

        # PASS 10: DETERMINISTIC FINAL VALIDATION (17 Hard Checks)
        if self.config.deterministic_validation:
            val_pass, val_issues = self.deterministic_validator.validate_project(project)
            self.audit_artifacts["final_validation.json"] = {
                "validation_passed": val_pass,
                "issues": [iss.model_dump() for iss in val_issues],
            }
            for iss in val_issues:
                issue_counts[iss.type] = issue_counts.get(iss.type, 0) + 1

        t_elapsed = time.time() - t_start
        metrics["pipeline_time_s"] = round(t_elapsed, 3)
        report.issue_counts = issue_counts
        report.metrics = metrics
        self.audit_artifacts["quality_summary.json"] = report.model_dump()

        if audit_output_dir:
            out_dir = Path(audit_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for filename, artifact_data in self.audit_artifacts.items():
                with open(out_dir / filename, "w", encoding="utf-8") as f:
                    json.dump(artifact_data, f, ensure_ascii=False, indent=2)
            logger.info("Saved all translation quality audit artifacts to %s", out_dir)

        self.last_report = report
        logger.info(
            "TranslationQualityPipeline V2 completed in %.3fs: total=%d, passed_1st=%d, repaired=%d, needs_review=%d",
            t_elapsed,
            report.total_cues,
            report.passed_first_attempt,
            report.repaired,
            report.needs_review,
        )
        return report
