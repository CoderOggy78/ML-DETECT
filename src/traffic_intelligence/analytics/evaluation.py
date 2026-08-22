"""
Self-Consistency Quality Evaluation (Without Ground Truth Annotations) & Optional MOT Metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from traffic_intelligence.schema import Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.evaluation")


class SelfConsistencyEvaluator:
    """
    Evaluates tracking reliability and reconstruction validity when NO human annotations exist.
    Evaluates: Track continuity, velocity smoothness, physical realism, jerk energy, and camera stabilization consistency.
    """

    @staticmethod
    def evaluate_unsupervised_consistency(
        trajectories: List[Trajectory],
        residual_stabilization_errors: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        if not trajectories:
            return {
                "overall_quality_score": 0.0,
                "valid_trajectory_ratio": 0.0,
                "mean_track_duration_s": 0.0,
                "mean_missing_ratio": 1.0,
                "physical_plausibility_rate": 0.0,
                "mean_jerk_energy": 0.0,
                "camera_stabilization_residual_px": 0.0,
            }

        valid_trajs = [t for t in trajectories if t.is_valid]
        valid_ratio = len(valid_trajs) / max(1, len(trajectories))
        durations = [t.duration_s for t in trajectories]
        missing_ratios = [t.missing_ratio for t in trajectories]
        quality_scores = [t.quality_score for t in trajectories]

        # Physical plausibility: fraction of trajectories without impossible velocity/acceleration jumps
        plausible_count = sum(
            1 for t in trajectories
            if t.max_speed_kmh <= 180.0 and t.max_acceleration_mps2 <= 12.0 and t.max_deceleration_mps2 >= -15.0
        )
        plausibility_rate = plausible_count / max(1, len(trajectories))

        # Jerk energy across all points
        jerks = []
        for t in trajectories:
            j_list = [p.jerk_mps3 for p in t.points if p.jerk_mps3 is not None]
            jerks.extend(j_list)

        mean_jerk = float(np.mean(np.abs(jerks))) if jerks else 0.0

        stab_err = 0.0
        if residual_stabilization_errors:
            err_vals = list(residual_stabilization_errors.values())
            stab_err = float(np.mean(err_vals)) if err_vals else 0.0

        overall_score = float(
            0.35 * np.mean(quality_scores)
            + 0.35 * plausibility_rate
            + 0.20 * valid_ratio
            + 0.10 * max(0.0, 1.0 - mean_jerk / 50.0)
        )

        metrics = {
            "overall_quality_score": round(overall_score, 4),
            "valid_trajectory_ratio": round(valid_ratio, 4),
            "total_reconstructed_trajectories": len(trajectories),
            "mean_track_duration_s": round(float(np.mean(durations)), 2),
            "mean_missing_ratio": round(float(np.mean(missing_ratios)), 4),
            "physical_plausibility_rate": round(plausibility_rate, 4),
            "mean_jerk_energy": round(mean_jerk, 3),
            "camera_stabilization_residual_px": round(stab_err, 3),
            "data_confidence_grade": (
                "EXCELLENT" if overall_score >= 0.85 else (
                    "GOOD" if overall_score >= 0.70 else (
                        "FAIR" if overall_score >= 0.50 else "POOR"
                    )
                )
            ),
        }

        logger.info(
            f"SelfConsistency Evaluation: Quality Score = {overall_score:.3f} | Grade = {metrics['data_confidence_grade']}"
        )
        return metrics
