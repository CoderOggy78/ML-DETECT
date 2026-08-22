"""
Traffic Summary Builder: Compiles executive KPI indicators and analytical summary dictionaries.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from traffic_intelligence.schema import ConflictEvent, TrafficEvent, Trajectory


class TrafficSummaryBuilder:
    """Consolidates trajectory intelligence into a high-level executive summary dictionary."""

    @staticmethod
    def build_summary(
        trajectories: List[Trajectory],
        conflicts: List[ConflictEvent],
        events: List[TrafficEvent],
        congestion_metrics: Dict[str, Any],
        quality_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        class_counts = {}
        for t in trajectories:
            c = t.class_name.value
            class_counts[c] = class_counts.get(c, 0) + 1

        conflict_by_type = {}
        conflict_by_severity = {}
        for c in conflicts:
            conflict_by_type[c.conflict_type] = conflict_by_type.get(c.conflict_type, 0) + 1
            conflict_by_severity[c.severity.value] = conflict_by_severity.get(c.severity.value, 0) + 1

        event_by_type = {}
        for e in events:
            event_by_type[e.event_type] = event_by_type.get(e.event_type, 0) + 1

        duration_s = (
            max(t.end_timestamp_s for t in trajectories) - min(t.start_timestamp_s for t in trajectories)
            if trajectories
            else 0.0
        )

        return {
            "metadata": {
                "total_vehicles_observed": len(trajectories),
                "total_video_duration_s": round(duration_s, 2),
                "class_breakdown": class_counts,
            },
            "flow_and_congestion": {
                "overall_mean_speed_kmh": round(congestion_metrics.get("overall_mean_speed_kmh", 0.0), 2),
                "p85_speed_kmh": round(congestion_metrics.get("p85_speed_kmh", 0.0), 2),
                "congestion_level": congestion_metrics.get("congestion_level", "FREE_FLOW"),
                "estimated_hourly_volume": round(len(trajectories) * (3600.0 / max(1.0, duration_s)), 1),
            },
            "surrogate_safety": {
                "total_conflicts": len(conflicts),
                "conflicts_by_type": conflict_by_type,
                "conflicts_by_severity": conflict_by_severity,
                "critical_conflict_rate_per_hour": round(
                    conflict_by_severity.get("CRITICAL", 0) * (3600.0 / max(1.0, duration_s)), 2
                ),
            },
            "behavioral_events": {
                "total_events_detected": len(events),
                "events_by_type": event_by_type,
            },
            "data_quality_and_integrity": quality_metrics,
        }
