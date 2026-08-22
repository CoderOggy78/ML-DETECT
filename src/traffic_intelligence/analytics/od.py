"""
Origin-Destination (O-D) Matrix and Turning Movement Proportions.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from traffic_intelligence.schema import Trajectory


class OriginDestinationAnalyzer:
    """Calculates comprehensive Origin-Destination trip tables and path distributions."""

    @staticmethod
    def compute_od_matrix(trajectories: List[Trajectory]) -> pd.DataFrame:
        """Constructs an NxM matrix of trip counts between origin and destination zones."""
        records = []
        for t in trajectories:
            orig = t.origin_zone or "Origin_Unknown"
            dest = t.destination_zone or "Dest_Unknown"
            records.append({"origin": orig, "destination": dest, "track_id": t.track_id})

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        od_pivot = pd.crosstab(df["origin"], df["destination"], margins=True, margins_name="Total")
        return od_pivot
