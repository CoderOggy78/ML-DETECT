"""
Automated Executive Traffic Intelligence Report Generator (Self-Contained HTML).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from jinja2 import Template

from traffic_intelligence.schema import ConflictEvent, TrafficEvent, Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.report")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aerial Traffic Intelligence Report - {{ title }}</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-orange: #fb923c;
            --accent-red: #f87171;
            --accent-purple: #c084fc;
            --border-color: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 30px;
        }
        .container { max-width: 1300px; margin: 0 auto; }
        .header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid var(--border-color);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 28px; color: var(--accent-blue); margin-bottom: 6px; }
        .header p { color: var(--text-secondary); font-size: 14px; }
        .badge-grade {
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 16px;
            text-align: center;
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid var(--accent-blue);
            color: var(--accent-blue);
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s;
        }
        .kpi-card:hover { transform: translateY(-3px); }
        .kpi-card .val { font-size: 28px; font-weight: 700; color: var(--text-primary); margin: 6px 0; }
        .kpi-card .lbl { font-size: 13px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
        
        .section-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 26px;
            margin-bottom: 30px;
        }
        .section-card h2 {
            font-size: 20px;
            color: var(--accent-blue);
            margin-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 14px;
        }
        th, td {
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        th {
            background: rgba(15, 23, 42, 0.6);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
        }
        tr:hover { background: rgba(51, 65, 85, 0.4); }
        .sev-CRITICAL { color: var(--accent-red); font-weight: 600; }
        .sev-HIGH { color: var(--accent-orange); font-weight: 600; }
        .sev-MEDIUM { color: #facc15; }
        .sev-LOW { color: var(--accent-green); }
        .recommendation-list { list-style: none; margin-top: 10px; }
        .recommendation-list li {
            position: relative;
            padding-left: 26px;
            margin-bottom: 12px;
            color: var(--text-primary);
        }
        .recommendation-list li::before {
            content: "✓";
            position: absolute;
            left: 0;
            color: var(--accent-green);
            font-weight: 700;
        }
        .footer {
            text-align: center;
            color: var(--text-secondary);
            font-size: 13px;
            margin-top: 40px;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>AERIAL TRAFFIC INTELLIGENCE REPORT</h1>
                <p>Generated: {{ generated_at }} | Pipeline Version: 1.0.0 | Source: {{ source_path }}</p>
            </div>
            <div class="badge-grade">
                <div>DATA QUALITY GRADE</div>
                <div style="font-size: 20px; margin-top: 4px;">{{ quality_grade }}</div>
            </div>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="lbl">Observed Vehicles</div>
                <div class="val" style="color: var(--accent-blue);">{{ total_vehicles }}</div>
                <div class="lbl">Road Users</div>
            </div>
            <div class="kpi-card">
                <div class="lbl">Estimated Hourly Volume</div>
                <div class="val" style="color: var(--accent-green);">{{ hourly_flow }}</div>
                <div class="lbl">Veh / Hour</div>
            </div>
            <div class="kpi-card">
                <div class="lbl">85th Percentile Speed</div>
                <div class="val" style="color: var(--accent-purple);">{{ p85_speed }}</div>
                <div class="lbl">km/h</div>
            </div>
            <div class="kpi-card">
                <div class="lbl">Congestion State</div>
                <div class="val" style="color: #38bdf8;">{{ congestion_state }}</div>
                <div class="lbl">Level of Service</div>
            </div>
            <div class="kpi-card">
                <div class="lbl">Surrogate Conflicts</div>
                <div class="val" style="color: var(--accent-orange);">{{ conflict_count }}</div>
                <div class="lbl">{{ critical_conflict_count }} Critical</div>
            </div>
            <div class="kpi-card">
                <div class="lbl">Anomalies Discovered</div>
                <div class="val" style="color: var(--accent-red);">{{ anomaly_count }}</div>
                <div class="lbl">Open-Vocabulary</div>
            </div>
        </div>

        <!-- Class Breakdown -->
        <div class="section-card">
            <h2>🚗 Class-Wise Volume & Speed Breakdown</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Count</th>
                        <th>Share (%)</th>
                        <th>Mean Speed (km/h)</th>
                        <th>P85 Speed (km/h)</th>
                        <th>Max Speed (km/h)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in class_table %}
                    <tr>
                        <td><strong>{{ row.class_name }}</strong></td>
                        <td>{{ row.count }}</td>
                        <td>{{ "%.1f"|format(row.share) }}%</td>
                        <td>{{ "%.1f"|format(row.mean_speed_kmh) }}</td>
                        <td>{{ "%.1f"|format(row.p85_speed_kmh) }}</td>
                        <td>{{ "%.1f"|format(row.max_speed_kmh) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Surrogate Safety & Conflict Events -->
        <div class="section-card">
            <h2>⚠️ Surrogate Safety & Conflict Analysis (TTC / PET / DRAC)</h2>
            <p style="color: var(--text-secondary); margin-bottom: 12px;">
                Identified interactions evaluated under standard Time-To-Collision (TTC) and Deceleration Rate to Avoid Collision (DRAC) thresholds.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Event ID</th>
                        <th>Time (s)</th>
                        <th>Primary vs Secondary</th>
                        <th>Conflict Type</th>
                        <th>Severity</th>
                        <th>TTC (s)</th>
                        <th>DRAC (m/s²)</th>
                        <th>Distance (m)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for c in conflict_list[:12] %}
                    <tr>
                        <td>{{ c.event_id }}</td>
                        <td>{{ "%.2f"|format(c.timestamp_s) }}</td>
                        <td>#{{ c.primary_track_id }} ({{ c.primary_class.value }}) ↔ #{{ c.secondary_track_id }} ({{ c.secondary_class.value }})</td>
                        <td>{{ c.conflict_type }}</td>
                        <td class="sev-{{ c.severity.value }}">{{ c.severity.value }}</td>
                        <td>{{ "%.2f"|format(c.ttc_s) if c.ttc_s is not none else "N/A" }}</td>
                        <td>{{ "%.2f"|format(c.drac_mps2) if c.drac_mps2 is not none else "N/A" }}</td>
                        <td>{{ "%.1f"|format(c.separation_distance_m) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Behavioral & Anomaly Events -->
        <div class="section-card">
            <h2>🔍 Behavioral Events & Open-Vocabulary Trajectory Anomalies</h2>
            <table>
                <thead>
                    <tr>
                        <th>Event ID</th>
                        <th>Type</th>
                        <th>Track ID</th>
                        <th>Severity</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {% for e in event_list[:10] %}
                    <tr>
                        <td>{{ e.event_id }}</td>
                        <td><strong>{{ e.event_type }}</strong></td>
                        <td>{{ e.involved_track_ids|join(", ") }}</td>
                        <td class="sev-{{ e.severity.value }}">{{ e.severity.value }}</td>
                        <td>{{ e.description }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- Data Quality & Limitations -->
        <div class="section-card">
            <h2>📊 Data Quality & System Limitations</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h3 style="font-size: 15px; color: var(--accent-blue); margin-bottom: 8px;">Self-Consistency Quality Metrics</h3>
                    <ul style="color: var(--text-secondary); font-size: 14px; line-height: 1.8;">
                        <li>Overall Trajectory Quality Score: <strong>{{ quality_metrics.overall_quality_score }}</strong></li>
                        <li>Valid Trajectory Ratio: <strong>{{ "%.1f"|format(quality_metrics.valid_trajectory_ratio * 100) }}%</strong></li>
                        <li>Mean Track Continuity / Duration: <strong>{{ quality_metrics.mean_track_duration_s }}s</strong></li>
                        <li>Physical Realism Rate: <strong>{{ "%.1f"|format(quality_metrics.physical_plausibility_rate * 100) }}%</strong></li>
                        <li>Camera Motion Residual Jitter: <strong>{{ quality_metrics.camera_stabilization_residual_px }} px</strong></li>
                    </ul>
                </div>
                <div>
                    <h3 style="font-size: 15px; color: var(--accent-orange); margin-bottom: 8px;">Known Model Limitations</h3>
                    <ul style="color: var(--text-secondary); font-size: 14px; line-height: 1.8;">
                        <li>Extreme non-planar elevation (e.g. multi-layer flyovers) assumes ground-plane homography.</li>
                        <li>Zero-shot appearance Re-ID is susceptible to occlusion lasting > 15 seconds without GPS.</li>
                        <li>Tiny pedestrians (< 15 pixels) require tiled sliced inference enabled.</li>
                    </ul>
                </div>
            </div>
        </div>

        <!-- Actionable ITS Recommendations -->
        <div class="section-card">
            <h2>💡 Actionable ITS Recommendations</h2>
            <ul class="recommendation-list">
                <li>Optimize signal timing split for dominant corridor flows to alleviate observed queue persistence.</li>
                <li>Implement enhanced crosswalk signage or refuge islands at conflict hotspot coordinates.</li>
                <li>Enforce lane delineation at merge/cut-in zones to mitigate high deceleration (DRAC > 4.0 m/s²) events.</li>
                <li>Deploy automated wrong-way LED warnings at intersection egress arms where anomalous turning was detected.</li>
            </ul>
        </div>

        <div class="footer">
            Generated autonomously by the Aerial Traffic Intelligence Platform &copy; 2026. Research Grade Trajectory Intelligence.
        </div>
    </div>
</body>
</html>
"""


class HTMLReportGenerator:
    """Renders a standalone, self-contained HTML executive traffic intelligence report."""

    @staticmethod
    def generate_report(
        output_path: Path,
        source_path: str,
        trajectories: List[Trajectory],
        conflicts: List[ConflictEvent],
        events: List[TrafficEvent],
        summary_dict: Dict[str, Any],
        class_summary_df: pd.DataFrame,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        template = Template(HTML_TEMPLATE)

        # Prepare class table data
        total_veh = max(1, len(trajectories))
        class_table_rows = []
        if not class_summary_df.empty:
            for _, r in class_summary_df.iterrows():
                class_table_rows.append(
                    {
                        "class_name": r.get("class_name", "UNKNOWN"),
                        "count": int(r.get("count", 0)),
                        "share": float(r.get("count", 0)) / total_veh * 100.0,
                        "mean_speed_kmh": float(r.get("mean_speed_kmh", 0.0)),
                        "p85_speed_kmh": float(r.get("p85_speed_kmh", 0.0)),
                        "max_speed_kmh": float(r.get("max_speed_kmh", 0.0)),
                    }
                )

        meta = summary_dict.get("metadata", {})
        flow_meta = summary_dict.get("flow_and_congestion", {})
        safety_meta = summary_dict.get("surrogate_safety", {})
        qual_meta = summary_dict.get("data_quality_and_integrity", {})

        critical_conflicts = sum(1 for c in conflicts if c.severity.value == "CRITICAL")
        anomalies_count = sum(1 for e in events if "ANOMALY" in e.event_type)

        html_content = template.render(
            title=Path(source_path).name,
            source_path=source_path,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            quality_grade=qual_meta.get("data_confidence_grade", "GOOD"),
            total_vehicles=len(trajectories),
            hourly_flow=flow_meta.get("estimated_hourly_volume", 0),
            p85_speed=f"{flow_meta.get('p85_speed_kmh', 0.0):.1f}",
            congestion_state=flow_meta.get("congestion_level", "FREE_FLOW"),
            conflict_count=len(conflicts),
            critical_conflict_count=critical_conflicts,
            anomaly_count=anomalies_count,
            class_table=class_table_rows,
            conflict_list=conflicts,
            event_list=events,
            quality_metrics=qual_meta,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated standalone HTML executive report: {output_path}")
        return output_path
