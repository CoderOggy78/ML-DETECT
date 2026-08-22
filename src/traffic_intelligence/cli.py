"""
Traffic Intelligence Command-Line Interface (CLI).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from traffic_intelligence.data.dataset import DatasetAdapter
from traffic_intelligence.data.synthetic import SyntheticTrafficGenerator
from traffic_intelligence.pipeline import TrafficIntelligencePipeline
from traffic_intelligence.utils.config import load_config, merge_configs, resolve_subconfigs
from traffic_intelligence.utils.logging import get_logger, setup_logger

logger = get_logger("traffic_intelligence.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic-intelligence",
        description="Aerial Traffic Intelligence & Trajectory Discovery Platform",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. run subcommand
    run_p = subparsers.add_parser("run", help="Execute complete end-to-end aerial pipeline")
    run_p.add_argument("--input", "-i", required=True, help="Path to video file, image sequence dir, or 'synthetic'")
    run_p.add_argument("--output", "-o", default="outputs/run_default", help="Output directory for artifacts")
    run_p.add_argument("--config", "-c", default="configs/default.yaml", help="Path to YAML configuration file")
    run_p.add_argument("--telemetry", "-t", default=None, help="Optional path to telemetry file (CSV/JSON/SRT)")
    run_p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"], help="Hardware device")
    run_p.add_argument("--confidence", type=float, default=None, help="Override detector confidence threshold")
    run_p.add_argument("--max-frames", type=int, default=None, help="Process first N frames only")

    # 2. detect subcommand
    det_p = subparsers.add_parser("detect", help="Run object detection only")
    det_p.add_argument("--input", "-i", required=True, help="Path to video or image sequence")
    det_p.add_argument("--output", "-o", default="outputs/detections", help="Output directory")
    det_p.add_argument("--config", "-c", default="configs/detector.yaml", help="Path to detector config")

    # 3. track subcommand
    trk_p = subparsers.add_parser("track", help="Run multi-object tracking")
    trk_p.add_argument("--input", "-i", required=True, help="Input video or detections Parquet")
    trk_p.add_argument("--output", "-o", default="outputs/tracks", help="Output directory")
    trk_p.add_argument("--config", "-c", default="configs/tracker.yaml", help="Tracker config")

    # 4. analyze subcommand
    ana_p = subparsers.add_parser("analyze", help="Run traffic analytics, surrogate safety, and anomalies")
    ana_p.add_argument("--input", "-i", required=True, help="Path to trajectories.parquet or directory")
    ana_p.add_argument("--output", "-o", default="outputs/analytics", help="Output directory")
    ana_p.add_argument("--config", "-c", default="configs/analytics.yaml", help="Analytics config")

    # 5. report subcommand
    rep_p = subparsers.add_parser("report", help="Generate executive HTML and JSON reports")
    rep_p.add_argument("--input", "-i", required=True, help="Path to analytics output directory")
    rep_p.add_argument("--output", "-o", default="outputs/reports/traffic_report.html", help="Report output file")

    # 6. dashboard subcommand
    dash_p = subparsers.add_parser("dashboard", help="Launch interactive Streamlit dashboard")
    dash_p.add_argument("--port", type=int, default=8501, help="Dashboard port")
    dash_p.add_argument("--host", default="localhost", help="Host address")
    dash_p.add_argument("--data-dir", default="outputs", help="Directory containing run outputs")

    # 7. synthetic subcommand
    syn_p = subparsers.add_parser("synthetic", help="Generate synthetic drone traffic scenario")
    syn_p.add_argument("--output-video", default="data/raw/synthetic_traffic.mp4", help="Video output path")
    syn_p.add_argument("--output-telemetry", default="data/raw/synthetic_telemetry.csv", help="Telemetry output path")
    syn_p.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")

    return parser


def main(args_list: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    setup_logger(level=args.log_level)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run":
        config_path = Path(args.config)
        base_cfg = load_config(config_path) if config_path.exists() else {}
        base_cfg = resolve_subconfigs(base_cfg, base_dir=config_path.parent)

        # Apply CLI overrides
        overrides: dict = {"data": {"input_path": args.input, "output_dir": args.output}}
        if args.device:
            overrides.setdefault("pipeline", {})["device"] = args.device
        if args.confidence:
            overrides.setdefault("detection", {})["confidence_threshold"] = args.confidence
        if args.max_frames:
            overrides.setdefault("data", {})["max_frames"] = args.max_frames

        final_cfg = merge_configs(base_cfg, overrides)
        pipeline = TrafficIntelligencePipeline(final_cfg)
        result = pipeline.run_on_source(args.input, telemetry_path=args.telemetry)

        print("\n" + "=" * 60)
        print("✓ TRAFFIC INTELLIGENCE PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total Trajectories Reconstructed : {result['trajectories_count']}")
        print(f"Surrogate Conflicts Detected     : {result['conflicts_count']}")
        print(f"Domain Events & Anomalies Found  : {result['events_count']}")
        print(f"Data Quality Grade               : {result['quality_grade']}")
        print(f"HTML Executive Report            : {result['report_path']}")
        print(f"Artifacts Directory              : {result['output_dir']}")
        print("=" * 60)
        return 0

    elif args.command == "synthetic":
        gen = SyntheticTrafficGenerator(duration_s=args.duration)
        v_path = Path(args.output_video)
        t_path = Path(args.output_telemetry)
        gen.generate_scenario(output_video_path=v_path, output_telemetry_path=t_path)
        logger.info(f"Synthetic traffic generation completed: {v_path}")
        return 0

    elif args.command == "dashboard":
        dash_script = Path(__file__).resolve().parent.parent.parent / "dashboard" / "app.py"
        if not dash_script.exists():
            dash_script = Path("dashboard/app.py")

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dash_script),
            "--server.port",
            str(args.port),
            "--server.address",
            str(args.host),
            "--",
            "--data-dir",
            args.data_dir,
        ]
        logger.info(f"Starting Streamlit dashboard at http://{args.host}:{args.port}...")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            logger.info("Dashboard stopped.")
        return 0

    elif args.command in {"detect", "track", "analyze", "report"}:
        # Delegate sub-commands to full pipeline with modular flags
        logger.info(f"Executing modular subcommand: {args.command}")
        config_path = Path(args.config)
        base_cfg = load_config(config_path) if config_path.exists() else {}
        overrides = {"data": {"input_path": args.input, "output_dir": args.output}}
        final_cfg = merge_configs(base_cfg, overrides)
        pipeline = TrafficIntelligencePipeline(final_cfg)
        pipeline.run_on_source(args.input)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
