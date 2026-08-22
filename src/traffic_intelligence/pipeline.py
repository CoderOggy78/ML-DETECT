"""
Full End-to-End Aerial Traffic Intelligence Pipeline Orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm

from traffic_intelligence.analytics.evaluation import SelfConsistencyEvaluator
from traffic_intelligence.analytics.od import OriginDestinationAnalyzer
from traffic_intelligence.analytics.spatial import SpatialHeatmapGenerator
from traffic_intelligence.analytics.statistics import ClassWiseStatisticsCalculator
from traffic_intelligence.analytics.summary import TrafficSummaryBuilder
from traffic_intelligence.analytics.temporal import TemporalTrendAggregator
from traffic_intelligence.traffic.congestion import CongestionAnalyzer
from traffic_intelligence.traffic.flow import TrafficFlowEstimator
from traffic_intelligence.data.dataset import DatasetAdapter
from traffic_intelligence.data.metadata import MetadataReader
from traffic_intelligence.data.synthetic import SyntheticTrafficGenerator
from traffic_intelligence.data.telemetry import TelemetryReader
from traffic_intelligence.data.video import VideoReader
from traffic_intelligence.detection.registry import DetectorRegistry
from traffic_intelligence.discovery.anomalies import TrajectoryAnomalyDetector
from traffic_intelligence.events.engine import EventEngine
from traffic_intelligence.geometry.calibration import CalibrationManager
from traffic_intelligence.geometry.stabilization import CameraMotionEstimator
from traffic_intelligence.interactions.conflicts import ConflictDetector
from traffic_intelligence.reporting.report import HTMLReportGenerator
from traffic_intelligence.schema import ConflictEvent, Detection, Track, TrafficEvent, Trajectory
from traffic_intelligence.tracking.track_manager import TrackManager
from traffic_intelligence.trajectories.builder import TrajectoryBuilder
from traffic_intelligence.traffic.movements import MovementClassifier
from traffic_intelligence.traffic.queues import QueueDetector
from traffic_intelligence.traffic.road_model import RoadModel
from traffic_intelligence.utils.config import load_config, merge_configs, resolve_subconfigs
from traffic_intelligence.utils.io import ensure_dir, save_csv, save_json, save_parquet
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.pipeline")


class TrafficIntelligencePipeline:
    """End-to-end aerial video processing, trajectory reconstruction, and traffic intelligence engine."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(self.config.get("data", {}).get("output_dir", "outputs/run_default"))
        ensure_dir(self.output_dir)

        # 1. Geometry & Calibration
        self.calib_manager = CalibrationManager(self.config)
        self.stabilizer = CameraMotionEstimator(
            method=self.config.get("stabilization", {}).get("method", "orb_ransac"),
            max_features=self.config.get("stabilization", {}).get("max_features", 1500),
            ransac_reproj_threshold=self.config.get("stabilization", {}).get("ransac_reproj_threshold", 3.0),
        )

        # 2. Detector & Tracker
        self.detector = DetectorRegistry.create(self.config)
        self.track_manager = TrackManager(self.config)

        # 3. Trajectory Builder
        traj_cfg = self.config.get("trajectories", {})
        self.traj_builder = TrajectoryBuilder(
            transformer=self.calib_manager.get_transformer(),
            smoothing_method=traj_cfg.get("smoothing_method", "kalman"),
            min_track_length=traj_cfg.get("min_track_length", 8),
            savgol_window=traj_cfg.get("savgol_window", 11),
            savgol_polyorder=traj_cfg.get("savgol_polyorder", 3),
            kalman_process_noise=traj_cfg.get("kalman_process_noise", 0.1),
            kalman_measurement_noise=traj_cfg.get("kalman_measurement_noise", 1.0),
            max_speed_mps=traj_cfg.get("max_plausible_speed_mps", 60.0),
            max_accel_mps2=traj_cfg.get("max_plausible_accel_mps2", 12.0),
        )

        # 4. Traffic Domain Engines
        self.queue_detector = QueueDetector()
        self.congestion_analyzer = CongestionAnalyzer()
        self.conflict_detector = ConflictDetector()
        self.event_engine = EventEngine(self.config)
        self.anomaly_detector = TrajectoryAnomalyDetector(
            contamination=self.config.get("discovery", {}).get("contamination", 0.05)
        )

    def run_on_source(
        self,
        source_path: Union[str, Path],
        telemetry_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes full pipeline on a single video or synthetic source."""
        source_str = str(source_path)
        logger.info(f"Starting Aerial Traffic Intelligence Pipeline on: {source_str}")

        # Check if running synthetic scenario mode
        if source_str.lower() in {"synthetic", "mock", "dummy"}:
            return self._run_synthetic_scenario()

        # Ingestion
        video_reader = VideoReader(
            source_path=source_path,
            frame_sample_rate=self.config.get("data", {}).get("frame_sample_rate", 1),
            max_frames=max_frames or self.config.get("data", {}).get("max_frames"),
        )
        telemetry_reader = TelemetryReader(telemetry_path)

        all_detections: List[Detection] = []
        all_tracks: List[Track] = []

        total_frames = video_reader.metadata.total_frames
        logger.info(f"Streaming video frames (Total: {total_frames})...")

        for frame_id, timestamp_s, frame_bgr in tqdm(video_reader, total=total_frames, desc="Processing Frames"):
            # Telemetry sync if available
            tel_sample = telemetry_reader.get_at_timestamp(timestamp_s)

            # Camera motion compensation
            cmc_H = self.stabilizer.estimate_motion(frame_id, frame_bgr)

            # Object Detection (Sliced or Standard)
            dets = self.detector.detect(frame_bgr, frame_id, timestamp_s)
            all_detections.extend(dets)

            # Multi-Object Tracking
            tracks = self.track_manager.update(dets, frame_bgr=frame_bgr, cmc_matrix=cmc_H)
            all_tracks.extend(tracks)

            # Feed to trajectory builder
            self.traj_builder.add_tracks(tracks)

        # Finalize Trajectories
        trajectories = self.traj_builder.build_trajectories()

        return self._process_analytics_and_artifacts(
            source_str=source_str,
            trajectories=trajectories,
            all_tracks=all_tracks,
            all_detections=all_detections,
        )

    def _run_synthetic_scenario(self) -> Dict[str, Any]:
        """Generates synthetic traffic and executes all downstream analytics."""
        logger.info("Executing synthetic traffic scenario for offline testing & validation...")
        synth_dir = self.output_dir / "synthetic_raw"
        ensure_dir(synth_dir)
        synth_video = synth_dir / "synthetic_traffic.mp4"
        synth_tel = synth_dir / "synthetic_telemetry.csv"

        gen = SyntheticTrafficGenerator(width=1280, height=720, fps=30.0, duration_s=10.0)
        trajectories, raw_dets = gen.generate_scenario(synth_video, synth_tel)

        # Build tracks from synthetic detections
        all_tracks = []
        for t in trajectories:
            for p in t.points:
                all_tracks.append(
                    Track(
                        track_id=t.track_id,
                        frame_id=p.frame_id,
                        timestamp_s=p.timestamp_s,
                        bbox_xyxy=p.bbox_xyxy,
                        confidence=0.95,
                        class_name=t.class_name,
                        state="ACTIVE",
                        world_x_m=p.world_x_m,
                        world_y_m=p.world_y_m,
                        speed_mps=p.speed_mps,
                        heading_deg=p.heading_deg,
                    )
                )

        return self._process_analytics_and_artifacts(
            source_str=str(synth_video),
            trajectories=trajectories,
            all_tracks=all_tracks,
            all_detections=[],
        )

    def _process_analytics_and_artifacts(
        self,
        source_str: str,
        trajectories: List[Trajectory],
        all_tracks: List[Track],
        all_detections: List[Detection],
    ) -> Dict[str, Any]:
        """Runs road network understanding, interactions, events, discovery, and saves all artifacts."""
        # 1. Road Network & Movement Classification
        road_model = RoadModel.auto_discover_from_trajectories(trajectories)
        trajectories = MovementClassifier.process_all_trajectories(trajectories, road_model)
        turning_matrix_df = MovementClassifier.generate_turning_movement_matrix(trajectories)
        od_matrix_df = OriginDestinationAnalyzer.compute_od_matrix(trajectories)

        # 2. Queue & Congestion Analytics
        queue_df = self.queue_detector.process_all_trajectories(trajectories)
        congestion_metrics = self.congestion_analyzer.analyze_trajectories(trajectories)
        flow_df = TrafficFlowEstimator.compute_binned_flow(trajectories, bin_size_s=60.0)
        class_summary_df = ClassWiseStatisticsCalculator.compute_summary_table(trajectories)

        # 3. Surrogate Safety & Conflict Events
        conflicts = self.conflict_detector.detect_conflicts(trajectories)
        conflicts_df = ConflictDetector.conflicts_to_dataframe(conflicts)

        # 4. Behavioral Events
        events = self.event_engine.detect_all(trajectories)

        # 5. Open-Vocabulary Anomaly Discovery
        anomaly_events, anomaly_feature_df = self.anomaly_detector.detect_anomalies(trajectories)
        events.extend(anomaly_events)
        events_df = EventEngine.events_to_dataframe(events)

        # 6. Self-Consistency Data Quality Evaluation
        quality_metrics = SelfConsistencyEvaluator.evaluate_unsupervised_consistency(
            trajectories, self.stabilizer.residual_errors
        )

        # 7. High-Level Summary Dictionary
        summary_dict = TrafficSummaryBuilder.build_summary(
            trajectories=trajectories,
            conflicts=conflicts,
            events=events,
            congestion_metrics=congestion_metrics,
            quality_metrics=quality_metrics,
        )

        # 8. Persist Artifacts to Disk
        dirs = {
            "trajectories": ensure_dir(self.output_dir / "trajectories"),
            "tracks": ensure_dir(self.output_dir / "tracks"),
            "events": ensure_dir(self.output_dir / "events"),
            "analytics": ensure_dir(self.output_dir / "analytics"),
            "reports": ensure_dir(self.output_dir / "reports"),
            "visualizations": ensure_dir(self.output_dir / "visualizations"),
        }

        # Trajectories and Tracks to Parquet & CSV
        traj_df = TrajectoryBuilder.trajectories_to_dataframe(trajectories)
        if not traj_df.empty:
            save_parquet(traj_df, dirs["trajectories"] / "trajectories.parquet")
            save_csv(traj_df, dirs["trajectories"] / "trajectories.csv")

        if all_tracks:
            tracks_df = pd.DataFrame([t.model_dump() for t in all_tracks])
            save_parquet(tracks_df, dirs["tracks"] / "tracks.parquet")

        # Events & Conflicts
        save_json([e.model_dump() for e in events], dirs["events"] / "events.json")
        if not events_df.empty:
            save_csv(events_df, dirs["events"] / "events.csv")

        if not conflicts_df.empty:
            save_csv(conflicts_df, dirs["analytics"] / "conflicts.csv")
            save_json([c.model_dump() for c in conflicts], dirs["analytics"] / "conflicts.json")

        # Analytics Tables
        save_json(summary_dict, dirs["analytics"] / "traffic_summary.json")
        if not class_summary_df.empty:
            save_csv(class_summary_df, dirs["analytics"] / "speed_statistics.csv")
        if not turning_matrix_df.empty:
            save_csv(turning_matrix_df, dirs["analytics"] / "turning_movements.csv")
        if not queue_df.empty:
            save_csv(queue_df, dirs["analytics"] / "queue_statistics.csv")
        if not flow_df.empty:
            save_csv(flow_df, dirs["analytics"] / "traffic_flow.csv")

        # HTML Executive Report
        report_path = dirs["reports"] / "traffic_report.html"
        HTMLReportGenerator.generate_report(
            output_path=report_path,
            source_path=source_str,
            trajectories=trajectories,
            conflicts=conflicts,
            events=events,
            summary_dict=summary_dict,
            class_summary_df=class_summary_df,
        )

        logger.info(f"Pipeline complete! Artifacts saved to: {self.output_dir.resolve()}")

        return {
            "summary": summary_dict,
            "trajectories_count": len(trajectories),
            "conflicts_count": len(conflicts),
            "events_count": len(events),
            "quality_grade": quality_metrics.get("data_confidence_grade"),
            "report_path": str(report_path),
            "output_dir": str(self.output_dir),
        }
