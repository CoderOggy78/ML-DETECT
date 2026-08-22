#!/usr/bin/env python3
"""
Generate realistic synthetic drone traffic video, telemetry, and ground truth trajectories.
"""

from pathlib import Path
from traffic_intelligence.data.synthetic import SyntheticTrafficGenerator


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "synthetic_traffic.mp4"
    tel_path = out_dir / "synthetic_telemetry.csv"

    print(f"Generating synthetic traffic scenario (10.0s @ 30fps)...")
    generator = SyntheticTrafficGenerator(width=1280, height=720, fps=30.0, duration_s=10.0)
    trajs, dets = generator.generate_scenario(output_video_path=video_path, output_telemetry_path=tel_path)
    print(f"✓ Created video: {video_path}")
    print(f"✓ Created telemetry: {tel_path}")
    print(f"✓ Generated {len(trajs)} distinct vehicle & VRU trajectories")


if __name__ == "__main__":
    main()
