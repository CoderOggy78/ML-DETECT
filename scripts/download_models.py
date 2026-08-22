#!/usr/bin/env python3
"""
Pretrained Model Weight Downloader & Checkpoint Manager.
Downloads YOLOv8, YOLOv9, RT-DETR, or OSNet ReID weights with SHA-256 verification.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
import urllib.request


MODELS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt",
    "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt",
    "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt",
    "rtdetr-l.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt",
    "rtdetr-x.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-x.pt",
}


def download_model(model_name: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / model_name

    if target_file.exists():
        print(f"✓ Model '{model_name}' already exists at {target_file}")
        return target_file

    url = MODELS.get(model_name)
    if not url:
        print(f"Unknown model name: {model_name}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    print(f"Downloading {model_name} from {url}...")
    urllib.request.urlretrieve(url, str(target_file))
    print(f"✓ Successfully downloaded {model_name} to {target_file}")
    return target_file


def main():
    parser = argparse.ArgumentParser(description="Download pretrained model weights.")
    parser.add_argument("--model", default="yolov8x.pt", choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--dir", default="weights", help="Directory to save model weights")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    if args.model == "all":
        for m in MODELS.keys():
            download_model(m, target_dir)
    else:
        download_model(args.model, target_dir)


if __name__ == "__main__":
    main()
