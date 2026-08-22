# Model Weights Download Guide

This guide explains how pretrained weights are retrieved and configured for the Aerial Traffic Intelligence platform.

The system runs out of the box in **zero-shot inference mode** using pretrained object detector weights. It also includes an offline **synthetic validation mode** (`--input synthetic`) that executes with zero downloaded model weights.

---

## 1. Automated Model Weight Download

Use the built-in downloader script to fetch checkpoints:

```bash
# Download default YOLOv8x model
python scripts/download_models.py --model yolov8x.pt --dir weights

# Download lightweight model for CPU / edge testing
python scripts/download_models.py --model yolov8n.pt --dir weights

# Download all supported models (YOLOv8, RT-DETR)
python scripts/download_models.py --model all --dir weights
```

---

## 2. Supported Pretrained Architectures

| Model Name | Target Architecture | Parameter Count | Primary Use Case | Download Link |
| :--- | :--- | :--- | :--- | :--- |
| **`yolov8x.pt`** | YOLOv8 Extra Large | 68.2 M | High-precision aerial detection (recommended default) | [Download](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt) |
| **`yolov8l.pt`** | YOLOv8 Large | 43.7 M | Balanced aerial inference | [Download](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt) |
| **`yolov8m.pt`** | YOLOv8 Medium | 25.9 M | Real-time edge inference | [Download](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt) |
| **`yolov8s.pt`** | YOLOv8 Small | 11.2 M | High-speed processing | [Download](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt) |
| **`yolov8n.pt`** | YOLOv8 Nano | 3.2 M | CPU / embedded debugging | [Download](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) |
| **`rtdetr-l.pt`** | RT-DETR Large | 32.0 M | Real-Time Transformer detector | [Download](https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-l.pt) |
| **`rtdetr-x.pt`** | RT-DETR Extra Large | 67.0 M | Transformer-based small object detection | [Download](https://github.com/ultralytics/assets/releases/download/v8.1.0/rtdetr-x.pt) |

---

## 3. Configuring Models in the Pipeline

Update `configs/detector.yaml` or pass parameters directly via CLI:

```yaml
detector:
  backend: "yolo"
  model_name: "weights/yolov8x.pt"
  confidence_threshold: 0.25
  iou_threshold: 0.50
  sliced_inference:
    enabled: true
    slice_height: 640
    slice_width: 640
    overlap_height_ratio: 0.20
    overlap_width_ratio: 0.20
```

Via CLI:
```bash
python -m traffic_intelligence run \
  --input /path/to/drone_video.mp4 \
  --config configs/default.yaml \
  --device cuda
```
