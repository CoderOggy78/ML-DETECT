# Aerial Traffic Intelligence Platform 🚁

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Status: Production-Grade](https://img.shields.io/badge/Status-Research--Grade%20%2F%20Production-green.svg)](PROJECT_STATUS.md)
[![Hardware: CPU / CUDA / Apple Silicon MPS](https://img.shields.io/badge/Hardware-CPU%20%7C%20CUDA%20%7C%20MPS-orange.svg)](src/traffic_intelligence/utils/device.py)

A modular, research-grade Python platform that transforms raw, unannotated aerial drone footage and optional telemetry into high-fidelity physical trajectories, interaction graphs, surrogate safety metrics (TTC, PET, DRAC), behavioral events, open-vocabulary anomaly discovery, and automated executive traffic intelligence reports.

---

## 🌟 The Core Paradigm

Traditional intelligent transportation systems often suffer from **closed-vocabulary alarm generation** or treat computer vision as merely bounding-box detection. This platform implements a hierarchical research paradigm:

$$\text{DETECTIONS} \xrightarrow{\text{Association}} \text{TRACKS} \xrightarrow{\text{Transformation}} \text{TRAJECTORIES} \xrightarrow{\text{Graph}} \text{INTERACTIONS} \xrightarrow{\text{Surrogate Safety}} \text{EVENTS} \xrightarrow{\text{Unsupervised}} \text{INSIGHTS}$$

1. **Detections** tell us *WHAT* is present.
2. **Tracks** tell us *WHO* moved *WHERE*.
3. **Trajectories** tell us *HOW* they moved physically in metric space.
4. **Interactions** model *HOW* road users dynamically influenced one another.
5. **Analytics & Discovery** uncover *WHY* traffic behaves the way it does.


<img width="1389" height="791" alt="Screenshot 2026-08-22 at 1 22 28 PM" src="https://github.com/user-attachments/assets/c4ca4eeb-ae60-4407-92e0-50f7912e335c" />


---

## 🏛️ System Architecture

```mermaid
graph TD
    A[Raw Drone Video / Image Sequence] --> B[Data Ingestion & Telemetry Reader]
    B --> C[Camera Motion Compensation & Video Stabilization]
    C --> D[Small-Object Sliced / Tiled Detector]
    D --> E[Multi-Object Tracker: ByteTrack / BoT-SORT + ReID]
    E --> F[Planar Homography / Telemetry Coordinate Transform]
    F --> G[World-Coordinate Trajectory Smoothing: RTS Kalman / Savitzky-Golay]
    G --> H[Physical Kinematics: Speed, Acceleration, Jerk, Heading]
    H --> I[Unsupervised Road Model & Movement Classification]
    H --> J[Queue Detection & Level of Service Congestion Analysis]
    H --> K[Spatial-Temporal Interaction Graph: KDTree O(N log N)]
    K --> L[Surrogate Safety Metrics: TTC, PET, DRAC]
    H & L --> M[Behavioral Event Engine: Braking, Cut-Ins, Wrong-Way]
    H --> N[Open-Vocabulary Discovery: 25-Dim Latent Embeddings & Isolation Forest]
    I & J & L & M & N --> O[Self-Consistency Quality Evaluator]
    O --> P[Artifact Persistence: Parquet, CSV, JSON, HTML Report]
    P --> Q[Interactive Streamlit Dashboard]
```

---

## 🔬 Mathematical Formulations

### 1. Camera Motion Compensation (CMC)
Drone camera movement between frames $t-1$ and $t$ is compensated via RANSAC homography $H_{t-1 \to t}$:
$$p_{0} = H_{0 \leftarrow t} \, p_{t} \quad \text{where} \quad H_{0 \leftarrow t} = H_{0 \leftarrow t-1} \cdot H_{t-1 \to t}^{-1}$$

### 2. Metric World Coordinate Projection
Image pixels $(u, v)$ are mapped to metric ground plane coordinates $(X, Y)$ via planar calibration matrix $H_{\text{pix}\to\text{world}}$:
$$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} \sim H_{\text{pix}\to\text{world}} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

### 3. RTS Forward-Backward Kalman Smoothing
Eliminates detector localization jitter while preserving true high-frequency braking/turning dynamics:
$$\hat{x}_{k|N} = \hat{x}_{k|k} + C_k \left( \hat{x}_{k+1|N} - \hat{x}_{k+1|k} \right), \quad C_k = P_{k|k} F^T P_{k+1|k}^{-1}$$

### 4. Surrogate Safety Measures
- **Time-To-Collision (TTC)**:
  $$\text{TTC} = \min \left\{ t > 0 \;\middle|\; \|\mathbf{r}_A(t) - \mathbf{r}_B(t)\| \le R_{\text{collision}} \right\}$$
- **Deceleration Rate to Avoid Collision (DRAC)**:
  $$\text{DRAC} = \frac{(\Delta v_{\text{closing}})^2}{2 (\Delta d - d_{\text{buffer}})}$$
- **Post-Encroachment Time (PET)**:
  $$\text{PET} = |t_{\text{entry}, B}(\Omega) - t_{\text{exit}, A}(\Omega)|$$

### 5. Trajectory-Based Queue Dynamics
Queues are detected by spatial headway clustering along approach vectors where $v \le v_{\text{queue}}$ and persistence $\Delta t \ge \tau$:
$$L_q(t) = \|\mathbf{x}_{\text{tail}}(t) - \mathbf{x}_{\text{head}}(t)\|, \quad \text{Growth Rate} = \frac{d L_q}{dt}$$

### 6. Open-Vocabulary Anomaly Discovery
Trajectories are mapped into a 25-dimensional geometric and kinematic descriptor space $\mathbf{f} \in \mathbb{R}^{25}$ (speed variance, acceleration skewness, tortuosity, jerk energy, heading entropy) and scored via Isolation Forests:
$$s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(n)}}$$

---

## ⚡ Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Hardware: NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU (auto-detected)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/traffic-intelligence.git
cd traffic-intelligence

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Running the Pipeline

### 1. Synthetic Offline Demo (Zero Downloaded Assets)
Run the full end-to-end pipeline instantly without needing any real video:
```bash
python -m traffic_intelligence run --input synthetic --output outputs/synthetic_run
```

### 2. Processing Arbitrary Drone Video
```bash
python -m traffic_intelligence run \
  --input data/raw/highway_drone.mp4 \
  --output outputs/highway_run \
  --config configs/default.yaml \
  --device auto
```

### 3. Launching the Interactive Dashboard
```bash
python -m traffic_intelligence dashboard --data-dir outputs/highway_run --port 8501
```
Open your browser at `http://localhost:8501`.

---

## 🗂️ Output Schema & Artifacts

All run outputs are systematically persisted in `outputs/<run_name>/`:

```
outputs/run_default/
├── trajectories/
│   ├── trajectories.parquet       # Strongly typed Arrow Parquet trajectories
│   └── trajectories.csv           # Flat tabular CSV representation
├── tracks/
│   └── tracks.parquet             # Frame-by-frame tracking observations
├── events/
│   ├── events.json                # Standardized domain & anomaly events
│   └── events.csv
├── analytics/
│   ├── traffic_summary.json       # Executive KPIs and metrics dictionary
│   ├── turning_movements.csv      # Intersection turning movement matrix
│   ├── speed_statistics.csv       # Class-wise speed distributions
│   ├── queue_statistics.csv       # Temporal queue evolution log
│   └── conflicts.csv              # Surrogate safety TTC/PET/DRAC conflict log
└── reports/
    └── traffic_report.html        # Standalone executive HTML report
```

### Trajectory Parquet Schema
| Column | Type | Description |
| :--- | :--- | :--- |
| `track_id` | `int64` | Persistent road-user identity |
| `class_name` | `string` | Standard class (`CAR`, `LGV`, `HGV`, `BUS`, `TRUCK`, `MOTORCYCLE`, `PEDESTRIAN`) |
| `frame_id` | `int64` | Video frame index |
| `timestamp_s` | `float64` | Precise elapsed timestamp in seconds |
| `world_x_m`, `world_y_m` | `float64` | Metric ground plane coordinates in meters |
| `velocity_x_mps`, `velocity_y_mps` | `float64` | Metric 2D velocity vector ($m/s$) |
| `speed_kmh` | `float64` | Instantaneous scalar speed ($km/h$) |
| `acceleration_mps2` | `float64` | Longitudinal acceleration/deceleration ($m/s^2$) |
| `jerk_mps3` | `float64` | Rate of change of acceleration ($m/s^3$) |
| `heading_deg` | `float64` | Direction of motion in degrees $[0, 360)$ |
| `movement_type` | `string` | Inferred movement (`STRAIGHT`, `LEFT_TURN`, `RIGHT_TURN`, `U_TURN`, `STOPPED`) |
| `quality_score` | `float64` | Reconstruction confidence score $[0.0, 1.0]$ |

---

## 🐳 Docker Deployment

### Run via Docker Compose
```bash
docker-compose up --build
```

### Run via Standalone Docker
```bash
docker build -t traffic-intelligence:latest .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs traffic-intelligence:latest run --input data/raw/video.mp4
```

---

## 🧪 Testing

Run the comprehensive unit and integration test suite:
```bash
pytest tests/ -v
```

---

## 📜 Citation

```bibtex
@software{aerial_traffic_intelligence_2026,
  author = {Traffic Intelligence Research Team},
  title = {Aerial Traffic Intelligence: Trajectory Reconstruction, Surrogate Safety, and Open-Vocabulary Discovery Platform},
  year = {2026},
  url = {https://github.com/your-org/traffic-intelligence}
}
```
