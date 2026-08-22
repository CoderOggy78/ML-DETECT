"""
Hardware acceleration detector and device manager for CPU, CUDA, and Apple Silicon MPS.
"""

from __future__ import annotations

from typing import Optional
import torch

from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.device")


class DeviceManager:
    """Detects and provides compute acceleration device handles across CUDA, MPS, and CPU."""

    @staticmethod
    def get_device(preferred_device: str = "auto") -> torch.device:
        pref = preferred_device.lower().strip()
        
        if pref == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using NVIDIA CUDA acceleration: {torch.cuda.get_device_name(0)}")
                return device
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using Apple Silicon MPS (Metal Performance Shaders) acceleration")
                return device
            else:
                device = torch.device("cpu")
                logger.info("Using CPU execution")
                return device

        elif pref == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")

        elif pref == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return torch.device("cpu")

        elif pref == "cpu":
            return torch.device("cpu")

        else:
            try:
                return torch.device(preferred_device)
            except Exception as e:
                logger.warning(f"Unrecognized device '{preferred_device}' ({e}). Defaulting to CPU.")
                return torch.device("cpu")


def get_optimal_device(preferred: str = "auto") -> torch.device:
    return DeviceManager.get_device(preferred)
