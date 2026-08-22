"""
Configuration loader, parser, and validator using PyYAML and Pydantic.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.config")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Safely loads a YAML configuration file from disk."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must parse into a dictionary.")

    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override_config into base_config."""
    result = copy.deepcopy(base_config)
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_subconfigs(config: Dict[str, Any], base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Resolves referenced sub-configuration YAML files if specified in paths."""
    resolved = copy.deepcopy(config)
    
    # Check for subconfig references
    subconfig_keys = [
        ("calibration", "config_path"),
        ("traffic", "config_path"),
        ("detection", "class_mapping_path")
    ]
    
    for section, path_key in subconfig_keys:
        if section in resolved and isinstance(resolved[section], dict):
            sub_path = resolved[section].get(path_key)
            if sub_path:
                full_path = Path(sub_path)
                if not full_path.is_absolute() and base_dir:
                    full_path = base_dir / full_path
                if full_path.exists():
                    sub_dict = load_config(full_path)
                    resolved[section] = merge_configs(sub_dict, resolved[section])
                    logger.debug(f"Loaded sub-configuration: {full_path}")
                    
    return resolved
