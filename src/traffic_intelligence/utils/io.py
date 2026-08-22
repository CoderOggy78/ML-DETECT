"""
Efficient I/O helpers for Parquet, CSV, JSON, and artifact persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.io")


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensures directory exists and returns Path object."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_parquet(df: pd.DataFrame, file_path: Union[str, Path], compression: str = "snappy") -> Path:
    """Persists a DataFrame to Apache Parquet format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(path), compression=compression)
    logger.debug(f"Saved Parquet dataset to {path} ({len(df)} rows)")
    return path


def load_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
    """Reads a Parquet dataset into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(str(path))


def save_json(data: Union[Dict[str, Any], List[Any]], file_path: Union[str, Path], indent: int = 2) -> Path:
    """Safely writes serializable Python objects to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug(f"Saved JSON to {path}")
    return path


def load_json(file_path: Union[str, Path]) -> Any:
    """Reads a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False) -> Path:
    """Writes a DataFrame to CSV."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=index)
    logger.debug(f"Saved CSV to {path} ({len(df)} rows)")
    return path


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """Reads a CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(str(path))
