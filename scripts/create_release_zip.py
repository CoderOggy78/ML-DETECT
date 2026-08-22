#!/usr/bin/env python3
"""
Packages the complete, clean repository into a deliverable ZIP archive.
Excludes virtual environments, pycache, temporary artifacts, heavy datasets, and model weights.
"""

import os
import zipfile
from pathlib import Path

EXCLUDE_DIRS = {
    ".venv", "venv", "env", "__pycache__", ".git", ".pytest_cache", ".tox", ".idea", ".vscode", "outputs", "weights"
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd", ".pt", ".onnx", ".zip", ".DS_Store"}


def create_zip(output_zip_path: str = "traffic_intelligence_release.zip", root_dir: str = "."):
    root_path = Path(root_dir).resolve()
    out_zip = Path(output_zip_path).resolve()

    print(f"Packaging repository from {root_path} into {out_zip}...")
    file_count = 0

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_path):
            # Modify dirs in-place to skip excluded folders
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in EXCLUDE_EXTS or file.startswith(".DS_Store") or file == out_zip.name:
                    continue

                rel_path = file_path.relative_to(root_path)
                zipf.write(file_path, arcname=str(rel_path))
                file_count += 1

    print(f"✓ Created ZIP archive: {out_zip} ({file_count} files, {out_zip.stat().st_size / 1024:.1f} KB)")
    return out_zip


if __name__ == "__main__":
    create_zip()
