from __future__ import annotations

import random
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_sources(source: str | Path) -> tuple[list[Path], list[Path]]:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    images = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    if not images and not videos:
        raise ValueError(f"No supported images/videos found in {source}")
    return images, videos
