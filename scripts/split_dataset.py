from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/pothole")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    images = sorted(p for p in (src / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {src / 'images'}")

    random.Random(a.seed).shuffle(images)
    n = len(images)
    n_test, n_val = int(n * a.test), int(n * a.val)
    splits = {"test": images[:n_test], "val": images[n_test:n_test + n_val], "train": images[n_test + n_val:]}

    for name, files in splits.items():
        (dst / "images" / name).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / name).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, dst / "images" / name / img.name)
            label = src / "labels" / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, dst / "labels" / name / label.name)
            else:
                (dst / "labels" / name / f"{img.stem}.txt").touch()
        print(f"{name:5s}: {len(files)} images")


if __name__ == "__main__":
    main()
