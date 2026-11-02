from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a YOLOv8 pothole detector")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="outputs/metrics.json")
    a = ap.parse_args()

    from ultralytics import YOLO

    metrics = YOLO(a.weights).val(data=a.data, split=a.split, imgsz=a.imgsz, device=a.device,
                                  project="runs/eval", name=a.split, exist_ok=True)
    result = {
        "split": a.split,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
