from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a YOLOv8 pothole detector")
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m.pt (pretrained COCO weights)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for first GPU, cpu for CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", default="runs/pothole")
    ap.add_argument("--name", default="yolov8n")
    a = ap.parse_args()

    from ultralytics import YOLO

    set_seed(a.seed)
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                workers=a.workers, patience=a.patience, seed=a.seed,
                project=str(Path(a.project).resolve()), name=a.name, exist_ok=True,
                hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=5.0, translate=0.1,
                scale=0.4, fliplr=0.5, mosaic=1.0)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nTraining finished. Best weights: {best}")


if __name__ == "__main__":
    main()
