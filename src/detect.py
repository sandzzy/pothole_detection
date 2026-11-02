from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.analytics import DET_COLUMNS, FRAME_COLUMNS, generate_report
from src.utils import collect_sources, ensure_dir


def _rows_from_result(result, source: str, sequence: str, frame_idx: int, ts: float, w: int, h: int) -> list[dict]:
    rows = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return rows
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        rows.append(dict(source=source, sequence=sequence, frame_idx=frame_idx, timestamp_s=round(ts, 3),
                         class_name=result.names[int(k)], confidence=float(c),
                         x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), img_w=w, img_h=h))
    return rows


def run_detection(weights, source, out_dir, conf=0.25, imgsz=640, device=None,
                  frame_stride=1, save_media=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ultralytics import YOLO  

    model = YOLO(str(weights))
    images, videos = collect_sources(source)
    out_dir = ensure_dir(out_dir)
    media_dir = ensure_dir(out_dir / "annotated") if save_media else None

    det_rows: list[dict] = []
    frame_rows: list[dict] = []
    predict_kw = dict(conf=conf, imgsz=imgsz, device=device, verbose=False)

    #images
    seq_name = Path(source).name if Path(source).is_dir() else "images"
    for idx, path in enumerate(tqdm(images, desc="images", disable=not images)):
        res = model.predict(str(path), **predict_kw)[0]
        h, w = res.orig_shape
        rows = _rows_from_result(res, path.name, seq_name, idx, float(idx), w, h)
        det_rows += rows
        frame_rows.append(dict(source=path.name, sequence=seq_name, frame_idx=idx, timestamp_s=float(idx),
                               n_detections=len(rows), img_w=w, img_h=h))
        if media_dir:
            cv2.imwrite(str(media_dir / path.name), res.plot())

    #videos
    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if media_dir:
            writer = cv2.VideoWriter(str(media_dir / f"{vid.stem}_annotated.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps / frame_stride, (w, h))
        n = 0
        with tqdm(total=total, desc=vid.name) as bar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if n % frame_stride == 0:
                    res = model.predict(frame, **predict_kw)[0]
                    rows = _rows_from_result(res, vid.name, vid.stem, n, n / fps, w, h)
                    det_rows += rows
                    frame_rows.append(dict(source=vid.name, sequence=vid.stem, frame_idx=n,
                                           timestamp_s=round(n / fps, 3), n_detections=len(rows), img_w=w, img_h=h))
                    if writer:
                        writer.write(res.plot())
                n += 1
                bar.update(1)
        cap.release()
        if writer:
            writer.release()

    det_df = pd.DataFrame(det_rows, columns=DET_COLUMNS)
    frames_df = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    det_df.to_csv(out_dir / "detections.csv", index=False)
    frames_df.to_csv(out_dir / "frames.csv", index=False)
    return det_df, frames_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Pothole detection + road analytics")
    ap.add_argument("--weights", required=True, help="path to trained .pt weights")
    ap.add_argument("--source", required=True, help="image, folder of images, or video")
    ap.add_argument("--out", default="outputs/run", help="output folder")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu")
    ap.add_argument("--frame-stride", type=int, default=1, help="process every Nth video frame")
    ap.add_argument("--segment-size", type=int, default=30, help="frames per road segment for scoring")
    ap.add_argument("--no-media", action="store_true", help="do not save annotated images/videos")
    a = ap.parse_args()

    det_df, frames_df = run_detection(a.weights, a.source, a.out, a.conf, a.imgsz, a.device,
                                      max(1, a.frame_stride), not a.no_media)
    summary = generate_report(det_df, frames_df, Path(a.out) / "analytics", a.segment_size)

    print("\n=== Road analytics summary ===")
    print(f"Frames analysed        : {summary['frames_analysed']}")
    print(f"Potholes detected      : {summary['total_detections']}  {summary['severity_counts']}")
    print(f"Frames with potholes   : {summary['pct_frames_with_potholes']}%")
    print(f"Overall road health    : {summary['overall_health_score']} ({summary['overall_rating']})")
    print(f"Results saved in       : {a.out}")


if __name__ == "__main__":
    main()
