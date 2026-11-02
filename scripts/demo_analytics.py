import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analytics import DET_COLUMNS, FRAME_COLUMNS, generate_report  # noqa: E402

rng = np.random.default_rng(0)
W, H, N_FRAMES = 1280, 720, 300

det_rows, frame_rows = [], []
for f in range(N_FRAMES):
    p = 0.6 if 100 <= f < 160 else 0.08
    n = rng.poisson(p)
    for _ in range(n):
        bw, bh = rng.uniform(30, 260), rng.uniform(20, 160)
        x1, y1 = rng.uniform(200, W - bw - 200), rng.uniform(H * 0.5, H - bh)
        det_rows.append(dict(source="demo.mp4", sequence="demo", frame_idx=f, timestamp_s=f / 30,
                             class_name="pothole", confidence=float(rng.beta(6, 2)),
                             x1=x1, y1=y1, x2=x1 + bw, y2=y1 + bh, img_w=W, img_h=H))
    frame_rows.append(dict(source="demo.mp4", sequence="demo", frame_idx=f, timestamp_s=f / 30,
                           n_detections=n, img_w=W, img_h=H))

out = Path("outputs/demo_synthetic")
summary = generate_report(pd.DataFrame(det_rows, columns=DET_COLUMNS),
                          pd.DataFrame(frame_rows, columns=FRAME_COLUMNS), out, frames_per_segment=30)
print(f"Overall health: {summary['overall_health_score']} ({summary['overall_rating']})")
print(f"Severity counts: {summary['severity_counts']}")
print(f"Report written to {out}/")
