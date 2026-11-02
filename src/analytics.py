from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEVERITY_ORDER = ["low", "medium", "high"]
DEFAULT_THRESHOLDS = {"medium": 0.005, "high": 0.02}  # bbox area / frame area
SEVERITY_WEIGHTS = {"low": 1.0, "medium": 2.5, "high": 5.0}
RATING_BANDS = [(80, "Good"), (60, "Fair"), (40, "Poor"), (0, "Critical")]
RATING_COLORS = {"Good": "#2e7d32", "Fair": "#f9a825", "Poor": "#ef6c00", "Critical": "#c62828"}

DET_COLUMNS = ["source", "sequence", "frame_idx", "timestamp_s", "class_name", "confidence",
               "x1", "y1", "x2", "y2", "img_w", "img_h"]
FRAME_COLUMNS = ["source", "sequence", "frame_idx", "timestamp_s", "n_detections", "img_w", "img_h"]


# features
def classify_severity(area_ratio: float, thresholds: dict | None = None) -> str:
    t = thresholds or DEFAULT_THRESHOLDS
    if area_ratio >= t["high"]:
        return "high"
    if area_ratio >= t["medium"]:
        return "medium"
    return "low"


def add_features(det_df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    df = det_df.copy()
    for c in ["x1", "y1", "x2", "y2", "img_w", "img_h", "confidence"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    w = (df["x2"] - df["x1"]).clip(lower=0)
    h = (df["y2"] - df["y1"]).clip(lower=0)
    frame_area = (df["img_w"] * df["img_h"]).replace(0, np.nan)
    df["area_px"] = w * h
    df["area_ratio"] = (df["area_px"] / frame_area).fillna(0.0)
    df["cx_norm"] = ((df["x1"] + df["x2"]) / 2) / df["img_w"].replace(0, np.nan)
    df["cy_norm"] = ((df["y1"] + df["y2"]) / 2) / df["img_h"].replace(0, np.nan)
    df["severity"] = [classify_severity(a, thresholds) for a in df["area_ratio"]]
    return df


# scoring
def health_rating(score: float) -> str:
    for lower, name in RATING_BANDS:
        if score >= lower:
            return name
    return "Critical"


def compute_segment_scores(
    det_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    frames_per_segment: int = 30,
    k: float = 1.0,
    weights: dict | None = None,
) -> pd.DataFrame:
    weights = weights or SEVERITY_WEIGHTS
    frames = frames_df.sort_values(["sequence", "frame_idx"]).copy()
    frames["segment"] = frames.groupby("sequence").cumcount() // max(1, frames_per_segment)

    seg = (
        frames.groupby(["sequence", "segment"])
        .agg(n_frames=("frame_idx", "size"),
             start_frame=("frame_idx", "min"),
             end_frame=("frame_idx", "max"),
             start_time_s=("timestamp_s", "min"))
        .reset_index()
    )

    extra_cols = ["n_potholes", "mean_conf", "damage"] + SEVERITY_ORDER
    if len(det_df):
        d = det_df.merge(frames[["sequence", "frame_idx", "segment"]],
                         on=["sequence", "frame_idx"], how="left")
        d["weight"] = d["severity"].map(weights)
        agg = d.groupby(["sequence", "segment"]).agg(
            n_potholes=("severity", "size"),
            mean_conf=("confidence", "mean"),
            damage=("weight", "sum"),
        ).reset_index()
        counts = (d.groupby(["sequence", "segment", "severity"]).size()
                  .unstack(fill_value=0).reindex(columns=SEVERITY_ORDER, fill_value=0).reset_index())
        seg = seg.merge(agg, on=["sequence", "segment"], how="left").merge(
            counts, on=["sequence", "segment"], how="left")
    else:
        for c in extra_cols:
            seg[c] = 0

    for c in extra_cols:
        seg[c] = seg[c].fillna(0)
    for c in ["n_potholes"] + SEVERITY_ORDER:
        seg[c] = seg[c].astype(int)

    seg["damage_per_frame"] = seg["damage"] / seg["n_frames"].clip(lower=1)
    seg["health_score"] = (100 * np.exp(-k * seg["damage_per_frame"])).round(1)
    seg["rating"] = seg["health_score"].apply(health_rating)
    return seg


def build_summary(det_df: pd.DataFrame, frames_df: pd.DataFrame, seg_df: pd.DataFrame) -> dict:
    n_frames = int(len(frames_df))
    frames_with = int((frames_df["n_detections"] > 0).sum()) if n_frames else 0
    weighted_score = float(np.average(seg_df["health_score"], weights=seg_df["n_frames"])) if len(seg_df) else 100.0
    sev_counts = {s: int((det_df["severity"] == s).sum()) if len(det_df) else 0 for s in SEVERITY_ORDER}

    worst = seg_df.sort_values("health_score").head(5)[
        ["sequence", "segment", "start_frame", "end_frame", "n_potholes", "health_score", "rating"]]
    return {
        "frames_analysed": n_frames,
        "frames_with_potholes": frames_with,
        "pct_frames_with_potholes": round(100 * frames_with / n_frames, 1) if n_frames else 0.0,
        "total_detections": int(len(det_df)),
        "avg_detections_per_frame": round(len(det_df) / n_frames, 3) if n_frames else 0.0,
        "mean_confidence": round(float(det_df["confidence"].mean()), 3) if len(det_df) else None,
        "severity_counts": sev_counts,
        "overall_health_score": round(weighted_score, 1),
        "overall_rating": health_rating(weighted_score),
        "worst_segments": worst.to_dict(orient="records"),
    }


#charts
def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_charts(det_df: pd.DataFrame, seg_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    #severity distribution
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = [int((det_df["severity"] == s).sum()) if len(det_df) else 0 for s in SEVERITY_ORDER]
    ax.bar(SEVERITY_ORDER, counts, color=["#fbc02d", "#ef6c00", "#c62828"])
    ax.set(title="Pothole severity distribution", ylabel="Detections")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    p = out_dir / "severity_distribution.png"; _save(fig, p); paths.append(p)

    #road health per segment
    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = [f"{r.sequence[:10]}#{r.segment}" for r in seg_df.itertuples()]
    ax.bar(range(len(seg_df)), seg_df["health_score"], color=[RATING_COLORS[r] for r in seg_df["rating"]])
    ax.set(title="Road health score per segment (100 = no potholes)", ylabel="Score", ylim=(0, 105))
    if len(seg_df) <= 25:
        ax.set_xticks(range(len(seg_df)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    else:
        ax.set_xlabel("Segment index")
    p = out_dir / "road_health_by_segment.png"; _save(fig, p); paths.append(p)

    #confidence histogram
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if len(det_df):
        ax.hist(det_df["confidence"], bins=15, range=(0, 1), color="#1565c0", edgecolor="white")
    ax.set(title="Detection confidence", xlabel="Confidence", ylabel="Count")
    p = out_dir / "confidence_hist.png"; _save(fig, p); paths.append(p)

    #spatial heatmap (where in the frame potholes appear)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    if len(det_df):
        hist, _, _ = np.histogram2d(det_df["cx_norm"], det_df["cy_norm"], bins=16, range=[[0, 1], [0, 1]])
        im = ax.imshow(hist.T, origin="upper", extent=[0, 1, 1, 0], cmap="magma", aspect="auto")
        fig.colorbar(im, ax=ax, label="Detections")
    ax.set(title="Where potholes appear in frame", xlabel="x (normalised)", ylabel="y (normalised)")
    p = out_dir / "spatial_heatmap.png"; _save(fig, p); paths.append(p)

    return paths


# entry point
def generate_report(
    det_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    out_dir: str | Path,
    frames_per_segment: int = 30,
    thresholds: dict | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    det = add_features(det_df, thresholds)
    seg = compute_segment_scores(det, frames_df, frames_per_segment)
    summary = build_summary(det, frames_df, seg)

    det.to_csv(out_dir / "detections_enriched.csv", index=False)
    seg.to_csv(out_dir / "segments.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    make_charts(det, seg, out_dir / "charts")
    return summary
