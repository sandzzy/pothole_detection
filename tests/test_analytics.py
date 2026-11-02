import pandas as pd
import pytest

from src.analytics import (DET_COLUMNS, FRAME_COLUMNS, add_features, classify_severity,
                           compute_segment_scores, generate_report, health_rating)


def _det(frame, x1, y1, x2, y2, conf=0.8, seq="vid"):
    return dict(source="vid.mp4", sequence=seq, frame_idx=frame, timestamp_s=frame / 10, class_name="pothole",
                confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2, img_w=640, img_h=480)


def _frames(n, seq="vid", dets=None):
    dets = dets or {}
    return pd.DataFrame([dict(source="vid.mp4", sequence=seq, frame_idx=i, timestamp_s=i / 10,
                              n_detections=dets.get(i, 0), img_w=640, img_h=480) for i in range(n)],
                        columns=FRAME_COLUMNS)


@pytest.mark.parametrize("ratio,expected", [(0.001, "low"), (0.005, "medium"), (0.019, "medium"), (0.02, "high"), (0.5, "high")])
def test_classify_severity(ratio, expected):
    assert classify_severity(ratio) == expected


def test_health_rating_bands():
    assert [health_rating(s) for s in (95, 70, 50, 10)] == ["Good", "Fair", "Poor", "Critical"]


def test_add_features_area_and_centre():
    df = add_features(pd.DataFrame([_det(0, 0, 0, 320, 240)]))
    assert df.loc[0, "area_ratio"] == pytest.approx(0.25)
    assert df.loc[0, "cx_norm"] == pytest.approx(0.25)
    assert df.loc[0, "severity"] == "high"


def test_clean_road_scores_100():
    det = add_features(pd.DataFrame(columns=DET_COLUMNS))
    seg = compute_segment_scores(det, _frames(60), frames_per_segment=30)
    assert len(seg) == 2 and (seg["health_score"] == 100).all()


def test_more_damage_lowers_score():
    det = add_features(pd.DataFrame([_det(0, 0, 0, 300, 300), _det(1, 0, 0, 300, 300)]))
    seg = compute_segment_scores(det, _frames(30, dets={0: 1, 1: 1}), frames_per_segment=30)
    assert seg.loc[0, "n_potholes"] == 2 and seg.loc[0, "high"] == 2
    assert seg.loc[0, "health_score"] < 100


def test_generate_report_writes_artifacts(tmp_path):
    det = pd.DataFrame([_det(2, 100, 200, 160, 240), _det(5, 0, 0, 400, 300, conf=0.9)])
    summary = generate_report(det, _frames(20, dets={2: 1, 5: 1}), tmp_path, frames_per_segment=10)
    assert summary["total_detections"] == 2
    for name in ("summary.json", "segments.csv", "detections_enriched.csv"):
        assert (tmp_path / name).exists()
    assert (tmp_path / "charts" / "spatial_heatmap.png").exists()


def test_generate_report_with_no_detections(tmp_path):
    summary = generate_report(pd.DataFrame(columns=DET_COLUMNS), _frames(10), tmp_path)
    assert summary["total_detections"] == 0 and summary["overall_health_score"] == 100
