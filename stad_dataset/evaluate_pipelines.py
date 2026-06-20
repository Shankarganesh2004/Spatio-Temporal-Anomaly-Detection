"""
Evaluate baseline vs CLIP+Chroma enhanced anomaly pipelines on held-out scenes.

Held-out labels are derived from temporal hotspot masks produced by the
change-detection module:
- Positive (anomaly): track center lands in hotspot mask
- Negative (normal): otherwise
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from models.anomaly import RiskScorer
from models.detector import YOLOv8OBBDetector
from models.semantic_memory import ClipChromaMemory
from models.tracker import ByteTracker
from models.transformer import TemporalAnalyzer
from utils.change_detection import compute_change_maps
from utils.config import get_config
from utils.real_data import load_highres_scenes
from utils.scene_manager import load_cached_scenes


REGIONS = {
    "san-diego-airport": {
        "min_lon": -117.205, "max_lon": -117.180,
        "min_lat": 32.728, "max_lat": 32.740,
        "center_lat": 32.734, "center_lon": -117.1925,
        "name": "San Diego Intl Airport",
    },
    "long-beach-port": {
        "min_lon": -118.230, "max_lon": -118.205,
        "min_lat": 33.742, "max_lat": 33.758,
        "center_lat": 33.750, "center_lon": -118.2175,
        "name": "Port of Long Beach",
    },
    "san-diego-naval": {
        "min_lon": -117.128, "max_lon": -117.112,
        "min_lat": 32.683, "max_lat": 32.696,
        "center_lat": 32.6895, "center_lon": -117.120,
        "name": "San Diego Naval Base",
    },
}


def _metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    acc = (tp + tn) / max(len(y_true), 1)

    return {
        "samples": len(y_true),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "accuracy": float(acc),
    }


def _track_label_from_hotspot(track: Dict, hotspot_mask: np.ndarray) -> int:
    h, w = hotspot_mask.shape[:2]
    x = int(np.clip(track.get("cx", 0), 0, w - 1))
    y = int(np.clip(track.get("cy", 0), 0, h - 1))
    return int(bool(hotspot_mask[y, x]))


def _run_comparison_on_sequence(scenes: List[Dict], config) -> Tuple[List[int], List[int], List[int]]:
    detector = YOLOv8OBBDetector(config.detector)
    tracker = ByteTracker(config.tracker)
    temporal = TemporalAnalyzer(config.transformer, device="cpu")
    risk = RiskScorer(config.anomaly)
    semantic_memory = ClipChromaMemory(config.semantic)

    change_maps = compute_change_maps(scenes)

    y_true: List[int] = []
    y_pred_baseline: List[int] = []
    y_pred_enhanced: List[int] = []

    for i, scene in enumerate(scenes):
        frame = scene["optical"]
        dets = detector.detect(frame)
        track_objs = tracker.update([d.to_dict() for d in dets])
        current_tracks = [t.to_dict() for t in track_objs]

        if i == 0:
            if semantic_memory.enabled:
                semantic_memory.ingest_tracks(frame, current_tracks, frame_id=i)
            continue

        trajectories = tracker.get_all_trajectories()
        transformer_scores = temporal.analyze(trajectories)

        baseline = risk.compute_risk_scores(
            trajectories,
            transformer_scores,
            current_tracks,
        )

        semantic_scores = None
        if semantic_memory.enabled:
            semantic_scores = semantic_memory.score_tracks(frame, current_tracks)

        enhanced = risk.compute_risk_scores(
            trajectories,
            transformer_scores,
            current_tracks,
            semantic_scores=semantic_scores,
        )

        hotspot_mask = change_maps[i - 1]["hotspot_mask"]
        threshold = config.anomaly.risk_threshold_medium

        for tr in current_tracks:
            tid = tr.get("track_id")
            if tid not in baseline or tid not in enhanced:
                continue
            y_true.append(_track_label_from_hotspot(tr, hotspot_mask))
            y_pred_baseline.append(int(baseline[tid]["risk_score"] >= threshold))
            y_pred_enhanced.append(int(enhanced[tid]["risk_score"] >= threshold))

        if semantic_memory.enabled:
            semantic_memory.ingest_tracks(frame, current_tracks, frame_id=i)

    return y_true, y_pred_baseline, y_pred_enhanced


def evaluate(region_key: str, max_scenes: int, holdout_ratio: float, semantic_weight: float) -> Dict:
    if region_key not in REGIONS:
        raise ValueError(f"Unknown region '{region_key}'. Available: {list(REGIONS.keys())}")

    region = REGIONS[region_key]

    config = get_config()
    config.image_size = (2048, 2048)
    config.num_demo_frames = max_scenes
    config.anomaly.semantic_weight = semantic_weight
    config.semantic.enabled = True

    scenes = load_highres_scenes(region=region, image_size=config.image_size, max_scenes=max_scenes)
    if not scenes:
        scenes = load_cached_scenes(region=region, image_size=config.image_size, max_scenes=max_scenes)
    if len(scenes) < 4:
        raise RuntimeError("Not enough scenes for held-out evaluation (need >= 4).")

    split = max(2, int(len(scenes) * (1.0 - holdout_ratio)))
    holdout = scenes[split:]

    y_true, y_base, y_enh = _run_comparison_on_sequence(holdout, config)

    out = {
        "region": region["name"],
        "total_scenes": len(scenes),
        "holdout_scenes": len(holdout),
        "holdout_date_range": [holdout[0]["date"], holdout[-1]["date"]],
        "risk_threshold": config.anomaly.risk_threshold_medium,
        "semantic_weight": semantic_weight,
        "baseline": _metrics(y_true, y_base),
        "enhanced_clip_chroma": _metrics(y_true, y_enh),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs CLIP+Chroma anomaly pipelines")
    parser.add_argument("--region", default="san-diego-airport", choices=list(REGIONS.keys()))
    parser.add_argument("--max-scenes", type=int, default=12)
    parser.add_argument("--holdout-ratio", type=float, default=0.3)
    parser.add_argument("--semantic-weight", type=float, default=0.25)
    parser.add_argument("--output", default="outputs/evaluation_compare.json")
    args = parser.parse_args()

    result = evaluate(
        region_key=args.region,
        max_scenes=args.max_scenes,
        holdout_ratio=args.holdout_ratio,
        semantic_weight=args.semantic_weight,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved evaluation report to {out_path}")


if __name__ == "__main__":
    main()
