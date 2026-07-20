"""Compare D43 detections from two Japan7 checkpoints at a fixed operating point."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402


TARGET_CLASS = 4  # D43 in Japan7


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def label_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def read_d43_labels(image_path: Path, image_shape: tuple[int, int]) -> list[dict]:
    height, width = image_shape
    labels = []
    path = label_path(image_path)
    if not path.exists():
        return labels
    for index, line in enumerate(path.read_text().splitlines()):
        values = line.split()
        if len(values) != 5 or int(values[0]) != TARGET_CLASS:
            continue
        _, cx, cy, box_w, box_h = map(float, values)
        xyxy = [
            (cx - box_w / 2) * width,
            (cy - box_h / 2) * height,
            (cx + box_w / 2) * width,
            (cy + box_h / 2) * height,
        ]
        ratio = max(box_w / max(box_h, 1e-12), box_h / max(box_w, 1e-12))
        labels.append(
            {
                "index": index,
                "xyxy": xyxy,
                "area_fraction": box_w * box_h,
                "long_short_ratio": ratio,
            }
        )
    return labels


def match(predictions: list[dict], labels: list[dict], conf: float, min_iou: float) -> tuple[dict, list[int], list[dict]]:
    used = set()
    true_positives = {}
    false_positives = []
    for prediction in sorted((p for p in predictions if p["conf"] >= conf), key=lambda p: p["conf"], reverse=True):
        choices = [(iou(prediction["xyxy"], label["xyxy"]), position) for position, label in enumerate(labels) if position not in used]
        best_iou, best_position = max(choices, default=(0.0, -1))
        if best_iou >= min_iou:
            used.add(best_position)
            true_positives[best_position] = prediction
        else:
            false_positives.append(prediction)
    return true_positives, [position for position in range(len(labels)) if position not in used], false_positives


def ap(predictions_by_image: dict[str, list[dict]], labels_by_image: dict[str, list[dict]], min_iou: float) -> float:
    predictions = sorted(
        ((prediction["conf"], image, prediction) for image, values in predictions_by_image.items() for prediction in values),
        reverse=True,
        key=lambda item: item[0],
    )
    total_labels = sum(len(values) for values in labels_by_image.values())
    if not total_labels:
        return 0.0
    used = {image: set() for image in labels_by_image}
    tp, fp = [], []
    for _, image, prediction in predictions:
        labels = labels_by_image[image]
        choices = [(iou(prediction["xyxy"], label["xyxy"]), position) for position, label in enumerate(labels) if position not in used[image]]
        best_iou, best_position = max(choices, default=(0.0, -1))
        if best_iou >= min_iou:
            used[image].add(best_position)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / total_labels
    precision = tp / np.maximum(tp + fp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.mean(np.interp(np.linspace(0, 1, 101), recall, envelope, left=1.0, right=0.0)))


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p25": None, "median": None, "p75": None}
    return {
        "n": len(values),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
    }


def run_model(weights: Path, image_dir: Path, device: int, conf: float, min_iou: float, names: dict[int, str]) -> tuple[dict, dict]:
    model = YOLO(str(weights))
    labels_by_image, target_predictions, all_predictions, outcomes = {}, {}, {}, {}
    for result in model.predict(
        source=str(image_dir),
        stream=True,
        imgsz=640,
        conf=0.001,
        iou=0.7,
        max_det=300,
        batch=32,
        device=device,
        verbose=False,
    ):
        image = str(Path(result.path))
        labels = read_d43_labels(Path(result.path), result.orig_shape)
        labels_by_image[image] = labels
        boxes = result.boxes
        predictions = []
        if boxes is not None:
            for xyxy, score, class_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                predictions.append({"xyxy": xyxy, "conf": float(score), "class_id": int(class_id)})
        target = [prediction for prediction in predictions if prediction["class_id"] == TARGET_CLASS]
        target_predictions[image] = target
        all_predictions[image] = predictions
        hits, misses, false_positives = match(target, labels, conf, min_iou)
        for position, label in enumerate(labels):
            key = f"{image}#{label['index']}"
            hit = hits.get(position)
            overlap = [p for p in predictions if p["class_id"] != TARGET_CLASS and p["conf"] >= conf and iou(p["xyxy"], label["xyxy"]) >= min_iou]
            confusion = names[max(overlap, key=lambda p: p["conf"])["class_id"]] if overlap and hit is None else None
            outcomes[key] = {
                **label,
                "image": image,
                "tp": hit is not None,
                "confidence": hit["conf"] if hit is not None else None,
                "confusion": confusion or ("no_overlap_detection" if hit is None else None),
            }
        outcomes[f"{image}#_fps"] = false_positives
    del model
    torch.cuda.empty_cache()
    return {"labels": labels_by_image, "target_predictions": target_predictions, "outcomes": outcomes}, all_predictions


def metrics(audit: dict, conf: float, min_iou: float) -> dict:
    labels = audit["labels"]
    outcomes = audit["outcomes"]
    gt = [value for key, value in outcomes.items() if not key.endswith("#_fps")]
    tp = [value for value in gt if value["tp"]]
    misses = [value for value in gt if not value["tp"]]
    fps = [value for key, values in outcomes.items() if key.endswith("#_fps") for value in values]
    return {
        "operating_point": {"confidence": conf, "iou": min_iou},
        "ground_truth": len(gt),
        "tp": len(tp),
        "fp": len(fps),
        "fn": len(misses),
        "precision": len(tp) / max(len(tp) + len(fps), 1),
        "recall": len(tp) / max(len(gt), 1),
        "ap50": ap(audit["target_predictions"], labels, 0.50),
        "ap75": ap(audit["target_predictions"], labels, 0.75),
        "miss_confusions": dict(Counter(value["confusion"] for value in misses)),
        "tp_confidence": summary([value["confidence"] for value in tp]),
        "fp_confidence": summary([value["conf"] for value in fps]),
        "tp_area_fraction": summary([value["area_fraction"] for value in tp]),
        "miss_area_fraction": summary([value["area_fraction"] for value in misses]),
        "tp_long_short_ratio": summary([value["long_short_ratio"] for value in tp]),
        "miss_long_short_ratio": summary([value["long_short_ratio"] for value in misses]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("b0_weights", type=Path)
    parser.add_argument("deep2_weights", type=Path)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "configs/japan7_remote.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()

    data = yaml.safe_load(args.data.read_text())
    root = Path(data["path"])
    image_dir = root / data["val"]
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise FileNotFoundError(f"No validation images found under {image_dir}")
    names = {int(key): value for key, value in data["names"].items()}
    args.output.mkdir(parents=True, exist_ok=True)

    b0, _ = run_model(args.b0_weights, image_dir, args.device, args.conf, args.iou, names)
    deep2, _ = run_model(args.deep2_weights, image_dir, args.device, args.conf, args.iou, names)
    b0_metrics, deep2_metrics = metrics(b0, args.conf, args.iou), metrics(deep2, args.conf, args.iou)

    b0_only, deep2_only, rows = [], [], []
    for key, b0_outcome in b0["outcomes"].items():
        if key.endswith("#_fps"):
            continue
        deep2_outcome = deep2["outcomes"][key]
        row = {
            "image": b0_outcome["image"],
            "gt_index": b0_outcome["index"],
            "gt_xyxy": json.dumps(b0_outcome["xyxy"]),
            "area_fraction": b0_outcome["area_fraction"],
            "long_short_ratio": b0_outcome["long_short_ratio"],
            "b0_tp": b0_outcome["tp"],
            "b0_confidence": b0_outcome["confidence"],
            "b0_miss_reason": b0_outcome["confusion"],
            "deep2_tp": deep2_outcome["tp"],
            "deep2_confidence": deep2_outcome["confidence"],
            "deep2_miss_reason": deep2_outcome["confusion"],
        }
        rows.append(row)
        if b0_outcome["tp"] and not deep2_outcome["tp"]:
            b0_only.append(row)
        if deep2_outcome["tp"] and not b0_outcome["tp"]:
            deep2_only.append(row)

    payload = {
        "models": {"b0": str(args.b0_weights), "deep2": str(args.deep2_weights)},
        "dataset": str(args.data),
        "d43": {"b0": b0_metrics, "deep2": deep2_metrics},
        "paired": {"b0_only_tp": len(b0_only), "deep2_only_tp": len(deep2_only)},
        "limitation": "This fixed-point audit uses conf=0.25 and IoU=0.50. It cannot recover per-class epoch-20/21 metrics because those checkpoints were not saved.",
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for name, values in (("per_gt.csv", rows), ("b0_only_tp.csv", b0_only), ("deep2_only_tp.csv", deep2_only)):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(values)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
