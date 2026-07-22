#!/usr/bin/env python3
"""Measure LiteRG region-prior behavior and D00/D10 failure modes on Japan7."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402

ULTRALYTICS_ROOT = Path(ultralytics.__file__).resolve().parent
assert ULTRALYTICS_ROOT.is_relative_to(ROOT), f"Imported ultralytics outside repository: {ULTRALYTICS_ROOT}"

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.utils.loss import build_region_targets  # noqa: E402


TARGET_CLASSES = {0: "D00", 1: "D10"}


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def schedule_rows(epochs: int, region_gain: float, region_floor: float) -> list[dict]:
    rows = []
    lambda_start = region_gain * 0.8
    lambda_end = region_gain * region_floor
    for epoch in range(1, epochs + 1):
        updates_before_epoch = epoch - 1
        o2m = max(1 - updates_before_epoch / max(epochs - 1, 1), 0) * (0.8 - 0.1) + 0.1
        rows.append(
            {
                "epoch": epoch,
                "updates_before_epoch": updates_before_epoch,
                "o2m": o2m,
                "o2o": 1.0 - o2m,
                "effective_lambda_current": region_gain * max(o2m, region_floor),
                "effective_lambda_fixed": region_gain,
                "effective_lambda_endpoint_normalized": lambda_end
                + (lambda_start - lambda_end) * ((o2m - 0.1) / (0.8 - 0.1)),
            }
        )
    return rows


def xywh_to_xyxy(box, width: int, height: int) -> list[float]:
    cx, cy, bw, bh = map(float, box)
    return [(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height]


def heatmap_panel(values: np.ndarray, size: tuple[int, int], title: str) -> np.ndarray:
    width, height = size
    resized = cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)
    normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    panel = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return panel


def render_panel(
    image: np.ndarray,
    ground_truth: list[dict],
    predictions: list[dict],
    region: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    names: dict[int, str],
    title: str,
) -> np.ndarray:
    height, width = image.shape[:2]
    raw = image.copy()
    cv2.putText(raw, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 255), 2)
    gt_panel = image.copy()
    for item in ground_truth:
        x1, y1, x2, y2 = map(int, item["xyxy"])
        cv2.rectangle(gt_panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(gt_panel, names[item["class_id"]], (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(gt_panel, "GT", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    pred_panel = image.copy()
    for item in predictions:
        x1, y1, x2, y2 = map(int, item["xyxy"])
        cv2.rectangle(pred_panel, (x1, y1), (x2, y2), (255, 160, 0), 2)
        label = f"{names[item['class_id']]} {item['confidence']:.2f}"
        cv2.putText(pred_panel, label, (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
    cv2.putText(pred_panel, "Prediction", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 160, 0), 2)
    return np.concatenate(
        (
            np.concatenate((raw, gt_panel, heatmap_panel(region, (width, height), "Region probability")), axis=1),
            np.concatenate((heatmap_panel(p3, (width, height), "P3 DRG response"), heatmap_panel(p4, (width, height), "P4 DRG response"), pred_panel), axis=1),
        ),
        axis=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--max-batches", type=int, default=0, help="0 analyzes the complete validation split.")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.data.is_file():
        raise FileNotFoundError(f"Missing checkpoint or data YAML: {args.checkpoint}, {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)
    visualization_dir = args.output / "region_visualizations"
    visualization_dir.mkdir(exist_ok=True)

    yolo = YOLO(str(args.checkpoint))
    core = yolo.model
    if getattr(core, "lite_rg", None) is None:
        raise RuntimeError("Loaded checkpoint does not contain LiteRG.")
    trainer = DetectionTrainer(
        overrides={
            "model": str(args.checkpoint),
            "data": str(args.data),
            "imgsz": 640,
            "batch": args.batch,
            "device": args.device,
            "workers": 0,
            "epochs": args.epochs,
            "plots": False,
        }
    )
    trainer.model = core.to(trainer.device)
    trainer.set_model_attributes()
    trainer.stride = max(int(core.stride.max()), 32)
    loader = trainer.get_dataloader(trainer.data["val"], args.batch, -1, "val")
    names = {int(key): value for key, value in trainer.data["names"].items()}
    config = core.yaml["lite_rg"]
    schedule = schedule_rows(args.epochs, float(config["region_gain"]), float(config["region_floor"]))
    write_csv(args.output / "progressive_region_schedule.csv", schedule)
    effective_lambda = schedule[args.epoch - 1]["effective_lambda_current"]

    responses = {}
    hooks = [
        core.lite_rg.drg3.register_forward_hook(
            lambda _module, _inputs, output: responses.__setitem__("p3", output.detach().float().abs().mean(1, keepdim=True))
        ),
        core.lite_rg.drg4.register_forward_hook(
            lambda _module, _inputs, output: responses.__setitem__("p4", output.detach().float().abs().mean(1, keepdim=True))
        ),
        core.lite_rg.drg3.horizontal.register_forward_hook(
            lambda _module, _inputs, output: responses.__setitem__(
                "drg3_horizontal", output.detach().float().abs().mean((1, 2, 3))
            )
        ),
        core.lite_rg.drg3.vertical.register_forward_hook(
            lambda _module, _inputs, output: responses.__setitem__(
                "drg3_vertical", output.detach().float().abs().mean((1, 2, 3))
            )
        ),
        core.lite_rg.drg4.horizontal.register_forward_hook(
            lambda _module, _inputs, output: responses.__setitem__(
                "drg4_horizontal", output.detach().float().abs().mean((1, 2, 3))
            )
        ),
        core.lite_rg.drg4.vertical.register_forward_hook(
            lambda _module, _inputs, output: responses.__setitem__(
                "drg4_vertical", output.detach().float().abs().mean((1, 2, 3))
            )
        ),
    ]
    core.eval()
    region_rows, geometry_rows, failure_rows = [], [], []
    class_geometry = defaultdict(lambda: {"areas": [], "aspect": [], "instances_per_image": []})
    failure_counts = {0: Counter(), 1: Counter()}
    false_positive_counts = {0: Counter(), 1: Counter()}
    confusion_counts = {0: Counter(), 1: Counter()}
    saved_categories = set()
    saved_failure_panels = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            batch = trainer.preprocess_batch(batch)
            output = core(batch["img"])
            decoded, raw = output
            logits = raw["region_logits"].float()
            probabilities = logits.sigmoid()
            targets = build_region_targets(
                batch,
                batch_size=logits.shape[0],
                height=logits.shape[-2],
                width=logits.shape[-1],
                device=logits.device,
                sigma_scale=float(config["sigma_scale"]),
                target_mode=str(config["target_mode"]),
            )
            bce_maps = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            intersections = (probabilities * targets).sum((1, 2, 3))
            denominators = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
            dice_losses = 1 - (2 * intersections + 1) / (denominators + 1)

            for image_index in range(batch["img"].shape[0]):
                image_path = str(batch["im_file"][image_index])
                image = (batch["img"][image_index].detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                height, width = image.shape[:2]
                image_mask = batch["batch_idx"].view(-1).long() == image_index
                image_boxes = batch["bboxes"][image_mask].detach().cpu().tolist()
                image_classes = batch["cls"][image_mask].view(-1).long().detach().cpu().tolist()
                ground_truth = []
                per_image_counts = Counter(image_classes)
                for label_index, (class_id, box) in enumerate(zip(image_classes, image_boxes)):
                    xyxy = xywh_to_xyxy(box, width, height)
                    area = float(box[2] * box[3])
                    signed_aspect = float(box[2] / max(box[3], 1e-12))
                    long_short = max(signed_aspect, 1 / max(signed_aspect, 1e-12))
                    ground_truth.append(
                        {
                            "class_id": class_id,
                            "xyxy": xyxy,
                            "label_index": label_index,
                            "area_fraction": area,
                            "signed_width_height_ratio": signed_aspect,
                            "long_short_ratio": long_short,
                        }
                    )
                    geometry_rows.append(
                        {
                            "image": image_path,
                            "class_id": class_id,
                            "class_name": names[class_id],
                            "area_fraction": area,
                            "signed_width_height_ratio": signed_aspect,
                            "long_short_ratio": long_short,
                            "instances_of_class_in_image": per_image_counts[class_id],
                        }
                    )
                    class_geometry[class_id]["areas"].append(area)
                    class_geometry[class_id]["aspect"].append(signed_aspect)
                for class_id in set(image_classes):
                    class_geometry[class_id]["instances_per_image"].append(per_image_counts[class_id])

                predictions = []
                for row in decoded[image_index].detach().float().cpu().tolist():
                    if row[4] < args.conf:
                        continue
                    predictions.append({"xyxy": row[:4], "confidence": float(row[4]), "class_id": int(row[5])})

                probability = probabilities[image_index, 0]
                target = targets[image_index, 0]
                binary_prediction = probability > 0.5
                binary_target = target > 0.5
                intersection = (binary_prediction & binary_target).sum().float()
                union = (binary_prediction | binary_target).sum().float()
                foreground = target > 0
                background = ~foreground
                raw_bce = float(bce_maps[image_index].mean().cpu())
                raw_dice = float(dice_losses[image_index].cpu())
                region_rows.append(
                    {
                        "image": image_path,
                        "raw_bce": raw_bce,
                        "raw_dice": raw_dice,
                        "raw_bce_plus_dice": raw_bce + raw_dice,
                        "effective_lambda": effective_lambda,
                        "weighted_region_loss": (raw_bce + raw_dice) * effective_lambda,
                        "logit_mean": float(logits[image_index].mean().cpu()),
                        "logit_std": float(logits[image_index].std(unbiased=False).cpu()),
                        "logit_min": float(logits[image_index].min().cpu()),
                        "logit_max": float(logits[image_index].max().cpu()),
                        "probability_mean": float(probability.mean().cpu()),
                        "probability_over_0_5": float(binary_prediction.float().mean().cpu()),
                        "gt_positive_over_0": float(foreground.float().mean().cpu()),
                        "gt_positive_over_0_5": float(binary_target.float().mean().cpu()),
                        "binary_iou_0_5": float((intersection / union.clamp_min(1)).cpu()),
                        "soft_dice": 1.0 - raw_dice,
                        "foreground_response": float(probability[foreground].mean().cpu()) if foreground.any() else None,
                        "background_response": float(probability[background].mean().cpu()) if background.any() else None,
                        "drg3_horizontal_response": float(responses["drg3_horizontal"][image_index].cpu()),
                        "drg3_vertical_response": float(responses["drg3_vertical"][image_index].cpu()),
                        "drg4_horizontal_response": float(responses["drg4_horizontal"][image_index].cpu()),
                        "drg4_vertical_response": float(responses["drg4_vertical"][image_index].cpu()),
                        "contains_D00": int(0 in image_classes),
                        "contains_D10": int(1 in image_classes),
                    }
                )

                image_failures = []
                for gt in ground_truth:
                    class_id = gt["class_id"]
                    if class_id not in TARGET_CLASSES:
                        continue
                    same = [(box_iou(gt["xyxy"], pred["xyxy"]), pred) for pred in predictions if pred["class_id"] == class_id]
                    other = [(box_iou(gt["xyxy"], pred["xyxy"]), pred) for pred in predictions if pred["class_id"] != class_id]
                    same_iou, same_pred = max(same, default=(0.0, None), key=lambda item: item[0])
                    other_iou, other_pred = max(other, default=(0.0, None), key=lambda item: item[0])
                    if same_iou >= args.match_iou:
                        reason = "true_positive"
                    elif other_iou >= args.match_iou:
                        reason = "classification_confusion"
                        confusion_counts[class_id][names[other_pred["class_id"]]] += 1
                    elif same_iou >= 0.10:
                        reason = "localization_error"
                    else:
                        reason = "miss"
                    failure_counts[class_id][reason] += 1
                    if reason != "true_positive":
                        row = {
                            "case_type": "ground_truth_failure",
                            "image": image_path,
                            "target_class": names[class_id],
                            "label_index": gt["label_index"],
                            "reason": reason,
                            "best_same_class_iou": same_iou,
                            "best_same_class_confidence": same_pred["confidence"] if same_pred else None,
                            "confused_with": names[other_pred["class_id"]] if reason == "classification_confusion" else None,
                            "best_other_iou": other_iou,
                            "area_fraction": gt["area_fraction"],
                            "signed_width_height_ratio": gt["signed_width_height_ratio"],
                            "long_short_ratio": gt["long_short_ratio"],
                        }
                        failure_rows.append(row)
                        image_failures.append(row)

                for prediction_index, prediction in enumerate(predictions):
                    class_id = prediction["class_id"]
                    if class_id not in TARGET_CLASSES:
                        continue
                    same_gt = [
                        box_iou(prediction["xyxy"], gt["xyxy"])
                        for gt in ground_truth
                        if gt["class_id"] == class_id
                    ]
                    best_same_iou = max(same_gt, default=0.0)
                    if best_same_iou >= args.match_iou:
                        continue
                    other_gt = [
                        (box_iou(prediction["xyxy"], gt["xyxy"]), gt)
                        for gt in ground_truth
                        if gt["class_id"] != class_id
                    ]
                    best_other_iou, best_other_gt = max(other_gt, default=(0.0, None), key=lambda item: item[0])
                    if best_same_iou >= 0.10:
                        reason = "localization_false_positive"
                    elif best_other_iou >= args.match_iou:
                        reason = "class_confusion_false_positive"
                    else:
                        reason = "background_false_positive"
                    false_positive_counts[class_id][reason] += 1
                    row = {
                        "case_type": "false_positive",
                        "image": image_path,
                        "target_class": names[class_id],
                        "label_index": prediction_index,
                        "reason": reason,
                        "best_same_class_iou": best_same_iou,
                        "best_same_class_confidence": prediction["confidence"],
                        "confused_with": names[best_other_gt["class_id"]] if best_other_gt else None,
                        "best_other_iou": best_other_iou,
                        "area_fraction": None,
                        "signed_width_height_ratio": None,
                        "long_short_ratio": None,
                    }
                    failure_rows.append(row)
                    image_failures.append(row)

                for class_id, class_name in names.items():
                    if class_name in saved_categories or not any(gt["class_id"] == class_id for gt in ground_truth):
                        continue
                    panel = render_panel(
                        image,
                        ground_truth,
                        predictions,
                        probability.cpu().numpy(),
                        responses["p3"][image_index, 0].cpu().numpy(),
                        responses["p4"][image_index, 0].cpu().numpy(),
                        names,
                        f"representative_{class_name}",
                    )
                    cv2.imwrite(str(visualization_dir / f"class_{class_name}.jpg"), panel)
                    saved_categories.add(class_name)
                if image_failures and saved_failure_panels < 20:
                    reason = image_failures[0]["reason"]
                    target = image_failures[0]["target_class"]
                    panel = render_panel(
                        image,
                        ground_truth,
                        predictions,
                        probability.cpu().numpy(),
                        responses["p3"][image_index, 0].cpu().numpy(),
                        responses["p4"][image_index, 0].cpu().numpy(),
                        names,
                        f"failure_{target}_{reason}",
                    )
                    cv2.imwrite(str(visualization_dir / f"failure_{saved_failure_panels + 1:02d}_{target}_{reason}.jpg"), panel)
                    saved_failure_panels += 1

    for hook in hooks:
        hook.remove()
    write_csv(args.output / "region_prior_per_image.csv", region_rows)
    write_csv(args.output / "japan7_geometry.csv", geometry_rows)
    write_csv(args.output / "d00_d10_failures.csv", failure_rows)

    numeric_region_keys = [key for key, value in region_rows[0].items() if key != "image" and isinstance(value, (int, float))]
    region_summary = {key: quantiles([float(row[key]) for row in region_rows if row[key] is not None]) for key in numeric_region_keys}
    directional_by_target = {
        class_name: {
            key: quantiles([float(row[key]) for row in region_rows if row[f"contains_{class_name}"]])
            for key in (
                "drg3_horizontal_response",
                "drg3_vertical_response",
                "drg4_horizontal_response",
                "drg4_vertical_response",
            )
        }
        for class_name in ("D00", "D10")
    }
    class_summary = {}
    for class_id, class_name in names.items():
        values = class_geometry[class_id]
        class_summary[class_name] = {
            "instances": len(values["areas"]),
            "area_fraction": quantiles(values["areas"]),
            "signed_width_height_ratio": quantiles(values["aspect"]),
            "instances_per_positive_image": quantiles(values["instances_per_image"]),
        }
    target_failure_summary = {}
    for class_id, class_name in TARGET_CLASSES.items():
        counts = failure_counts[class_id]
        total = sum(counts.values())
        target_failure_summary[class_name] = {
            "counts": dict(counts),
            "proportions": {key: value / max(total, 1) for key, value in counts.items()},
            "confusions": dict(confusion_counts[class_id]),
            "false_positive_counts": dict(false_positive_counts[class_id]),
            "false_positives": sum(false_positive_counts[class_id].values()),
        }
    summary = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.data),
        "images": len(region_rows),
        "region_config": config,
        "effective_lambda_at_requested_epoch": effective_lambda,
        "region_prior": region_summary,
        "directional_response_by_target_image": directional_by_target,
        "class_geometry": class_summary,
        "d00_d10_operating_point": {"confidence": args.conf, "iou": args.match_iou},
        "d00_d10_failures": target_failure_summary,
        "failure_rows": len(failure_rows),
        "failure_visualizations": saved_failure_panels,
        "semantic_confuser_limitation": (
            "Japan7 has no shadow/lane-line/patch confuser labels. The script saves non-cherry-picked false/missed cases; "
            "semantic confuser tags require manual review and are not inferred from pixels."
        ),
    }
    (args.output / "region_prior_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
