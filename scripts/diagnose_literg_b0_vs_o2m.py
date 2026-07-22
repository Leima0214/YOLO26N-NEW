#!/usr/bin/env python3
"""Compare B0 O2O with B5 O2M and audit the learned LiteRG prior without training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402

ULTRALYTICS_ROOT = Path(ultralytics.__file__).resolve().parent
assert ULTRALYTICS_ROOT.is_relative_to(ROOT), f"Imported ultralytics outside repository: {ULTRALYTICS_ROOT}"

from diagnose_literg_100e import matched_gt  # noqa: E402
from eval_literg_full import metrics_payload, use_one_to_many  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.utils.loss import build_region_targets  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write reviewed rows without overwriting an earlier diagnostic."""
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(checkpoint: Path, data: Path, output: Path, label: str, branch: str, args) -> tuple[dict, dict[int, str]]:
    """Run one explicit, unfused validation call shared by all three comparisons."""
    model = YOLO(str(checkpoint))
    if branch == "O2M":
        use_one_to_many(model)
    with patch.object(type(model.model), "fuse", lambda self, verbose=True: self):
        result = model.val(
            data=str(data),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            plots=False,
            project=str(output / "validation"),
            name=label,
            exist_ok=False,
            verbose=False,
        )
    names = {int(key): value for key, value in model.names.items()}
    payload = metrics_payload(result, names)
    payload.update(
        {
            "checkpoint": str(checkpoint.resolve()),
            "branch": branch,
            "runtime_fusion_disabled": True,
            "validation_parameters": {
                "data": str(data.resolve()),
                "split": "val",
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "conf": args.conf,
                "iou": args.iou,
                "max_det": args.max_det,
            },
            "class_mapping": names,
        }
    )
    return payload, names


def extract_region_logits(output) -> torch.Tensor:
    """Retrieve the auxiliary prior logits from an unfused LiteRG forward."""
    if isinstance(output, dict):
        parsed = output
    elif isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
        parsed = output[1]
    else:
        raise TypeError(f"Unexpected LiteRG output type: {type(output)!r}")
    logits = parsed.get("region_logits")
    if logits is None:
        raise KeyError("LiteRG forward did not return region_logits")
    return logits.float()


def scalar_values(core) -> dict[str, float]:
    """Read the four learned residual coefficients from the loaded best checkpoint."""
    lite_rg = getattr(core, "lite_rg", None)
    if lite_rg is None:
        raise RuntimeError("B5 checkpoint has no lite_rg module")
    values = {}
    for name in ("gamma3", "gamma4", "eta3", "eta4"):
        parameter = getattr(lite_rg, name, None)
        if parameter is None:
            raise RuntimeError(f"Missing LiteRG scalar {name}")
        values[name] = float(parameter.detach().cpu().item())
    return values


def summarize_categories(counter: Counter, denominator: int) -> dict:
    """Return every requested category, including explicit zero counts."""
    keys = ("both_hit", "b0_hit_b5_o2m_miss", "b0_miss_b5_o2m_hit", "both_miss")
    return {key: {"count": int(counter[key]), "proportion": counter[key] / max(denominator, 1)} for key in keys}


def compare_gt_and_region(
    b0_checkpoint: Path,
    b5_checkpoint: Path,
    data: Path,
    args,
) -> tuple[dict, list[dict], dict]:
    """Compare per-GT hits at IoU 0.50/0.75 and aggregate region-prior quality."""
    b0 = YOLO(str(b0_checkpoint))
    b5 = YOLO(str(b5_checkpoint))
    use_one_to_many(b5)
    core_b0, core_b5 = b0.model, b5.model

    with tempfile.TemporaryDirectory(prefix="literg-b0-o2m-loader-") as temporary_project:
        trainer = DetectionTrainer(
            overrides={
                "model": str(b0_checkpoint),
                "data": str(data),
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "epochs": 1,
                "project": temporary_project,
                "name": "loader",
                "exist_ok": True,
                "save": False,
                "plots": False,
            }
        )
        core_b0 = core_b0.to(trainer.device).eval()
        core_b5 = core_b5.to(trainer.device).eval()
        trainer.model = core_b0
        trainer.set_model_attributes()
        trainer.stride = max(int(core_b0.stride.max()), 32)
        loader = trainer.get_dataloader(trainer.data["val"], args.batch, -1, "val")
        raw_names = trainer.data["names"]
        names = {
            int(key): value for key, value in (raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names))
        }

        category_counts = {threshold: Counter() for threshold in (0.50, 0.75)}
        category_by_class = {threshold: defaultdict(Counter) for threshold in (0.50, 0.75)}
        gt_rows = []
        images = 0
        total_gt = 0

        region_sums = Counter()
        region_counts = Counter()
        intersection_soft = probability_sum = target_sum = 0.0
        intersection_binary = union_binary = predicted_binary = target_binary = 0
        bce_sum = 0.0
        pixel_count = 0

        with torch.inference_mode():
            for batch in loader:
                batch = trainer.preprocess_batch(batch)
                b0_output = core_b0(batch["img"])
                b5_output = core_b5(batch["img"])
                decoded_b0 = b0_output[0]
                decoded_b5_o2m = b5_output[0]
                post_b5_o2m = non_max_suppression(
                    decoded_b5_o2m,
                    args.conf,
                    args.iou,
                    multi_label=True,
                    max_det=args.max_det,
                    end2end=False,
                )

                region_logits = extract_region_logits(b5_output)
                region_probability = region_logits.sigmoid()
                region_target = build_region_targets(
                    batch,
                    batch_size=region_logits.shape[0],
                    height=region_logits.shape[-2],
                    width=region_logits.shape[-1],
                    device=region_logits.device,
                    sigma_scale=float(core_b5.yaml["lite_rg"]["sigma_scale"]),
                    target_mode=str(core_b5.yaml["lite_rg"]["target_mode"]),
                )
                foreground_core = region_target >= 0.5
                foreground_box = region_target > 0
                background = region_target == 0
                for label, mask in (
                    ("foreground_core", foreground_core),
                    ("foreground_box", foreground_box),
                    ("background", background),
                ):
                    region_sums[label] += float(region_probability[mask].sum().item())
                    region_counts[label] += int(mask.sum().item())
                region_sums["target_weighted_foreground"] += float((region_probability * region_target).sum().item())
                region_counts["target_weighted_foreground"] += float(region_target.sum().item())

                soft_intersection = float((region_probability * region_target).sum().item())
                intersection_soft += soft_intersection
                probability_sum += float(region_probability.sum().item())
                target_sum += float(region_target.sum().item())
                predicted_mask = region_probability >= 0.5
                target_mask = foreground_core
                intersection_binary += int((predicted_mask & target_mask).sum().item())
                union_binary += int((predicted_mask | target_mask).sum().item())
                predicted_binary += int(predicted_mask.sum().item())
                target_binary += int(target_mask.sum().item())
                bce_sum += float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        region_logits, region_target, reduction="sum"
                    ).item()
                )
                pixel_count += int(region_target.numel())

                height, width = batch["img"].shape[2:]
                scale = torch.tensor((width, height, width, height), device=trainer.device)
                for image_index in range(batch["img"].shape[0]):
                    mask = batch["batch_idx"].view(-1).long() == image_index
                    gt_cls = batch["cls"][mask].view(-1).long().cpu().numpy()
                    gt_boxes = (xywh2xyxy(batch["bboxes"][mask]) * scale).cpu().numpy()
                    pred_b0 = decoded_b0[image_index]
                    pred_b0 = pred_b0[pred_b0[:, 4] > args.conf].detach().float().cpu().numpy()
                    pred_b5 = post_b5_o2m[image_index].detach().float().cpu().numpy()

                    hit_vectors = {}
                    for threshold in (0.50, 0.75):
                        hit_b0 = matched_gt(pred_b0, gt_boxes, gt_cls, args.operating_conf, threshold)
                        hit_b5 = matched_gt(pred_b5, gt_boxes, gt_cls, args.operating_conf, threshold)
                        hit_vectors[threshold] = (hit_b0, hit_b5)
                        for gt_index, class_id in enumerate(gt_cls):
                            if hit_b0[gt_index] and hit_b5[gt_index]:
                                category = "both_hit"
                            elif hit_b0[gt_index]:
                                category = "b0_hit_b5_o2m_miss"
                            elif hit_b5[gt_index]:
                                category = "b0_miss_b5_o2m_hit"
                            else:
                                category = "both_miss"
                            category_counts[threshold][category] += 1
                            category_by_class[threshold][names[int(class_id)]][category] += 1

                    hit50_b0, hit50_b5 = hit_vectors[0.50]
                    hit75_b0, hit75_b5 = hit_vectors[0.75]
                    for gt_index, class_id in enumerate(gt_cls):
                        gt_rows.append(
                            {
                                "image": str(batch["im_file"][image_index]),
                                "class": names[int(class_id)],
                                "gt_index": gt_index,
                                "b0_hit_iou50": int(hit50_b0[gt_index]),
                                "b5_o2m_hit_iou50": int(hit50_b5[gt_index]),
                                "b0_hit_iou75": int(hit75_b0[gt_index]),
                                "b5_o2m_hit_iou75": int(hit75_b5[gt_index]),
                            }
                        )
                    total_gt += len(gt_cls)
                    images += 1

    gt_summary = {
        "images": images,
        "ground_truth_instances": total_gt,
        "operating_confidence": args.operating_conf,
        "nms_iou": args.iou,
        "by_match_iou": {},
    }
    for threshold in (0.50, 0.75):
        key = f"{threshold:.2f}"
        gt_summary["by_match_iou"][key] = {
            "all": summarize_categories(category_counts[threshold], total_gt),
            "by_class": {
                class_name: summarize_categories(counter, sum(counter.values()))
                for class_name, counter in category_by_class[threshold].items()
            },
        }

    foreground_core_mean = region_sums["foreground_core"] / max(region_counts["foreground_core"], 1)
    foreground_box_mean = region_sums["foreground_box"] / max(region_counts["foreground_box"], 1)
    background_mean = region_sums["background"] / max(region_counts["background"], 1)
    weighted_foreground_mean = region_sums["target_weighted_foreground"] / max(
        region_counts["target_weighted_foreground"], 1e-12
    )
    region_summary = {
        "definitions": {
            "foreground_core": "soft target >= 0.5",
            "foreground_box": "soft target > 0 (inside any GT box)",
            "background": "soft target == 0 (outside all GT boxes)",
            "soft_iou_dice": "probability map compared with morphology-adaptive soft target over all validation pixels",
            "binary_iou_dice": "probability >= 0.5 compared with soft target >= 0.5",
        },
        "response": {
            "foreground_core_mean_probability": foreground_core_mean,
            "foreground_box_mean_probability": foreground_box_mean,
            "target_weighted_foreground_mean_probability": weighted_foreground_mean,
            "background_mean_probability": background_mean,
            "foreground_core_minus_background": foreground_core_mean - background_mean,
            "foreground_box_minus_background": foreground_box_mean - background_mean,
        },
        "soft_iou": intersection_soft / max(probability_sum + target_sum - intersection_soft, 1e-12),
        "soft_dice": 2.0 * intersection_soft / max(probability_sum + target_sum, 1e-12),
        "binary_iou": intersection_binary / max(union_binary, 1),
        "binary_dice": 2.0 * intersection_binary / max(predicted_binary + target_binary, 1),
        "bce": bce_sum / max(pixel_count, 1),
        "pixel_counts": {key: int(value) for key, value in region_counts.items() if key != "target_weighted_foreground"},
    }
    return gt_summary, gt_rows, region_summary


def metric_rows(metrics: dict) -> list[dict]:
    rows = []
    for label, payload in metrics.items():
        rows.append(
            {
                "model_branch": label,
                "class": "all",
                "precision": payload["precision"],
                "recall": payload["recall"],
                "map50": payload["map50"],
                "map75": payload["map75"],
                "map50_95": payload["map50_95"],
            }
        )
        for class_name, class_metrics in payload["classes"].items():
            rows.append(
                {
                    "model_branch": label,
                    "class": class_name,
                    "precision": "",
                    "recall": "",
                    "map50": class_metrics["map50"],
                    "map75": class_metrics["map75"],
                    "map50_95": class_metrics["map50_95"],
                }
            )
    return rows


def region_rows(region: dict, scalars: dict[str, float]) -> list[dict]:
    rows = [{"metric": name, "value": value} for name, value in scalars.items()]
    rows.extend({"metric": name, "value": value} for name, value in region["response"].items())
    rows.extend(
        {"metric": name, "value": region[name]}
        for name in ("soft_iou", "soft_dice", "binary_iou", "binary_dice", "bce")
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--b5", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--operating-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    for path in (args.b0, args.b5, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=False)

    metrics = {}
    mappings = []
    for label, checkpoint, branch in (
        ("B0 O2O", args.b0, "O2O"),
        ("B5 O2O", args.b5, "O2O"),
        ("B5 O2M", args.b5, "O2M"),
    ):
        metrics[label], names = evaluate(checkpoint, args.data, args.output, label.lower().replace(" ", "_"), branch, args)
        mappings.append(names)
    if not all(mapping == mappings[0] for mapping in mappings[1:]):
        raise RuntimeError(f"Class mapping mismatch: {mappings}")
    validation_parameters = [payload["validation_parameters"] for payload in metrics.values()]
    if not all(parameters == validation_parameters[0] for parameters in validation_parameters[1:]):
        raise RuntimeError(f"Validation parameter mismatch: {validation_parameters}")

    gt_summary, gt_rows, region_summary = compare_gt_and_region(args.b0, args.b5, args.data, args)
    b5_loaded = YOLO(str(args.b5))
    scalars = scalar_values(b5_loaded.model)
    payload = {
        "protocol_identical": True,
        "validation_parameters": validation_parameters[0],
        "class_mapping": mappings[0],
        "metrics": metrics,
        "b0_vs_b5_o2m_gt": gt_summary,
        "b5_literg": {"scalars": scalars, "region_prior": region_summary},
    }
    with (args.output / "b0_vs_b5_o2m_diagnostic.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(args.output / "matched_metrics.csv", metric_rows(metrics))
    write_csv(args.output / "b0_vs_b5_o2m_gt_instances.csv", gt_rows)
    write_csv(args.output / "region_prior_metrics.csv", region_rows(region_summary, scalars))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
