#!/usr/bin/env python3
"""Audit a LiteRG 100e run and diagnose same-checkpoint O2O/O2M gaps without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
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

from eval_literg_full import metrics_payload, use_one_to_many  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402
from ultralytics.utils.ops import xywh2xyxy  # noqa: E402


METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
PROTOCOL_KEYS = (
    "epochs",
    "batch",
    "imgsz",
    "device",
    "workers",
    "pretrained",
    "optimizer",
    "seed",
    "deterministic",
    "amp",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "mosaic",
    "mixup",
    "copy_paste",
    "close_mosaic",
    "conf",
    "iou",
    "max_det",
    "resume",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_results(path: Path) -> tuple[list[dict], dict, dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No result rows in {path}")
    best = max(rows, key=lambda row: float(row[METRIC_COLUMNS["map50_95"]]))
    return rows, best, rows[-1]


def result_metrics(row: dict) -> dict:
    return {key: float(row[column]) for key, column in METRIC_COLUMNS.items()}


def checkpoint_facts(path: Path) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    core = raw.get("ema") or raw.get("model")
    head = core.model[-1]
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "checkpoint_epoch": raw.get("epoch"),
        "date": raw.get("date"),
        "git": raw.get("git"),
        "unfused_one_to_many_present": getattr(head, "cv2", None) is not None and getattr(head, "cv3", None) is not None,
        "unfused_one_to_one_present": hasattr(head, "one2one_cv2") and hasattr(head, "one2one_cv3"),
        "lite_rg_present": getattr(core, "lite_rg", None) is not None,
    }


def run_audit(run_dir: Path, metadata_name: str | None = None) -> dict:
    args_path = run_dir / "args.yaml"
    results_path = run_dir / "results.csv"
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    for path in (args_path, results_path, best_path, last_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    args = YAML.load(args_path)
    rows, best_row, final_row = read_results(results_path)
    best = checkpoint_facts(best_path)
    last = checkpoint_facts(last_path)
    metadata = {}
    if metadata_name and (run_dir / metadata_name).is_file():
        metadata = json.loads((run_dir / metadata_name).read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir.resolve()),
        "args_path": str(args_path.resolve()),
        "results_path": str(results_path.resolve()),
        "epochs_recorded": len(rows),
        "best_epoch": int(float(best_row["epoch"])),
        "best_results_row": result_metrics(best_row),
        "final_epoch": int(float(final_row["epoch"])),
        "final_results_row": result_metrics(final_row),
        "best": best,
        "last": last,
        "best_last_identical": best["sha256"] == last["sha256"],
        "args": {key: args.get(key) for key in PROTOCOL_KEYS},
        "data": str(Path(args["data"]).resolve()),
        "data_sha256": sha256(Path(args["data"])),
        "started_from_yolo26n_pretrained_not_resume": bool(
            args.get("resume") is False
            and str(args.get("pretrained", "")).endswith("yolo26n.pt")
            and len(rows) == int(args["epochs"])
            and (not metadata or str(metadata.get("weights", "")).endswith("yolo26n.pt"))
        ),
        "metadata": metadata,
    }


def box_iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0, dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.clip(right_bottom - left_top, 0, None).prod(1)
    area_a = np.clip(box[2:] - box[:2], 0, None).prod()
    area_b = np.clip(boxes[:, 2:] - boxes[:, :2], 0, None).prod(1)
    return intersection / np.clip(area_a + area_b - intersection, 1e-9, None)


def matched_gt(predictions: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray, conf: float, iou: float) -> np.ndarray:
    hits = np.zeros(len(gt_boxes), dtype=bool)
    if not len(predictions) or not len(gt_boxes):
        return hits
    pairs = []
    for pred_index, pred in enumerate(predictions):
        if pred[4] < conf:
            continue
        eligible = np.flatnonzero(gt_cls == int(pred[5]))
        for local_index, overlap in enumerate(box_iou_one(pred[:4], gt_boxes[eligible])):
            if overlap >= iou:
                pairs.append((float(pred[4]), float(overlap), pred_index, int(eligible[local_index])))
    used_predictions = set()
    for _confidence, _overlap, pred_index, gt_index in sorted(pairs, reverse=True):
        if pred_index not in used_predictions and not hits[gt_index]:
            used_predictions.add(pred_index)
            hits[gt_index] = True
    return hits


def o2o_miss_reason(raw: np.ndarray, gt_box: np.ndarray, class_id: int, eval_conf: float, op_conf: float, match_iou: float) -> dict:
    boxes, scores = raw[:, :4], raw[:, 4:]
    overlaps = box_iou_one(gt_box, boxes)
    same = scores[:, class_id]
    matched = overlaps >= match_iou
    best_correct_score = float(same[matched].max()) if matched.any() else 0.0
    wrong = scores.copy()
    wrong[:, class_id] = -1
    best_wrong_score = float(wrong[matched].max()) if matched.any() else 0.0
    confident = same >= op_conf
    best_confident_iou = float(overlaps[confident].max()) if confident.any() else 0.0
    if best_correct_score >= op_conf:
        reason = "postprocess_or_duplicate_conflict"
    elif best_wrong_score >= op_conf:
        reason = "classification_confusion"
    elif best_correct_score >= eval_conf:
        reason = "low_confidence_correct_candidate"
    elif best_confident_iou >= 0.10:
        reason = "localization_error"
    else:
        reason = "no_effective_candidate_assignment_or_head"
    return {
        "reason": reason,
        "best_correct_score_at_iou50": best_correct_score,
        "best_wrong_score_at_iou50": best_wrong_score,
        "best_iou_at_operating_conf": best_confident_iou,
        "best_iou_any_score": float(overlaps.max()) if len(overlaps) else 0.0,
    }


def gap_analysis(checkpoint: Path, data: Path, device: str, batch_size: int, workers: int, eval_conf: float, op_conf: float, match_iou: float, nms_iou: float, max_det: int) -> dict:
    o2o = YOLO(str(checkpoint))
    o2m = YOLO(str(checkpoint))
    use_one_to_many(o2m)
    core_o2o = o2o.model
    core_o2m = o2m.model
    trainer = DetectionTrainer(
        overrides={
            "model": str(checkpoint),
            "data": str(data),
            "imgsz": 640,
            "batch": batch_size,
            "device": device,
            "workers": workers,
            "epochs": 1,
            "plots": False,
        }
    )
    core_o2o = core_o2o.to(trainer.device).eval()
    core_o2m = core_o2m.to(trainer.device).eval()
    trainer.model = core_o2o
    trainer.set_model_attributes()
    trainer.stride = max(int(core_o2o.stride.max()), 32)
    loader = trainer.get_dataloader(trainer.data["val"], batch_size, -1, "val")
    raw_names = trainer.data["names"]
    names = {int(key): value for key, value in (raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names))}
    categories = Counter()
    categories_by_class = defaultdict(Counter)
    reasons = Counter()
    reasons_by_class = defaultdict(Counter)
    detail_rows = []
    prediction_counts = {"one_to_one": [], "one_to_many": []}
    images = 0

    with torch.inference_mode():
        for batch in loader:
            batch = trainer.preprocess_batch(batch)
            decoded_o2o, raw_o2o = core_o2o(batch["img"])
            decoded_o2m = core_o2m(batch["img"])[0]
            post_o2m = non_max_suppression(
                decoded_o2m,
                eval_conf,
                nms_iou,
                multi_label=True,
                max_det=max_det,
                end2end=False,
            )
            head = core_o2o.model[-1]
            dense_o2o = head._inference(raw_o2o["one2one"]).permute(0, 2, 1)
            height, width = batch["img"].shape[2:]
            scale = torch.tensor((width, height, width, height), device=trainer.device)
            for image_index in range(batch["img"].shape[0]):
                mask = batch["batch_idx"].view(-1).long() == image_index
                gt_cls = batch["cls"][mask].view(-1).long().cpu().numpy()
                gt_boxes = xywh2xyxy(batch["bboxes"][mask]) * scale
                gt_boxes = gt_boxes.cpu().numpy()
                pred_o2o = decoded_o2o[image_index]
                pred_o2o = pred_o2o[pred_o2o[:, 4] > eval_conf].detach().float().cpu().numpy()
                pred_o2m = post_o2m[image_index].detach().float().cpu().numpy()
                raw = dense_o2o[image_index].detach().float().cpu().numpy()
                prediction_counts["one_to_one"].append(len(pred_o2o))
                prediction_counts["one_to_many"].append(len(pred_o2m))
                hit_o2o = matched_gt(pred_o2o, gt_boxes, gt_cls, op_conf, match_iou)
                hit_o2m = matched_gt(pred_o2m, gt_boxes, gt_cls, op_conf, match_iou)
                for gt_index, class_id in enumerate(gt_cls):
                    if hit_o2m[gt_index] and hit_o2o[gt_index]:
                        category = "both_hit"
                    elif hit_o2m[gt_index]:
                        category = "o2m_hit_o2o_miss"
                    elif hit_o2o[gt_index]:
                        category = "o2o_hit_o2m_miss"
                    else:
                        category = "both_miss"
                    class_name = names[int(class_id)]
                    categories[category] += 1
                    categories_by_class[class_name][category] += 1
                    if category == "o2m_hit_o2o_miss":
                        detail = o2o_miss_reason(raw, gt_boxes[gt_index], int(class_id), eval_conf, op_conf, match_iou)
                        reasons[detail["reason"]] += 1
                        reasons_by_class[class_name][detail["reason"]] += 1
                        detail_rows.append(
                            {
                                "image": str(batch["im_file"][image_index]),
                                "class": class_name,
                                "gt_index": gt_index,
                                **detail,
                            }
                        )
                images += 1

    total = sum(categories.values())
    return {
        "images": images,
        "ground_truth_instances": total,
        "evaluation_confidence": eval_conf,
        "operating_confidence": op_conf,
        "match_iou": match_iou,
        "nms_iou": nms_iou,
        "prediction_counts": {
            branch: {
                "total": int(sum(values)),
                "mean_per_image": float(np.mean(values)),
                "median_per_image": float(np.median(values)),
            }
            for branch, values in prediction_counts.items()
        },
        "categories": {
            key: {"count": value, "proportion": value / max(total, 1)} for key, value in categories.items()
        },
        "categories_by_class": {
            class_name: {
                key: {"count": value, "proportion": value / max(sum(counter.values()), 1)}
                for key, value in counter.items()
            }
            for class_name, counter in categories_by_class.items()
        },
        "o2m_hit_o2o_miss_reasons": {
            key: {"count": value, "proportion": value / max(sum(reasons.values()), 1)} for key, value in reasons.items()
        },
        "o2m_hit_o2o_miss_reasons_by_class": {
            class_name: {
                key: {"count": value, "proportion": value / max(sum(counter.values()), 1)}
                for key, value in counter.items()
            }
            for class_name, counter in reasons_by_class.items()
        },
        "detail_rows": detail_rows,
        "attribution_limit": (
            "No-effective-candidate cases are consistent with assignment or O2O-head learning failure, but inference "
            "outputs alone cannot prove the training-time assigner was the cause."
        ),
    }


def evaluate_baseline(checkpoint: Path, data: Path, output: Path, args) -> dict:
    model = YOLO(str(checkpoint))
    with patch.object(type(model.model), "fuse", lambda self, verbose=True: self):
        result = model.val(
            data=str(data),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.eval_conf,
            iou=args.iou,
            max_det=args.max_det,
            plots=False,
            project=str(output / "validation"),
            name="b0_one_to_one",
            exist_ok=False,
            verbose=False,
        )
    payload = metrics_payload(result, {int(key): value for key, value in model.names.items()})
    payload["runtime_fusion_disabled"] = True
    return payload


def delta(left: dict, right: dict) -> dict:
    return {key: float(left[key] - right[key]) for key in ("precision", "recall", "map50", "map75", "map50_95")}


def write_csv_exclusive(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(b0: dict, o2o: dict, o2m: dict, gap: dict) -> list[dict]:
    rows = []
    for model_name, branch, metrics in (("B0", "O2O", b0), ("B5", "O2O", o2o), ("B5", "O2M", o2m)):
        overall = {
            "model": model_name,
            "branch": branch,
            "class": "all",
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "map50": metrics["map50"],
            "map75": metrics["map75"],
            "map50_95": metrics["map50_95"],
            "avg_predictions_per_image": "" if model_name == "B0" else gap["prediction_counts"]["one_to_one" if branch == "O2O" else "one_to_many"]["mean_per_image"],
            "preprocess_ms": metrics["speed_ms_per_image"]["preprocess"],
            "inference_ms": metrics["speed_ms_per_image"]["inference"],
            "postprocess_ms": metrics["speed_ms_per_image"]["postprocess"],
        }
        rows.append(overall)
        for class_name, class_metrics in metrics["classes"].items():
            rows.append(
                {
                    "model": model_name,
                    "branch": branch,
                    "class": class_name,
                    "precision": "",
                    "recall": "",
                    "map50": class_metrics["map50"],
                    "map75": class_metrics["map75"],
                    "map50_95": class_metrics["map50_95"],
                    "avg_predictions_per_image": "",
                    "preprocess_ms": "",
                    "inference_ms": "",
                    "postprocess_ms": "",
                }
            )
    return rows


def gap_rows(gap: dict) -> list[dict]:
    rows = []
    for class_name, counters in (("all", gap["categories"]), *gap["categories_by_class"].items()):
        for category, values in counters.items():
            rows.append({"scope": "hit_category", "class": class_name, "category": category, **values})
    for class_name, counters in (("all", gap["o2m_hit_o2o_miss_reasons"]), *gap["o2m_hit_o2o_miss_reasons_by_class"].items()):
        for category, values in counters.items():
            rows.append({"scope": "o2m_only_reason", "class": class_name, "category": category, **values})
    return rows


def self_check() -> None:
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    classes = np.array([0, 1])
    preds = np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 25, 25, 0.8, 1]], dtype=np.float32)
    assert np.allclose(box_iou_one(gt[0], gt), [1.0, 0.0])
    assert matched_gt(preds, gt, classes, 0.25, 0.5).tolist() == [True, False]
    print("DIAGNOSTIC_SELF_CHECK_OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dual-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-conf", type=float, default=0.001)
    parser.add_argument("--operating-conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all((args.run_dir, args.baseline_run, args.data, args.dual_json, args.output)):
        parser.error("run_dir, --baseline-run, --data, --dual-json and --output are required")
    for path in (args.run_dir, args.baseline_run, args.data, args.dual_json, args.output):
        if not path.exists():
            raise FileNotFoundError(path)

    b5_audit = run_audit(args.run_dir, "literg_run_metadata.json")
    b0_audit = run_audit(args.baseline_run)
    dual = json.loads(args.dual_json.read_text(encoding="utf-8"))["same_checkpoint_dual_branch"]
    o2o_metrics, o2m_metrics = dual["one_to_one"], dual["one_to_many"]
    b0_metrics = evaluate_baseline(args.baseline_run / "weights" / "best.pt", args.data, args.output, args)
    gap = gap_analysis(
        args.run_dir / "weights" / "best.pt",
        args.data,
        args.device,
        args.batch,
        args.workers,
        args.eval_conf,
        args.operating_conf,
        args.match_iou,
        args.iou,
        args.max_det,
    )
    protocol = {
        key: {"b0": b0_audit["args"][key], "b5": b5_audit["args"][key], "match": b0_audit["args"][key] == b5_audit["args"][key]}
        for key in PROTOCOL_KEYS
    }
    protocol["data_sha256"] = {
        "b0": b0_audit["data_sha256"],
        "b5": b5_audit["data_sha256"],
        "match": b0_audit["data_sha256"] == b5_audit["data_sha256"],
    }
    comparisons = {
        "b5_o2o_minus_b0_o2o": delta(o2o_metrics, b0_metrics),
        "b5_o2m_minus_b0_o2o": delta(o2m_metrics, b0_metrics),
        "b5_o2m_minus_b5_o2o": delta(o2m_metrics, o2o_metrics),
        "per_class_map50_95": {
            class_name: {
                "b5_o2o_minus_b0": o2o_metrics["classes"][class_name]["map50_95"] - b0_metrics["classes"][class_name]["map50_95"],
                "b5_o2m_minus_b0": o2m_metrics["classes"][class_name]["map50_95"] - b0_metrics["classes"][class_name]["map50_95"],
                "b5_o2m_minus_b5_o2o": o2m_metrics["classes"][class_name]["map50_95"] - o2o_metrics["classes"][class_name]["map50_95"],
            }
            for class_name in b0_metrics["classes"]
        },
    }
    case = "C" if o2m_metrics["map50_95"] <= b0_metrics["map50_95"] + 0.003 else "B"
    recommendation = {
        "decision_case": case,
        "m2o_kd_now": case == "B",
        "train_separate_o2m": False,
        "next_experiment": "B2 soft-region auxiliary supervision only, matched 30e" if case == "C" else "M2O-KD matched 30e",
        "command": (
            "PYTHONPATH=. /opt/conda/bin/python scripts/train_literg_ablation.py --stage B2 "
            "--data configs/japan7_remote.yaml --weights /root/YOLO26N-NEW/yolo26n.pt --epochs 30 "
            "--batch 32 --device 0 --workers 8 --seed 42 --name literg_b2_soft_region_only_japan7_30e_seed42"
            if case == "C"
            else "M2O-KD command is intentionally unavailable until the training-only loss is implemented and preflighted."
        ),
    }
    payload = {
        "source": {
            "repository": str(ROOT),
            "ultralytics": str(ULTRALYTICS_ROOT),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status_short": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        },
        "b5_run_audit": b5_audit,
        "b0_run_audit": b0_audit,
        "protocol_comparison": protocol,
        "metrics": {"b0_o2o": b0_metrics, "b5_o2o": o2o_metrics, "b5_o2m": o2m_metrics},
        "comparisons": comparisons,
        "gap_analysis": {key: value for key, value in gap.items() if key != "detail_rows"},
        "recommendation": recommendation,
    }
    metrics_json = args.output / "dual_head_100e_metrics.json"
    metrics_csv = args.output / "dual_head_100e_metrics.csv"
    gap_csv = args.output / "dual_head_100e_gap.csv"
    detail_csv = args.output / "o2m_hit_o2o_miss_details.csv"
    with metrics_json.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv_exclusive(metrics_csv, metric_rows(b0_metrics, o2o_metrics, o2m_metrics, gap))
    write_csv_exclusive(gap_csv, gap_rows(gap))
    if gap["detail_rows"]:
        write_csv_exclusive(detail_csv, gap["detail_rows"])
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
