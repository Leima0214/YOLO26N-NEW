#!/usr/bin/env python3
"""Diagnose per-GT task-aligned assignment for a completed Paper 1 smoke run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import select_device

AR_BINS = (("lt2", 0.0, 2.0), ("2to3", 2.0, 3.0), ("3to5", 3.0, 5.0), ("5to8", 5.0, 8.0), ("ge8", 8.0, math.inf))
BRANCHES = ("one2many", "one2one")
OUTCOMES = ("both_adequate", "iou_low_score_adequate", "iou_adequate_score_low", "both_low")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed smoke run directory inside the repository.")
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0, help="0 scans the full split.")
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    return parser.parse_args()


def resolve_within_root(value: str, label: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} must stay inside the project root: {value}")
    return path


def safe_load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def normalize_data_config(path: Path, split: str) -> tuple[dict, str | list[str]]:
    data = safe_load_yaml(path)
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("data YAML must define non-empty names")
    names = {int(key): str(value) for key, value in names.items()}
    if sorted(names) != list(range(len(names))):
        raise ValueError("class IDs in names must be contiguous from zero")
    data["names"] = names
    data["nc"] = len(names)
    data["channels"] = int(data.get("channels", 3))

    root = Path(data.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    value = data.get(split)
    if not value:
        raise ValueError(f"data YAML does not define split: {split}")

    def resolve_split(item: str) -> str:
        candidate = Path(item)
        return str(candidate if candidate.is_absolute() else (root / candidate).resolve())

    split_path = [resolve_split(item) for item in value] if isinstance(value, list) else resolve_split(value)
    data["path"] = root
    data[split] = split_path
    return data, split_path


def ar_bin(aspect_ratio: float) -> str:
    return next(label for label, lower, upper in AR_BINS if lower <= aspect_ratio < upper)


def parse_branch(criterion, predictions: dict[str, torch.Tensor], batch: dict) -> dict[tuple[int, int], dict]:
    pred_distri = predictions["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = predictions["scores"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(predictions["feats"], criterion.stride, 0.5)
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(predictions["feats"][0].shape[2:], device=criterion.device, dtype=pred_scores.dtype)
    imgsz = imgsz * criterion.stride[0]
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = criterion.preprocess(
        targets.to(criterion.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
    )
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    scores = pred_scores.detach().sigmoid()
    boxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype)
    anchors = anchor_points * stride_tensor

    _, _, _, fg_mask, target_gt_idx = criterion.assigner(
        scores, boxes, anchors, gt_labels, gt_bboxes, mask_gt
    )
    mask_in_gts = criterion.assigner.select_candidates_in_gts(anchors, gt_bboxes, mask_gt)
    _, overlaps = criterion.assigner.get_box_metrics(
        scores, boxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
    )

    scores = scores.cpu()
    overlaps = overlaps.cpu()
    mask_in_gts = mask_in_gts.bool().cpu()
    fg_mask = fg_mask.bool().cpu()
    target_gt_idx = target_gt_idx.long().cpu()
    gt_labels = gt_labels.long().cpu()
    gt_bboxes = gt_bboxes.float().cpu()
    mask_gt = mask_gt.bool().cpu()

    records = {}
    for image_index in range(batch_size):
        valid_count = int(mask_gt[image_index].sum().item())
        for gt_index in range(valid_count):
            class_id = int(gt_labels[image_index, gt_index, 0].item())
            inside = mask_in_gts[image_index, gt_index]
            assigned = fg_mask[image_index] & target_gt_idx[image_index].eq(gt_index)
            box = gt_bboxes[image_index, gt_index]
            width, height = float((box[2] - box[0]).item()), float((box[3] - box[1]).item())
            if inside.any():
                best_iou = float(overlaps[image_index, gt_index, inside].max().item())
                best_score = float(scores[image_index, inside, class_id].max().item())
            else:
                best_iou = 0.0
                best_score = 0.0
            records[(image_index, gt_index)] = {
                "class_id": class_id,
                "width": width,
                "height": height,
                "aspect_ratio": max(width / height, height / width),
                "positive_count": int(assigned.sum().item()),
                "inside_anchor_count": int(inside.sum().item()),
                "best_candidate_iou": best_iou,
                "best_class_score": best_score,
            }
    return records


def outcome(row: dict, branch: str, iou_threshold: float, score_threshold: float) -> str:
    iou_ok = row[f"{branch}_best_candidate_iou"] >= iou_threshold
    score_ok = row[f"{branch}_best_class_score"] >= score_threshold
    if iou_ok and score_ok:
        return "both_adequate"
    if score_ok:
        return "iou_low_score_adequate"
    if iou_ok:
        return "iou_adequate_score_low"
    return "both_low"


def summarize(rows: list[dict], branch: str, iou_threshold: float, score_threshold: float) -> dict:
    positive_key = f"{branch}_positive_count"
    iou_key = f"{branch}_best_candidate_iou"
    score_key = f"{branch}_best_class_score"

    def group_summary(group: list[dict]) -> dict:
        if not group:
            return {"gt_count": 0}
        outcomes = defaultdict(int, {name: 0 for name in OUTCOMES})
        for row in group:
            outcomes[outcome(row, branch, iou_threshold, score_threshold)] += 1
        total = len(group)
        return {
            "gt_count": total,
            "zero_positive_count": sum(row[positive_key] == 0 for row in group),
            "zero_positive_rate": sum(row[positive_key] == 0 for row in group) / total,
            "positive_count_mean": mean(row[positive_key] for row in group),
            "positive_count_median": median(row[positive_key] for row in group),
            "best_candidate_iou_mean": mean(row[iou_key] for row in group),
            "best_candidate_iou_median": median(row[iou_key] for row in group),
            "best_class_score_mean": mean(row[score_key] for row in group),
            "best_class_score_median": median(row[score_key] for row in group),
            "outcome_rates": {key: value / total for key, value in sorted(outcomes.items())},
        }

    by_class = {row["class_name"]: None for row in rows}
    by_ar = {label: None for label, _, _ in AR_BINS}
    return {
        "overall": group_summary(rows),
        "by_class": {
            name: group_summary([row for row in rows if row["class_name"] == name]) for name in by_class
        },
        "by_ar": {label: group_summary([row for row in rows if row["ar_bin"] == label]) for label in by_ar},
    }


def csv_text(rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def markdown(summary: dict) -> str:
    lines = [
        "# Paper 1 Per-GT Assignment Diagnostic",
        "",
        f"- status: `{summary['status']}`",
        f"- split/images/GT: `{summary['split']}` / `{summary['images']}` / `{summary['gt_count']}`",
        f"- weights SHA256: `{summary['weights_sha256']}`",
        f"- assignment overlap metric: `{summary['overlap_metric']}`",
        f"- thresholds: IoU `{summary['iou_threshold']}`, class score `{summary['score_threshold']}`",
        "",
    ]
    for branch in BRANCHES:
        lines.extend(
            [
                f"## {branch}",
                "",
                "| group | GT | zero positive | mean positives | median IoU | median class score |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        groups = {"all": summary["branches"][branch]["overall"]}
        groups.update(summary["branches"][branch]["by_class"])
        groups.update({f"AR:{key}": value for key, value in summary["branches"][branch]["by_ar"].items()})
        for name, values in groups.items():
            lines.append(
                f"| {name} | {values['gt_count']} | {values.get('zero_positive_rate', 0):.4f} | "
                f"{values.get('positive_count_mean', 0):.3f} | {values.get('best_candidate_iou_median', 0):.4f} | "
                f"{values.get('best_class_score_median', 0):.4f} |"
            )
        lines.extend(["", "| overall outcome | rate |", "| --- | ---: |"])
        for name, rate in summary["branches"][branch]["overall"]["outcome_rates"].items():
            lines.append(f"| {name} | {rate:.4f} |")
        lines.append("")
    lines.append("Use one2many as the primary early-training assignment diagnostic; one2one is a secondary E2E check.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.imgsz < 32 or args.imgsz % 32:
        raise SystemExit("--imgsz must be at least 32 and divisible by 32")
    if args.batch < 1 or args.workers < 0 or args.max_batches < 0:
        raise SystemExit("batch/workers/max-batches values are invalid")
    if not 0 <= args.iou_threshold <= 1 or not 0 <= args.score_threshold <= 1:
        raise SystemExit("diagnostic thresholds must be in [0, 1]")

    run_dir = resolve_within_root(args.run_dir, "run directory")
    data_yaml = resolve_within_root(args.data, "data YAML")
    weights = run_dir / "weights" / "best.pt"
    if not (run_dir / "COMPLETED").is_file() or not weights.is_file():
        raise SystemExit("run directory must contain COMPLETED and weights/best.pt")
    data, split_path = normalize_data_config(data_yaml, args.split)

    device = select_device(args.device)
    yolo = YOLO(str(weights))
    model = yolo.model.to(device)
    if int(model.model[-1].nc) != data["nc"]:
        raise SystemExit(f"model/data class mismatch: {model.model[-1].nc} vs {data['nc']}")
    model.names = data["names"]
    model.args = get_cfg()
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    criterion = model.init_criterion()

    loader_args = get_cfg(
        overrides={
            "data": str(data_yaml),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "rect": True,
            "cache": False,
            "task": "detect",
            "mode": "val",
            "model": str(weights),
        }
    )
    stride = max(int(model.stride.max()), 32)
    dataset = build_yolo_dataset(
        loader_args, split_path, args.batch, data, mode="val", rect=True, stride=stride
    )
    loader = build_dataloader(
        dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=-1, drop_last=False
    )

    rows = []
    image_offset = 0
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches and batch_index >= args.max_batches:
                break
            images = batch["img"].to(device, non_blocking=True).float() / 255
            batch_tensors = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                raw_predictions = model(images)
            parsed = criterion.one2many.parse_output(raw_predictions)
            branch_records = {
                branch: parse_branch(getattr(criterion, branch), parsed[branch], batch_tensors)
                for branch in BRANCHES
            }
            for key, common in branch_records["one2many"].items():
                image_index, gt_index = key
                class_id = common["class_id"]
                row = {
                    "image_index": image_offset + image_index,
                    "image_path": str(batch["im_file"][image_index]),
                    "gt_index": gt_index,
                    "class_id": class_id,
                    "class_name": data["names"][class_id],
                    "width_img640": common["width"],
                    "height_img640": common["height"],
                    "aspect_ratio": common["aspect_ratio"],
                    "ar_bin": ar_bin(common["aspect_ratio"]),
                }
                for branch in BRANCHES:
                    values = branch_records[branch][key]
                    for metric in ("positive_count", "inside_anchor_count", "best_candidate_iou", "best_class_score"):
                        row[f"{branch}_{metric}"] = values[metric]
                rows.append(row)
            image_offset += images.shape[0]
            if (batch_index + 1) % 20 == 0:
                print(f"processed_batches={batch_index + 1} images={image_offset} gt={len(rows)}")

    if not rows:
        raise RuntimeError("diagnostic produced no GT rows")
    summary = {
        "status": "PASS",
        "git_commit": git_commit(),
        "command": shlex.join(sys.argv),
        "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "weights_sha256": file_sha256(weights),
        "data_yaml": str(data_yaml.relative_to(ROOT)).replace("\\", "/"),
        "split": args.split,
        "images": image_offset,
        "gt_count": len(rows),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "amp": amp_enabled,
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
        "overlap_metric": "CIoU clamped to [0, 1], matching TaskAlignedAssigner",
        "branches": {
            branch: summarize(rows, branch, args.iou_threshold, args.score_threshold) for branch in BRANCHES
        },
    }
    atomic_write(run_dir / "gt_assignment_per_gt.csv", csv_text(rows))
    atomic_write(run_dir / "gt_assignment_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_write(run_dir / "gt_assignment_summary.md", markdown(summary))
    atomic_write(run_dir / "gt_assignment_command.txt", shlex.join(sys.argv) + "\n")
    print(f"PASS: images={image_offset} gt={len(rows)} report={run_dir / 'gt_assignment_summary.md'}")


if __name__ == "__main__":
    main()
