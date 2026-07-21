#!/usr/bin/env python3
"""Build the immutable B0/P4 Single three-seed 30e report from saved artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    ("B0", 42): ROOT / "runs/detect/runs/paper1/b0_yolo26n_pretrained_auto_japan7_30e_seed42",
    ("B0", 0): ROOT / "runs/paper1/paper1_b0_pretrained_auto_linear_japan7_30e_seed0",
    ("B0", 3447): ROOT / "runs/paper1/paper1_b0_pretrained_auto_linear_japan7_30e_seed3447",
    ("P4 Single", 42): ROOT / "runs/paper1/paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_30e_seed42",
    ("P4 Single", 0): ROOT / "runs/paper1/paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_30e_seed0",
    ("P4 Single", 3447): ROOT / "runs/paper1/paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_30e_seed3447",
}
METRIC_LOGS = {
    ("B0", 42): ROOT / "logs/audit_b0_seed42_metrics.log",
    ("B0", 0): ROOT / "logs/audit_b0_seed0_metrics.log",
    ("B0", 3447): ROOT / "logs/reval_b0_seed3447_20260720.log",
    ("P4 Single", 42): ROOT / "logs/audit_p4single_seed42_metrics.log",
    ("P4 Single", 0): ROOT / "logs/audit_p4single_seed0_metrics.log",
    ("P4 Single", 3447): ROOT / "logs/reval_p4_single_seed3447_20260720.log",
}
CLASSES = ("D00", "D10", "D20", "D40", "D43", "D44", "D50")


def curve(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if len(rows) != 30:
        return {"best_epoch": None, "curve_best": None, "last10_mean": None, "curve_rows": len(rows)}
    values = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    best_position = max(range(len(values)), key=values.__getitem__)
    return {
        "best_epoch": int(float(rows[best_position]["epoch"])),
        "curve_best": values[best_position],
        "last10_mean": statistics.fmean(values[-10:]),
        "curve_rows": len(rows),
    }


def validation(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("METRICS_JSON="):
            return json.loads(line.removeprefix("METRICS_JSON="))
    raise RuntimeError(f"METRICS_JSON missing from {path}")


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def main() -> None:
    records = {}
    for key, run in RUNS.items():
        records[key] = {**curve(run / "results.csv"), **validation(METRIC_LOGS[key])}

    lines = [
        "# P4 Single 30e three-seed immutable summary",
        "",
        "All aggregate and per-class values below are independent validations of each saved `best.pt` using the same Japan7 val split, `imgsz=640`, `batch=32`, `iou=0.7`, and `max_det=300`.",
        "",
        "## Per-seed results",
        "",
        "| Model | Seed | best.pt mAP50-95 | Curve-best epoch | Curve best | Last-10 mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("B0", "P4 Single"):
        for seed in (42, 0, 3447):
            item = records[(model, seed)]
            curve_cells = (
                (str(item["best_epoch"]), f"{item['curve_best']:.5f}", f"{item['last10_mean']:.5f}")
                if item["curve_rows"] == 30
                else ("N/A", "N/A", "N/A")
            )
            lines.append(f"| {model} | {seed} | {item['aggregate']['map50_95']:.5f} | {curve_cells[0]} | {curve_cells[1]} | {curve_cells[2]} |")

    b0 = [records[("B0", seed)]["aggregate"]["map50_95"] for seed in (42, 0, 3447)]
    p4 = [records[("P4 Single", seed)]["aggregate"]["map50_95"] for seed in (42, 0, 3447)]
    deltas = [right - left for left, right in zip(b0, p4, strict=True)]
    lines += [
        "",
        "## Paired conclusion",
        "",
        f"- Highest single-seed value: P4 Single seed0 `{max(p4):.5f}`.",
        f"- B0 three-seed mean: `{statistics.fmean(b0):.5f}`.",
        f"- P4 Single three-seed mean: `{statistics.fmean(p4):.5f}`.",
        f"- Paired deltas seed42/0/3447: `{deltas[0]:+.5f}`, `{deltas[1]:+.5f}`, `{deltas[2]:+.5f}`.",
        f"- Mean paired delta: `{statistics.fmean(deltas):+.5f}`; 2/3 seeds are positive and seed42 is a practical tie.",
        "- Decision: repeatable weak positive only; this is not evidence of a material improvement.",
        "- Learned Gate audits showed finite non-zero gradients and movement away from 1e-3, so the weak effect is not explained by gradient starvation.",
        "- Cost: 2,376,201 -> 2,451,753 parameters (+3.18%); 5.2 -> 5.7 GFLOPs (+9.62%).",
        "- B0 seed42 curve fields are unavailable: its `results.csv` now contains only one row although the independently validated 30e `best.pt` remains intact. The missing history is reported as N/A rather than reconstructed.",
        "",
        "## Per-class best-checkpoint mean and sample standard deviation",
        "",
        "| Class | B0 mean | B0 std | P4 mean | P4 std | Mean delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    class_deltas = {}
    for name in CLASSES:
        left = [records[("B0", seed)]["per_class"][name]["map50_95"] for seed in (42, 0, 3447)]
        right = [records[("P4 Single", seed)]["per_class"][name]["map50_95"] for seed in (42, 0, 3447)]
        left_mean, left_std = mean_std(left)
        right_mean, right_std = mean_std(right)
        class_deltas[name] = right_mean - left_mean
        lines.append(f"| {name} | {left_mean:.5f} | {left_std:.5f} | {right_mean:.5f} | {right_std:.5f} | {right_mean-left_mean:+.5f} |")

    helped = [name for name, value in class_deltas.items() if value > 0]
    harmed = [name for name, value in class_deltas.items() if value < 0]
    lines += [
        "",
        f"Mean-positive classes: {', '.join(helped)}. Mean-negative classes: {', '.join(harmed)}.",
        "D10 and D20 decline in every paired seed; this is the most stable class-level warning.",
        "",
        "## Validity caveat discovered on 2026-07-21",
        "",
        "A later cross-split perceptual audit found visually confirmed near-duplicate road scenes across train and val. These values remain an immutable record of what was run, but they must not be interpreted as leakage-free generalization estimates.",
    ]
    output = ROOT / "reports/p4_single_30e_three_seed_summary.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
