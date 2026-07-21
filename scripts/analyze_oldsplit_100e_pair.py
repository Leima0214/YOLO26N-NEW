#!/usr/bin/env python3
"""Analyze the completed exploratory old-split B0/P4 100e pair."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/paper1/exploratory_oldsplit"
LOG_ROOT = ROOT / "logs/exploratory_oldsplit"
REPORT_JSON = ROOT / "reports/oldsplit_exploratory_100e_pair_analysis.json"
REPORT_MD = ROOT / "reports/oldsplit_exploratory_100e_pair_analysis.md"
MAP_KEY = "metrics/mAP50-95(B)"
RUNS = {
    "B0": "b0_japan7_oldsplit_exploratory_100e_seed42",
    "P4 Single": "p4single_japan7_oldsplit_exploratory_100e_seed42",
}
SHARED_ARGS = (
    "data",
    "epochs",
    "imgsz",
    "batch",
    "workers",
    "seed",
    "deterministic",
    "amp",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "warmup_epochs",
    "weight_decay",
    "mosaic",
    "mixup",
    "copy_paste",
    "close_mosaic",
    "iou",
    "max_det",
    "resume",
    "pretrained",
    "split_status",
    "result_scope",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def read_revalidation(name: str) -> dict:
    log = LOG_ROOT / f"{name}_reval.log"
    for line in reversed(log.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if "METRICS_JSON=" in line:
            return json.loads(line.split("METRICS_JSON=", 1)[1])
    raise ValueError(f"No METRICS_JSON in {log}")


def curve_summary(rows: list[dict[str, str]]) -> dict:
    if len(rows) != 100:
        raise ValueError(f"Expected 100 epochs, found {len(rows)}")
    values = [float(row[MAP_KEY]) for row in rows]
    best_index = max(range(len(values)), key=values.__getitem__)
    best = values[best_index]
    plateau_index = next(index for index in range(len(values)) if best - max(values[: index + 1]) < 0.001)
    return {
        "best_map50_95": best,
        "best_epoch": int(float(rows[best_index]["epoch"])),
        "epoch100_map50_95": values[-1],
        "last10_mean_map50_95": fmean(values[-10:]),
        "last20_mean_map50_95": fmean(values[-20:]),
        "plateau_epoch_within_0_001_of_eventual_best": int(float(rows[plateau_index]["epoch"])),
    }


def telemetry_summary(path: Path) -> dict:
    rows = read_csv(path)
    if len(rows) != 100:
        raise ValueError(f"Expected 100 telemetry rows, found {len(rows)}")
    return {
        "peak_gpu_gb": max(float(row["peak_gpu_gb"]) for row in rows),
        "mean_epoch_seconds": fmean(float(row["epoch_seconds"]) for row in rows),
        "mean_iterations_per_second": fmean(float(row["iterations_per_second"]) for row in rows),
    }


def main() -> None:
    summaries = {}
    args_by_model = {}
    for label, name in RUNS.items():
        run = RUN_ROOT / name
        args_by_model[label] = yaml.safe_load((run / "args.yaml").read_text(encoding="utf-8"))
        summaries[label] = {
            "curve": curve_summary(read_csv(run / "results.csv")),
            "telemetry": telemetry_summary(run / "telemetry.csv"),
            "best_checkpoint_revalidation": read_revalidation(name),
        }

    mismatches = {
        key: {label: values.get(key) for label, values in args_by_model.items()}
        for key in SHARED_ARGS
        if len({json.dumps(values.get(key), sort_keys=True) for values in args_by_model.values()}) != 1
    }
    if mismatches:
        raise ValueError(f"Unfair paired arguments: {mismatches}")

    b0, p4 = summaries["B0"], summaries["P4 Single"]
    best_delta = p4["curve"]["best_map50_95"] - b0["curve"]["best_map50_95"]
    last10_delta = p4["curve"]["last10_mean_map50_95"] - b0["curve"]["last10_mean_map50_95"]
    if best_delta < 0.001:
        decision = "STOP_P4_STRUCTURE_LINE"
        run_p4_v2 = False
    elif best_delta >= 0.002 and last10_delta > 0:
        decision = "STRONG_ENOUGH_FOR_JAPAN7_V2_CONFIRMATION"
        run_p4_v2 = True
    else:
        decision = "WEAK_OR_MIXED_POSITIVE_ONE_JAPAN7_V2_CONFIRMATION"
        run_p4_v2 = True

    payload = {
        "scope": "exploratory_oldsplit_not_final_benchmark",
        "paired_args_match": True,
        "summaries": summaries,
        "paired_deltas": {
            "best_map50_95": best_delta,
            "epoch100_map50_95": p4["curve"]["epoch100_map50_95"] - b0["curve"]["epoch100_map50_95"],
            "last10_mean_map50_95": last10_delta,
            "last20_mean_map50_95": p4["curve"]["last20_mean_map50_95"] - b0["curve"]["last20_mean_map50_95"],
        },
        "decision": decision,
        "run_p4_v2": run_p4_v2,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# exploratory_oldsplit 100e paired analysis — not a final benchmark",
        "",
        "> This split contains documented near-duplicate train/val scenes. Results answer only the paired long-convergence question.",
        "",
        "## Aggregate curve",
        "",
        "| Model | best | best epoch | epoch100 | last10 mean | last20 mean | plateau epoch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in RUNS:
        curve = summaries[label]["curve"]
        lines.append(
            f"| {label} | {curve['best_map50_95']:.6f} | {curve['best_epoch']} | "
            f"{curve['epoch100_map50_95']:.6f} | {curve['last10_mean_map50_95']:.6f} | "
            f"{curve['last20_mean_map50_95']:.6f} | {curve['plateau_epoch_within_0_001_of_eventual_best']} |"
        )
    delta = payload["paired_deltas"]
    lines.extend(
        [
            "",
            f"Paired deltas (P4-B0): best {delta['best_map50_95']:+.6f}, epoch100 {delta['epoch100_map50_95']:+.6f}, "
            f"last10 {delta['last10_mean_map50_95']:+.6f}, last20 {delta['last20_mean_map50_95']:+.6f}.",
            "",
            "## Best-checkpoint revalidation by class",
            "",
            "| Class | B0 AP50-95 | P4 AP50-95 | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    b0_classes = b0["best_checkpoint_revalidation"]["per_class"]
    p4_classes = p4["best_checkpoint_revalidation"]["per_class"]
    for class_name in b0_classes:
        left = b0_classes[class_name]["map50_95"]
        right = p4_classes[class_name]["map50_95"]
        lines.append(f"| {class_name} | {left:.6f} | {right:.6f} | {right-left:+.6f} |")

    lines.extend(["", "## Cost and speed", "", "| Model | Params | GFLOPs | peak GPU GB | train it/s | inference ms/image |", "|---|---:|---:|---:|---:|---:|"])
    for label in RUNS:
        reval = summaries[label]["best_checkpoint_revalidation"]
        telemetry = summaries[label]["telemetry"]
        lines.append(
            f"| {label} | {reval['model']['parameters']:,} | {reval['model']['gflops_640']:.3f} | "
            f"{telemetry['peak_gpu_gb']:.3f} | {telemetry['mean_iterations_per_second']:.3f} | "
            f"{reval['speed_ms_per_image']['inference']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `{decision}`",
            f"- Run P4 on Japan7-v2: `{str(run_p4_v2).lower()}`",
            "- Plateau definition: first epoch whose best-so-far is within 0.001 of the eventual curve best.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["paired_deltas"], indent=2))
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
