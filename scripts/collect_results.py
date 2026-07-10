#!/usr/bin/env python3
"""Collect exact formal runs listed in a CSV manifest; never infer run directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50",
    "metrics/mAP50-95(B)": "map50_95",
}


def read_best_metrics(results_csv: Path) -> dict[str, str]:
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row.get("metrics/mAP50-95(B)", "-inf")))
    return {name: f"{float(best[column]):.6f}" for column, name in METRIC_COLUMNS.items() if best.get(column)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect formal experiment results from an explicit run manifest")
    parser.add_argument("--manifest", required=True, help="CSV with model, run_dir, and optional protocol columns")
    parser.add_argument("--output", default="experiments/formal_results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = (ROOT / args.manifest).resolve()
    output_path = (ROOT / args.output).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Missing run manifest: {args.manifest}")

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    if not runs or not {"model", "run_dir"}.issubset(runs[0]):
        raise SystemExit("Manifest must contain at least: model,run_dir")

    lines = [
        "# Formal Experiment Results",
        "",
        f"Manifest: `{manifest_path.relative_to(ROOT).as_posix()}`",
        "Best epoch is selected by mAP50-95 from each exact `results.csv` path.",
        "",
        "| model | initialization | protocol | params | FLOPs | P | R | mAP50 | mAP50-95 | run_dir | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run in runs:
        run_dir = (ROOT / run["run_dir"]).resolve()
        if ROOT not in run_dir.parents or not (run_dir / "results.csv").exists():
            lines.append(f"| {run['model']} | {run.get('initialization', '')} | {run.get('protocol', '')} |  |  |  |  |  |  | {run['run_dir']} | MISSING_RESULTS |")
            continue
        metrics = {name: "" for name in METRIC_COLUMNS.values()}
        metrics.update(read_best_metrics(run_dir / "results.csv"))
        lines.append(
            "| {model} | {initialization} | {protocol} | {params} | {flops} | {precision} | {recall} | {map50} | {map50_95} | {run_dir} | COMPLETED |".format(
                initialization=run.get("initialization", ""),
                protocol=run.get("protocol", ""),
                params=run.get("params", ""),
                flops=run.get("flops", ""),
                **metrics,
                **run,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
