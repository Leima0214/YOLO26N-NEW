#!/usr/bin/env python3
"""
Collect results from runs/baseline/*/results.csv and generate a Markdown table.

Output: experiments/baseline_table.md (overwrites template)

Usage:
    python scripts/collect_results.py
"""

from pathlib import Path

RESULTS_DIR = Path("runs/baseline")
OUTPUT = Path("experiments/baseline_table.md")

EXPECTED = {
    "yolov8n_japan7_e100_img640_b32_seed422": ("YOLOv8n", "yolov8n.pt"),
    "yolo11n_japan7_e100_img640_b32_seed42":  ("YOLO11n", "yolo11n.pt"),
    "yolo26n_japan7_e100_img640_b32_seed42":  ("YOLO26n", "yolo26n.pt"),
    "yolo26s_japan7_e100_img640_b32_seed42":  ("YOLO26s", "yolo26s.pt"),
}


def read_last_metrics(csv_path: Path) -> dict:
    """Read the last row of results.csv and extract key metrics."""
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return {}
    header = lines[0].strip().split(",")
    last = lines[-1].strip().split(",")
    if len(header) != len(last):
        return {}
    row = dict(zip(header, last))

    metrics = {}
    col_map = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
    }
    for col, key in col_map.items():
        if col in row:
            try:
                metrics[key] = f"{float(row[col]):.4f}"
            except ValueError:
                metrics[key] = row[col]
    return metrics


def main():
    if not RESULTS_DIR.is_dir():
        print(f"No results directory found: {RESULTS_DIR}")
        print("Run training first, then re-run this script.")
        return

    lines = [
        "# Japan Baseline Results",
        "",
        "| Method | Weight | Data | Epochs | ImgSize | Precision | Recall | mAP50 | mAP50-95 | Params | FLOPs | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for exp_name, (method, weight) in EXPECTED.items():
        csv_path = RESULTS_DIR / exp_name / "results.csv"
        if not csv_path.exists():
            lines.append(f"| {method} | {weight} | Japan | — | — | — | — | — | — | — | — | results.csv missing |")
            print(f"  SKIP: {csv_path} not found")
            continue

        m = read_last_metrics(csv_path)
        precision = m.get("precision", "—")
        recall = m.get("recall", "—")
        map50 = m.get("map50", "—")
        map50_95 = m.get("map50_95", "—")

        lines.append(
            f"| {method} | {weight} | Japan | 100 | 640 | {precision} | {recall} | {map50} | {map50_95} | | | |"
        )
        print(f"  OK: {exp_name}  mAP50={map50}  mAP50-95={map50_95}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nTable written to {OUTPUT}")


if __name__ == "__main__":
    main()
