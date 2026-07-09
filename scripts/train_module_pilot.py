#!/usr/bin/env python3
"""
train_module_pilot.py — 3-epoch pilot training for a single YOLO26 variant YAML.

Usage:
    python scripts/train_module_pilot.py \
        --model-yaml ultralytics/cfg/models/26/yolo26-CBAM.yaml \
        --data configs/japan7_remote.yaml \
        --device 0

Output naming:
    module_CBAM_japan7_e3_img640_b16_seed42

Records:
    experiments/module_scan/pilot_report.csv
"""

import argparse
import csv
import sys
import traceback
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("experiments/module_scan")
PILOT_CSV = OUT_DIR / "pilot_report.csv"


def extract_module_name(yaml_path: str) -> str:
    """yolo26-CBAM.yaml -> CBAM"""
    stem = Path(yaml_path).stem
    if stem.startswith("yolo26-"):
        return stem[7:]
    return stem


def main():
    parser = argparse.ArgumentParser(description="3-epoch pilot for YOLO26 module variant")
    parser.add_argument("--model-yaml", required=True, help="Path to YAML config")
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default=None, help="Override run name")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    args = parser.parse_args()

    yaml_p = Path(args.model_yaml)
    if not yaml_p.exists():
        print(f"ERROR: YAML not found: {args.model_yaml}")
        sys.exit(1)

    module_name = extract_module_name(args.model_yaml)
    run_name = args.name or f"module_{module_name}_japan7_e3_img640_b16_seed42"

    print(f"=== Module Pilot ===")
    print(f"  YAML:     {args.model_yaml}")
    print(f"  Module:   {module_name}")
    print(f"  Data:     {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Run name: {run_name}")
    print(f"  Device:   {args.device}")
    print()

    # ── Build model ────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        model = YOLO(str(yaml_p))
        print("Model build OK")
    except Exception as e:
        print(f"BUILD FAILED: {e}")
        sys.exit(1)

    # ── Train ──────────────────────────────────────────────────────────────────
    results = None
    error_msg = ""
    status = "UNKNOWN"

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=42,
            amp=not args.no_amp,
            project="runs/baseline",
            name=run_name,
            exist_ok=True,
            plots=False,
        )
        status = "COMPLETED"
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "OOM" in msg:
            status = "OOM"
            error_msg = "GPU out of memory"
        elif "NaN" in msg or "nan" in msg:
            status = "NaN"
            error_msg = "NaN detected during training"
        else:
            status = "RUNTIME_ERROR"
            error_msg = msg[:200]
    except Exception as e:
        status = "ERROR"
        error_msg = f"{type(e).__name__}: {str(e)[:200]}"

    # ── Collect metrics ────────────────────────────────────────────────────────
    map50 = ""
    map50_95 = ""
    has_results_csv = False
    has_best_pt = False
    has_args_yaml = False
    params_info = ""

    exp_dir = Path("runs/baseline") / run_name

    if status == "COMPLETED" and results is not None:
        try:
            map50 = f"{results.results_dict.get('metrics/mAP50(B)', 0):.4f}"
            map50_95 = f"{results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}"
        except Exception:
            pass

    if exp_dir.exists():
        has_results_csv = (exp_dir / "results.csv").exists()
        has_best_pt = (exp_dir / "weights" / "best.pt").exists()
        has_args_yaml = (exp_dir / "args.yaml").exists()

    # ── Write pilot report row ─────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp", "module_name", "yaml_path", "run_name",
        "status", "error_message",
        "map50", "map50_95",
        "has_results_csv", "has_best_pt", "has_args_yaml",
        "params_info", "recommended_next_step",
    ]

    row = {
        "timestamp": datetime.now().isoformat(),
        "module_name": module_name,
        "yaml_path": args.model_yaml,
        "run_name": run_name,
        "status": status,
        "error_message": error_msg[:200],
        "map50": map50,
        "map50_95": map50_95,
        "has_results_csv": str(has_results_csv),
        "has_best_pt": str(has_best_pt),
        "has_args_yaml": str(has_args_yaml),
        "params_info": params_info,
        "recommended_next_step": "",
    }

    if status == "COMPLETED":
        try:
            m50 = float(map50)
            m95 = float(map50_95)
            if m50 > 0.3 and m95 > 0.15:
                row["recommended_next_step"] = "promote — proceed to 20-epoch signal test"
            elif m50 > 0:
                row["recommended_next_step"] = "review — metrics low, check loss curve before promoting"
            else:
                row["recommended_next_step"] = "skip — zero/NaN metrics"
        except ValueError:
            row["recommended_next_step"] = "investigate — could not parse metrics"
    elif status == "OOM":
        row["recommended_next_step"] = f"retry with --batch {max(4, args.batch // 4)}"
    elif status == "NaN":
        row["recommended_next_step"] = "skip — NaN training, likely module incompatible with training"
    else:
        row["recommended_next_step"] = "investigate"

    file_exists = PILOT_CSV.exists()
    with open(PILOT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n=== Pilot Result ===")
    print(f"  Status:        {status}")
    print(f"  mAP50:         {map50}")
    print(f"  mAP50-95:      {map50_95}")
    print(f"  results.csv:   {has_results_csv}")
    print(f"  best.pt:       {has_best_pt}")
    print(f"  Next:          {row['recommended_next_step']}")
    print(f"  Report:        {PILOT_CSV}")


if __name__ == "__main__":
    main()
