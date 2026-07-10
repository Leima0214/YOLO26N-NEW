#!/usr/bin/env python3
"""Run one 3-epoch YOLO26 module pilot and append a report row."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "experiments" / "module_scan"
PILOT_CSV = OUT_DIR / "pilot_report.csv"
PILOT_MD = OUT_DIR / "pilot_report.md"
PROJECT = ROOT / "runs" / "module_scan"
MIN_TRANSFER_RATIO = 0.10

FIELDS = [
    "timestamp",
    "yaml_path",
    "pretrained",
    "transferred_items",
    "transfer_total_items",
    "transfer_ratio",
    "run_name",
    "run_dir",
    "status",
    "error_type",
    "error_message_short",
    "results_csv_exists",
    "best_pt_exists",
    "args_yaml_exists",
    "oom",
    "nan_detected",
    "loss_decreased",
    "map50_nonzero",
    "map50",
    "map50_95",
    "params_if_available",
    "flops_if_available",
    "recommended_next_step",
]


def short(text: object, limit: int = 180) -> str:
    return " ".join(str(text).split())[:limit]


def safe_load_yaml(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        yaml.safe_load(f)


def module_name(yaml_path: Path) -> str:
    stem = yaml_path.stem
    return stem.removeprefix("yolo26-").replace("_", "-")


def unique_name(base: str) -> str:
    if not (PROJECT / base).exists():
        return base
    return f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_text(command: list[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False).stdout
    except Exception as e:
        return f"{type(e).__name__}: {e}\n"


def save_repro_files(run_dir: Path, data_yaml: Path, command: str, pretrained: Path | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "git_commit.txt").write_text(run_text(["git", "rev-parse", "HEAD"]), encoding="utf-8")
    (run_dir / "git_branch.txt").write_text(run_text(["git", "branch", "--show-current"]), encoding="utf-8")
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "pretrained.txt").write_text(f"{pretrained or 'scratch'}\n", encoding="utf-8")
    (run_dir / "python_version.txt").write_text(sys.version + "\n", encoding="utf-8")
    (run_dir / "torch_info.txt").write_text(torch_info(), encoding="utf-8")
    (run_dir / "nvidia_smi.txt").write_text(run_text(["nvidia-smi"]), encoding="utf-8")
    (run_dir / "pip_freeze.txt").write_text(run_text([sys.executable, "-m", "pip", "freeze"]), encoding="utf-8")
    if data_yaml.exists():
        shutil.copy2(data_yaml, run_dir / "data_yaml_snapshot.yaml")


def torch_info() -> str:
    try:
        import torch

        return "\n".join(
            [
                f"torch={torch.__version__}",
                f"cuda_available={torch.cuda.is_available()}",
                f"cuda_version={torch.version.cuda}",
                f"device_count={torch.cuda.device_count()}",
            ]
        ) + "\n"
    except Exception as e:
        return f"{type(e).__name__}: {e}\n"


def count_params(model) -> str:
    module = getattr(model, "model", model)
    try:
        return str(sum(p.numel() for p in module.parameters()))
    except Exception:
        return ""


def model_stats(model) -> tuple[str, str]:
    params = count_params(model)
    flops = ""
    try:
        info = model.info(detailed=False, verbose=False)
    except Exception:
        return params, flops
    if isinstance(info, tuple):
        if len(info) > 1 and info[1] is not None:
            params = str(info[1])
        if len(info) > 3 and info[3] is not None:
            flops = str(info[3])
    elif info is not None and not params:
        params = short(info)
    return params, flops


def transfer_stats(model) -> tuple[int, int]:
    """Count the same-name, same-shape items used by Ultralytics weight transfer."""
    checkpoint = getattr(model, "ckpt", None)
    checkpoint_model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if checkpoint_model is None:
        return 0, len(model.model.state_dict())
    from ultralytics.utils.torch_utils import intersect_dicts

    target = model.model.state_dict()
    return len(intersect_dicts(checkpoint_model.float().state_dict(), target)), len(target)


def save_transfer_metadata(run_dir: Path, pretrained: Path, transferred: int, total: int) -> None:
    ratio = transferred / total if total else 0.0
    (run_dir / "pretrained.txt").write_text(
        f"{pretrained}\ntransferred_items={transferred}\ntotal_items={total}\ntransfer_ratio={ratio:.6f}\n",
        encoding="utf-8",
    )


def read_results(results_csv: Path) -> dict[str, object]:
    out = {"nan_detected": False, "loss_decreased": False, "map50": "", "map50_95": "", "map50_nonzero": False}
    if not results_csv.exists():
        return out
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    if not rows:
        return out

    numeric_rows: list[dict[str, float]] = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            try:
                parsed[key.strip()] = float(value)
            except (TypeError, ValueError):
                continue
        numeric_rows.append(parsed)

    out["nan_detected"] = any(math.isnan(v) for row in numeric_rows for v in row.values())
    last = numeric_rows[-1]
    map50 = last.get("metrics/mAP50(B)", last.get("metrics/mAP50"))
    map95 = last.get("metrics/mAP50-95(B)", last.get("metrics/mAP50-95"))
    if map50 is not None:
        out["map50"] = f"{map50:.6f}"
        out["map50_nonzero"] = map50 > 0
    if map95 is not None:
        out["map50_95"] = f"{map95:.6f}"

    loss_keys = [k for k in numeric_rows[0] if k.startswith("train/") and k.endswith("_loss")]
    if loss_keys and len(numeric_rows) > 1:
        first_loss = sum(numeric_rows[0].get(k, 0.0) for k in loss_keys)
        last_loss = sum(last.get(k, 0.0) for k in loss_keys)
        out["loss_decreased"] = last_loss < first_loss
    return out


def write_pilot_markdown(rows: list[dict[str, str]]) -> None:
    lines = [
        "# YOLO26 Module Pilot Report",
        "",
        "One row per module run. Older rows may have blank transfer fields.",
        "",
        "| yaml_path | pretrained | transfer | run_name | status | mAP50 | mAP50-95 | OOM | NaN | loss_decreased | next_step |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {yaml_path} | {pretrained} | {transferred_items}/{transfer_total_items} ({transfer_ratio}) | "
            "{run_name} | {status} | {map50} | {map50_95} | {oom} | {nan_detected} | "
            "{loss_decreased} | {recommended_next_step} |".format(**row)
        )
    PILOT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_templates() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    migrated = False
    if PILOT_CSV.exists():
        with PILOT_CSV.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [{field: row.get(field, "") for field in FIELDS} for row in reader]
            migrated = reader.fieldnames != FIELDS
        if migrated:
            with PILOT_CSV.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
    else:
        with PILOT_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    if migrated or not PILOT_MD.exists():
        write_pilot_markdown(rows)


def append_row(row: dict[str, object]) -> None:
    ensure_templates()
    with PILOT_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    with PILOT_MD.open("a", encoding="utf-8") as f:
        f.write(
            "| {yaml_path} | {pretrained} | {transferred_items}/{transfer_total_items} ({transfer_ratio}) | "
            "{run_name} | {status} | {map50} | {map50_95} | {oom} | {nan_detected} | "
            "{loss_decreased} | {recommended_next_step} |\n".format(**row)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one YOLO26 module pilot")
    parser.add_argument("--model-yaml", required=True)
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name")
    parser.add_argument(
        "--pretrained",
        default="yolo26n.pt",
        help="Checkpoint transferred into matching YAML layers; use 'none' for scratch training.",
    )
    parser.add_argument(
        "--allow-low-transfer",
        action="store_true",
        help="Allow a pretrained run with under 10%% matching state items.",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_path = (ROOT / args.model_yaml).resolve()
    data_yaml = (ROOT / args.data).resolve()
    if not yaml_path.exists():
        raise SystemExit(f"Missing model YAML: {args.model_yaml}")
    if not data_yaml.exists():
        raise SystemExit(f"Missing data YAML: {args.data}")
    pretrained_arg = args.pretrained.strip()
    pretrained_path = None if pretrained_arg.lower() in {"", "none", "null"} else (ROOT / pretrained_arg).resolve()
    if pretrained_path is not None and not pretrained_path.exists():
        raise SystemExit(f"Missing pretrained checkpoint: {args.pretrained}")
    safe_load_yaml(yaml_path)
    safe_load_yaml(data_yaml)

    base = args.name or f"module_{module_name(yaml_path)}_japan7_e{args.epochs}_img{args.imgsz}_b{args.batch}_seed42"
    run_name = unique_name(base)
    run_dir = PROJECT / run_name
    command = " ".join(sys.argv)
    save_repro_files(run_dir, data_yaml, command, pretrained_path)

    row = {field: "" for field in FIELDS}
    row.update(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        yaml_path=str(yaml_path.relative_to(ROOT)).replace("\\", "/"),
        pretrained=str(pretrained_path.relative_to(ROOT)).replace("\\", "/") if pretrained_path else "scratch",
        run_name=run_name,
        run_dir=str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        status="ERROR",
        oom=False,
        nan_detected=False,
        loss_decreased=False,
        map50_nonzero=False,
    )

    try:
        from ultralytics import YOLO

        model = YOLO(str(yaml_path))
        if pretrained_path is not None:
            model.load(str(pretrained_path))
            transferred, total = transfer_stats(model)
            ratio = transferred / total if total else 0.0
            row.update(transferred_items=transferred, transfer_total_items=total, transfer_ratio=f"{ratio:.6f}")
            save_transfer_metadata(run_dir, pretrained_path, transferred, total)
            if ratio < MIN_TRANSFER_RATIO and not args.allow_low_transfer:
                raise RuntimeError(
                    f"Only {transferred}/{total} checkpoint items match this YAML. "
                    "Use architecture-native weights, train with --pretrained none, or explicitly pass --allow-low-transfer."
                )
        row["params_if_available"], row["flops_if_available"] = model_stats(model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=42,
            amp=not args.no_amp,
            project=str(PROJECT),
            name=run_name,
            # save_repro_files() intentionally creates this unique directory before training.
            exist_ok=True,
            plots=False,
        )
        row["status"] = "COMPLETED"
    except RuntimeError as e:
        msg = short(e)
        row.update(error_type=type(e).__name__, error_message_short=msg, oom="out of memory" in msg.lower())
        row["status"] = "OOM" if row["oom"] else "RUNTIME_ERROR"
    except Exception as e:
        row.update(status="ERROR", error_type=type(e).__name__, error_message_short=short(e))

    results_csv = run_dir / "results.csv"
    checks = read_results(results_csv)
    row.update(checks)
    row.update(
        results_csv_exists=results_csv.exists(),
        best_pt_exists=(run_dir / "weights" / "best.pt").exists(),
        args_yaml_exists=(run_dir / "args.yaml").exists(),
    )
    if pretrained_path is not None and row["transfer_ratio"] and float(row["transfer_ratio"]) < MIN_TRANSFER_RATIO:
        row["recommended_next_step"] = "low transfer coverage: use architecture-native pretraining or label as scratch"
    elif row["status"] == "COMPLETED" and args.epochs >= 100:
        row["recommended_next_step"] = "review against the protocol-matched baseline"
    elif row["status"] == "COMPLETED" and row["map50_nonzero"] and row["loss_decreased"] and not row["nan_detected"]:
        row["recommended_next_step"] = "consider for 20/30 epoch signal test"
    elif row["status"] == "OOM":
        row["recommended_next_step"] = f"retry smaller batch, e.g. {max(1, args.batch // 2)}"
    else:
        row["recommended_next_step"] = "review before promotion"
    append_row(row)
    print(f"Wrote {PILOT_CSV}")
    print(f"Wrote {PILOT_MD}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
