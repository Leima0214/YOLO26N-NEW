#!/usr/bin/env python3
"""Run one 3-epoch YOLO26 module pilot and append a report row."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "experiments" / "module_scan"
PILOT_CSV = OUT_DIR / "pilot_report.csv"
PILOT_MD = OUT_DIR / "pilot_report.md"
PROJECT = ROOT / "runs" / "module_scan"
MIN_TRANSFER_RATIO = 0.80

FIELDS = [
    "timestamp",
    "yaml_path",
    "pretrained",
    "transferred_items",
    "transfer_total_items",
    "transfer_ratio",
    "transferred_numel",
    "transfer_total_numel",
    "transfer_numel_ratio",
    "backbone_transfer_ratio",
    "neck_transfer_ratio",
    "detect_transfer_ratio",
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


def safe_load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def module_name(yaml_path: Path) -> str:
    stem = yaml_path.stem
    return stem.removeprefix("yolo26-").replace("_", "-")


def validate_run_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) or name in {".", ".."}:
        raise ValueError("run name must be 1-128 characters using only letters, digits, dot, underscore, and hyphen")
    return name


def reserve_run(base: str) -> tuple[str, Path]:
    """Atomically reserve a unique run directory across concurrent processes."""
    PROJECT.mkdir(parents=True, exist_ok=True)
    base = validate_run_name(base)
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}_{attempt}"
        run_name = f"{base}{suffix}"
        run_dir = PROJECT / run_name
        try:
            run_dir.mkdir(exist_ok=False)
            return run_name, run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Unable to reserve a unique run directory for {base}")


@contextmanager
def report_lock(timeout=30.0):
    """Serialize report updates and recover locks left by dead processes."""
    lock_path = OUT_DIR / ".pilot_report.lock"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            break
        except FileExistsError:
            if lock_path.exists() and time.time() - lock_path.stat().st_mtime > 600:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for report lock: {lock_path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def run_text(command: list[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False).stdout
    except Exception as e:
        return f"{type(e).__name__}: {e}\n"


def save_repro_files(run_dir: Path, data_yaml: Path, command: str, pretrained: Path | None) -> None:
    if not run_dir.is_dir():
        raise RuntimeError(f"Run directory was not reserved: {run_dir}")
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
    code = (
        "import torch; "
        "print(f'torch={torch.__version__}'); "
        "print(f'cuda_available={torch.cuda.is_available()}'); "
        "print(f'cuda_version={torch.version.cuda}'); "
        "print(f'device_count={torch.cuda.device_count()}')"
    )
    return run_text([sys.executable, "-c", code])


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


PREFIX_PATTERN = re.compile(r"^model\.\d+(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|\d+))*$")


def validate_pretrained_map(mapping: object) -> list[tuple[str, str]]:
    """Validate and order a semantic state-dict prefix mapping."""
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("pretrained_map must be a non-empty mapping")
    if len(mapping) > 128:
        raise ValueError(f"pretrained_map has too many entries: {len(mapping)} > 128")
    pairs = []
    targets = set()
    for source, target in mapping.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("pretrained_map prefixes must be strings")
        if not PREFIX_PATTERN.fullmatch(source) or not PREFIX_PATTERN.fullmatch(target):
            raise ValueError(f"Unsafe pretrained_map prefix: {source!r} -> {target!r}")
        if target in targets:
            raise ValueError(f"Duplicate pretrained_map target prefix: {target}")
        targets.add(target)
        pairs.append((source, target))
    return sorted(pairs, key=lambda item: len(item[0]), reverse=True)


def remap_state_dict(source: dict, mapping: list[tuple[str, str]], keep_unmapped: bool) -> tuple[dict, int]:
    """Rename source keys atomically, rejecting collisions before model mutation."""
    renamed = {}
    changed = 0
    for key, value in source.items():
        mapped = None
        for source_prefix, target_prefix in mapping:
            if key == source_prefix or key.startswith(f"{source_prefix}."):
                mapped = f"{target_prefix}{key[len(source_prefix):]}"
                changed += mapped != key
                break
        if mapped is None:
            if not keep_unmapped:
                continue
            mapped = key
        if mapped in renamed:
            raise ValueError(f"Checkpoint mapping produced duplicate key: {mapped}")
        renamed[mapped] = value
    return renamed, changed


def transfer_coverage(model, matched: dict) -> dict[str, object]:
    """Report item and trainable-parameter coverage globally and by model region."""
    target_state = model.model.state_dict()
    target_parameters = dict(model.model.named_parameters())
    matched_keys = set(matched)
    backbone_layers = len(model.model.yaml["backbone"])
    detect_index = len(model.model.model) - 1

    def region(key):
        match = re.match(r"model\.(\d+)", key)
        if not match:
            return "other"
        index = int(match.group(1))
        return "backbone" if index < backbone_layers else "detect" if index == detect_index else "neck"

    total_numel = sum(parameter.numel() for parameter in target_parameters.values())
    matched_numel = sum(target_parameters[key].numel() for key in matched_keys if key in target_parameters)
    region_totals = {name: 0 for name in ("backbone", "neck", "detect")}
    region_matched = region_totals.copy()
    for key, parameter in target_parameters.items():
        name = region(key)
        if name in region_totals:
            region_totals[name] += parameter.numel()
            if key in matched_keys:
                region_matched[name] += parameter.numel()

    report = {
        "transferred_items": len(matched),
        "transfer_total_items": len(target_state),
        "transfer_ratio": len(matched) / len(target_state) if target_state else 0.0,
        "transferred_numel": matched_numel,
        "transfer_total_numel": total_numel,
        "transfer_numel_ratio": matched_numel / total_numel if total_numel else 0.0,
        "unmatched_keys": tuple(sorted(set(target_state) - matched_keys)),
    }
    for name in region_totals:
        total = region_totals[name]
        report[f"{name}_transfer_ratio"] = region_matched[name] / total if total else 1.0
    return report


def save_transfer_metadata(run_dir: Path, pretrained: Path, report: dict, checkpoint_remap: str) -> None:
    unmatched_keys = report["unmatched_keys"]
    (run_dir / "pretrained.txt").write_text(
        "\n".join(
            [
                str(pretrained),
                f"transferred_items={report['transferred_items']}",
                f"total_items={report['transfer_total_items']}",
                f"transfer_ratio={report['transfer_ratio']:.6f}",
                f"transferred_parameter_numel={report['transferred_numel']}",
                f"total_parameter_numel={report['transfer_total_numel']}",
                f"transfer_parameter_numel_ratio={report['transfer_numel_ratio']:.6f}",
                f"backbone_transfer_ratio={report['backbone_transfer_ratio']:.6f}",
                f"neck_transfer_ratio={report['neck_transfer_ratio']:.6f}",
                f"detect_transfer_ratio={report['detect_transfer_ratio']:.6f}",
                f"checkpoint_remap={checkpoint_remap or 'none'}",
                f"remapped_items={report['remapped_items']}",
                f"unmatched_target_items={len(unmatched_keys)}",
                *[f"unmatched_target_key={key}" for key in unmatched_keys],
                "",
            ]
        ),
        encoding="utf-8",
    )


def load_pretrained(model, pretrained: Path, checkpoint_remap: str, semantic_map: object = None) -> dict:
    """Load weights through a validated semantic map and return item/parameter coverage."""
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import intersect_dicts

    checkpoint_model, checkpoint = load_checkpoint(pretrained)
    source = checkpoint_model.float().state_dict()
    target = model.model.state_dict()
    mode = checkpoint_remap.strip().lower()
    if semantic_map and mode != "auto":
        raise ValueError("This YAML defines pretrained_map; legacy/manual remapping is forbidden. Use --checkpoint-remap auto.")
    if mode == "auto" and semantic_map:
        mapping = validate_pretrained_map(semantic_map)
        renamed, remapped_items = remap_state_dict(source, mapping, keep_unmapped=False)
    elif mode in {"", "auto", "none"}:
        renamed, remapped_items = source, 0
    else:
        try:
            source_prefix, target_prefix = checkpoint_remap.split(":", 1)
        except ValueError as error:
            raise ValueError("--checkpoint-remap must be auto or SOURCE_PREFIX:TARGET_PREFIX") from error
        mapping = validate_pretrained_map({source_prefix: target_prefix})
        renamed, remapped_items = remap_state_dict(source, mapping, keep_unmapped=True)
    matched = intersect_dicts(renamed, target)
    model.model.load_state_dict(matched, strict=False)
    model.ckpt = checkpoint
    model.overrides["pretrained"] = str(pretrained)
    report = transfer_coverage(model, matched)
    report["remapped_items"] = remapped_items
    return report


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
        "| yaml_path | pretrained | item transfer | parameter transfer | run_name | status | mAP50 | mAP50-95 | OOM | NaN | loss_decreased | next_step |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {yaml_path} | {pretrained} | {transferred_items}/{transfer_total_items} ({transfer_ratio}) | "
            "{transferred_numel}/{transfer_total_numel} ({transfer_numel_ratio}) | "
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
    with report_lock():
        ensure_templates()
        with PILOT_CSV.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        with PILOT_MD.open("a", encoding="utf-8") as f:
            f.write(
                "| {yaml_path} | {pretrained} | {transferred_items}/{transfer_total_items} ({transfer_ratio}) | "
                "{transferred_numel}/{transfer_total_numel} ({transfer_numel_ratio}) | "
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
        help="Allow a pretrained run with under 80%% matching trainable parameter values.",
    )
    parser.add_argument(
        "--checkpoint-remap",
        default="auto",
        help="Use the YAML semantic map (auto) or one legacy SOURCE_PREFIX:TARGET_PREFIX remap.",
    )
    parser.add_argument(
        "--expect-transfer",
        default="",
        help="Abort before training unless transfer coverage equals ITEMS/TOTAL, e.g. 708/714.",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def resolve_within_root(value: str, label: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} must stay inside the project root: {value}")
    return path


def main() -> None:
    args = parse_args()
    if not 1 <= args.epochs <= 1000:
        raise SystemExit("--epochs must be in [1, 1000]")
    if not 32 <= args.imgsz <= 4096:
        raise SystemExit("--imgsz must be in [32, 4096]")
    if not 1 <= args.batch <= 1024:
        raise SystemExit("--batch must be in [1, 1024]")
    if not 0 <= args.workers <= 128:
        raise SystemExit("--workers must be in [0, 128]")
    yaml_path = resolve_within_root(args.model_yaml, "model YAML")
    data_yaml = resolve_within_root(args.data, "data YAML")
    if not yaml_path.exists():
        raise SystemExit(f"Missing model YAML: {args.model_yaml}")
    if not data_yaml.exists():
        raise SystemExit(f"Missing data YAML: {args.data}")
    pretrained_arg = args.pretrained.strip()
    pretrained_path = (
        None if pretrained_arg.lower() in {"", "none", "null"} else resolve_within_root(pretrained_arg, "checkpoint")
    )
    if pretrained_path is not None and not pretrained_path.exists():
        raise SystemExit(f"Missing pretrained checkpoint: {args.pretrained}")
    model_config = safe_load_yaml(yaml_path)
    safe_load_yaml(data_yaml)

    base = args.name or f"module_{module_name(yaml_path)}_japan7_e{args.epochs}_img{args.imgsz}_b{args.batch}_seed42"
    run_name, run_dir = reserve_run(base)
    command = shlex.join(sys.argv)
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
            report = load_pretrained(model, pretrained_path, args.checkpoint_remap, model_config.get("pretrained_map"))
            row.update(
                transferred_items=report["transferred_items"],
                transfer_total_items=report["transfer_total_items"],
                transfer_ratio=f"{report['transfer_ratio']:.6f}",
                transferred_numel=report["transferred_numel"],
                transfer_total_numel=report["transfer_total_numel"],
                transfer_numel_ratio=f"{report['transfer_numel_ratio']:.6f}",
                backbone_transfer_ratio=f"{report['backbone_transfer_ratio']:.6f}",
                neck_transfer_ratio=f"{report['neck_transfer_ratio']:.6f}",
                detect_transfer_ratio=f"{report['detect_transfer_ratio']:.6f}",
            )
            save_transfer_metadata(run_dir, pretrained_path, report, args.checkpoint_remap)
            print(
                f"Checkpoint transfer: items={report['transferred_items']}/{report['transfer_total_items']}, "
                f"parameter_numel={report['transferred_numel']}/{report['transfer_total_numel']}, "
                f"backbone={report['backbone_transfer_ratio']:.3%}, neck={report['neck_transfer_ratio']:.3%}, "
                f"detect={report['detect_transfer_ratio']:.3%}"
            )
            actual_transfer = f"{report['transferred_items']}/{report['transfer_total_items']}"
            if args.expect_transfer and args.expect_transfer != actual_transfer:
                raise RuntimeError(
                    f"Expected checkpoint transfer {args.expect_transfer}, got {actual_transfer}. Training aborted."
                )
            if report["transfer_numel_ratio"] < MIN_TRANSFER_RATIO and not args.allow_low_transfer:
                raise RuntimeError(
                    f"Only {report['transfer_numel_ratio']:.1%} of target parameter values match this YAML. "
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
    if pretrained_path is not None and row["transfer_numel_ratio"] and float(row["transfer_numel_ratio"]) < MIN_TRANSFER_RATIO:
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
    if row["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
