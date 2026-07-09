#!/usr/bin/env python3
"""Build-only scan for selected YOLO26 module YAMLs."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "experiments" / "module_scan"
CSV_PATH = OUT_DIR / "buildability_report.csv"
MD_PATH = OUT_DIR / "buildability_report.md"

CANDIDATES = [
    "ultralytics/cfg/models/26/yolo26-CPUBoneNano-P2Lite.yaml",
    "ultralytics/cfg/models/26/yolo26-CARAFE.yaml",
    "ultralytics/cfg/models/26/yolo26-BiFPN.yaml",
    "ultralytics/cfg/models/26/yolo26-BiFPN1.yaml",
    "ultralytics/cfg/models/26/yolo26-EMA_attention.yaml",
    "ultralytics/cfg/models/26/yolo26-SEAttention.yaml",
    "ultralytics/cfg/models/26/yolo26-CBAM.yaml",
    "ultralytics/cfg/models/26/yolo26-LaplacianConv.yaml",
    "ultralytics/cfg/models/26/yolo26-FDConv.yaml",
    "ultralytics/cfg/models/26/yolo26-SPDConv.yaml",
    "ultralytics/cfg/models/26/yolo26-FFAFusion-Neck.yaml",
    "ultralytics/cfg/models/26/yolo26-HVIEnhanceStem.yaml",
    "ultralytics/cfg/models/26/yolo26-ContextAggregation.yaml",
]

FIELDS = [
    "yaml_path",
    "exists",
    "build_ok",
    "error_type",
    "error_message_short",
    "params_if_available",
    "flops_if_available",
    "recommended_next_step",
]


def short(text: object, limit: int = 180) -> str:
    return " ".join(str(text).split())[:limit]


def safe_load_yaml(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        yaml.safe_load(f)


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
        if len(info) > 1:
            params = str(info[1])
        if len(info) > 3:
            flops = str(info[3])
    else:
        if info is not None and not params:
            params = short(info)
    return params, flops


def scan_one(yaml_path: str) -> dict[str, object]:
    row = {field: "" for field in FIELDS}
    row.update({"yaml_path": yaml_path, "exists": False, "build_ok": False})

    path = ROOT / yaml_path
    if not path.exists():
        row.update(
            error_type="FileNotFound",
            error_message_short="YAML file not found",
            recommended_next_step="skip: file missing",
        )
        return row

    row["exists"] = True
    try:
        safe_load_yaml(path)
        from ultralytics import YOLO

        model = YOLO(str(path))
        params, flops = model_stats(model)
        row.update(
            build_ok=True,
            error_type="",
            params_if_available=params,
            flops_if_available=flops,
            recommended_next_step="pilot: run 3 epoch single-module test",
        )
    except Exception as e:
        row.update(
            error_type=type(e).__name__,
            error_message_short=short(e),
            recommended_next_step="skip until build error is fixed",
        )
    return row


def write_csv(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, object]]) -> None:
    ok = sum(1 for r in rows if r["build_ok"])
    lines = [
        "# YOLO26 Module Buildability Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Candidates: {len(rows)}",
        f"Build OK: {ok}",
        f"Build failed: {len(rows) - ok}",
        "",
        "| yaml_path | exists | build_ok | error_type | error_message_short | params | flops | recommended_next_step |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            "| {yaml_path} | {exists} | {build_ok} | {error_type} | {error_message_short} | {params_if_available} | "
            "{flops_if_available} | {recommended_next_step} |".format(**r)
        )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [scan_one(path) for path in CANDIDATES]
    write_csv(rows)
    write_md(rows)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
