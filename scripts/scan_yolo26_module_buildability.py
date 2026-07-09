#!/usr/bin/env python3
"""
scan_yolo26_module_buildability.py — Try to build YOLO26 variant models from YAMLs.

Reads a list of candidate YAML paths, calls YOLO(yaml) for each, and records:
  - Whether the file exists
  - Whether the model builds (no crash)
  - Error type and message if it fails
  - Estimated params / FLOPs if available

Output:
  experiments/module_scan/buildability_report.csv
  experiments/module_scan/buildability_report.md

Usage:
    python scripts/scan_yolo26_module_buildability.py
"""

import csv
import sys
import traceback
from pathlib import Path
from datetime import datetime

# ── Candidate YAMLs ────────────────────────────────────────────────────────────
CANDIDATES = {
    "paper1": [
        "ultralytics/cfg/models/26/yolo26-4D.yaml",
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
    ],
    "paper2": [
        "ultralytics/cfg/models/26/yolo26-HVIEnhanceStem.yaml",
        "ultralytics/cfg/models/26/yolo26-FFAFusion-Neck.yaml",
        "ultralytics/cfg/models/26/yolo26-ContextAggregation.yaml",
    ],
}

SKIPPED = [
    "yolo26-MambaYOLORG.yaml",
    "yolo26-MobileMamba-Backbone.yaml",
    "yolo26-SwinTransformer.yaml",
    "yolo26-XRestormerPP.yaml",
    "yolo26-EfficientViM-Backbone.yaml",
]

OUT_DIR = Path("experiments/module_scan")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def count_params(model) -> tuple:
    """Try to extract param count and FLOPs from model info."""
    try:
        info = model.info(detailed=False, verbose=False)
        return info
    except Exception:
        return None


def build_one(yaml_path: str) -> dict:
    """Attempt to build a model from a YAML path. Never trains."""
    result = {
        "yaml_path": yaml_path,
        "exists": False,
        "build_ok": False,
        "error_type": "",
        "error_message_short": "",
        "params_if_available": "",
        "flops_if_available": "",
        "recommended_next_step": "",
    }

    p = Path(yaml_path)
    if not p.exists():
        result["error_type"] = "FileNotFound"
        result["error_message_short"] = f"YAML file not found: {yaml_path}"
        result["recommended_next_step"] = "skip — file missing"
        return result

    result["exists"] = True

    try:
        from ultralytics import YOLO
        model = YOLO(str(p))
        result["build_ok"] = True
        result["error_type"] = "none"
        result["error_message_short"] = ""
        result["recommended_next_step"] = "pilot — 3-epoch smoke test"

        # Try to get param/FLOP info
        try:
            info = model.info(detailed=False, verbose=False)
        except Exception:
            info = None

        if info is not None:
            result["params_if_available"] = str(info)
    except ModuleNotFoundError as e:
        result["error_type"] = "ModuleNotFound"
        msg = str(e)[:200]
        result["error_message_short"] = msg
        result["recommended_next_step"] = f"skip — missing Python module: {msg.split(chr(39))[1] if chr(39) in msg else msg[:80]}"
    except ImportError as e:
        result["error_type"] = "ImportError"
        result["error_message_short"] = str(e)[:200]
        result["recommended_next_step"] = "skip — import chain broken"
    except KeyError as e:
        result["error_type"] = "KeyError"
        result["error_message_short"] = f"Missing module registration: {e}"
        result["recommended_next_step"] = "skip — module not registered in tasks.py"
    except AttributeError as e:
        result["error_type"] = "AttributeError"
        result["error_message_short"] = str(e)[:200]
        result["recommended_next_step"] = "skip — module class missing attribute"
    except TypeError as e:
        result["error_type"] = "TypeError"
        result["error_message_short"] = str(e)[:200]
        result["recommended_next_step"] = "skip — YAML structure / args mismatch"
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message_short"] = str(e)[:300]
        result["recommended_next_step"] = "investigate — unexpected error"

    return result


def main():
    # Flatten all unique YAMLs
    all_yamls = []
    seen = set()
    for paper, paths in CANDIDATES.items():
        for p in paths:
            if p not in seen:
                seen.add(p)
                all_yamls.append((paper, p))

    print(f"Scanning {len(all_yamls)} unique YAMLs...")
    print()

    rows = []
    build_ok = 0
    build_fail = 0

    for paper, yp in all_yamls:
        print(f"  [{paper}] {Path(yp).name:50s} ... ", end="", flush=True)
        row = build_one(yp)
        row["paper"] = paper
        rows.append(row)

        if row["build_ok"]:
            build_ok += 1
            print("BUILD OK")
        else:
            build_fail += 1
            print(f"FAIL ({row['error_type']})")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "buildability_report.csv"
    fieldnames = [
        "paper", "yaml_path", "exists", "build_ok", "error_type",
        "error_message_short", "params_if_available", "flops_if_available",
        "recommended_next_step",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # ── Write Markdown ─────────────────────────────────────────────────────────
    md_path = OUT_DIR / "buildability_report.md"
    lines = [
        "# Module Buildability Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total candidates: {len(rows)}",
        f"BUILD OK: {build_ok}",
        f"BUILD FAIL: {build_fail}",
        "",
        "## Skipped (not in this scan)",
        "",
    ]
    for s in SKIPPED:
        lines.append(f"- `{s}` — too heavy / not relevant for Paper 1/2")
    lines.append("")

    # Paper 1
    lines.append("## Paper 1 Candidates")
    lines.append("")
    lines.append("| YAML | Build | Error | Next Step |")
    lines.append("| --- | --- | --- | --- |")
    for r in rows:
        if r["paper"] == "paper1":
            status = "✅ OK" if r["build_ok"] else f"❌ {r['error_type']}"
            lines.append(f"| {Path(r['yaml_path']).name} | {status} | {r['error_message_short'][:80]} | {r['recommended_next_step']} |")

    # Paper 2
    lines.append("")
    lines.append("## Paper 2 Candidates")
    lines.append("")
    lines.append("| YAML | Build | Error | Next Step |")
    lines.append("| --- | --- | --- | --- |")
    for r in rows:
        if r["paper"] == "paper2":
            status = "✅ OK" if r["build_ok"] else f"❌ {r['error_type']}"
            lines.append(f"| {Path(r['yaml_path']).name} | {status} | {r['error_message_short'][:80]} | {r['recommended_next_step']} |")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| | Paper 1 | Paper 2 |")
    lines.append(f"| --- | ---: | ---: |")
    p1_ok = sum(1 for r in rows if r["paper"] == "paper1" and r["build_ok"])
    p1_fail = sum(1 for r in rows if r["paper"] == "paper1" and not r["build_ok"])
    p2_ok = sum(1 for r in rows if r["paper"] == "paper2" and r["build_ok"])
    p2_fail = sum(1 for r in rows if r["paper"] == "paper2" and not r["build_ok"])
    lines.append(f"| BUILD OK | {p1_ok} | {p2_ok} |")
    lines.append(f"| BUILD FAIL | {p1_fail} | {p2_fail} |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReports written:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"\nBuild OK: {build_ok} / {len(rows)}")


if __name__ == "__main__":
    main()
