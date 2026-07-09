#!/usr/bin/env python3
"""
check_dataset.py — Validate YOLO dataset before training.

Checks:
  - images/train and images/val exist
  - labels/train and labels/val exist (auto-derived)
  - image-label pairing
  - YOLO label format (>= 5 columns per line)
  - class ID distribution
  - out-of-range class IDs vs declared nc

Usage:
    python scripts/check_dataset.py --data configs/japan7_local.yaml
    python scripts/check_dataset.py --data configs/japan7_remote.yaml
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml(path: str) -> dict:
    """Load YAML file. Requires PyYAML."""
    if yaml is None:
        print("ERROR: PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def detect_class_names(dataset_root: Path, fallback_nc: int, fallback_names: dict) -> tuple:
    """Try to read nc/names from dataset_root/dataset.yaml or data.yaml.
    Falls back to the values from the config file if none found.
    """
    for candidate in ["dataset.yaml", "data.yaml"]:
        candidate_path = dataset_root / candidate
        if candidate_path.exists():
            try:
                ds_cfg = load_yaml(str(candidate_path))
                if "names" in ds_cfg and isinstance(ds_cfg["names"], list):
                    nc = len(ds_cfg["names"])
                    names = {i: n for i, n in enumerate(ds_cfg["names"])}
                    print(f"  [INFO] Loaded {nc} classes from {candidate_path}")
                    return nc, names
                if "nc" in ds_cfg:
                    fallback_nc = ds_cfg["nc"]
            except Exception:
                pass
    return fallback_nc, fallback_names


def check_split(img_dir: Path, lbl_dir: Path, declared_nc: int) -> dict:
    """Check one split (train or val)."""
    result = {
        "num_images": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "class_counts": defaultdict(int),
        "bad_lines": 0,
        "bad_class_ids": set(),
    }

    if not img_dir.is_dir():
        print(f"  [ERROR] Image directory not found: {img_dir}")
        return result

    img_files = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    result["num_images"] = len(img_files)

    for img_path in img_files:
        stem = img_path.stem
        lbl_path = lbl_dir / (stem + ".txt")

        if not lbl_path.exists():
            result["missing_labels"] += 1
            continue

        try:
            with open(lbl_path) as f:
                lines = f.readlines()
        except Exception:
            result["missing_labels"] += 1
            continue

        if not lines or all(l.strip() == "" for l in lines):
            result["empty_labels"] += 1
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                result["bad_lines"] += 1
                continue
            try:
                cls_id = int(parts[0])
            except ValueError:
                result["bad_lines"] += 1
                continue

            if cls_id < 0 or cls_id >= declared_nc:
                result["bad_class_ids"].add(cls_id)
            result["class_counts"][cls_id] += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="Check YOLO dataset structure")
    parser.add_argument("--data", required=True, help="Path to YOLO data YAML")
    args = parser.parse_args()

    cfg = load_yaml(args.data)
    dataset_path = Path(cfg.get("path", "."))
    train_rel = cfg.get("train", "images/train")
    val_rel = cfg.get("val", "images/val")
    fallback_nc = cfg.get("nc", 5)
    fallback_names = cfg.get("names", {})

    train_img_dir = dataset_path / train_rel
    val_img_dir = dataset_path / val_rel
    train_lbl_dir = Path(str(train_img_dir).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))
    val_lbl_dir = Path(str(val_img_dir).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))

    nc, names = detect_class_names(dataset_path, fallback_nc, fallback_names)

    print("[DATA]", args.data)
    print("[ROOT]", dataset_path.resolve())
    print("[NAMES]")
    for i in sorted(names.keys()):
        print(f"  {i}: {names[i]}")
    print(f"[NC] {nc}")
    print()

    # --- Train ---
    print("[train]")
    train_stats = check_split(train_img_dir, train_lbl_dir, nc)
    print(f"  num_images:     {train_stats['num_images']}")
    print(f"  missing_labels: {train_stats['missing_labels']}")
    print(f"  empty_labels:   {train_stats['empty_labels']}")
    print(f"  bad_lines:      {train_stats['bad_lines']}")
    print(f"  class_counts:")
    for cid in sorted(train_stats["class_counts"].keys()):
        print(f"    class {cid}: {train_stats['class_counts'][cid]}")

    # --- Val ---
    print()
    print("[val]")
    val_stats = check_split(val_img_dir, val_lbl_dir, nc)
    print(f"  num_images:     {val_stats['num_images']}")
    print(f"  missing_labels: {val_stats['missing_labels']}")
    print(f"  empty_labels:   {val_stats['empty_labels']}")
    print(f"  bad_lines:      {val_stats['bad_lines']}")
    print(f"  class_counts:")
    for cid in sorted(val_stats["class_counts"].keys()):
        print(f"    class {cid}: {val_stats['class_counts'][cid]}")

    # --- All class IDs ---
    all_ids = set(train_stats["class_counts"].keys()) | set(val_stats["class_counts"].keys())
    bad_ids = train_stats["bad_class_ids"] | val_stats["bad_class_ids"]

    print()
    print("[ALL CLASS IDS]")
    print(f"  class id range: [{min(all_ids) if all_ids else 'N/A'}, {max(all_ids) if all_ids else 'N/A'}]")
    print(f"  unique ids:     {sorted(all_ids) if all_ids else 'N/A'}")
    if bad_ids:
        print(f"  bad class ids:  {sorted(bad_ids)}  <-- OUT OF RANGE (nc={nc})")
    else:
        print(f"  bad class ids:  []")

    if bad_ids:
        print(f"\nERROR: Found class IDs {sorted(bad_ids)} >= nc={nc}. Fix labels or update nc/names in config!")
        sys.exit(1)

    # Check for empty dataset
    total_imgs = train_stats["num_images"] + val_stats["num_images"]
    if total_imgs == 0:
        print("\nWARNING: No images found. Dataset directory may not exist yet.")
        print("This is expected if you are checking configs before uploading data to GPU.")
    else:
        total_boxes = sum(train_stats["class_counts"].values()) + sum(val_stats["class_counts"].values())
        print(f"\n  Total images:  {total_imgs}")
        print(f"  Total boxes:   {total_boxes}")

    print("\nCheck complete.")


if __name__ == "__main__":
    main()
