#!/usr/bin/env python3
"""
build_remapped_yolo_dataset.py — Remap class IDs and drop unwanted classes.

Reads a YOLO dataset, applies a class mapping YAML, and writes a new dataset.

Usage:
    python scripts/build_remapped_yolo_dataset.py \
        --src F:/deeplearning/YOLO26-probe/datasets/japan_yolo \
        --dst F:/deeplearning/YOLO26-probe/datasets_derived/japan7 \
        --mapping configs/mappings/japan7.yaml \
        --mode copy \
        --keep-empty

    # Remote GPU (symlink):
    python scripts/build_remapped_yolo_dataset.py \
        --src /yolo26-probe/japan_yolo \
        --dst /yolo26-probe/derived/japan7 \
        --mapping configs/mappings/japan7.yaml \
        --mode symlink \
        --keep-empty
"""

import argparse
import shutil
import sys
from pathlib import Path
from collections import defaultdict

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


MODE_HANDLERS = {
    "copy": shutil.copy2,
    "symlink": lambda src, dst: dst.symlink_to(src.resolve()),
    "hardlink": lambda src, dst: (dst.unlink(missing_ok=True), dst.hardlink_to(src))[0] or dst,
}


def main():
    parser = argparse.ArgumentParser(description="Remap YOLO dataset classes")
    parser.add_argument("--src", required=True, help="Source dataset root")
    parser.add_argument("--dst", required=True, help="Destination dataset root")
    parser.add_argument("--mapping", required=True, help="Class mapping YAML")
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument("--keep-empty", action="store_true", default=True,
                        help="Keep images with no remaining labels (background)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    mapping = load_yaml(args.mapping)

    class_map = {int(k): int(v) for k, v in mapping["class_map"].items()}
    drop_classes = {int(k): v for k, v in mapping.get("drop_classes", {}).items()}
    target_names = mapping.get("target_names", {})
    target_nc = len(target_names)

    link_fn = MODE_HANDLERS.get(args.mode, shutil.copy2)
    print(f"  Mode:    {args.mode}")
    print(f"  Mapping: {len(class_map)} classes remapped, {len(drop_classes)} dropped")
    print(f"  Target:  {target_nc} classes ({list(target_names.values())})")
    print()

    stats_all = {}

    for split in ["train", "val"]:
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        dst_img_dir = dst_root / "images" / split
        dst_lbl_dir = dst_root / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.is_dir():
            print(f"  [{split}] SKIP — {src_img_dir} not found")
            continue

        count_images = 0
        count_empty = 0
        src_box_counts = defaultdict(int)
        tgt_box_counts = defaultdict(int)
        dropped_counts = defaultdict(int)

        for img_path in sorted(src_img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            count_images += 1

            stem = img_path.stem
            src_lbl = src_lbl_dir / (stem + ".txt")
            dst_img = dst_img_dir / img_path.name
            dst_lbl = dst_lbl_dir / (stem + ".txt")

            # Copy/symlink image
            if not dst_img.exists():
                try:
                    link_fn(img_path, dst_img)
                except FileExistsError:
                    pass

            # Remap labels
            new_lines = []
            if src_lbl.exists():
                for line in src_lbl.read_text().strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    src_box_counts[cls_id] += 1

                    if cls_id in drop_classes:
                        dropped_counts[cls_id] += 1
                        continue
                    if cls_id in class_map:
                        new_id = class_map[cls_id]
                        parts[0] = str(new_id)
                        new_lines.append(" ".join(parts))
                        tgt_box_counts[new_id] += 1

            if new_lines or args.keep_empty:
                dst_lbl.write_text("\n".join(new_lines) + "\n" if new_lines else "")
            if not new_lines:
                count_empty += 1

        stats_all[split] = {
            "images": count_images,
            "empty": count_empty,
            "src_boxes": dict(src_box_counts),
            "tgt_boxes": dict(tgt_box_counts),
            "dropped": dict(dropped_counts),
        }

        print(f"  [{split}] images={count_images}  empty_labels={count_empty}")
        print(f"         src boxes by class: {dict(sorted(src_box_counts.items()))}")
        print(f"         tgt boxes by class: {dict(sorted(tgt_box_counts.items()))}")
        print(f"         dropped by class:   {dict(sorted(dropped_counts.items()))}")
        print()

    # Write dataset.yaml
    yaml_path = dst_root / "dataset.yaml"
    yaml_content = {
        "path": str(dst_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": target_nc,
        "names": [target_names[i] for i in range(target_nc)],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  Wrote {yaml_path}")
    print(f"  Done. Derived dataset at {dst_root.resolve()}")


if __name__ == "__main__":
    main()
