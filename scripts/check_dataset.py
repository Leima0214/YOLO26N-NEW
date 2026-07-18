#!/usr/bin/env python3
"""Fail-fast validation for a YOLO detection dataset."""

import argparse
import math
from collections import Counter
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def check_split(root: Path, split: str, nc: int) -> dict:
    image_dir, label_dir = root / "images" / split, root / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing split: {image_dir} or {label_dir}")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    stats = {
        "images": len(images),
        "missing_labels": 0,
        "empty_labels": 0,
        "bad_lines": 0,
        "bad_class_ids": set(),
        "out_of_bounds": 0,
        "duplicate_lines": 0,
        "class_counts": Counter(),
    }

    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            stats["missing_labels"] += 1
            continue

        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            stats["empty_labels"] += 1
            continue
        stats["duplicate_lines"] += len(lines) - len(set(lines))

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                stats["bad_lines"] += 1
                continue
            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                stats["bad_lines"] += 1
                continue

            stats["class_counts"][class_id] += 1
            if not 0 <= class_id < nc:
                stats["bad_class_ids"].add(class_id)
            finite = all(math.isfinite(v) for v in (x, y, w, h))
            inside = (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
                and x - w / 2 >= -1e-6
                and y - h / 2 >= -1e-6
                and x + w / 2 <= 1 + 1e-6
                and y + h / 2 <= 1 + 1e-6
            )
            if not finite or not inside:
                stats["out_of_bounds"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root, nc = Path(cfg["path"]), int(cfg["nc"])
    names = cfg["names"]
    print(f"root = {root.resolve()}")
    print(f"classes = {nc}")
    print(f"names = {list(names.values()) if isinstance(names, dict) else names}")

    all_ids = set()
    total_boxes = 0
    failed = False
    for split in ("train", "val"):
        stats = check_split(root, split, nc)
        all_ids.update(stats["class_counts"])
        total_boxes += sum(stats["class_counts"].values())
        print(f"{split} images = {stats['images']}")
        for key in ("missing_labels", "empty_labels", "bad_lines", "out_of_bounds", "duplicate_lines"):
            print(f"{split} {key} = {stats[key]}")
        print(f"{split} class_counts = {dict(sorted(stats['class_counts'].items()))}")
        failed |= bool(
            stats["missing_labels"]
            or stats["bad_lines"]
            or stats["bad_class_ids"]
            or stats["out_of_bounds"]
            or stats["duplicate_lines"]
        )

    broken_links = sum(1 for path in root.rglob("*") if path.is_symlink() and not path.exists())
    print(f"total boxes = {total_boxes}")
    print(f"class ids = {min(all_ids) if all_ids else 'N/A'}-{max(all_ids) if all_ids else 'N/A'}")
    print(f"broken links = {broken_links}")
    failed |= all_ids != set(range(nc)) or broken_links > 0
    if failed:
        raise SystemExit("Dataset check failed")


if __name__ == "__main__":
    main()
