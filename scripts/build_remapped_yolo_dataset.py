#!/usr/bin/env python3
"""Build a remapped YOLO dataset with copied, hard-linked, or symlinked images."""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml


def link_image(src: Path, dst: Path, mode: str) -> None:
    if dst.is_symlink() and not dst.exists():
        dst.unlink()
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy")
    parser.add_argument("--keep-empty", action="store_true")
    args = parser.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    mapping = yaml.safe_load(Path(args.mapping).read_text(encoding="utf-8"))
    class_map = {int(k): int(v) for k, v in mapping["class_map"].items()}
    drop_classes = {int(k) for k in mapping.get("drop_classes", {})}
    target_names = {int(k): v for k, v in mapping["target_names"].items()}
    known_classes = set(class_map) | drop_classes

    for split in ("train", "val"):
        src_images, src_labels = src_root / "images" / split, src_root / "labels" / split
        if not src_images.is_dir() or not src_labels.is_dir():
            raise FileNotFoundError(f"Missing source split: {src_images} or {src_labels}")

        dst_images, dst_labels = dst_root / "images" / split, dst_root / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        images = empty = 0
        boxes = Counter()

        for image in sorted(p for p in src_images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            label = src_labels / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing source label: {label}")

            remapped = []
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if len(parts) < 5:
                    raise ValueError(f"Malformed label line in {label}: {line!r}")
                source_id = int(parts[0])
                if source_id not in known_classes:
                    raise ValueError(f"Unmapped class {source_id} in {label}")
                if source_id in class_map:
                    target_id = class_map[source_id]
                    parts[0] = str(target_id)
                    remapped.append(" ".join(parts))
                    boxes[target_id] += 1

            if remapped or args.keep_empty:
                link_image(image, dst_images / image.name, args.mode)
                (dst_labels / f"{image.stem}.txt").write_text(
                    "\n".join(remapped) + ("\n" if remapped else ""), encoding="utf-8"
                )
                images += 1
                empty += not remapped

        print(f"{split}: images={images} empty_labels={empty} boxes={dict(sorted(boxes.items()))}")

    dataset = {
        "path": str(dst_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(target_names),
        "names": [target_names[i] for i in range(len(target_names))],
    }
    (dst_root / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
