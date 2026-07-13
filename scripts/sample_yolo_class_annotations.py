#!/usr/bin/env python3
"""Sample labeled boxes from a YOLO dataset for manual annotation review."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import yaml

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--class-name", default="D10")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def safe_load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def class_id_for(data: dict, class_name: str) -> int:
    names = data.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    matches = [int(class_id) for class_id, name in names.items() if str(name) == class_name]
    if len(matches) != 1:
        raise ValueError(f"class name must match exactly once: {class_name}")
    return matches[0]


def resolve_split(data_yaml: Path, data: dict, split: str) -> list[Path]:
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    values = data.get(split)
    if not values:
        raise ValueError(f"dataset does not define split: {split}")
    values = values if isinstance(values, list) else [values]
    images = []
    for value in values:
        source = Path(value)
        source = source if source.is_absolute() else root / source
        if source.is_dir():
            images.extend(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        elif source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                images.append(candidate if candidate.is_absolute() else root / candidate)
        elif source.suffix.lower() in IMAGE_SUFFIXES:
            images.append(source)
        else:
            raise FileNotFoundError(f"unsupported or missing split source: {source}")
    # Keep derived-dataset symlink paths so labels come from the remapped dataset, not the source dataset.
    return sorted(set(path.absolute() for path in images))


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"image path does not contain an images directory: {image_path}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def target_boxes(images: list[Path], class_id: int) -> list[dict]:
    boxes = []
    for image_path in images:
        annotation = label_path(image_path)
        if not annotation.is_file():
            continue
        for line_number, line in enumerate(annotation.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) != 5 or int(float(fields[0])) != class_id:
                continue
            x, y, width, height = map(float, fields[1:])
            if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)) or width <= 0 or height <= 0:
                continue
            boxes.append(
                {
                    "image_path": image_path,
                    "label_path": annotation,
                    "line_number": line_number,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )
    return boxes


def crop_bounds(image_width: int, image_height: int, xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    box_width, box_height = max(x2 - x1, 1), max(y2 - y1, 1)
    crop_width = min(image_width, max(int(box_width * 1.5), 192))
    crop_height = min(image_height, max(int(box_height * 1.5), 192))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    left = max(0, min(center_x - crop_width // 2, image_width - crop_width))
    top = max(0, min(center_y - crop_height // 2, image_height - crop_height))
    return left, top, left + crop_width, top + crop_height


def write_sample(box: dict, output: Path, index: int, class_name: str) -> dict | None:
    image = cv2.imread(str(box["image_path"]))
    if image is None:
        return None
    image_height, image_width = image.shape[:2]
    center_x, center_y = box["x"] * image_width, box["y"] * image_height
    box_width, box_height = box["width"] * image_width, box["height"] * image_height
    xyxy = (
        max(0, round(center_x - box_width / 2)),
        max(0, round(center_y - box_height / 2)),
        min(image_width, round(center_x + box_width / 2)),
        min(image_height, round(center_y + box_height / 2)),
    )
    left, top, right, bottom = crop_bounds(image_width, image_height, xyxy)
    crop = image[top:bottom, left:right].copy()
    relative_box = (xyxy[0] - left, xyxy[1] - top, xyxy[2] - left, xyxy[3] - top)
    cv2.rectangle(crop, relative_box[:2], relative_box[2:], (0, 0, 255), 2)
    aspect_ratio = max(box_width / max(box_height, 1e-6), box_height / max(box_width, 1e-6))
    caption = f"{index:03d} {class_name} AR={aspect_ratio:.2f} line={box['line_number']}"
    cv2.rectangle(crop, (0, 0), (min(crop.shape[1], 390), 25), (0, 0, 0), -1)
    cv2.putText(crop, caption, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    filename = f"{index:03d}_{box['image_path'].stem}_line{box['line_number']}.jpg"
    if not cv2.imwrite(str(output / filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"failed to write sample: {output / filename}")
    return {
        "sample": index,
        "image_path": str(box["image_path"]),
        "label_path": str(box["label_path"]),
        "line_number": box["line_number"],
        "aspect_ratio": f"{aspect_ratio:.6f}",
        "sample_image": filename,
        "incomplete_box": "",
        "excess_background": "",
        "inconsistent_scale": "",
        "split_or_merge_issue": "",
        "nearby_missed_label": "",
        "notes": "",
    }


def self_check() -> None:
    assert crop_bounds(100, 80, (40, 30, 60, 50)) == (0, 0, 100, 80)
    assert crop_bounds(640, 480, (300, 200, 340, 240)) == (224, 124, 416, 316)
    print("self-check passed")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.samples < 1 or args.output is None:
        raise SystemExit("--samples must be positive and --output is required")
    data_yaml = Path(args.data)
    data_yaml = data_yaml if data_yaml.is_absolute() else Path.cwd() / data_yaml
    data_yaml = data_yaml.resolve()
    data = safe_load(data_yaml)
    class_id = class_id_for(data, args.class_name)
    boxes = target_boxes(resolve_split(data_yaml, data, args.split), class_id)
    if len(boxes) < args.samples:
        raise SystemExit(f"requested {args.samples} boxes but found only {len(boxes)}")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    random.Random(args.seed).shuffle(boxes)
    rows = []
    for box in boxes:
        row = write_sample(box, output, len(rows) + 1, args.class_name)
        if row is not None:
            rows.append(row)
        if len(rows) == args.samples:
            break
    if len(rows) != args.samples:
        raise SystemExit(f"only {len(rows)} readable samples were written")

    manifest = output / "review_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} {args.class_name} samples to {output}")
    print(f"Review manifest: {manifest}")


if __name__ == "__main__":
    main()
