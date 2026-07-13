#!/usr/bin/env python3
"""Summarize YOLO box geometry after square letterbox scaling."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FIELDS = [
    "split",
    "class_id",
    "class_name",
    "images",
    "boxes",
    "boxes_per_image",
    "median_width_px",
    "median_height_px",
    "p25_area_px2",
    "median_area_px2",
    "p75_area_px2",
    "small_lt_32sq_pct",
    "medium_32_to_96sq_pct",
    "large_ge_96sq_pct",
    "min_side_lt_8px_pct",
    "min_side_lt_16px_pct",
    "median_aspect_ratio",
    "aspect_ge_3_pct",
    "aspect_ge_5_pct",
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def label_dir_for(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    indexes = [index for index, part in enumerate(parts) if part == "images"]
    if not indexes:
        raise ValueError(f"Image directory has no 'images' component: {image_dir}")
    parts[indexes[-1]] = "labels"
    return Path(*parts)


def normalize_names(raw_names) -> dict[int, str]:
    if isinstance(raw_names, list):
        return dict(enumerate(map(str, raw_names)))
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    raise ValueError("data YAML must define names as a list or mapping")


def box_at_imgsz(image_width: int, image_height: int, normalized_width: float, normalized_height: float, imgsz: int):
    scale = min(imgsz / image_width, imgsz / image_height)
    width = normalized_width * image_width * scale
    height = normalized_height * image_height * scale
    return width, height


def summarize(split: str, class_id: int, class_name: str, records: list[tuple], image_count: int) -> dict:
    widths = [record[0] for record in records]
    heights = [record[1] for record in records]
    areas = [width * height for width, height in zip(widths, heights)]
    aspects = [max(width / height, height / width) for width, height in zip(widths, heights)]
    count = len(records)

    def percentage(matches: int) -> float:
        return 100.0 * matches / count if count else 0.0

    return {
        "split": split,
        "class_id": class_id,
        "class_name": class_name,
        "images": image_count,
        "boxes": count,
        "boxes_per_image": count / image_count if image_count else 0.0,
        "median_width_px": statistics.median(widths) if widths else 0.0,
        "median_height_px": statistics.median(heights) if heights else 0.0,
        "p25_area_px2": percentile(areas, 0.25),
        "median_area_px2": percentile(areas, 0.50),
        "p75_area_px2": percentile(areas, 0.75),
        "small_lt_32sq_pct": percentage(sum(area < 32**2 for area in areas)),
        "medium_32_to_96sq_pct": percentage(sum(32**2 <= area < 96**2 for area in areas)),
        "large_ge_96sq_pct": percentage(sum(area >= 96**2 for area in areas)),
        "min_side_lt_8px_pct": percentage(sum(min(width, height) < 8 for width, height in zip(widths, heights))),
        "min_side_lt_16px_pct": percentage(sum(min(width, height) < 16 for width, height in zip(widths, heights))),
        "median_aspect_ratio": statistics.median(aspects) if aspects else 0.0,
        "aspect_ge_3_pct": percentage(sum(aspect >= 3 for aspect in aspects)),
        "aspect_ge_5_pct": percentage(sum(aspect >= 5 for aspect in aspects)),
    }


def analyze_split(root: Path, entry: str, split: str, names: dict[int, str], imgsz: int):
    image_dir = Path(entry)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    image_dir = image_dir.resolve()
    label_dir = label_dir_for(image_dir)
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing split directories: {image_dir} or {label_dir}")

    records = defaultdict(list)
    class_images = defaultdict(set)
    totals = {"images": 0, "boxes": 0, "missing_labels": 0, "empty_labels": 0, "invalid_lines": 0}
    image_paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    totals["images"] = len(image_paths)

    for image_path in image_paths:
        relative = image_path.relative_to(image_dir)
        label_path = (label_dir / relative).with_suffix(".txt")
        if not label_path.exists():
            totals["missing_labels"] += 1
            continue
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            totals["empty_labels"] += 1
            continue
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            totals["invalid_lines"] += len(lines)
            continue

        for line in lines:
            fields = line.split()
            try:
                class_id = int(fields[0])
                center_x, center_y, normalized_width, normalized_height = map(float, fields[1:5])
            except (IndexError, ValueError):
                totals["invalid_lines"] += 1
                continue
            if (
                class_id not in names
                or not all(math.isfinite(value) for value in (center_x, center_y, normalized_width, normalized_height))
                or not (0 <= center_x <= 1 and 0 <= center_y <= 1)
                or not (0 < normalized_width <= 1 and 0 < normalized_height <= 1)
            ):
                totals["invalid_lines"] += 1
                continue
            width, height = box_at_imgsz(image_width, image_height, normalized_width, normalized_height, imgsz)
            records[class_id].append((width, height))
            class_images[class_id].add(relative.as_posix())
            totals["boxes"] += 1

    rows = [summarize(split, class_id, names[class_id], records[class_id], len(class_images[class_id])) for class_id in names]
    return rows, totals


def write_reports(rows: list[dict], totals: dict[str, dict], data_path: Path, imgsz: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"japan7_box_geometry_img{imgsz}.csv"
    md_path = output_dir / f"japan7_box_geometry_img{imgsz}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Japan7 Box Geometry Diagnostic",
        "",
        f"- data: `{data_path}`",
        f"- square letterbox size: `{imgsz}`",
        "- size bins are diagnostic COCO-style thresholds after letterbox scaling, not AP_small/AP_medium/AP_large results",
        "",
        "| split | class | images | boxes | median w/h | median area | small % | min side <16 % | median aspect | aspect >=3 % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['class_name']} | {row['images']} | {row['boxes']} | "
            f"{row['median_width_px']:.1f}/{row['median_height_px']:.1f} | {row['median_area_px2']:.1f} | "
            f"{row['small_lt_32sq_pct']:.1f} | {row['min_side_lt_16px_pct']:.1f} | "
            f"{row['median_aspect_ratio']:.2f} | {row['aspect_ge_3_pct']:.1f} |"
        )
    lines.extend(["", "## Split Integrity", "", "| split | images | boxes | missing labels | empty labels | invalid lines |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for split, values in totals.items():
        lines.append(
            f"| {split} | {values['images']} | {values['boxes']} | {values['missing_labels']} | "
            f"{values['empty_labels']} | {values['invalid_lines']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def self_check():
    width, height = box_at_imgsz(1920, 1080, 0.1, 0.2, 640)
    assert abs(width - 64) < 1e-9 and abs(height - 72) < 1e-9
    row = summarize("val", 0, "D00", [(8.0, 64.0), (64.0, 64.0)], 2)
    assert row["boxes"] == 2 and row["small_lt_32sq_pct"] == 50.0 and row["aspect_ge_5_pct"] == 50.0
    print("SELF_CHECK_OK")


def main():
    parser = argparse.ArgumentParser(description="Analyze YOLO box geometry after square letterbox scaling")
    parser.add_argument("--data", help="YOLO data YAML")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", default="experiments/dataset_diagnostics/japan7")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.data:
        parser.error("--data is required unless --self-check is used")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")

    data_path = Path(args.data).resolve()
    with data_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("data YAML must contain a mapping")
    names = normalize_names(config.get("names"))
    root = Path(config.get("path", ".")).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    rows = []
    totals = {}
    for split in ("train", "val"):
        split_rows, split_totals = analyze_split(root, config.get(split, f"images/{split}"), split, names, args.imgsz)
        rows.extend(split_rows)
        totals[split] = split_totals
    csv_path, md_path = write_reports(rows, totals, data_path, args.imgsz, Path(args.output_dir))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    for row in rows:
        if row["class_name"] in {"D00", "D10"}:
            print(
                f"{row['split']} {row['class_name']}: boxes={row['boxes']} small={row['small_lt_32sq_pct']:.1f}% "
                f"min_side_lt16={row['min_side_lt_16px_pct']:.1f}% aspect_ge3={row['aspect_ge_3_pct']:.1f}%"
            )


if __name__ == "__main__":
    main()
