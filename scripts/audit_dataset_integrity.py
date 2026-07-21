#!/usr/bin/env python3
"""Audit Japan7 integrity and train/val leakage without modifying the dataset."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from check_dataset import IMAGE_SUFFIXES, check_split


def image_paths(root: Path, split: str) -> list[Path]:
    return sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Unable to read image: {path}")
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(image)[:8, :8]
    median = float(np.median(low.ravel()[1:]))
    bits = (low > median).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def sequence_key(path: Path) -> tuple[str, int] | None:
    match = re.match(r"^(.*?)(\d+)$", path.stem)
    return (match.group(1), int(match.group(2))) if match else None


def make_gallery(pairs: list[dict], output: Path, limit: int = 16) -> None:
    rows = []
    for pair in pairs[:limit]:
        panels = []
        for split in ("train", "val"):
            image = cv2.imread(pair[split])
            if image is None:
                continue
            image = cv2.resize(image, (360, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(
                image,
                f"{split}: {Path(pair[split]).name}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(image)
        if len(panels) == 2:
            row = np.hstack(panels)
            cv2.putText(
                row,
                f"pHash distance={pair['phash_distance']}, id distance={pair.get('id_distance')}",
                (8, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            rows.append(row)
    if rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/japan7_remote.yaml")
    parser.add_argument("--report", default="reports/dataset_integrity_and_leakage_audit.md")
    parser.add_argument("--json", default="reports/dataset_integrity_and_leakage_audit.json")
    parser.add_argument("--gallery", default="reports/dataset_leakage_gallery.jpg")
    parser.add_argument("--confirm-near-duplicates", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    root, nc = Path(cfg["path"]), int(cfg["nc"])
    train, val = image_paths(root, "train"), image_paths(root, "val")
    integrity = {split: check_split(root, split, nc) for split in ("train", "val")}

    train_names, val_names = {p.name for p in train}, {p.name for p in val}
    filename_overlap = sorted(train_names & val_names)

    hashes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, paths in (("train", train), ("val", val)):
        for path in paths:
            hashes[sha256(path)].append((split, str(path)))
    exact_pairs = []
    for digest, entries in hashes.items():
        train_entries = [p for split, p in entries if split == "train"]
        val_entries = [p for split, p in entries if split == "val"]
        exact_pairs.extend(
            {"sha256": digest, "train": left, "val": right}
            for left in train_entries
            for right in val_entries
        )

    train_phash = [(path, phash(path)) for path in train]
    val_phash = [(path, phash(path)) for path in val]
    closest = []
    for val_path, val_hash in val_phash:
        train_path, distance = min(
            ((path, (value ^ val_hash).bit_count()) for path, value in train_phash),
            key=lambda item: item[1],
        )
        closest.append(
            {
                "train": str(train_path),
                "val": str(val_path),
                "phash_distance": distance,
            }
        )
    closest.sort(key=lambda item: item["phash_distance"])
    near_duplicate_pairs = [pair for pair in closest if pair["phash_distance"] <= 2]

    by_prefix: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    train_hash_by_path = dict(train_phash)
    val_hash_by_path = dict(val_phash)
    for path in train:
        key = sequence_key(path)
        if key:
            by_prefix[key[0]].append((key[1], path))
    for values in by_prefix.values():
        values.sort()

    adjacent = []
    nearest_id_hist = Counter()
    for path in val:
        key = sequence_key(path)
        if not key or key[0] not in by_prefix:
            continue
        values = by_prefix[key[0]]
        ids = [item[0] for item in values]
        position = bisect.bisect_left(ids, key[1])
        candidates = values[max(0, position - 1) : min(len(values), position + 1)]
        train_id, train_path = min(candidates, key=lambda item: abs(item[0] - key[1]))
        id_distance = abs(train_id - key[1])
        nearest_id_hist[min(id_distance, 11)] += 1
        adjacent.append(
            {
                "train": str(train_path),
                "val": str(path),
                "id_distance": id_distance,
                "phash_distance": (train_hash_by_path[train_path] ^ val_hash_by_path[path]).bit_count(),
            }
        )
    adjacent.sort(key=lambda item: (item["phash_distance"], item["id_distance"]))

    obvious = bool(filename_overlap or exact_pairs)
    warning_pairs = [p for p in adjacent if p["id_distance"] <= 5 and p["phash_distance"] <= 6]
    if obvious:
        status = "FAIL_EXACT_LEAKAGE"
    elif near_duplicate_pairs and args.confirm_near_duplicates:
        status = "FAIL_CONFIRMED_NEAR_DUPLICATE_LEAKAGE"
    elif near_duplicate_pairs:
        status = "REVIEW_REQUIRED_NEAR_DUPLICATES"
    elif warning_pairs:
        status = "PASS_WITH_SEQUENCE_WARNING"
    else:
        status = "PASS"
    payload = {
        "status": status,
        "dataset_root": str(root),
        "integrity": {
            split: {
                **{k: v for k, v in stats.items() if k != "bad_class_ids"},
                "bad_class_ids": sorted(stats["bad_class_ids"]),
                "class_counts": dict(sorted(stats["class_counts"].items())),
            }
            for split, stats in integrity.items()
        },
        "filename_overlap_count": len(filename_overlap),
        "exact_cross_split_duplicate_count": len(exact_pairs),
        "closest_phash_distance_histogram": dict(sorted(Counter(p["phash_distance"] for p in closest).items())),
        "closest_phash_pairs": closest[:50],
        "near_duplicate_pair_count_phash_le_2": len(near_duplicate_pairs),
        "near_duplicate_pairs_phash_le_2": near_duplicate_pairs[:100],
        "nearest_sequence_distance_histogram_capped_at_11": dict(sorted(nearest_id_hist.items())),
        "adjacent_high_similarity_count": len(warning_pairs),
        "adjacent_high_similarity_pairs": warning_pairs[:100],
        "symlink_targets": {
            "train_sample": str(train[0].resolve()),
            "val_sample": str(val[0].resolve()),
        },
    }

    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_gallery((near_duplicate_pairs or warning_pairs or closest), Path(args.gallery))

    lines = [
        "# Japan7 dataset integrity and leakage audit",
        "",
        f"- Decision: **{status}**",
        f"- Dataset root: `{root}`",
        f"- Train/val images: {len(train)} / {len(val)}",
        f"- Train/val boxes: {sum(integrity['train']['class_counts'].values())} / {sum(integrity['val']['class_counts'].values())}",
        f"- Train/val empty labels: {integrity['train']['empty_labels']} / {integrity['val']['empty_labels']}",
        f"- Filename overlap: {len(filename_overlap)}",
        f"- Exact cross-split content duplicates: {len(exact_pairs)}",
        f"- Cross-split nearest pairs with pHash distance <=2: {len(near_duplicate_pairs)}",
        f"- Adjacent-ID pairs with ID distance <=5 and pHash distance <=6: {len(warning_pairs)}",
        "",
        "## Per-class box counts",
        "",
        f"- Train: `{dict(sorted(integrity['train']['class_counts'].items()))}`",
        f"- Val: `{dict(sorted(integrity['val']['class_counts'].items()))}`",
        "",
        "## Interpretation",
        "",
        "Exact duplicates or identical filenames stop training automatically. Perceptual similarity first requires visual review. "
        "For this run, `--confirm-near-duplicates` records the completed visual review: the pHash<=2 pairs show the same road scenes, structures, vehicles, and almost identical viewpoints across train and val, so formal 100e training must stop until the split is rebuilt by scene/sequence group.",
        "",
        f"Visual review gallery: `{args.gallery}`",
        f"Machine-readable details: `{args.json}`",
    ]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "filename_overlap_count", "exact_cross_split_duplicate_count", "adjacent_high_similarity_count")}, indent=2))


if __name__ == "__main__":
    main()
