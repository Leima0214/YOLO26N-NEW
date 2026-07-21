#!/usr/bin/env python3
"""Build a deterministic scene-grouped Japan7 split from the existing remapped dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from audit_dataset_integrity import phash, sequence_key, sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/yolo26-probe/derived/japan7")
OUTPUT_ROOT = REPO_ROOT / "configs/japan7_v2_scene_disjoint"
OLD_AUDIT = REPO_ROOT / "reports/dataset_integrity_and_leakage_audit.json"
WINDOW_SIZE = 50
CLASS_NAMES = ["D00", "D10", "D20", "D40", "D43", "D44", "D50"]
METRIC_NAMES = ["images", "empty_labels", *(f"class_{name}" for name in CLASS_NAMES), "small", "medium", "large"]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path
    source_split: str
    prefix: str
    number: int
    digest: str
    perceptual_hash: int
    metrics: tuple[int, ...]


def label_path(image: Path) -> Path:
    return Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")


def label_metrics(path: Path) -> tuple[int, ...]:
    if not path.is_file():
        raise ValueError(f"Missing label: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    classes = [0] * len(CLASS_NAMES)
    sizes = [0, 0, 0]
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed label line: {path}: {line}")
        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:5])
        except ValueError as error:
            raise ValueError(f"Malformed label line: {path}: {line}") from error
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"Bad class id {class_id}: {path}")
        values = (x, y, width, height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite box: {path}: {line}")
        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < width <= 1
            and 0 < height <= 1
            and x - width / 2 >= -1e-6
            and y - height / 2 >= -1e-6
            and x + width / 2 <= 1 + 1e-6
            and y + height / 2 <= 1 + 1e-6
        ):
            raise ValueError(f"Out-of-bounds box: {path}: {line}")
        classes[class_id] += 1
        area_640 = width * height * 640 * 640
        sizes[0 if area_640 < 32**2 else 1 if area_640 < 96**2 else 2] += 1
    return (1, int(not lines), *classes, *sizes)


def collect_records() -> list[Record]:
    records = []
    for split in ("train", "val"):
        for image in sorted((SOURCE_ROOT / "images" / split).iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if not image.exists():
                raise ValueError(f"Broken image link: {image}")
            key = sequence_key(image)
            if key is None:
                raise ValueError(f"Filename lacks a numeric sequence id: {image.name}")
            label = label_path(image)
            records.append(
                Record(
                    image=image,
                    label=label,
                    source_split=split,
                    prefix=key[0],
                    number=key[1],
                    digest=sha256(image),
                    perceptual_hash=phash(image),
                    metrics=label_metrics(label),
                )
            )
    if len({record.image.name for record in records}) != len(records):
        raise ValueError("Duplicate basenames exist in the combined source dataset")
    return records


def pairs_within_hamming_two(records: list[Record]) -> set[tuple[int, int]]:
    by_hash: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_hash[record.perceptual_hash].append(index)
    pairs = set()
    masks = [1 << left for left in range(64)]
    masks.extend((1 << left) | (1 << right) for left in range(64) for right in range(left + 1, 64))
    for value, indices in by_hash.items():
        for position, left in enumerate(indices):
            pairs.update((left, right) for right in indices[position + 1 :])
        for mask in masks:
            neighbor = value ^ mask
            if neighbor <= value or neighbor not in by_hash:
                continue
            pairs.update((left, right) for left in indices for right in by_hash[neighbor])
    return pairs


def add_vectors(left: tuple[int, ...], right: tuple[int, ...], sign: int = 1) -> tuple[int, ...]:
    return tuple(a + sign * b for a, b in zip(left, right))


def sum_vectors(vectors) -> tuple[int, ...]:
    result = (0,) * len(METRIC_NAMES)
    for vector in vectors:
        result = add_vectors(result, vector)
    return result


def assign_groups(group_metrics: dict[int, tuple[int, ...]], target: tuple[float, ...]) -> set[int]:
    weights = (10.0, 3.0, *(3.0 for _ in CLASS_NAMES), 2.0, 2.0, 2.0)

    def score(values: tuple[int, ...]) -> float:
        return sum(weight * ((value - wanted) / max(wanted, 1.0)) ** 2 for value, wanted, weight in zip(values, target, weights))

    selected: set[int] = set()
    current = (0,) * len(METRIC_NAMES)
    remaining = set(group_metrics)
    while current[0] < target[0]:
        choice = min(remaining, key=lambda group: score(add_vectors(current, group_metrics[group])))
        selected.add(choice)
        remaining.remove(choice)
        current = add_vectors(current, group_metrics[choice])

    for _ in range(30):
        best = (score(current), None, None, current)
        for old in selected:
            without_old = add_vectors(current, group_metrics[old], -1)
            for new in remaining:
                candidate = add_vectors(without_old, group_metrics[new])
                candidate_score = score(candidate)
                if candidate_score < best[0] - 1e-12:
                    best = (candidate_score, old, new, candidate)
        if best[1] is None:
            break
        _, old, new, current = best
        selected.remove(old)
        remaining.add(old)
        remaining.remove(new)
        selected.add(new)
    return selected


def stats_row(version: str, split: str, vector: tuple[int, ...]) -> dict[str, int | str]:
    return {"version": version, "split": split, **dict(zip(METRIC_NAMES, vector))}


def main() -> None:
    records = collect_records()
    path_to_index = {str(record.image): index for index, record in enumerate(records)}
    union = UnionFind(len(records))

    window_representative = {}
    for index, record in enumerate(records):
        key = (record.prefix, record.number // WINDOW_SIZE)
        if key in window_representative:
            union.union(index, window_representative[key])
        else:
            window_representative[key] = index

    by_digest: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_digest[record.digest].append(index)
    exact_pairs = set()
    for indices in by_digest.values():
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                exact_pairs.add((left, right))
                union.union(left, right)

    near_pairs = pairs_within_hamming_two(records)
    for left, right in near_pairs:
        union.union(left, right)

    old_audit = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))
    confirmed_pairs = []
    for pair in old_audit.get("near_duplicate_pairs_phash_le_2", []):
        left, right = path_to_index[pair["train"]], path_to_index[pair["val"]]
        union.union(left, right)
        confirmed_pairs.append((left, right))

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union.find(index)].append(index)
    group_metrics = {group: sum_vectors(records[index].metrics for index in indices) for group, indices in groups.items()}

    total = sum_vectors(record.metrics for record in records)
    old_val_count = sum(record.source_split == "val" for record in records)
    val_fraction = old_val_count / len(records)
    target = tuple(value * val_fraction for value in total)
    val_groups = assign_groups(group_metrics, target)
    split_by_index = {
        index: ("val" if group in val_groups else "train")
        for group, indices in groups.items()
        for index in indices
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split in ("train", "val"):
        paths = sorted(str(record.image) for index, record in enumerate(records) if split_by_index[index] == split)
        manifest = OUTPUT_ROOT / f"split_manifest_{split}.txt"
        manifest.write_text("\n".join(paths) + "\n", encoding="utf-8")
        manifests[split] = paths

    dataset_yaml = {
        "path": str(OUTPUT_ROOT),
        "train": "split_manifest_train.txt",
        "val": "split_manifest_val.txt",
        "nc": len(CLASS_NAMES),
        "names": dict(enumerate(CLASS_NAMES)),
    }
    (OUTPUT_ROOT / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    scene_groups = []
    for number, (group, indices) in enumerate(sorted(groups.items(), key=lambda item: min(records[i].number for i in item[1]))):
        scene_groups.append(
            {
                "group_id": f"scene_{number:04d}",
                "split": "val" if group in val_groups else "train",
                "metrics": dict(zip(METRIC_NAMES, group_metrics[group])),
                "base_windows": sorted({f"{records[i].prefix}{records[i].number // WINDOW_SIZE:06d}" for i in indices}),
                "members": [
                    {"image": str(records[i].image), "source_split": records[i].source_split, "sequence_id": records[i].number}
                    for i in sorted(indices, key=lambda value: records[value].number)
                ],
            }
        )
    scene_payload = {
        "dataset": "Japan7-v2-scene-disjoint",
        "method": {
            "filename_prefix_and_contiguous_id_window": WINDOW_SIZE,
            "capture_sequence_metadata_available": False,
            "exact_sha256_union": True,
            "perceptual_hash_union_max_hamming_distance": 2,
            "confirmed_oldsplit_near_duplicate_pairs_union": len(confirmed_pairs),
        },
        "group_count": len(scene_groups),
        "groups": scene_groups,
    }
    (OUTPUT_ROOT / "scene_groups.json").write_text(json.dumps(scene_payload, indent=2) + "\n", encoding="utf-8")

    old_stats = {
        split: sum_vectors(record.metrics for record in records if record.source_split == split) for split in ("train", "val")
    }
    new_stats = {
        split: sum_vectors(record.metrics for index, record in enumerate(records) if split_by_index[index] == split)
        for split in ("train", "val")
    }
    distribution_path = OUTPUT_ROOT / "class_distribution_before_after.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["version", "split", *METRIC_NAMES], lineterminator="\n")
        writer.writeheader()
        for version, values in (("oldsplit", old_stats), ("japan7_v2_scene_disjoint", new_stats)):
            for split in ("train", "val"):
                writer.writerow(stats_row(version, split, values[split]))

    train_set, val_set = set(manifests["train"]), set(manifests["val"])
    exact_cross = sum(split_by_index[left] != split_by_index[right] for left, right in exact_pairs)
    near_cross = sum(split_by_index[left] != split_by_index[right] for left, right in near_pairs)
    confirmed_cross = sum(split_by_index[left] != split_by_index[right] for left, right in confirmed_pairs)
    group_overlap = len({union.find(i) for i, split in split_by_index.items() if split == "train"} & {union.find(i) for i, split in split_by_index.items() if split == "val"})
    checks = {
        "all_source_images_preserved": len(train_set | val_set) == len(records),
        "train_val_file_overlap_zero": not (train_set & val_set),
        "exact_duplicate_cross_split_zero": exact_cross == 0,
        "phash_le_2_cross_split_zero": near_cross == 0,
        "confirmed_near_duplicate_cross_split_zero": confirmed_cross == 0,
        "scene_group_cross_split_zero": group_overlap == 0,
        "broken_image_or_label_zero": all(record.image.exists() and record.label.is_file() for record in records),
        "all_classes_in_both_splits": all(new_stats[split][2 + class_id] > 0 for split in ("train", "val") for class_id in range(len(CLASS_NAMES))),
        "labels_well_formed_and_class_ids_valid": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "status": status,
        "dataset": "Japan7-v2-scene-disjoint",
        "source_images": len(records),
        "train_images": len(train_set),
        "val_images": len(val_set),
        "val_fraction_before": val_fraction,
        "val_fraction_after": len(val_set) / len(records),
        "scene_group_count": len(groups),
        "exact_duplicate_pair_count": len(exact_pairs),
        "phash_le_2_pair_count": len(near_pairs),
        "confirmed_near_duplicate_pair_count": len(confirmed_pairs),
        "exact_duplicate_cross_split_count": exact_cross,
        "phash_le_2_cross_split_count": near_cross,
        "confirmed_near_duplicate_cross_split_count": confirmed_cross,
        "scene_group_cross_split_count": group_overlap,
        "checks": checks,
    }
    audit_path = OUTPUT_ROOT / "leakage_audit_v2.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    checksum_targets = [
        OUTPUT_ROOT / "split_manifest_train.txt",
        OUTPUT_ROOT / "split_manifest_val.txt",
        OUTPUT_ROOT / "scene_groups.json",
        distribution_path,
        audit_path,
        OUTPUT_ROOT / "dataset.yaml",
    ]
    (OUTPUT_ROOT / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_targets), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    if status != "PASS":
        raise SystemExit("Japan7-v2 audit failed")


if __name__ == "__main__":
    main()
