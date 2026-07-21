# Japan7 dataset integrity and leakage audit

- Decision: **FAIL_CONFIRMED_NEAR_DUPLICATE_LEAKAGE**
- Dataset root: `/yolo26-probe/derived/japan7`
- Train/val images: 8387 / 2119
- Train/val boxes: 19752 / 5000
- Train/val empty labels: 638 / 156
- Filename overlap: 0
- Exact cross-split content duplicates: 0
- Cross-split nearest pairs with pHash distance <=2: 8
- Adjacent-ID pairs with ID distance <=5 and pHash distance <=6: 0

## Per-class box counts

- Train: `{0: 3238, 1: 3192, 2: 4964, 3: 1818, 4: 598, 5: 3118, 6: 2824}`
- Val: `{0: 811, 1: 787, 2: 1234, 3: 425, 4: 138, 5: 877, 6: 728}`

## Interpretation

Exact duplicates or identical filenames stop training automatically. Perceptual similarity first requires visual review. For this run, `--confirm-near-duplicates` records the completed visual review: the pHash<=2 pairs show the same road scenes, structures, vehicles, and almost identical viewpoints across train and val, so formal 100e training must stop until the split is rebuilt by scene/sequence group.

Visual review gallery: `reports/dataset_leakage_gallery.jpg`
Machine-readable details: `reports/dataset_integrity_and_leakage_audit.json`
