# B0 100e reuse audit

## Decision

**RERUN_REQUIRED**

No existing B0 Japan7 100e seed42 run directory or report was found under `/root/YOLO26N-NEW/runs` or `/root/YOLO26N-NEW/reports` on 2026-07-21. Therefore there is no artifact that can satisfy the exact-match requirements for code, weights, dataset split, seed, schedule, augmentation, and validation settings.

The required rerun must not start on the current split because `reports/dataset_integrity_and_leakage_audit.md` records visually confirmed train/val near-duplicate road scenes. After rebuilding the split by scene or sequence group, both B0 and P4 must start from the same audited `yolo26n.pt` and use the shared guarded 100e protocol.
