# Next-candidate decision matrix

## Decision

The only candidate allowed to proceed is **Japan7 Scene/Sequence Group Split v1**. It is a data correction, not a new model module.

Reason: the 2026-07-21 audit found no exact file duplicates but found eight train/val nearest pairs with pHash distance <=2; visual inspection confirmed the same road scenes, structures, vehicles, markings, and nearly identical viewpoints. This activates first-principles case **E: data/evaluation validity before model changes**.

## Candidate ranking

| Priority | Candidate | Evidence required | Current evidence | Decision |
|---:|---|---|---|---|
| 0 | Scene/sequence-grouped Japan7 split | Cross-split exact and near-duplicate audit | Confirmed leakage | **SELECTED** |
| 1 | TIDE-style unified error diagnosis | Leakage-free paired predictions | Missing | Deferred until clean pair |
| 2 | Native P2 detection path | Small-object/short-side miss concentration | Missing | Deferred |
| 3 | NWD assignment/loss | IoU-driven errors concentrated in tiny boxes | Missing | Deferred |
| 4 | DeepCrack-inspired auxiliary line supervision | D10/D20 thin-crack continuity errors with usable masks | Missing | Deferred |
| 5 | Varifocal/GFL/uncertainty loss | Calibration or box-uncertainty evidence | Missing; partial repository overlap | Deferred |
| 6 | PCGrad or grouped classification experts | Stable negative class-gradient cosine similarity | Not measured on valid data | Deferred |
| 7 | More P4/attention/Laplacian stacking | Material clean-data P4 gain | Not established | Stop |

## Minimal single-variable design

1. Preserve all source images and labels; create a new derived manifest/symlink dataset, never overwrite the current split.
2. Build scene groups from exact SHA256 matches, visually reviewed pHash-near pairs, and contiguous capture-sequence windows. All members of a group must stay in one split.
3. Assign whole groups to train/val while preserving approximate 80/20 size and per-class box distributions.
4. Acceptance gates: zero basename overlap, zero SHA256 overlap, zero visually confirmed pHash<=2 cross-split pairs, zero group overlap, every class present in both splits, no broken links or invalid labels.
5. Run B0 and P4 from the same `yolo26n.pt`, first through build/AMP/backward smoke. Then execute the original paired 100e seed42 protocol without changing optimizer, LR, augmentation, epoch count, or validation settings.

## Pretraining and deployment

There is no model change, so pretrained mapping remains identical to the existing B0/P4 audit. Parameters, FLOPs, ONNX, and TensorRT behavior are unchanged. The cost is experimental: all previous metrics become historical results on a leaky split and cannot serve as clean generalization baselines.

## Stop rule

Do not start the prepared 100e scripts while `reports/dataset_integrity_and_leakage_audit.json` is not exactly `PASS`. The shared training entry enforces this rule programmatically.
