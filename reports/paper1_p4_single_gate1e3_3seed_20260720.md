# Paper 1 Japan7: P4 Single Gate-1e-3 three-seed record (2026-07-20)

## Scope

This is the preserved reference implementation for the Paper 1 OverLoCK line.
It keeps the pretrained YOLO26n backbone, neck, and detection head intact, and
replaces only the original P4/16 `C3k2` stage with `OverLoCKStage`.

- Model YAML: `ultralytics/cfg/models/26/yolo26n-OverLoCK-ProjectFit-P4-Gate1e3.yaml`
- Added operation: a residual OverLoCK local-plus-overview refinement after the
  pretrained-compatible P4 `C3k2` output.
- Gate: per-channel `LayerScale2d`, initialized to `1e-3`.
- Pretraining: `model.load("yolo26n.pt")`; original YOLO26n parameter shapes
  remain compatible and the new OverLoCK branch is the expected newly initialized part.

This is **not** the full OverLoCK backbone experiment.  It is a small,
pretrained-compatible P4 adapter.

## Fixed protocol

- Dataset: `configs/japan7_remote.yaml` (Japan7, 7 classes)
- Epochs / image size / batch: `30 / 640 / 32`
- Device / workers: `0 / 8`
- Optimizer: `auto`; `lr0=0.01`, `lrf=0.01`, `momentum=0.937`,
  `weight_decay=0.0005`, `warmup_epochs=3`
- Augmentation: `mosaic=1.0`, `close_mosaic=10`, `mixup=0.0`,
  `copy_paste=0.0`
- Other controls: `deterministic=True`, `amp=True`, `cos_lr=False`
- Paired seeds: `42`, `0`, `3447`

## Independently validated aggregate results

| Seed | B0 mAP50-95 | P4 Single mAP50-95 | Paired delta | Interpretation |
|---:|---:|---:|---:|---|
| 42 | 0.31799 | 0.31790 | -0.00009 | practical tie |
| 0 | 0.31932 | 0.32059 | +0.00127 | weak positive |
| 3447 | 0.31709 | 0.31769 | +0.00059 | weak positive |
| Mean | 0.31813 | 0.31872 | **+0.00059** | weak, not conclusive |

The P4 branch is non-destructive across these three paired runs (two weakly
positive results and one tie), but the mean gain is below the `+0.002` practical
effect threshold.  It is therefore the correct preserved OverLoCK reference,
not a final claimed accuracy improvement.

## Cost

| Model | Parameters | FLOPs |
|---|---:|---:|
| YOLO26n B0 | 2,376,201 | 5.2 GFLOPs |
| P4 Single Gate-1e-3 | 2,451,753 | 5.7 GFLOPs |
| Delta | +75,552 | +0.5 GFLOPs |

## Saved remote artifacts

The Git repository stores source code, configurations, reports, and diagnostic
scripts only.  Checkpoints and `runs/` remain outside Git and must be archived
separately.

- `runs/paper1/paper1_b0_pretrained_auto_linear_japan7_30e_seed3447/`
- `runs/paper1/paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_30e_seed3447/`
- P4 `best.pt` SHA256: seed0
  `66401d51eb8af5ba7017179b3db7ae5b66e4ac6e2feeb610de26e096cf00a47b`; seed42
  `efea3942b463e8d206805d27ed5234ae397653ee7c77519bf9ae077c32d779d9`; seed3447
  `bbdbb39483081d50ed96d24472754d5e278875fd373a0f4ebbab76a14770301b`.
- P4 seed3447 `last.pt` SHA256:
  `4086bda101dc9cc46e2c398cc801573f9f49f38db189548161a6bed6395828f2`.

## Files required to reproduce or diagnose this reference

- `ultralytics/nn/yolo26_2025_backbones/modules.py`
- `ultralytics/cfg/models/26/yolo26n-OverLoCK-ProjectFit-P4-Gate1e3.yaml`
- `scripts/verify_overlock_projectfit.py`
- `scripts/inspect_overlock_gate.py`
- `scripts/validate_japan7_checkpoint.py`
- `scripts/audit_d43_pair.py`

`train.py` is deliberately not part of this preservation commit because it is
currently being used as a switchable local launcher for another experiment.
Use the YAML path above with the fixed protocol instead.

## Next experiment gate

Do not stack another module directly onto this branch without a paired B0
comparison.  A new candidate should first pass build, AMP forward, one-batch
backward, and one-epoch smoke checks; then use the same three-seed protocol if
its seed-42 result is promising.
