# P4 Single 30e three-seed immutable summary

All aggregate and per-class values below are independent validations of each saved `best.pt` using the same Japan7 val split, `imgsz=640`, `batch=32`, `iou=0.7`, and `max_det=300`.

## Per-seed results

| Model | Seed | best.pt mAP50-95 | Curve-best epoch | Curve best | Last-10 mean |
|---|---:|---:|---:|---:|---:|
| B0 | 42 | 0.31799 | N/A | N/A | N/A |
| B0 | 0 | 0.31932 | 30 | 0.31928 | 0.31074 |
| B0 | 3447 | 0.31709 | 30 | 0.31706 | 0.30932 |
| P4 Single | 42 | 0.31790 | 30 | 0.31801 | 0.30878 |
| P4 Single | 0 | 0.32059 | 30 | 0.32047 | 0.31370 |
| P4 Single | 3447 | 0.31769 | 30 | 0.31760 | 0.30952 |

## Paired conclusion

- Highest single-seed value: P4 Single seed0 `0.32059`.
- B0 three-seed mean: `0.31813`.
- P4 Single three-seed mean: `0.31872`.
- Paired deltas seed42/0/3447: `-0.00009`, `+0.00127`, `+0.00059`.
- Mean paired delta: `+0.00059`; 2/3 seeds are positive and seed42 is a practical tie.
- Decision: repeatable weak positive only; this is not evidence of a material improvement.
- Learned Gate audits showed finite non-zero gradients and movement away from 1e-3, so the weak effect is not explained by gradient starvation.
- Cost: 2,376,201 -> 2,451,753 parameters (+3.18%); 5.2 -> 5.7 GFLOPs (+9.62%).
- B0 seed42 curve fields are unavailable: its `results.csv` now contains only one row although the independently validated 30e `best.pt` remains intact. The missing history is reported as N/A rather than reconstructed.

## Per-class best-checkpoint mean and sample standard deviation

| Class | B0 mean | B0 std | P4 mean | P4 std | Mean delta |
|---|---:|---:|---:|---:|---:|
| D00 | 0.18957 | 0.00341 | 0.18892 | 0.00313 | -0.00065 |
| D10 | 0.12763 | 0.00463 | 0.12503 | 0.00377 | -0.00260 |
| D20 | 0.34578 | 0.00193 | 0.34304 | 0.00237 | -0.00274 |
| D40 | 0.18262 | 0.00854 | 0.18760 | 0.00499 | +0.00498 |
| D43 | 0.53839 | 0.00288 | 0.54516 | 0.01579 | +0.00677 |
| D44 | 0.44066 | 0.01043 | 0.43471 | 0.00534 | -0.00595 |
| D50 | 0.40230 | 0.01498 | 0.40662 | 0.00610 | +0.00433 |

Mean-positive classes: D40, D43, D50. Mean-negative classes: D00, D10, D20, D44.
D10 and D20 decline in every paired seed; this is the most stable class-level warning.

## Validity caveat discovered on 2026-07-21

A later cross-split perceptual audit found visually confirmed near-duplicate road scenes across train and val. These values remain an immutable record of what was run, but they must not be interpreted as leakage-free generalization estimates.
