# Module Scan Workflow

This directory records buildability and pilot reports for the YOLO26 module queue.

## Reports

- `buildability_report.csv`: build-only scan results for selected YAMLs.
- `buildability_report.md`: markdown summary of the build-only scan.
- `pilot_report.csv`: one appended row per 3 epoch pilot.
- `pilot_report.md`: markdown pilot summary.

## Rules

`BUILD OK` is not an effective module result. It only means the YAML file can be loaded safely and `YOLO(yaml)` can construct the model.

A 3 epoch pilot only checks basic training stability:

- run completed
- no OOM
- no NaN
- `results.csv` exists
- `weights/best.pt` exists
- `args.yaml` exists
- loss decreases normally
- mAP50 is nonzero
- Params/FLOPs are not out of control

Pilot results are not paper results. Use them only to decide whether a module deserves a 20/30 epoch signal experiment.

100 epoch formal runs require a separate selection step and are limited to at most 1-2 models. Do not launch formal runs just because a YAML is buildable.

## Paper 1 Signal Update

Current 30 epoch single-module decision:

| module | mAP50 | mAP50-95 | params | FLOPs | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| EMA_attention | 0.202 | 0.0884 | 2.377M | 5.2G | 100e formal |
| CPUBoneNano-P2Lite | 0.153 | 0.0674 | 3.672M | 6.6G | 100e formal |
| SPDConv | 0.129 | 0.0516 | 2.600M | 1.5G | optional 100e |

Current 3 epoch composite pilot:

| model | Recall | mAP50 | mAP50-95 | params | FLOPs | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P2Lite | 0.0714 | 0.000306 | 0.0000811 | 3.672M | 6.6G | stronger than current composites |
| P2Lite + EMA | 0.0306 | 0.000128 | 0.0000280 | 3.673M | 6.6G | keep only as exploration |
| P2Lite + SPDConv + EMA | 0.0601 | 0.000230 | 0.0000569 | 4.254M | 7.7G | keep only as exploration |

Both composite YAMLs are trainable, but neither should be promoted to 30 epoch or 100 epoch in the current insertion design. Future composite work should revisit EMA/SPDConv placement.

## Frozen Build Failures

The current failed candidates are frozen on this branch:

- `yolo26-CARAFE.yaml`
- `yolo26-FDConv.yaml`
- `yolo26-ContextAggregation.yaml`

Their failures come from `mmcv` API compatibility. Fix them only on a separate `fix/mmcv-compat-module-build` branch.
