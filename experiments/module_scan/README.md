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

## Frozen Build Failures

The current failed candidates are frozen on this branch:

- `yolo26-CARAFE.yaml`
- `yolo26-FDConv.yaml`
- `yolo26-ContextAggregation.yaml`

Their failures come from `mmcv` API compatibility. Fix them only on a separate `fix/mmcv-compat-module-build` branch.
