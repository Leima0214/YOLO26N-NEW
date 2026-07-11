# Module Scan Workflow

This directory records buildability and pilot reports for the YOLO26 module queue.

For a single-page baseline vs `30e+` Paper 1 comparison record, see [docs/paper1_japan7_yolo26n/experiment_record_30e_plus.md](../../docs/paper1_japan7_yolo26n/experiment_record_30e_plus.md).

## Reports

- `buildability_report.csv`: build-only scan results for selected YAMLs.
- `buildability_report.md`: markdown summary of the build-only scan.
- `pilot_report.csv`: one appended row per 3 epoch pilot.
- `pilot_report.md`: markdown pilot summary.
- `paper1_tiera_buildability_report.csv` / `.md`: build-only audit of the 12 Paper 1 Tier A three-module YAMLs.
- `paper1_tiera_adversarial_audit.csv` / `.md`: full-model forward/backward, identity initialization, fused inference, concurrency, malicious input, and semantic parameter-transfer audit.

## Formal Result Collection

`scripts/collect_results.py` accepts only an explicit CSV manifest; it never searches or guesses run directories. The manifest must contain `model,run_dir` and may add `initialization,protocol,params,flops`.

```bash
python scripts/collect_results.py --manifest experiments/run_manifest.csv
```

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

## Protocol Correction: 2026-07-10

The stored YOLO26n baseline is a COCO-pretrained `yolo26n.pt` fine-tune. Earlier module pilots and the two completed 100 epoch module runs constructed `YOLO(custom.yaml)` from scratch, so they are not directly comparable to that baseline.

`train_module_pilot.py` now defaults to `--pretrained yolo26n.pt`, records item and parameter-weighted transfer coverage in `pretrained.txt`, and atomically reserves a unique run directory. Use `--pretrained none` only for an explicitly labeled scratch experiment. Tier A YAMLs must use `--checkpoint-remap auto`; their embedded semantic mappings make numeric-prefix remaps invalid. Parameter coverage below 80% is rejected by default. Future formal comparisons must keep the same initialization, AMP mode, Japan7 split, seed, image size, batch, and epoch budget.

Scratch 100 epoch archive:

| module | AMP | best epoch | mAP50 | mAP50-95 | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| EMA_attention | False | 65 | 0.58704 | 0.31410 | archive; not comparable to pretrained baseline |
| CPUBoneNano-P2Lite | True | 82 | 0.58718 | 0.31373 | archive; not comparable to pretrained baseline |

## Paper 1 Signal Update

Historical 30 epoch scratch signal:

| module | mAP50 | mAP50-95 | params | FLOPs | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| EMA_attention | 0.202 | 0.0884 | 2.377M | 5.2G | rerun only under pretrained protocol |
| CPUBoneNano-P2Lite | 0.153 | 0.0674 | 3.672M | 6.6G | architecture-native pretraining required |
| SPDConv | 0.129 | 0.0516 | 2.600M | 1.5G | do not promote before fair signal |

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

## Tier A Composite Build Update: 2026-07-11

The unused `mmcv` imports in CARAFE and FDConv were removed, and the selected slimneck blocks were clean-ported with parser registration. The 12 generated Tier A composite YAMLs now build successfully; see `paper1_tiera_buildability_report.md`. This supersedes the frozen status for CARAFE and FDConv only within these audited composites. `ContextAggregation` remains frozen.

## Tier A Adversarial Correction: 2026-07-11

The first A01-A12 implementation was not launch-ready despite building. Attention/fusion paths were non-identity at initialization, FFA was over-inserted, FDConv violated the replaced Conv contract, BiFPN pulled an unrelated dependency graph, and item-count remapping did not prove meaningful pretrained inheritance.

The corrected generator and modules now pass `scripts/audit_paper1_tiera_models.py` for all 12 YAMLs. See `paper1_tiera_adversarial_audit.md` and `docs/paper1_japan7_yolo26n/paper1_tiera_adversarial_review.md`. Any result produced before this correction remains historical evidence for its old model definition and must not be attributed to the corrected YAML with the same filename.
