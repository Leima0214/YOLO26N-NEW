# Paper 1 Formal Protocol Correction: 2026-07-10

## Finding

The original Japan7 YOLO26n baseline was trained from `YOLO("yolo26n.pt")` with AMP enabled. The original module runner instead used `YOLO(custom.yaml)`, which starts from random weights. This was a major confounder in the initial module comparison.

## Completed Scratch Archive

| model | initialization | AMP | best epoch | mAP50 | mAP50-95 | Params | FLOPs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26n baseline | `yolo26n.pt` | True | 100 | 0.62300 | 0.34100 | 2.376M | 5.2G |
| EMA_attention | scratch | False | 65 | 0.58704 | 0.31410 | 2.377M | 5.2G |
| CPUBoneNano-P2Lite | scratch | True | 82 | 0.58718 | 0.31373 | 3.672M | 6.6G |

EMA and P2Lite have near-identical scratch best scores despite different AMP modes, so AMP is not the leading explanation for the gap. The supported interpretation is that the models need a protocol-matched pretrained transfer before they can be compared with the baseline.

## Implementation Change

`scripts/train_module_pilot.py` now defaults to `--pretrained yolo26n.pt`. Ultralytics transfers only same-name, same-shape parameters; new or incompatible layers remain initialized for training. The script writes `pretrained.txt`, keeps the exact requested run directory with `exist_ok=True`, and records the initialization in the pilot report.

Use `--pretrained none` only when deliberately studying scratch training.

The remote project must contain the same ignored `yolo26n.pt` checkpoint at its root before a pretrained run starts. Do not add that file to Git.

## Transfer Audit

| custom YAML | transferred items from `yolo26n.pt` | interpretation |
| --- | ---: | --- |
| EMA_attention | 468 / 714 | eligible for a YOLO26n-pretrained fine-tune |
| CPUBoneNano-P2Lite | 8 / 881 | not a meaningful YOLO26n-pretrained fine-tune |

P2Lite replaces the original backbone, so the available YOLO26n checkpoint cannot initialize its central representation. The [official CPUBone repository](https://github.com/altair199797/CPUBone) publishes ImageNet classification checkpoints, but this project has not validated a compatible conversion into `CPUBoneP2BackboneYOLO`. Do not load those weights directly into this branch or describe P2Lite as pretrained until a separate conversion validation reports coverage and a build check.

## Next Queue

1. EMA_attention, 1 epoch pretrained+AMP smoke. Confirm a `Transferred 468/714 items` log line and `args.yaml` with `amp: true`.
2. EMA_attention, 30 epoch pretrained+AMP signal. Compare mAP50-95 and D00/D10 against the pretrained baseline.
3. Promote EMA to a new 100 epoch run only if the 30 epoch signal is positive.
4. Keep P2Lite, composites, and SPDConv paused. P2Lite requires a separately validated CPUBone checkpoint conversion.

## Evidence

- [Ultralytics fine-tuning guide](https://docs.ultralytics.com/guides/finetuning-guide) distinguishes pretrained fine-tuning from random initialization and documents automatic compatible-weight transfer.
- [Ultralytics model YAML guide](https://docs.ultralytics.com/guides/model-yaml-config) documents `model.load()` for custom YAMLs and notes that only matching layers load.
- [PyTorch TorchVision object-detection fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html) demonstrates retaining pretrained features while replacing incompatible task heads.
- [CPUBone official repository](https://github.com/altair199797/CPUBone) documents its separate ImageNet classification checkpoints; this does not itself prove compatibility with the project-specific P2Lite wrapper.

The research-lookup skill could not run because this environment has neither `PARALLEL_API_KEY` nor `OPENROUTER_API_KEY`; the sources above are primary official documentation.
