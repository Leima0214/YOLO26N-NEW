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
- `paper1_tierb_adversarial_audit.csv` / `.md`: 12-model Tier B audit covering 640x640 gradients, transfer coverage, fusion, boundaries, concurrent generation/inference, recovery, and abuse resistance.
- `paper1_s4_wpformer_wdr_audit.md`: 640x640 numerical, gradient, baseline-equivalence, fusion, and transfer audit for the WCA-inspired single-module candidate.

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

`train_module_pilot.py` now defaults to `--pretrained yolo26n.pt`, records item and parameter-weighted transfer coverage in `pretrained.txt`, and atomically reserves a unique run directory. Use `--pretrained none` only for an explicitly labeled scratch experiment. Generated Tier A/Tier B YAMLs must use `--checkpoint-remap auto`; their embedded semantic mappings make numeric-prefix remaps invalid. Parameter coverage below 80% is rejected by default. Future formal comparisons must keep the same initialization, AMP mode, Japan7 split, seed, image size, batch, and epoch budget.

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

## Tier B Adversarial Build Update: 2026-07-12

`scripts/generate_paper1_tierb_composites.py` materializes B13-B24 with explicit layer assignment for repeated detail operators. `scripts/audit_paper1_tierb_models.py` passes all 12 YAMLs in normal and `python -O` modes, including 640x640 forward/backward, six boundary shapes per model, semantic pretrained transfer, real custom BatchNorm fusion, two-step gated-branch gradients, malformed-input recovery, fixed-shape concurrent inference, and concurrent atomic generation/reporting. No dataset training or CUDA smoke was performed.

The audit found and fixed axial FFA wrapping, FDConv duplicate convolution, CARAFE patch replication, unstable weighted concatenation, missing custom BN fusion, optimized-mode assertion loss, Windows concurrent replace, report atomicity, cross-tier generation mistakes, and untrusted Tier B checkpoint loading. The BiFPN-labelled YAMLs use BiFPN-style positive weighted concatenation, not classic additive BiFPN. THOP GFLOPs are lower-bound estimates; measured CUDA latency, FPS, peak VRAM, and iteration time are required before efficiency claims.

Tier B pilots require the SHA256-pinned project-root `yolo26n.pt`, `imgsz <= 640` divisible by 32, `batch <= 32`, and `--checkpoint-remap auto`. Runs preserve model/data YAML snapshots, hashes, Git status, and RUNNING/COMPLETED/FAILED markers. Remaining risks are scientific: P2 adds a new Detect branch, SPDConv cannot inherit the baseline stride-2 kernel, and overlapping roles may still reduce accuracy. See `paper1_tierb_adversarial_audit.md` before selecting a smoke run.

## A2 Bounded Shape Loss Gate: 2026-07-13

The corrected A2 keeps CIoU and adds only a bounded, geometry-gated log-aspect term. `paper1_a2_adversarial_audit.md` and `.json` record 251 passing checks in normal and optimized Python, including CUDA AMP extreme boxes and exact weight-zero full-criterion/gradient equivalence. Training commands must pin `--expect-loss ciou_bounded_elongation`, `--expect-elongation-weight 0.1`, and the trusted checkpoint SHA256. The first assigned-positive batch writes `a2_penalty_diagnostics.json` with per-branch CIoU, penalty, AR-bin means, and gradient-norm ratios. This authorizes one 1e smoke only; per-GT assignment diagnostics are mandatory before deciding whether a 30e run is scientifically justified.

The completed A2 smoke passed those gates, but its matched 30e signal tied B0 at `0.574/0.319` while D10 degraded from `0.324/0.130` to `0.305/0.121` AP50/AP50-95. B0/A2 best-checkpoint assignment summaries show no D10 CIoU or confidence improvement. A2 is rejected from 100e, weight sweeps, and composites; B1 remains unnecessary because D10 receives normal positive assignment. The next task is B2 design and adversarial audit, not another training run.

## B2 Quality-Aware Hard Positive Gate: 2026-07-13

`yolo26n-Paper1-B2-QualityHardPositive.yaml` preserves the YOLO26n architecture and applies a detached, class-agnostic hard-positive weight only to the assigned target-class BCE. The weight is bounded to `[1.0, 1.25]`; negatives and localization are unchanged, and weight zero executes the exact baseline BCE branch.

`paper1_b2_adversarial_audit.md` and `.json` record `65/65` checks passing under normal Python and `python -O` on the remote RTX 4090. The audit covers exact weight-zero full-loss and parameter-gradient equivalence, both E2E branches, `708/708` bitwise checkpoint transfer, CUDA AMP, empty targets, malformed strengths, bounded weighting, and finite gradients. The synthetic assigned-positive probe added `7.07%` and `12.52%` of the baseline classification-gradient norm in the one-to-many and one-to-one branches, respectively.

The B2 smoke passed its artifact and numerical gates, but the matched 30e signal reached `0.571/0.317` mAP50/mAP50-95 versus B0 `0.574/0.319`; D10 fell `0.324/0.130 -> 0.297/0.117` AP50/AP50-95. B2 is rejected from 100e, lambda sweeps, and B2+C combinations.
