# Module Selection

This branch records the module screening rules for YOLO26-probe. The current goal is an auditable Paper 1/Paper 2 candidate queue, not immediate module stacking.

Model-structure scanning is frozen on this branch. Localization/classification research changes must be minimal, configurable, baseline-equivalent when disabled, and adversarially audited before training. Do not restore old stashes or reopen `CARAFE`, `FDConv`, or `ContextAggregation` here.

## 2026-07-13 Current Paper 1 Decision

A1 Shape-IoU finished at `0.573/0.318` and A2 bounded log-aspect at `0.574/0.319`, versus the matched B0 control at `0.574/0.319`. A2 also reduced D10 from `0.324/0.130` to `0.305/0.121` AP50/AP50-95. Both localization-loss candidates are rejected from 100e and composites.

Per-GT diagnostics show complete D10 positive assignment and nearly identical B0/A2 candidate CIoU, so do not change the assigner and do not sweep A2 weights. The next candidate is B2: a bounded, class-agnostic weighting mechanism for high-quality matched positives whose classification confidence remains low. B2 requires a disabled-path baseline equivalence audit and 1e smoke before any 30e signal.

## 2026-07-10 Protocol Correction

The stored YOLO26n baseline was fine-tuned from `yolo26n.pt`, but the earlier module runner constructed custom YAMLs from scratch. The completed scratch formal runs therefore do not test whether EMA or P2Lite improves the pretrained YOLO26n baseline.

| run | initialization | AMP | best epoch | mAP50 | mAP50-95 | disposition |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| EMA_attention 100e | scratch | False | 65 | 0.58704 | 0.31410 | archive as exploratory |
| P2Lite 100e | scratch | True | 82 | 0.58718 | 0.31373 | archive as exploratory |

The near-identical best scores despite different architectures and AMP modes identify random initialization as the first confounder to remove. The migration audit found `468/714` compatible state items for EMA_attention but only `8/881` for P2Lite. `scripts/train_module_pilot.py` now records this coverage in `pretrained.txt` and the pilot report.

The next queue is EMA pretrained+AMP smoke followed by a 30 epoch pretrained EMA signal. P2Lite is paused: use an architecture-native CPUBone pretrained checkpoint only after a separate conversion/build validation. Do not run SPDConv or any composite until one signal is protocol-matched and positive.

## Current Scan Result

The current buildability scan covers 13 YAML candidates:

- Candidates: 13
- Build OK: 10
- Build failed: 3

`BUILD OK` only means `yaml.safe_load()` and `YOLO(yaml)` can construct the model. It does not mean the module is effective.

## Protocols

Paper 1 uses Japan7:

- Classes: D00, D10, D20, D40, D43, D44, D50
- Main metric: mAP50-95
- Secondary metrics: mAP50, Precision, Recall, Params, FLOPs, FPS
- Key classes: D00, D10

Paper 2 uses Common4:

- Classes: D00, D10, D20, D40
- Source domain: Japan
- Target domains: Czech, India, China_MB / United States if available
- Strict DG: train=Japan_train, val=Japan_val, test=Czech/India/China_MB/United States
- Loose transfer: train=Japan_train, val=target_val

If target-domain validation selects `best.pt`, do not call it strict domain generalization.

## Long-Term Direction

The earlier high-level module stories remain valid only as long-term combination candidates:

- Paper 1: `YOLO26n + P2-like shallow detail + SPDConv + EMA`
- Paper 2: `YOLO26n + HVI + ContextAggregation + hwd / wavelet-frequency enhancement`

They must not be trained directly as three-module combinations. Run one module at a time first.

Historical scratch signal decision:

- `EMA_attention` had the strongest scratch signal, but must be rerun with pretrained initialization before promotion.
- `CPUBoneNano-P2Lite` had the strongest shallow-detail scratch signal, but requires architecture-native CPUBone pretraining before another formal run.
- `SPDConv` remains paused; do not spend another 100 epoch run on the old scratch protocol.
- Current composite YAMLs are trainable, but they should not enter 30 epoch or 100 epoch experiments yet.

## Paper 1 Route

Paper 1 targets Japan7 single-domain detection. The focus is YOLO26n weakness on D00/D10 thin cracks, low contrast damage, and small road defects.

Historical 30 epoch scratch ranking:

| rank | module | mAP50 | mAP50-95 | params | FLOPs | decision |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | EMA_attention | 0.202 | 0.0884 | 2.377M | 5.2G | archive; rerun pretrained+AMP signal |
| 2 | CPUBoneNano-P2Lite | 0.153 | 0.0674 | 3.672M | 6.6G | archive; CPUBone pretraining required |
| 3 | SPDConv | 0.129 | 0.0516 | 2.600M | 1.5G | archive; paused |

First 3 epoch pilot batch:

- `yolo26-EMA_attention.yaml`: lightweight attention for damage-region response and background texture suppression.
- `yolo26-LaplacianConv.yaml`: edge/detail enhancement for D00/D10 crack boundaries.
- `yolo26-BiFPN1.yaml`: lightweight multi-scale fusion with the smallest parameter increase among BiFPN variants.

Second 3 epoch pilot batch:

- `yolo26-SPDConv.yaml`: detail-preserving downsampling for small targets and thin cracks.
- `yolo26-CPUBoneNano-P2Lite.yaml`: P2-like shallow detail path; useful but parameter increase is high.
- `yolo26-FFAFusion-Neck.yaml`: neck feature fusion; backup multi-scale fusion direction.

Controls and backups:

- `SEAttention`: backup to EMA.
- `CBAM`: traditional attention control, not the main innovation.
- `BiFPN`: lower priority than `BiFPN1` because `BiFPN1` is lighter.

Frozen until a separate mmcv compatibility fix:

- `yolo26-CARAFE.yaml`
- `yolo26-FDConv.yaml`
- `yolo26-ContextAggregation.yaml`

## Paper 1 Ablation

Do not start from a stacked model. Use this order:

| ID | Model |
| --- | --- |
| B0 | YOLO26n baseline |
| B1 | YOLO26n + EMA_attention |
| B2 | YOLO26n + LaplacianConv |
| B3 | YOLO26n + BiFPN1 |
| B4 | YOLO26n + SPDConv |
| B5 | YOLO26n + CPUBoneNano-P2Lite / P2-like shallow detail |
| B6 | YOLO26n + best single module + best shallow detail module |
| B7 | Ours-YOLO26n |

The old `P2 + SPDConv + EMA` idea can only become a later combination if:

- The P2-like module works alone.
- `SPDConv` works alone.
- `EMA_attention` works alone.
- Pairwise combinations do not reduce mAP50-95.
- Params/FLOPs/FPS still support a lightweight detection story.

Current 3 epoch composite pilot result:

| model | Recall | mAP50 | mAP50-95 | params | FLOPs | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P2Lite | 0.0714 | 0.000306 | 0.0000811 | 3.672M | 6.6G | keep as stronger single-module path |
| P2Lite + EMA | 0.0306 | 0.000128 | 0.0000280 | 3.673M | 6.6G | do not promote |
| P2Lite + SPDConv + EMA | 0.0601 | 0.000230 | 0.0000569 | 4.254M | 7.7G | trainable but do not promote |

Both composite YAMLs can train, but neither beats P2Lite in the 3 epoch pilot. Keep them as exploration candidates only. Future composite work should revisit EMA and SPDConv insertion positions before any 30 epoch or 100 epoch run.

## Paper 2 Route

Paper 2 uses Common4 cross-domain evaluation and must not reuse Japan7 seven-class results as the main table.

Current first Paper 2 candidate:

- `yolo26-HVIEnhanceStem.yaml`: color/light robustness for cross-domain road scenes.

Deferred Paper 2 candidates:

- `ContextAggregation`: currently build failed; needs mmcv compatibility work later.
- `FDConv / hwd / wavelet-frequency enhancement`: not currently validated as a buildable Paper 2 candidate.
- `HVI + ContextAggregation + hwd`: late-stage combination idea only, not a current direct training target.

## Promotion Rules

3 epoch pilot promotion requires:

- completed status
- no OOM
- no NaN
- `results.csv` exists
- `weights/best.pt` exists
- `args.yaml` exists
- loss decreases normally
- mAP50 is nonzero
- Params/FLOPs do not explode

20/30 epoch signal promotion requires:

- mAP50-95 is not clearly worse than YOLO26n baseline trend
- D00 or D10 has a positive signal
- parameter increase is explainable
- training is stable
- the module has a clear paper narrative

100 epoch formal is limited to at most 1-2 models. Before promotion, ensure the run is a `yolo26n.pt` transfer fine-tune with AMP enabled and inspect the log line reporting transferred items. Prioritize clear D00/D10 improvement, stable mAP50-95, acceptable Params/FLOPs/FPS, and a defensible paper motivation.

## Do Not Do

- Do not train `P2 + SPDConv + EMA` directly.
- Do not promote current `P2Lite + EMA` or `P2Lite + SPDConv + EMA` composites to 30e/100e.
- Do not train `HVI + ContextAggregation + hwd` directly.
- Do not fix `CARAFE`, `FDConv`, or `ContextAggregation` on this branch.
- Do not restore old stashes into this clean branch.
- Do not run all buildable modules for 100 epochs.
- Do not treat `BUILD OK` as proof of effectiveness.
- Do not compare a scratch custom YAML run against the pretrained YOLO26n baseline.
- Do not judge only by total mAP; inspect D00/D10.
- Do not mix Paper 1 Japan7 and Paper 2 Common4 in one main result table.
- Do not commit `runs`, `datasets`, `.pt` weights, or large images.
- Do not change model structure on the baseline branch.

## 2026-07-13 B2 Quality-Aware Hard Positive Gate

A1 and A2 are closed after matched 30 epoch results failed the targeted D10 gate. B1 assigner work is also unsupported: B0 gives every evaluated D10 GT at least one positive and approximately ten one-to-many positives per GT. The next single-mechanism candidate is B2, not another model module.

B2 changes only positive-class BCE during training:

`weight = 1 + lambda * clamp(CIoU, 0, 1) * (1 - correct_class_confidence)`

The adaptive factors are detached, class-agnostic, and applied only to the assigned target class. Negative classes, TaskAlignedAssigner, box/DFL loss, model architecture, Params, FLOPs, and inference are unchanged. The candidate YAML pins `lambda=0.25`; `lambda=0` is exactly the baseline code path.

The adversarial audit passed 65 checks in normal and optimized Python on the remote RTX 4090. It verified CUDA AMP, empty targets, finite gradients, unchanged state names/shapes, bitwise `708/708` checkpoint transfer, and exact weight-zero loss and parameter-gradient equivalence in both end-to-end branches. Measured added/base classification-gradient ratios were `7.07%` for one-to-many and `12.52%` for one-to-one, so the candidate is active but bounded.

The B2 smoke passed every mechanical gate, but its matched 30 epoch signal reached only `0.571/0.317` mAP50/mAP50-95 versus B0 `0.574/0.319`. D10 fell from `0.324/0.130` to `0.297/0.117` AP50/AP50-95 and Recall fell by `0.008`. Reject B2 from 100e, lambda sweeps, and all composites. The next candidate must change representation, not localization, positive assignment, or low-confidence BCE weighting.
