# Module Selection

This branch records the module screening rules for YOLO26-probe. The current goal is an auditable Paper 1/Paper 2 candidate queue, not immediate module stacking.

Do not modify Ultralytics core model code on this branch. Do not train. Do not restore old stashes. Do not fix `CARAFE`, `FDConv`, or `ContextAggregation` here; use a separate `fix/mmcv-compat-module-build` branch for that.

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

## Paper 1 Route

Paper 1 targets Japan7 single-domain detection. The focus is YOLO26n weakness on D00/D10 thin cracks, low contrast damage, and small road defects.

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

100 epoch formal is limited to at most 1-2 models. Prioritize clear D00/D10 improvement, stable mAP50-95, acceptable Params/FLOPs/FPS, and a defensible paper motivation.

## Do Not Do

- Do not train `P2 + SPDConv + EMA` directly.
- Do not train `HVI + ContextAggregation + hwd` directly.
- Do not fix `CARAFE`, `FDConv`, or `ContextAggregation` on this branch.
- Do not restore old stashes into this clean branch.
- Do not run all buildable modules for 100 epochs.
- Do not treat `BUILD OK` as proof of effectiveness.
- Do not judge only by total mAP; inspect D00/D10.
- Do not mix Paper 1 Japan7 and Paper 2 Common4 in one main result table.
- Do not commit `runs`, `datasets`, `.pt` weights, or large images.
- Do not change model structure on the baseline branch.
