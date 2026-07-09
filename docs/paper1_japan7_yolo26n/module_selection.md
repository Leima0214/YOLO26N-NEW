# Module Selection

This branch is for building an auditable module queue, not for stacking modules.

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
- Strict DG: train=Japan_train, val=Japan_val, test=target domain
- Loose transfer: train=Japan_train, val=target_val

If target-domain validation selects `best.pt`, do not call it strict domain generalization.

## Phase 1: Buildability

Run:

```bash
python scripts/scan_yolo26_module_buildability.py
```

Outputs:

- `experiments/module_scan/buildability_report.csv`
- `experiments/module_scan/buildability_report.md`

Buildability means `yaml.safe_load()` succeeds and `YOLO(yaml)` constructs the model. It does not train.

## Phase 2: 3 Epoch Pilot

Run only `build_ok=True` YAMLs, one module per run:

```bash
python scripts/train_module_pilot.py \
    --model-yaml ultralytics/cfg/models/26/yolo26-CBAM.yaml \
    --data configs/japan7_remote.yaml \
    --epochs 3 --imgsz 640 --batch 16 --device 0 --workers 8
```

Default run name:

```text
module_{module_name}_japan7_e3_img640_b16_seed42
```

If that directory already exists, the script appends a timestamp to avoid overwriting old runs.

Pilot checks:

- `results.csv` exists
- `weights/best.pt` exists
- `args.yaml` exists
- OOM
- NaN
- training loss decreased
- mAP50 is nonzero
- Params/FLOPs are available when Ultralytics reports them

Outputs:

- `experiments/module_scan/pilot_report.csv`
- `experiments/module_scan/pilot_report.md`

## Phase 3: 20/30 Epoch Signal

Promote at most 3 modules. Priority order:

1. Shallow detail / P2: `yolo26-CPUBoneNano-P2Lite.yaml`
2. Lightweight attention: `yolo26-EMA_attention.yaml`, `yolo26-SEAttention.yaml`, `yolo26-CBAM.yaml`
3. Edge/frequency/detail: `yolo26-LaplacianConv.yaml`, `yolo26-FDConv.yaml`
4. Multi-scale fusion: `yolo26-CARAFE.yaml`, `yolo26-BiFPN.yaml`, `yolo26-FFAFusion-Neck.yaml`

`CBAM` is a traditional attention control, not the main innovation. Official P2-style references must state that no direct `yolo26n.pt` pretrained-weight comparison is fair unless weights are aligned.

## Phase 4: 100 Epoch Formal

Only 1-2 single modules may enter formal training. Combination models wait until single modules show signal.

Recommended ablation table:

| ID | Model |
| --- | --- |
| B0 | YOLO26n baseline |
| B1 | YOLO26n + shallow detail / P2 idea |
| B2 | YOLO26n + EMA_attention |
| B3 | YOLO26n + FDConv or LaplacianConv |
| B4 | YOLO26n + shallow detail + best single module |
| B5 | Ours-YOLO26n |

Each formal row must record Params, FLOPs, FPS, P, R, mAP50, mAP50-95, D00/D10 AP50, D00/D10 AP50-95, `best.pt` path, `results.csv` path, `args.yaml` path, git commit, and command.

## First Scan Candidates

Paper 1:

- `ultralytics/cfg/models/26/yolo26-CPUBoneNano-P2Lite.yaml`
- `ultralytics/cfg/models/26/yolo26-CARAFE.yaml`
- `ultralytics/cfg/models/26/yolo26-BiFPN.yaml`
- `ultralytics/cfg/models/26/yolo26-BiFPN1.yaml`
- `ultralytics/cfg/models/26/yolo26-EMA_attention.yaml`
- `ultralytics/cfg/models/26/yolo26-SEAttention.yaml`
- `ultralytics/cfg/models/26/yolo26-CBAM.yaml`
- `ultralytics/cfg/models/26/yolo26-LaplacianConv.yaml`
- `ultralytics/cfg/models/26/yolo26-FDConv.yaml`
- `ultralytics/cfg/models/26/yolo26-SPDConv.yaml`
- `ultralytics/cfg/models/26/yolo26-FFAFusion-Neck.yaml`

Paper 2:

- `ultralytics/cfg/models/26/yolo26-HVIEnhanceStem.yaml`
- `ultralytics/cfg/models/26/yolo26-FFAFusion-Neck.yaml`
- `ultralytics/cfg/models/26/yolo26-ContextAggregation.yaml`

Removed from the local candidate set:

- Large transformer/backbone replacement variants
- Module-zoo YAMLs outside the Paper 1 and Paper 2 candidate lists
