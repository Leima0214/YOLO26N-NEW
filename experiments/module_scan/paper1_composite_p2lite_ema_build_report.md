# Paper 1 P2Lite-EMA Build Report

Generated: 2026-07-09

## Candidate

- YAML: `ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-EMA.yaml`
- Base: `yolo26-CPUBoneNano-P2Lite.yaml`
- Added: one `EMA_attention` block before the final P5 Detect input
- Role: current main Paper 1 composite candidate

## Motivation

30 epoch signal ranking now places `EMA_attention` first and `P2Lite` second among single modules:

| module | mAP50 | mAP50-95 | params | FLOPs |
| --- | ---: | ---: | ---: | ---: |
| EMA_attention | 0.202 | 0.0884 | 2.377M | 5.2G |
| P2Lite | 0.153 | 0.0674 | 3.672M | 6.6G |
| SPDConv | 0.129 | 0.0516 | 2.600M | 1.5G |

This makes `P2Lite + EMA` the current preferred Paper 1 combination: P2Lite supplies shallow-detail and P2/4 detection, while EMA supplies the strongest single-module attention signal.

## Buildability

| check | result |
| --- | --- |
| YAML exists | OK |
| `yaml.safe_load()` | OK |
| `YOLO(yaml)` build | OK |
| params | 3,931,052 |
| Detect strides | `[4.0, 8.0, 16.0, 32.0]` |
| Detect inputs | P2, P3, P4, P5 |
| training run | not run |

## Remote GPU Commands

1 epoch smoke, batch 32:

```bash
python scripts/train_module_pilot.py \
  --model-yaml ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-EMA.yaml \
  --data configs/japan7_remote.yaml \
  --epochs 1 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 8 \
  --name module_Paper1-P2Lite-EMA_japan7_e1_img640_b32_seed42
```

3 epoch pilot, batch 32:

```bash
python scripts/train_module_pilot.py \
  --model-yaml ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-EMA.yaml \
  --data configs/japan7_remote.yaml \
  --epochs 3 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 8 \
  --name module_Paper1-P2Lite-EMA_japan7_e3_img640_b32_seed42
```

## Decision

The `P2Lite + EMA` YAML is buildable and ready for remote smoke/pilot only. Do not run 30 epoch or 100 epoch training until the smoke/pilot results are reviewed.
