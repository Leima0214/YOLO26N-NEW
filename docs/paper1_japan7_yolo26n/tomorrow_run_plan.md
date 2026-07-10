# Paper 1 Next Run Plan

> Superseded on 2026-07-10. Do not run the scratch 100 epoch commands below. The baseline uses `yolo26n.pt`, while those commands constructed custom YAMLs from scratch. Follow [formal_protocol_correction_20260710.md](formal_protocol_correction_20260710.md): EMA is the only current pretrained-transfer candidate; P2Lite needs a separate CPUBone checkpoint conversion validation.

Date: 2026-07-09

Branch: `codex/yolo26-module-scan-cleanup`

Shutdown baseline commit: `3388dc83a293319d7e4c2c9c4af110d86458e636`

## Current State

The GPU can be shut down. Do not start more training tonight. Composite YAMLs are kept for exploration, but the current formal Paper 1 route returns to single-module 100 epoch comparisons.

## Completed Experiments

| run | epochs | model | precision | recall | mAP50 | mAP50-95 | decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `pilot_Paper1_EMA_attention_japan7_e3_img640_b32_seed42` | 3 | EMA_attention | 0.00038 | 0.04343 | 0.00024 | 0.00005 | completed pilot |
| `Paper1_LaplacianConv_japan7_e3_img640_b32_seed42` | 3 | LaplacianConv | 0.00027 | 0.05453 | 0.00017 | 0.00004 | completed pilot |
| `Paper1_BiFPN1_japan7_e3_img640_b32_seed42` | 3 | BiFPN1 | 0.00035 | 0.04209 | 0.00020 | 0.00004 | completed pilot |
| `Paper1_SPDConv_japan7_e3_img640_b32_seed42` | 3 | SPDConv | 0.00034 | 0.03886 | 0.00022 | 0.00005 | completed pilot |
| `Paper1_P2Lite_japan7_e3_img640_b32_seed42` | 3 | P2Lite | 0.00041 | 0.0714 | 0.000306 | 0.0000811 | best early P2 signal |
| `signal_Paper1_EMA_attention_japan7_e30_img640_b32_seed42` | 30 | EMA_attention | n/a | n/a | 0.202 | 0.0884 | strongest single module |
| `signal_Paper1_P2Lite_japan7_e30_img640_b32_seed42` | 30 | P2Lite | n/a | n/a | 0.153 | 0.0674 | best shallow-detail module |
| `signal_Paper1_SPDConv_japan7_e30_img640_b32_seed42` | 30 | SPDConv | n/a | n/a | 0.129 | 0.0516 | efficient optional candidate |
| `module_Paper1-P2Lite-EMA_japan7_e3_img640_b32_seed42` | 3 | P2Lite + EMA | n/a | 0.0306 | 0.000128 | 0.0000280 | trainable, do not promote |
| `module_Paper1_P2Lite_SPDConv_EMA_japan7_e3_img640_b32_seed42` | 3 | P2Lite + SPDConv + EMA | n/a | 0.0601 | 0.000230 | 0.0000569 | trainable, do not promote |

## Historical Single-Module Ranking (Superseded)

| rank | module | mAP50 | mAP50-95 | params | FLOPs | next action |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | EMA_attention | 0.202 | 0.0884 | 2.376873M | 5.2G | archive; rerun pretrained+AMP signal |
| 2 | P2Lite | 0.153 | 0.0674 | 3.672344M | 6.6G | archive; CPUBone pretraining required |
| 3 | SPDConv | 0.129 | 0.0516 | 2.600377M | 1.5G | archive; paused |

## Decisions

EMA_attention should run first tomorrow because it has the strongest 30 epoch mAP50 and mAP50-95 while keeping Params and FLOPs low.

P2Lite should run second because it is the best shallow-detail/P2 candidate and directly supports the Paper 1 small-target and thin-crack story.

SPDConv is optional because it is weaker than EMA and P2Lite, but it is efficient and may still be useful as a lightweight detail-preserving ablation.

The current composite models are paused. Both `P2Lite + EMA` and `P2Lite + SPDConv + EMA` can train, but their 3 epoch signals are below P2Lite alone. Future composite work should revisit EMA and SPDConv insertion positions before any 30e or 100e run.

## Current Recommended Order

1. Pull the latest branch on the GPU machine.
2. Run EMA_attention pretrained+AMP smoke.
3. Only if smoke confirms `Transferred 468/714 items`, run its 30 epoch pretrained signal.
4. Do not run P2Lite, SPDConv, or composite models until their initialization is scientifically valid.

## Current Commands

EMA_attention 1 epoch smoke:

```bash
python scripts/train_module_pilot.py \
  --model-yaml ultralytics/cfg/models/26/yolo26-EMA_attention.yaml \
  --data configs/japan7_remote.yaml \
  --pretrained yolo26n.pt \
  --epochs 1 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 8 \
  --name smoke_Paper1_EMA_attention_japan7_e1_img640_b32_pretrained_amp_seed42
```

EMA_attention 30 epoch signal, only after successful smoke:

```bash
python scripts/train_module_pilot.py \
  --model-yaml ultralytics/cfg/models/26/yolo26-EMA_attention.yaml \
  --data configs/japan7_remote.yaml \
  --pretrained yolo26n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 8 \
  --name signal_Paper1_EMA_attention_japan7_e30_img640_b32_pretrained_amp_seed42
```

## Manual Checks Before GPU Shutdown

- Confirm no Python training process is still running.
- Confirm `paper1_logs_results_docs_20260709.tar.gz`, `paper1_best_weights_20260709.tar.gz`, and `backup_manifest_20260709.txt` exist locally.
- Confirm the GitHub branch contains this shutdown plan.
- Confirm no uncommitted `runs`, `datasets`, or `.pt` files are waiting to be committed.
