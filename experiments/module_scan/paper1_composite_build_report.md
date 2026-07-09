# Paper 1 P2Lite-SPDConv-EMA Build Report

Generated: 2026-07-09

## Candidate

- YAML: `ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-SPDConv-EMA.yaml`
- Base: `yolo26-CPUBoneNano-P2Lite.yaml`
- Added: SPDConv-style `space_to_depth + Conv(stride=1)` bottom-up downsampling
- Added: one `EMA_attention` block before the final P5 Detect input
- Role: enhancement candidate after `P2Lite + EMA`

## Current Ranking Context

30 epoch single-module signal now ranks `EMA_attention` first, `P2Lite` second, and `SPDConv` third. Therefore the main Paper 1 composite candidate is `P2Lite + EMA`; this three-module YAML is retained as the stronger enhancement candidate.

## Buildability

| check | result |
| --- | --- |
| YAML exists | OK |
| `yaml.safe_load()` | OK |
| `YOLO(yaml)` build | OK |
| params | 4,511,660 |
| Detect strides | `[4.0, 8.0, 16.0, 32.0]` |
| Detect inputs | P2, P3, P4, P5 |
| training run | not run |

Command used for build-only check:

```bash
python - <<'PY'
from pathlib import Path
import yaml
from ultralytics import YOLO

path = Path("ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-SPDConv-EMA.yaml")
with path.open("r", encoding="utf-8") as f:
    yaml.safe_load(f)
model = YOLO(str(path))
params = sum(p.numel() for p in model.model.parameters())
print(params)
print(model.model.stride)
PY
```

## Remote GPU Commands

1 epoch smoke, batch 32:

```bash
python scripts/train_module_pilot.py \
  --model-yaml ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-SPDConv-EMA.yaml \
  --data configs/japan7_remote.yaml \
  --epochs 1 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 8 \
  --name module_Paper1-P2Lite-SPDConv-EMA_japan7_e1_img640_b32_seed42
```

3 epoch pilot, batch 32:

```bash
python scripts/train_module_pilot.py \
  --model-yaml ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-SPDConv-EMA.yaml \
  --data configs/japan7_remote.yaml \
  --epochs 3 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --workers 8 \
  --name module_Paper1-P2Lite-SPDConv-EMA_japan7_e3_img640_b32_seed42
```

## Decision

The composite YAML is buildable and ready for remote smoke/pilot only. Do not run 30 epoch or 100 epoch training until the `P2Lite + EMA` result is reviewed and the extra SPDConv complexity is justified.
