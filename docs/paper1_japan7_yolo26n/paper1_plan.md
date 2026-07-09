# Paper 1 Plan

**Title (draft)**: Lightweight Detail Enhancement for Road Crack Damage Detection Based on YOLO26n

**Target**: Domestic journal, 3rd/4th tier
**Dataset**: Japan7 (7 classes from Japan domain)

## Timeline

| Phase | Duration | Tasks |
| --- | --- | --- |
| 1. Baseline | ✅ Done | 4 models trained, results archived |
| 2. Analysis | Current | Bottleneck identification, failure case collection |
| 3. Method design | 1–2 weeks | Ablation candidates, module selection |
| 4. Experiments | 2–3 weeks | B0–B5 ablations, hyperparameter tuning |
| 5. Writing | 2 weeks | Draft, tables, figures |
| 6. Revision | 1 week | Internal review, polish |

## Core narrative

1. YOLO26n is lightweight (2.376M params) but underperforms on thin cracks (D00 mAP50-95 = 0.183, D10 = 0.148)
2. YOLO26s (4× params) shows limited gain → capacity alone insufficient
3. Propose lightweight detail enhancement module(s)
4. Ours-YOLO26n surpasses YOLO26n baseline, approaches YOLOv8n/YOLO11n while staying lightweight

## Success criteria

- Ours-YOLO26n mAP50-95 > YOLO26n (0.341)
- D00 mAP50-95 > 0.200
- D10 mAP50-95 > 0.170
- Params increase < 30% over YOLO26n
- FLOPs increase < 30% over YOLO26n
