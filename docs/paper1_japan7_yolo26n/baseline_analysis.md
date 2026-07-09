# Baseline Analysis

## Overall performance (Japan7, best.pt validation)

| Model | Params | FLOPs | P | R | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 3.007M | 8.1G | 0.647 | 0.606 | 0.642 | **0.353** |
| YOLO11n | 2.584M | 6.3G | 0.639 | **0.616** | 0.642 | 0.349 |
| YOLO26s | 9.468M | 20.5G | **0.678** | 0.591 | 0.630 | 0.347 |
| YOLO26n | **2.376M** | **5.2G** | 0.644 | 0.597 | 0.623 | 0.341 |

## Bottleneck: D00 and D10

| Model | D00 mAP50-95 | D10 mAP50-95 | D20 mAP50-95 | D43 mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | 0.219 | 0.167 | 0.338 | 0.570 |
| YOLO11n | 0.202 | 0.158 | 0.342 | 0.562 |
| YOLO26n | 0.183 | 0.148 | 0.346 | 0.541 |
| YOLO26s | 0.173 | 0.150 | 0.338 | 0.584 |

D00 and D10 account for ~55% of training instances (D00=3238, D10=3192 out of 19753 total).

### Why D00/D10 are hard

1. **Thin and elongated**: Cracks are often <5px wide, spanning 50–200px
2. **Low contrast**: Asphalt cracks blend into road texture
3. **Multi-scale**: Same crack type can appear at vastly different scales
4. **Background clutter**: Road markings, shadows, patches resemble cracks
5. **Shallow feature loss**: Standard FPN may lose fine crack detail in deep layers

## YOLO26n vs YOLO26s: Capacity alone is insufficient

YOLO26s has **4× params** (9.468M vs 2.376M) and **4× FLOPs** (20.5G vs 5.2G) but only gains **+0.006 mAP50-95**. This strongly suggests:

> Simply increasing model capacity is ineffective for fine-grained crack detection. Architectural changes targeting shallow detail preservation are needed.

This is a key motivation for Paper 1.

## Target improvement

| Metric | YOLO26n (current) | Target |
| --- | ---: | ---: |
| mAP50-95 | 0.341 | >0.360 |
| D00 mAP50-95 | 0.183 | >0.220 |
| D10 mAP50-95 | 0.148 | >0.180 |
| Params | 2.376M | <3.0M |
| FLOPs | 5.2G | <6.5G |
