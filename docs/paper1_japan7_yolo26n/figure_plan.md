# Figure Plan

## Figure 1: Detection challenge illustration

Composite figure showing why D00/D10 are hard:
- (a) Thin longitudinal crack (<5px wide) — easy to miss at 640×640
- (b) Low-contrast pothole crack blending into wet asphalt
- (c) Crack occluded by road marking / shadow
- (d) Successful detection example (D20/D40)

Source: Val set, pick representative examples.

## Figure 2: Per-class mAP50-95 comparison

Grouped bar chart: 4 models × 7 classes = 28 bars.
X-axis: D00, D10, D20, D40, D43, D44, D50.
4 bars per class (one per model).
Red box highlighting D00/D10 as bottleneck.

## Figure 3: Proposed architecture

Diagram of Ours-YOLO26n:
- YOLO26n backbone + neck
- Highlight added module(s): P2 branch / attention / frequency enhancement
- Label dimensions at each stage

## Figure 4: Ablation results

Table or bar chart showing B0–B5:
- X-axis: B0 (baseline), B1, B2, B3, B4, B5 (Ours)
- Y-axis: mAP50-95 + D00/D10 per-class AP
- Secondary axis: Params / FLOPs

## Figure 5: Detection examples

3×3 grid:
- Rows: D00, D10, D20
- Columns: Ground truth, YOLO26n, Ours-YOLO26n
- Green boxes on GT, red circles on missed detections, green boxes on improvements

## Figure 6: Confusion matrix

Normalized confusion matrix for Ours-YOLO26n.
7×7 heatmap.

## Figure 7: PR curves

Precision-Recall curves for D00 and D10:
- YOLO26n (dashed)
- Ours-YOLO26n (solid)
- Show AP improvement area.

## Figure 8: FLOPs vs mAP50-95

Scatter plot:
- 7 models: YOLOv8n, YOLO11n, YOLO26n, YOLO26s, B1, B2, Ours
- X-axis: FLOPs (G)
- Y-axis: mAP50-95
- Pareto frontier highlighting Ours as best lightweight option.
