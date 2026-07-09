# Ablation Plan — YOLO26n Improvements

## Models

| ID | Name | Description | Expected ΔmAP50-95 |
| --- | --- | --- | ---: |
| B0 | YOLO26n | Baseline (no modification) | 0 (reference) |
| B1 | YOLO26n + P2 | Add P2 shallow detail branch | +0.010–0.020 |
| B2 | YOLO26n + Attn | Lightweight channel/spatial attention | +0.005–0.015 |
| B3 | YOLO26n + WFA | Wavelet frequency attention | +0.010–0.020 |
| B4 | YOLO26n + P2 + Attn | P2 + lightweight attention | +0.015–0.025 |
| B5 | Ours-YOLO26n | Best combination of B1–B4 | Target: >0.360 mAP50-95 |

## Metrics tracked per model

| Metric | Priority |
| --- | --- |
| Params (M) | Must stay <3.0M |
| FLOPs (G) | Must stay <7.0G |
| mAP50 | Secondary |
| mAP50-95 | **Primary** |
| D00 mAP50 | Per-class |
| D00 mAP50-95 | Per-class |
| D10 mAP50 | Per-class |
| D10 mAP50-95 | Per-class |
| Training time (h) | Reference only |

## Experiment workflow per model

1. 3-epoch smoke test (verify no crash, loss decreasing)
2. 100-epoch full train
3. Record best.pt validation metrics
4. Save results.csv + args.yaml
5. Run `scripts/collect_results.py`
6. Compare D00/D10 improvements vs B0

## Success criteria

- B5 (Ours) mAP50-95 > B0 (0.341) by ≥ 0.020
- D00 mAP50-95 > 0.200
- D10 mAP50-95 > 0.170
- Params < 3.0M
- FLOPs < 7.0G

## Notes

- All ablations use the same Japan7 dataset and training protocol (epochs=100, imgsz=640, batch=32, seed=42)
- Do NOT modify training hyperparameters between ablations
- Each ablation should be a separate script in `scripts/ablation_*.py`
- Results go to `experiments/ablation_baseline_*.csv`
