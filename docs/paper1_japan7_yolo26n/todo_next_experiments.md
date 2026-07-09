# Next Experiments TODO

## Immediate (this week)

- [ ] Collect failure case images (D00/D10 miss) from val set
- [ ] Save confusion matrices from all 4 baseline runs
- [ ] Save PR curves from all 4 baseline runs
- [ ] Create `feature/yolo26n-paper1` branch from `baseline/japan-baseline-engineering`
- [ ] Merge amp fix into `baseline/japan-baseline-engineering`

## Short-term (1–2 weeks)

- [ ] Implement P2 shallow detail branch (B1)
- [ ] 3-epoch smoke test B1
- [ ] 100-epoch train B1
- [ ] Evaluate B1 vs B0 (YOLO26n baseline)
- [ ] Implement lightweight attention (B2)
- [ ] 3-epoch smoke + 100-epoch train B2
- [ ] Evaluate B2 vs B0

## Medium-term (2–3 weeks)

- [ ] Implement WFA (B3) if B1/B2 insufficient
- [ ] Combine best modules (B4/B5)
- [ ] Multi-seed validation (seed=0,42,3407) for B0 + B5
- [ ] Optional: train/val/test 70/15/15 split

## Before paper submission

- [ ] Multi-seed mean±std for ALL reported models
- [ ] Failure case visualization (Figure 5)
- [ ] Confusion matrix (Figure 6)
- [ ] FLOPs computation verification (all models use same tool)
- [ ] Params count verification
- [ ] Write paper draft
- [ ] Internal review

## DO NOT

- Do NOT modify baseline scripts on `baseline/japan-baseline-engineering`
- Do NOT re-run baseline trainings unless protocol changes
- Do NOT delete any run directory
- Do NOT commit .pt / datasets / runs
- Do NOT claim YOLO26 > YOLOv8
