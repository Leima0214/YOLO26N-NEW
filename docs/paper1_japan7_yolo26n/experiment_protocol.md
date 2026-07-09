# Experiment Protocol — Japan7 Baseline

## Training protocol (all models)

```
epochs=100
imgsz=640
batch=32
device=0
workers=8
seed=42
amp=True
```

All other hyperparameters use Ultralytics 8.4.2 defaults (MuSGD optimizer, auto lr).

See `experiments/direct_train_equivalence.md` for full audit confirming all scripts are thin wrappers around `YOLO().train()`.

## Evaluation protocol

- **Metric source**: `best.pt` final validation (NOT last.pt)
- **Primary metric**: mAP50-95
- **Secondary metrics**: mAP50, Precision, Recall, Params, FLOPs
- **Per-class metrics**: D00–D50 mAP50, mAP50-95

## Hardware

| Component | Spec |
| --- | --- |
| GPU | NVIDIA RTX 4090 24GB |
| CUDA | 12.4 |
| PyTorch | 2.5.1+cu124 |
| Python | 3.11.10 |
| Ultralytics | 8.4.2 |

## Limitations (acknowledged in paper)

1. **No independent test set**: Val used for both model selection and reporting.
   Mitigation: Report as "validation performance", not "test performance".
2. **Single seed (42)**: Model ranking may vary with seed.
   Mitigation: Multi-seed validation planned for final submission.
3. **Single domain (Japan)**: Results may not generalize.
   Mitigation: Paper 2 addresses cross-domain separately.

## Reproducibility checklist

- [x] Data config YAML committed
- [x] Mapping YAML committed
- [x] Training scripts committed
- [x] Training commands documented
- [x] Results CSV saved (remote server)
- [x] args.yaml saved (remote server)
- [x] Environment info documented
- [ ] Multi-seed validation (planned)
- [ ] Test set (planned)
