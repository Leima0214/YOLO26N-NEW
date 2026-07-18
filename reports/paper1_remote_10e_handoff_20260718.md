# Paper 1 remote 10-epoch handoff (2026-07-18)

## Fixed protocol

- GPU: NVIDIA GeForce RTX 4090
- Python/PyTorch: 3.11.10 / 2.5.1+cu124
- Model: `ultralytics/cfg/models/26/yolo26-MobileMamba-Backbone.yaml`
- Data: `configs/japan7_remote.yaml`
- Classes: D00, D10, D20, D40, D43, D44, D50
- Train/val images: 8,387 / 2,119
- Train/val boxes: 19,752 / 5,000
- Class IDs: 0-6
- Training: 640 px, batch 32, MuSGD, seed 42, AMP, 8 workers

Dataset validation found no missing labels, malformed lines, out-of-bounds boxes,
duplicate label lines, corrupt images, or broken links. Empty labels (638 train,
156 val) are valid background images after the three excluded classes are removed.

## 10-epoch smoke result

- Completed normally in 0.254 hours.
- Last/best: precision 0.508, recall 0.306, mAP50 0.28790, mAP50-95 0.13207.
- Best checkpoint SHA256:
  `9c2e6b8ad80774084e78c3a74533f2bfa4337119a10a4807f6f0468fa1df775d`
- Last checkpoint SHA256:
  `d29c99b1320c408ad7fca6efe6b15c1a40be36303e13eef9c9e3c90c654e8aca`
- Remote output:
  `/root/YOLO26N-NEW/runs/paper1/mobilemamba_backbone_japan7_10e`

This is a runtime smoke test, not a paper-quality comparison.

## VSCode remote terminal

Install or refresh dependencies without replacing the CUDA-enabled PyTorch image:

```bash
cd /root/YOLO26N-NEW
/opt/conda/bin/python -m pip install -r requirements-remote.txt
```

Recheck Japan7 before every new experiment window:

```bash
cd /root/YOLO26N-NEW
/opt/conda/bin/python scripts/check_dataset.py --data configs/japan7_remote.yaml
find -L /yolo26-probe/derived/japan7 -type l | head
```

Run the requested 50 epochs in the VSCode terminal:

```bash
cd /root/YOLO26N-NEW
EPOCHS=50 RUN_NAME=mobilemamba_backbone_japan7_50e \
  /opt/conda/bin/python train.py
```

Monitor from a second VSCode terminal:

```bash
watch -n 5 nvidia-smi
tail -f /root/YOLO26N-NEW/runs/paper1/mobilemamba_backbone_japan7_50e/results.csv
```

Resume only an interrupted 50-epoch run:

```bash
cd /root/YOLO26N-NEW
/opt/conda/bin/python - <<'PY'
from ultralytics import YOLO

YOLO("runs/paper1/mobilemamba_backbone_japan7_50e/weights/last.pt").train(resume=True)
PY
```

## Next-task gates

1. Keep the Japan7 split, class order, image size, batch, seed, optimizer,
   augmentation, AMP, workers, and validation thresholds identical.
2. Treat the 50-epoch MobileMamba run as exploratory until a yolo26n control is
   rerun under the same final protocol and initialization policy.
3. Compare class-level results, especially D10, in addition to aggregate mAP.
4. Reject runtime failures, NaN/Inf loss, class/split drift, or a final
   mAP50-95 below the matched control; do not stack another module before the
   single-change comparison is decided.
5. Preserve `args.yaml`, `results.csv`, `best.pt`, `last.pt`, and the training
   log together for every accepted or rejected run.
