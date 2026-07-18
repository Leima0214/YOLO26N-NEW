# MobileMamba scratch + explicit MuSGD, Japan7, 50 epochs, seed 42

## Identity

- Canonical experiment ID:
  `mobilemamba_scratch_explicit_musgd_japan7_50e_seed42`
- Original remote directory:
  `/root/YOLO26N-NEW/runs/paper1/mobilemamba_backbone_japan7_50e`
- Launch checkout: `299536c`
- Model: `ultralytics/cfg/models/26/yolo26-MobileMamba-Backbone.yaml`
- Data: `configs/japan7_remote.yaml`
- Initialization: scratch from YAML
- Optimizer: explicit `MuSGD`, `lr0=0.01`, `momentum=0.937`
- Schedule: linear, `lrf=0.01`, 3 warmup epochs, warmup bias LR 0.1
- Training: 50 epochs, 640 px, batch 32, seed 42, deterministic, AMP

`args.yaml` contains the framework default `pretrained: true`, but no checkpoint was
loaded: the launch code constructed the YAML directly and the run has no
`pretrained_transfer.txt`. This result is therefore scratch, not partial-pretrained.

## Result

- Runtime: 1.216 hours
- Best epoch: 50
- Precision / recall: 0.57573 / 0.55068
- mAP50 / mAP50-95: 0.56168 / 0.29534
- Fused model: 2,505,981 parameters, 6.2 GFLOPs
- Inference: 0.8 ms/image on the recorded RTX 4090 validation

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| D00 | 0.519 | 0.324 | 0.359 | 0.168 |
| D10 | 0.450 | 0.270 | 0.306 | 0.116 |
| D20 | 0.622 | 0.605 | 0.627 | 0.318 |
| D40 | 0.543 | 0.431 | 0.438 | 0.195 |
| D43 | 0.673 | 0.775 | 0.778 | 0.472 |
| D44 | 0.610 | 0.717 | 0.706 | 0.431 |
| D50 | 0.615 | 0.731 | 0.716 | 0.366 |

## Artifact hashes

- `best.pt`:
  `48b5bf5428158959ec15cf8bc19b7a8fa859b229e75f8e6d8181d0666ae4cd8d`
- `last.pt`:
  `c46299a1b285b994488d78e005eb679a735c3d69a6da6b4a8dbe17772e3390e1`

## Decision

Reject as a Paper 1 candidate. It is 0.02366 below the prior 30-epoch B0
mAP50-95 anchor of 0.319, and D10 mAP50-95 is 0.014 below the 0.130 reference.
Because initialization, optimizer protocol, and epoch count differ from B0, this is an
exploratory result rather than a fair causal estimate of the MobileMamba backbone.
