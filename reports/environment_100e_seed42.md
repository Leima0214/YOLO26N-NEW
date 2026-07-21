# Paper 1 100e seed42 environment audit

- Audit time: 2026-07-21 10:22 CST
- Host: `bitahub-a20509641536434176801806`
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB; idle at audit time
- Driver: 580.76.05
- Python: 3.11.10
- PyTorch / torchvision: 2.5.1+cu124 / 0.20.1+cu124
- CUDA runtime / cuDNN: 12.4 / 9.1.0
- CUDA capability: 8.9
- System RAM available: 372 GiB
- Disk available under `/root`: 27 GiB
- Residual training processes: none
- Environment decision: **PASS**

## Git and artifact identity

- Branch: `codex/p4-single-gate1e3`
- HEAD: `09a63f65d637cf3b6fe739ad4c986cf866f841c7`
- Upstream: `origin/codex/p4-single-gate1e3`
- `yolo26n.pt`: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- B0 YAML: `b1d1fa0c69eced64f9939536038bd8d697c32e72200b34187a767de431af7ef0`
- P4 YAML: `fa153beb263d5e9497af9ef90f818342cb8eb9e285ced1516cb8fbedd79425cf`
- Dataset YAML: `f7818ae85148d0adfd09ecb8ddd09dcca49b980434ce46deda305585d3ced3f9`

## Working-tree warning

The branch commit is preserved remotely, but the active training checkout is dirty from earlier experiments. Source-relevant tracked differences at audit time were:

- `train.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/nn/yolo26_2025_backbones/__init__.py`
- `best.pt`, `yolo11n.pt`, `yolo26n.pt`

There are also many generated `__pycache__`, `runs/`, logs, and untracked historical experiment files. None were deleted or overwritten. A future formal run should use a clean worktree at the recorded commit and copy in only the audited `yolo26n.pt` plus the corrected dataset configuration.

## Training state

No 100e training was started. The environment passed, but the later dataset audit returned `FAIL_CONFIRMED_NEAR_DUPLICATE_LEAKAGE`, which is a mandatory stop condition.
