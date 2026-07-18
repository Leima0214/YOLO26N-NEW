# YOLO26N-NEW Paper 1 protocol

This checkout trains Japan7 models from `/yolo26-probe/derived/japan7` with class order
`D00, D10, D20, D40, D43, D44, D50`.

## Experiment identity

- Historical scratch result:
  `mobilemamba_scratch_explicit_musgd_japan7_50e_seed42`.
- Next partial-pretrained result:
  `mobilemamba_partial_pretrained_auto_japan7_30e_seed42`.
- The historical scratch artifacts remain in
  `/root/YOLO26N-NEW/runs/paper1/mobilemamba_backbone_japan7_50e` so their original
  `args.yaml` and checkpoint metadata are not rewritten.

Pulling a commit or editing `train.py` does not change an already running Python
process. Initialization, optimizer groups, and run name are fixed when that process
starts.

Never resume a scratch checkpoint as a partial-pretrained experiment. Resume is valid
only when `last.pt`, `args.yaml`, initialization policy, model YAML, dataset, optimizer,
seed, and run identity all belong to the same interrupted run.

## Audits before training

```bash
/opt/conda/bin/python scripts/check_dataset.py --data configs/japan7_remote.yaml
CUDA_VISIBLE_DEVICES=-1 /opt/conda/bin/python scripts/audit_optimizer_groups.py
```

The optimizer audit must report `model.23` for YOLO26n and `model.18` for
MobileMamba. A partial-pretrained MobileMamba run must also create
`pretrained_transfer.txt` with the checkpoint SHA256 and global/backbone/neck/Detect
coverage. The audited Japan7 target is 420/674 state items, 55.56% parameters,
27.70% backbone, 100% neck, and 62.93% of the actual `model.18` Detect head.
This is not initialization-equivalent to a fully pretrained B0 because the
MobileMamba-specific backbone remains random; report that limitation explicitly.

`yolo26n.pt` must be the real checkpoint, not a small Git LFS pointer file. The
training entry fails before model construction when it detects a pointer.

## Matched 30-epoch run

```bash
EPOCHS=1 RUN_NAME=mobilemamba_partial_pretrained_auto_japan7_1e_seed42 \
  /opt/conda/bin/python train.py

EPOCHS=30 RUN_NAME=mobilemamba_partial_pretrained_auto_japan7_30e_seed42 \
  /opt/conda/bin/python train.py
```

Run the one-epoch smoke first. Do not overwrite or reuse the historical scratch
directory.
