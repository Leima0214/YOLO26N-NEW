# YOLO26N-NEW Paper 1

Japan7 uses seven classes in this order:
`D00, D10, D20, D40, D43, D44, D50`.

## Current candidate

`yolo26n-Paper1-MobileMamba-P3.yaml` keeps the native YOLO26n 24-layer topology
and adds one MobileMamba residual to the P3 backbone stage. Its residual scale is
zero-initialized, so loading `yolo26n.pt` preserves B0 behavior exactly at step 0
while allowing the new branch to learn.

The rejected full-backbone replacement remains available only for historical
reproduction. Its 55.56% parameter coverage and random backbone produced:

- scratch MuSGD 50e: mAP50-95 `0.295`
- partial-pretrained auto 30e: mAP50-95 `0.191`
- scratch-best stage-2 run: stopped after epoch 25, best mAP50-95 `0.287`

Do not resume any of those checkpoints into the current candidate.

## Remote checks and run

```bash
/opt/conda/bin/python scripts/check_dataset.py --data configs/japan7_remote.yaml
CUDA_VISIBLE_DEVICES=-1 /opt/conda/bin/python scripts/audit_mobilemamba_identity.py
CUDA_VISIBLE_DEVICES=-1 /opt/conda/bin/python scripts/audit_optimizer_groups.py

/opt/conda/bin/python train.py
```

The identity audit must report at least 96% parameter coverage, exact step-0
equivalence, and a nonzero residual-gate gradient.

To switch experiments, edit only `MODEL` and `RUN_NAME` at the top of
`train.py`. Comment out `model.load("yolo26n.pt")` only for an intentional
scratch run.
