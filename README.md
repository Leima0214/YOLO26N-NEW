# YOLO26N-NEW Paper 1

Japan7 uses seven classes in this order:
`D00, D10, D20, D40, D43, D44, D50`.

## Clean experiment entry

`train.py` is intentionally generic. To switch experiments, edit only `MODEL`
and `RUN_NAME` at the top.

```python
model = YOLO(MODEL)
model.load("yolo26n.pt")
```

`model.load` is the standard Ultralytics pretrained path: matching parameter
names and shapes are loaded, while new or incompatible layers stay randomly
initialized. Comment out that line only for an intentional scratch run.

## B0 optimizer identity

The matched B0 protocol uses `optimizer="auto"`, `momentum=0.937`, and three
warmup epochs. Auto selects MuSGD with an internal initial momentum of `0.9`;
the warmup loop then interpolates optimizer groups from `0.8` to the unchanged
training argument `0.937`. Do not replace this with explicit
`MuSGD(momentum=0.9)`.

## Remote checks and run

```bash
/opt/conda/bin/python scripts/check_dataset.py --data configs/japan7_remote.yaml
CUDA_VISIBLE_DEVICES=-1 /opt/conda/bin/python scripts/audit_optimizer_groups.py
/opt/conda/bin/python train.py
```
