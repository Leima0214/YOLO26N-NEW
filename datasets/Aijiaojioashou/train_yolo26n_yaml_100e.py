import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(r"D:/Users/D/Desktop/ultralytics26-main5.26")
DATASET = Path(r"D:/Users/D/Desktop/Aijiaojioashou")
RUN_NAME = "Aijiaojioashou_yolo26n_yaml_100e"

model = YOLO(str(DATASET / "yolo26n.yaml"))

model.train(
    data=str(DATASET / "data.yaml"),
    epochs=100,
    imgsz=640,
    batch=8,
    optimizer="SGD",
    workers=0,
    device=0,
    pretrained=str(ROOT / "yolo26n.pt"),
    project=str(ROOT / "runs" / "detect"),
    name=RUN_NAME,
    exist_ok=True,
    plots=False,
    verbose=True,
    lr0=0.0001,
    lrf=0.01,
    warmup_bias_lr=0.0001,
    mosaic=0.0,
    erasing=0.0,
    auto_augment=None,
    patience=100,
)

last = ROOT / "runs" / "detect" / RUN_NAME / "weights" / "last.pt"
metrics = YOLO(str(last)).val(
    data=str(DATASET / "data.yaml"),
    imgsz=640,
    batch=8,
    workers=0,
    device=0,
    split="val",
    project=str(ROOT / "runs" / "detect"),
    name=RUN_NAME + "_val_last",
    exist_ok=True,
    plots=False,
)
out = DATASET / "metadata" / "yolo26n_100e_metrics.json"
payload = {
    "weights": str(last),
    "epochs": 100,
    "split": "val",
    "results_dict": metrics.results_dict,
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(metrics.results_dict)
