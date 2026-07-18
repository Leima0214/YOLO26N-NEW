
import os

from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26-MobileMamba-Backbone.yaml")

results = model.train(
    data="configs/japan7_remote.yaml",
    project="/root/YOLO26N-NEW/runs/paper1",
    name=os.getenv("RUN_NAME", "mobilemamba_backbone_japan7_10e"),
    epochs=int(os.getenv("EPOCHS", "10")),
    imgsz=640,
    batch=32,
    device=0,
    workers=8,
    seed=42,
    amp=True,
    optimizer="MuSGD",
)
