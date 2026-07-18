from ultralytics import YOLO

# 切换实验时只改这两行。
MODEL = "ultralytics/cfg/models/26/yolo26.yaml"
RUN_NAME = "paper1_model_pretrained_auto_japan7_30e_seed42"

model = YOLO(MODEL)
model.load("yolo26n.pt")  # 如需从零训练，删除或注释此行。

results = model.train(
    data="configs/japan7_remote.yaml",
    project="runs/paper1",
    name=RUN_NAME,
    epochs=30,
    imgsz=640,
    batch=32,
    device=0,
    workers=8,
    seed=42,
    deterministic=True,
    amp=True,
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    close_mosaic=10,
    iou=0.7,
    max_det=300,
)
