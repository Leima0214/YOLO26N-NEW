from pathlib import Path

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA, strip_optimizer


def test_yolo26_distillation_is_training_only(tmp_path: Path):
    student = DetectionModel("ultralytics/cfg/models/26/yolo26.yaml", nc=7, verbose=False)
    teacher = DetectionModel("ultralytics/cfg/models/26/yolo26s-Paper1-Teacher.yaml", nc=7, verbose=False)
    args = get_cfg()
    args.imgsz = 32
    args.dis = 6.0
    student.args = teacher.args = args
    student.names = teacher.names = {i: f"class_{i}" for i in range(7)}

    model = DistillationModel(teacher, student).train()
    batch = {
        "img": torch.rand(2, 3, 32, 32),
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
    }
    loss, loss_items = model(batch)
    assert loss.shape == loss_items.shape == (4,)
    assert torch.isfinite(loss).all() and torch.isfinite(loss_items).all()

    loss.sum().backward()
    assert all(param.grad is None for param in model.teacher_model.parameters())
    assert any(param.grad is not None for param in model.student_model.parameters() if param.requires_grad)
    assert any(param.grad is not None for param in model.projector.parameters())

    ema = ModelEMA(model)
    assert ema.ema.teacher_model is None
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": None, "ema": ema.ema, "train_args": vars(args)}, checkpoint)
    stripped = strip_optimizer(checkpoint)
    assert isinstance(stripped["model"], DetectionModel)
    assert not isinstance(stripped["model"], DistillationModel)
