import hashlib
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import intersect_dicts

MODEL = "ultralytics/cfg/models/26/yolo26-MobileMamba-Backbone.yaml"
PRETRAINED = os.getenv("PRETRAINED", "yolo26n.pt").strip()
MOBILEMAMBA_YAML = "yolo26-MobileMamba-Backbone.yaml"
MOBILEMAMBA_LAYER_MAP = {source: source - 5 for source in range(9, 24)}


def load_mobilemamba_pretrained(model, weights: str) -> dict[str, int | float | str]:
    """Transfer the compatible YOLO26n neck/head weights into the MobileMamba topology."""
    if len(model.model) != 19:
        raise RuntimeError(f"Unexpected MobileMamba topology: expected 19 layers, got {len(model.model)}")
    weights_path = Path(weights)
    if weights_path.exists():
        with weights_path.open("rb") as weights_file:
            if weights_file.read(32).startswith(b"version https://git-lfs"):
                raise RuntimeError(f"{weights} is a Git LFS pointer; replace it with the real checkpoint")

    source_model, _ = load_checkpoint(weights)
    remapped = {}
    for key, value in source_model.float().state_dict().items():
        parts = key.split(".", 2)
        if len(parts) == 3 and parts[0] == "model" and parts[1].isdigit():
            target_index = MOBILEMAMBA_LAYER_MAP.get(int(parts[1]))
            if target_index is not None:
                remapped[f"model.{target_index}.{parts[2]}"] = value

    target = model.state_dict()
    matched = intersect_dicts(remapped, target)
    model.load_state_dict(matched, strict=False)

    parameters = dict(model.named_parameters())
    matched_parameters = set(matched) & set(parameters)
    transferred = sum(parameters[key].numel() for key in matched_parameters)
    total = sum(parameter.numel() for parameter in parameters.values())
    ratio = transferred / total
    backbone_end, detect_index = len(model.yaml["backbone"]), len(model.model) - 1

    def region_ratio(indexes: set[int]) -> float:
        keys = [key for key in parameters if int(key.split(".", 2)[1]) in indexes]
        region_total = sum(parameters[key].numel() for key in keys)
        return sum(parameters[key].numel() for key in keys if key in matched_parameters) / region_total

    with weights_path.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    report = {
        "source_sha256": source_sha256,
        "matched_items": len(matched),
        "total_items": len(target),
        "parameter_ratio": ratio,
        "backbone_ratio": region_ratio(set(range(backbone_end))),
        "neck_ratio": region_ratio(set(range(backbone_end, detect_index))),
        "detect_ratio": region_ratio({detect_index}),
    }
    if report["parameter_ratio"] < 0.50 or report["neck_ratio"] < 0.99 or report["detect_ratio"] < 0.60:
        raise RuntimeError(f"Unsafe pretrained transfer coverage: {report}")
    LOGGER.info(
        f"MobileMamba pretrained transfer: {report['matched_items']}/{report['total_items']} items, "
        f"{report['parameter_ratio']:.2%} parameters, {report['neck_ratio']:.2%} neck, "
        f"{report['detect_ratio']:.2%} Detect"
    )
    return report


class MobileMambaTrainer(DetectionTrainer):
    """Build the final Japan7 model first, then apply the audited semantic transfer."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights=weights, verbose=verbose)
        if not weights and PRETRAINED:
            report = load_mobilemamba_pretrained(model, PRETRAINED)
            (self.save_dir / "pretrained_transfer.txt").write_text(
                "\n".join(
                    [
                        f"source={PRETRAINED}",
                        f"source_sha256={report['source_sha256']}",
                        "semantic_map=model.9-23 -> model.4-18",
                        f"matched_items={report['matched_items']}/{report['total_items']}",
                        f"parameter_ratio={report['parameter_ratio']:.6f}",
                        f"backbone_ratio={report['backbone_ratio']:.6f}",
                        f"neck_ratio={report['neck_ratio']:.6f}",
                        f"detect_ratio={report['detect_ratio']:.6f}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        return model


def train():
    epochs = int(os.getenv("EPOCHS", "30"))
    model = YOLO(MODEL)
    trainer = None
    is_mobilemamba = Path(MODEL).name == MOBILEMAMBA_YAML
    if PRETRAINED and is_mobilemamba:
        model.overrides["pretrained"] = PRETRAINED
        trainer = MobileMambaTrainer
    elif PRETRAINED and Path(MODEL).suffix in {".yaml", ".yml"}:
        model.load(PRETRAINED)

    initialization = (
        "partial_pretrained" if PRETRAINED and is_mobilemamba else "pretrained" if PRETRAINED else "scratch"
    )
    model_name = "mobilemamba" if is_mobilemamba else Path(MODEL).stem.lower()
    default_name = f"{model_name}_{initialization}_auto_japan7_{epochs}e_seed42"
    run_name = os.getenv("RUN_NAME", default_name)
    if initialization not in run_name:
        raise ValueError(f"RUN_NAME must contain initialization identity {initialization!r}: {run_name!r}")

    return model.train(
        data="configs/japan7_remote.yaml",
        project="/root/YOLO26N-NEW/runs/paper1",
        name=run_name,
        epochs=epochs,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        seed=42,
        deterministic=True,
        amp=True,
        optimizer="auto",
        lrf=0.01,
        warmup_epochs=3.0,
        weight_decay=0.0005,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        iou=0.7,
        max_det=300,
        trainer=trainer,
    )


if __name__ == "__main__":
    train()
