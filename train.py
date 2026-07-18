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


def load_mobilemamba_pretrained(model, weights: str) -> tuple[int, int, float]:
    """Transfer the compatible YOLO26n neck/head weights into the MobileMamba topology."""
    if len(model.model) != 19:
        raise RuntimeError(f"Unexpected MobileMamba topology: expected 19 layers, got {len(model.model)}")

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
    transferred = sum(parameters[key].numel() for key in matched if key in parameters)
    total = sum(parameter.numel() for parameter in parameters.values())
    ratio = transferred / total
    if ratio < 0.50:
        raise RuntimeError(f"Unsafe pretrained transfer coverage: {ratio:.2%} < 50%")
    LOGGER.info(f"MobileMamba pretrained transfer: {len(matched)}/{len(target)} items, {ratio:.2%} parameters")
    return len(matched), len(target), ratio


class MobileMambaTrainer(DetectionTrainer):
    """Build the final Japan7 model first, then apply the audited semantic transfer."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights=weights, verbose=verbose)
        if not weights and PRETRAINED:
            matched, total, ratio = load_mobilemamba_pretrained(model, PRETRAINED)
            (self.save_dir / "pretrained_transfer.txt").write_text(
                f"source={PRETRAINED}\nmatched_items={matched}/{total}\nparameter_ratio={ratio:.6f}\n",
                encoding="utf-8",
            )
        return model


def train():
    epochs = int(os.getenv("EPOCHS", "50"))
    model = YOLO(MODEL)
    trainer = None
    if PRETRAINED and Path(MODEL).name == MOBILEMAMBA_YAML:
        model.overrides["pretrained"] = PRETRAINED
        trainer = MobileMambaTrainer
    elif PRETRAINED and Path(MODEL).suffix in {".yaml", ".yml"}:
        model.load(PRETRAINED)

    return model.train(
        data="configs/japan7_remote.yaml",
        project="/root/YOLO26N-NEW/runs/paper1",
        name=os.getenv("RUN_NAME", f"mobilemamba_pretrained_auto_japan7_{epochs}e"),
        epochs=epochs,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,
        seed=42,
        amp=True,
        optimizer="auto",
        trainer=trainer,
    )


if __name__ == "__main__":
    train()
