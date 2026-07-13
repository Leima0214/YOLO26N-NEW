# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import copy_attr

from .tasks import load_checkpoint


class FeatureHook:
    """Picklable forward hook that stores a layer output in a shared dictionary."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """Train a YOLO student with score-weighted neck features from a frozen teacher."""

    def __init__(self, teacher_model: str | Path | nn.Module, student_model: nn.Module):
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
            if teacher_model.yaml.get("channels", 3) != ch:
                weights = teacher_model
                teacher_model = type(weights)(weights.yaml.copy(), ch=ch, nc=weights.yaml["nc"], verbose=False)
                teacher_model.load(weights)

        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.feats_idx = self.get_distill_layers(student_model)

        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()

        imgsz = student_model.args.imgsz
        student_model.eval()
        with torch.no_grad():
            im = torch.zeros(2, ch, imgsz, imgsz, device=device)
            self.teacher_model(im)
            student_model(im)
        student_model.train()
        teacher_output = [self._teacher_feats[idx] for idx in self.feats_idx]
        student_output = [self._student_feats[idx] for idx in self.feats_idx]

        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis
        projectors = []
        for student_out, teacher_out in zip(student_output[:-1], teacher_output[:-1]):
            student_dim = self.decouple_outputs(student_out).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out).shape[1]
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)

    def __getstate__(self):
        """Return picklable state without captured tensors or hook handles."""
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        """Restore feature dictionaries and hooks after unpickling."""
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self) -> None:
        for handle in self._student_hooks:
            handle.remove()
        self._student_hooks.clear()
        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()
            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    def _register_feature_hooks(self) -> None:
        self._remove_feature_hooks()
        for idx in self.feats_idx:
            self._clear_feature_hooks(self.student_model.model[idx])
            self._student_hooks.append(
                self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))
            )
            if self.teacher_model is not None:
                self._clear_feature_hooks(self.teacher_model.model[idx])
                self._teacher_hooks.append(
                    self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
                )

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        """Return the three Detect input layers and the Detect layer itself."""
        for module in model.model:
            if isinstance(module, Detect):
                return [*list(module.f), module.i]
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self) -> None:
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
            return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, loss_distill])

        self._teacher_feats.clear()
        self._student_feats.clear()
        with torch.no_grad():
            self.teacher_model(batch["img"])
        preds = self.student_model(batch["img"])

        regular_loss, regular_loss_detach = self.student_model.loss(batch, preds)
        teacher_head_feat = self._teacher_feats[self.feats_idx[-1]]
        teacher_scores = (
            self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
            + self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
        ) / 2
        neck_feats = [self._teacher_feats[idx] for idx in self.feats_idx[:-1]]
        parts = torch.split(teacher_scores, [feat.shape[-2] * feat.shape[-1] for feat in neck_feats], dim=-1)
        teacher_scores = tuple(part.sigmoid().max(dim=1, keepdim=True).values for part in parts)
        for i, feat_idx in enumerate(self.feats_idx[:-1]):
            teacher_feat = self.decouple_outputs(self._teacher_feats[feat_idx])
            student_feat = self.projector[i](self.decouple_outputs(self._student_feats[feat_idx]))
            loss_distill += self.loss_sl2(student_feat, teacher_feat, i, teacher_scores) * self.dis

        distill_loss_detach = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_distill]), torch.cat([regular_loss_detach, distill_loss_detach])

    @staticmethod
    def loss_sl2(
        student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
    ) -> torch.Tensor:
        """Compute the official score-weighted L2 feature loss for one neck level."""
        teacher_score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        return (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)

    @property
    def criterion(self):
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        self.student_model.criterion = value

    def init_criterion(self):
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        self.student_model.set_head_attr(**kwargs)

    @staticmethod
    def decouple_outputs(preds, branch: str = "one2one"):
        if isinstance(preds, tuple):
            preds = preds[1]
        if isinstance(preds, dict) and branch in preds:
            preds = preds[branch]
        return preds
