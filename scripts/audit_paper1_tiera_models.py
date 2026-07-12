"""Adversarial runtime audit for the Paper 1 Tier A composite models."""

from __future__ import annotations

import csv
import importlib.util
import io
import math
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import torch
import torch.nn.functional as F
from ultralytics.nn.CARAFE import CARAFE
from ultralytics.nn.CBAM import CBAM
from ultralytics.nn.EMA_attention import EMA_attention
from ultralytics.nn.FDConv import FDConv
from ultralytics.nn.LaplacianConv import LaplacianConv
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.se import SEAttention
from ultralytics.nn.spdconv import SPDConv
from ultralytics.nn.tasks import resolve_activation, yaml_model_load
from ultralytics.nn.weighted_concat import Concat_bifpn
from ultralytics.nn.yolo26_ffafusion import FFAFusionConcat
from ultralytics.nn.yolo26_ffafusion.modules import _wrap_axial


MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "26"
REPORT_CSV = ROOT / "experiments" / "module_scan" / "paper1_tiera_adversarial_audit.csv"
REPORT_MD = ROOT / "experiments" / "module_scan" / "paper1_tiera_adversarial_audit.md"


def require(condition, message):
    if isinstance(condition, torch.Tensor):
        condition = bool(condition.detach().all().item())
    if not condition:
        raise AssertionError(message)


def assert_close(actual, expected, name, tolerance=1e-6):
    if not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        error = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} is not identity-initialized, max error={error}")


def assert_finite(value):
    if isinstance(value, torch.Tensor):
        require(torch.isfinite(value).all(), "non-finite tensor detected")
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)


def tensor_sum(value):
    if isinstance(value, torch.Tensor):
        return value.float().mean()
    if isinstance(value, dict):
        return sum((tensor_sum(item) for item in value.values()), torch.tensor(0.0))
    if isinstance(value, (list, tuple)):
        return sum((tensor_sum(item) for item in value), torch.tensor(0.0))
    return torch.tensor(0.0)


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return sum((tensors(item) for item in value.values()), [])
    if isinstance(value, (list, tuple)):
        return sum((tensors(item) for item in value), [])
    return []


def audit_identity_and_boundaries():
    x = torch.randn(2, 16, 7, 9, requires_grad=True)
    modules = (EMA_attention(16, factor=4), SEAttention(16), CBAM(16, 16))
    for module in modules:
        module.eval()
        output = module(x)
        assert_close(output, x, module.__class__.__name__)
        output.sum().backward(retain_graph=True)
        require(torch.isfinite(x.grad).all(), f"{module.__class__.__name__} produced a non-finite input gradient")
        x.grad.zero_()

    reference = Conv(16, 32, 3, 2).eval()
    for candidate in (LaplacianConv(16, 32, 3, 2), FDConv(16, 32, 3, 2)):
        candidate.load_state_dict(reference.state_dict(), strict=False)
        candidate.eval()
        assert_close(candidate(x), reference(x), candidate.__class__.__name__, tolerance=2e-6)

    tensors = [torch.randn(2, 8, 5, 7), torch.randn(2, 12, 5, 7)]
    bifpn = Concat_bifpn(2, 1)
    assert_close(bifpn(tensors), torch.cat(tensors, 1), "Concat_bifpn")
    with torch.no_grad():
        bifpn.w.copy_(torch.tensor([1e30, -1e30]))
    assert_finite(bifpn(tensors))

    axial = _wrap_axial(torch.tensor([math.radians(178.0)]))
    assert_close(axial, torch.tensor([math.radians(-2.0)]), "axial angle wrapping")

    ffa = FFAFusionConcat([8, 12], 1, 7, 16, 0.0).eval()
    assert_close(ffa(tensors), torch.cat(tensors, 1), "FFAFusionConcat", tolerance=2e-6)

    carafe_input = torch.randn(2, 8, 3, 5, requires_grad=True)
    carafe = CARAFE(8, 8, 3, 2).eval()
    carafe_output = carafe(carafe_input)
    require(carafe_output.shape == (2, 8, 6, 10), f"CARAFE output shape mismatch: {carafe_output.shape}")
    assert_close(carafe_output, F.interpolate(carafe_input, scale_factor=2, mode="nearest"), "CARAFE")
    carafe_output.sum().backward()
    require(torch.isfinite(carafe_input.grad).all(), "CARAFE produced a non-finite input gradient")
    with torch.no_grad():
        carafe.gamma.fill_(0.3)
        logits = carafe.encoder(carafe.down(carafe_input.detach()))
        kernels = F.pixel_shuffle(logits, 2).softmax(dim=1)
        patches = F.unfold(carafe_input.detach(), 3, padding=1).reshape(2, 8, 9, 3, 5)
        patches = patches.repeat_interleave(2, 3).repeat_interleave(2, 4)
        legacy = (patches * kernels.unsqueeze(1)).sum(2)
        nearest = F.interpolate(carafe_input.detach(), scale_factor=2, mode="nearest")
        expected = carafe.out(nearest + torch.tanh(carafe.gamma) * (legacy - nearest))
        assert_close(carafe(carafe_input.detach()), expected, "CARAFE no-repeat reassembly", tolerance=2e-6)
    try:
        CARAFE(8, 8, 3, 2, max_workspace_bytes=1)(carafe_input.detach())
    except RuntimeError:
        pass
    else:
        raise AssertionError("CARAFE workspace guard did not reject an oversized allocation")

    spd_input = torch.randn(2, 8, 5, 7, requires_grad=True)
    spd_output = SPDConv(8, 16)(spd_input)
    require(spd_output.shape == (2, 16, 3, 4), f"SPDConv output shape mismatch: {spd_output.shape}")
    spd_output.sum().backward()
    require(torch.isfinite(spd_input.grad).all(), "SPDConv produced a non-finite input gradient")

    concurrent = EMA_attention(16, factor=4).eval()
    concurrent_input = torch.randn(1, 16, 5, 7)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: concurrent(concurrent_input), range(8)))
    for output in outputs:
        assert_close(output, concurrent_input, "EMA concurrent forward")

    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = FDConv(8, 8, 3, 1)(torch.randn(1, 8, 5, 7))
    assert_finite(mixed)
    fdconv = FDConv(8, 8, 3, 1).eval()
    with torch.no_grad():
        fdconv.gamma.fill_(0.5)
    conv_calls = []
    hook = fdconv.conv.register_forward_hook(lambda *_: conv_calls.append(1))
    fdconv(torch.randn(1, 8, 5, 7))
    hook.remove()
    require(len(conv_calls) == 1, f"FDConv executed {len(conv_calls)} convolutions instead of one")


def load_pilot_module():
    spec = importlib.util.spec_from_file_location("tier_a_pilot", ROOT / "scripts" / "train_module_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_security(pilot):
    try:
        resolve_activation("__import__('os').system('echo unsafe')")
    except ValueError:
        pass
    else:
        raise AssertionError("malicious activation was accepted")

    for mapping in ({"../../source": "model.1"}, {"model.1": "model.2", "model.3": "model.2"}):
        try:
            pilot.validate_pretrained_map(mapping)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe pretrained map was accepted: {mapping}")

    for name in ("../escape", "a/b", "", "x" * 129):
        try:
            pilot.validate_run_name(name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe run name was accepted: {name!r}")

    old_project = pilot.PROJECT
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        pilot.PROJECT = Path(directory)
        with ThreadPoolExecutor(max_workers=4) as pool:
            reservations = list(pool.map(lambda _: pilot.reserve_run("same-name"), range(8)))
        require(len({name for name, _ in reservations}) == 8, "concurrent run reservations collided")
        require(all(path.is_dir() for _, path in reservations), "reserved run directory is missing")
    pilot.PROJECT = old_project

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        directory = Path(directory)
        base = directory / "yolo26-audit.yaml"
        exact = directory / "yolo26n-audit.yaml"
        base.write_text("audit_marker: stale\n", encoding="utf-8")
        exact.write_text("audit_marker: exact\n", encoding="utf-8")
        require(yaml_model_load(exact)["audit_marker"] == "exact", "exact YAML path did not win")

    model_path = MODEL_DIR / "yolo26n-Paper1-TierA05-LaplacianConv-EMA-P3f8-BiFPN.yaml"
    model = YOLO(str(model_path), task="detect")
    before = next(model.model.parameters()).detach().clone()
    try:
        pilot.load_pretrained(model, ROOT / "yolo26n.pt", "auto", {"../../bad": "model.1"})
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe transfer did not fail")
    require(torch.equal(before, next(model.model.parameters()).detach()), "failed transfer mutated the model")


def audit_models_and_transfer(pilot):
    paths = sorted(MODEL_DIR.glob("yolo26n-Paper1-TierA*.yaml"))
    require(len(paths) == 12, f"expected 12 Tier A YAMLs, found {len(paths)}")
    checkpoint = ROOT / "yolo26n.pt"
    require(checkpoint.exists(), f"Missing audit checkpoint: {checkpoint}")
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        require(isinstance(config.get("pretrained_map"), dict), f"missing pretrained map: {path.name}")
        model = YOLO(str(path), task="detect")
        modules = list(model.model.modules())
        ffa_count = sum(module.__class__.__name__ == "FFAFusionConcat" for module in modules)
        if "FFAFusion" in path.name:
            require(ffa_count == 1, f"{path.name} has {ffa_count} FFA nodes")
        model.model.train()
        model_input = torch.randn(2, 3, 64, 64, requires_grad=True)
        output = model.model(model_input)
        assert_finite(output)
        tensor_sum(output).backward()
        require(
            model_input.grad is not None and torch.isfinite(model_input.grad).all(),
            f"{path.name} produced an invalid input gradient",
        )
        require(
            all(torch.isfinite(parameter.grad).all() for parameter in model.model.parameters() if parameter.grad is not None),
            f"{path.name} produced a non-finite parameter gradient",
        )
        model.model.zero_grad(set_to_none=True)
        report = pilot.load_pretrained(model, checkpoint, "auto", config["pretrained_map"])
        require(report["transfer_numel_ratio"] >= 0.85, f"low total transfer: {path.name}: {report}")
        require(report["backbone_transfer_ratio"] >= 0.98, f"low backbone transfer: {path.name}: {report}")
        require(report["detect_transfer_ratio"] >= 0.86, f"low Detect transfer: {path.name}: {report}")
        parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
        model.model.eval()
        fuse_input = torch.linspace(-1.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)
        with torch.no_grad():
            before_fuse = model.model(fuse_input)[0]
        model.model.fuse(verbose=False)
        with torch.no_grad():
            after_fuse = model.model(fuse_input)[0]
        # End-to-end Detect intentionally drops one2many debug tensors when fused; compare deploy predictions.
        assert_close(after_fuse, before_fuse, f"{path.name} fused inference", tolerance=1e-4)
        rows.append(
            (
                path.name,
                parameter_count,
                report["transfer_numel_ratio"],
                report["backbone_transfer_ratio"],
                report["neck_transfer_ratio"],
                report["detect_transfer_ratio"],
                ffa_count,
            )
        )
    return rows


def atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_reports(rows, pilot):
    fields = ["yaml", "params", "transfer_numel_ratio", "backbone", "neck", "detect", "ffa_instances"]
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(fields)
    writer.writerows(rows)
    lines = [
        "# Paper 1 Tier A Adversarial Runtime Audit",
        "",
        "All rows passed safe YAML loading, construction, finite 64x64 forward/backward, identity-init module checks,",
        "odd-size module checks, CPU mixed precision, fused inference, concurrent re-entry, semantic transfer, and malicious-input rejection.",
        "",
        "| YAML | Params | Parameter transfer | Backbone | Neck | Detect | FFA instances |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, params, overall, backbone, neck, detect, ffa_count in rows:
        lines.append(
            f"| `{name}` | {params} | {overall:.3%} | {backbone:.3%} | {neck:.3%} | {detect:.3%} | {ffa_count} |"
        )
    with pilot.report_lock():
        atomic_write_text(REPORT_CSV, csv_buffer.getvalue())
        atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    pilot = load_pilot_module()
    audit_identity_and_boundaries()
    audit_security(pilot)
    rows = audit_models_and_transfer(pilot)
    write_reports(rows, pilot)
    for name, params, overall, backbone, neck, detect, _ in rows:
        print(
            f"AUDIT_OK {name} params={params} transfer={overall:.3%} "
            f"backbone={backbone:.3%} neck={neck:.3%} detect={detect:.3%}"
        )
    print("PAPER1_TIER_A_ADVERSARIAL_AUDIT_OK=12")


if __name__ == "__main__":
    main()
