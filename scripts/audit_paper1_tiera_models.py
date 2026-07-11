"""Adversarial runtime audit for the Paper 1 Tier A composite models."""

from __future__ import annotations

import csv
import importlib.util
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


MODEL_DIR = ROOT / "ultralytics" / "cfg" / "models" / "26"
REPORT_CSV = ROOT / "experiments" / "module_scan" / "paper1_tiera_adversarial_audit.csv"
REPORT_MD = ROOT / "experiments" / "module_scan" / "paper1_tiera_adversarial_audit.md"


def assert_close(actual, expected, name, tolerance=1e-6):
    if not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        error = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} is not identity-initialized, max error={error}")


def assert_finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
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
        assert torch.isfinite(x.grad).all()
        x.grad.zero_()

    reference = Conv(16, 32, 3, 2).eval()
    for candidate in (LaplacianConv(16, 32, 3, 2), FDConv(16, 32, 3, 2)):
        candidate.load_state_dict(reference.state_dict(), strict=False)
        candidate.eval()
        assert_close(candidate(x), reference(x), candidate.__class__.__name__, tolerance=2e-6)

    tensors = [torch.randn(2, 8, 5, 7), torch.randn(2, 12, 5, 7)]
    bifpn = Concat_bifpn(2, 1)
    assert_close(bifpn(tensors), torch.cat(tensors, 1), "Concat_bifpn")

    ffa = FFAFusionConcat([8, 12], 1, 7, 16, 0.0).eval()
    assert_close(ffa(tensors), torch.cat(tensors, 1), "FFAFusionConcat", tolerance=2e-6)

    carafe_input = torch.randn(2, 8, 3, 5, requires_grad=True)
    carafe = CARAFE(8, 8, 3, 2).eval()
    carafe_output = carafe(carafe_input)
    assert carafe_output.shape == (2, 8, 6, 10)
    assert_close(carafe_output, F.interpolate(carafe_input, scale_factor=2, mode="nearest"), "CARAFE")
    carafe_output.sum().backward()
    assert torch.isfinite(carafe_input.grad).all()

    spd_input = torch.randn(2, 8, 5, 7, requires_grad=True)
    spd_output = SPDConv(8, 16)(spd_input)
    assert spd_output.shape == (2, 16, 3, 4)
    spd_output.sum().backward()
    assert torch.isfinite(spd_input.grad).all()

    concurrent = EMA_attention(16, factor=4).eval()
    concurrent_input = torch.randn(1, 16, 5, 7)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: concurrent(concurrent_input), range(8)))
    for output in outputs:
        assert_close(output, concurrent_input, "EMA concurrent forward")

    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = FDConv(8, 8, 3, 1)(torch.randn(1, 8, 5, 7))
    assert_finite(mixed)


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
        assert len({name for name, _ in reservations}) == 8
        assert all(path.is_dir() for _, path in reservations)
    pilot.PROJECT = old_project

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        directory = Path(directory)
        base = directory / "yolo26-audit.yaml"
        exact = directory / "yolo26n-audit.yaml"
        base.write_text("audit_marker: stale\n", encoding="utf-8")
        exact.write_text("audit_marker: exact\n", encoding="utf-8")
        assert yaml_model_load(exact)["audit_marker"] == "exact"

    model_path = MODEL_DIR / "yolo26n-Paper1-TierA05-LaplacianConv-EMA-P3f8-BiFPN.yaml"
    model = YOLO(str(model_path), task="detect")
    before = next(model.model.parameters()).detach().clone()
    try:
        pilot.load_pretrained(model, ROOT / "yolo26n.pt", "auto", {"../../bad": "model.1"})
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe transfer did not fail")
    assert torch.equal(before, next(model.model.parameters()).detach())


def audit_models_and_transfer(pilot):
    paths = sorted(MODEL_DIR.glob("yolo26n-Paper1-TierA*.yaml"))
    assert len(paths) == 12
    checkpoint = ROOT / "yolo26n.pt"
    assert checkpoint.exists(), f"Missing audit checkpoint: {checkpoint}"
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert isinstance(config.get("pretrained_map"), dict)
        model = YOLO(str(path), task="detect")
        modules = list(model.model.modules())
        ffa_count = sum(module.__class__.__name__ == "FFAFusionConcat" for module in modules)
        if "FFAFusion" in path.name:
            assert ffa_count == 1, (path.name, ffa_count)
        model.model.train()
        model_input = torch.randn(2, 3, 64, 64, requires_grad=True)
        output = model.model(model_input)
        assert_finite(output)
        tensor_sum(output).backward()
        assert model_input.grad is not None and torch.isfinite(model_input.grad).all()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.model.parameters() if parameter.grad is not None)
        model.model.zero_grad(set_to_none=True)
        report = pilot.load_pretrained(model, checkpoint, "auto", config["pretrained_map"])
        assert report["transfer_numel_ratio"] >= 0.85, (path.name, report)
        assert report["backbone_transfer_ratio"] >= 0.98, (path.name, report)
        assert report["detect_transfer_ratio"] >= 0.86, (path.name, report)
        parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
        model.model.eval()
        fuse_input = torch.randn(1, 3, 64, 64)
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


def write_reports(rows):
    fields = ["yaml", "params", "transfer_numel_ratio", "backbone", "neck", "detect", "ffa_instances"]
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
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
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    pilot = load_pilot_module()
    audit_identity_and_boundaries()
    audit_security(pilot)
    rows = audit_models_and_transfer(pilot)
    write_reports(rows)
    for name, params, overall, backbone, neck, detect, _ in rows:
        print(
            f"AUDIT_OK {name} params={params} transfer={overall:.3%} "
            f"backbone={backbone:.3%} neck={neck:.3%} detect={detect:.3%}"
        )
    print("PAPER1_TIER_A_ADVERSARIAL_AUDIT_OK=12")


if __name__ == "__main__":
    main()
