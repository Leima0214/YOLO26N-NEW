#!/usr/bin/env python3
"""Audit the WPFormer-WCA-inspired Paper 1 single-module candidate."""

from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import torch

from ultralytics.nn.wpformer_wavelet import (
    WaveletDetailRefinement,
    haar_decompose,
    haar_reconstruct,
    wavelet_context,
)


MODEL_YAML = ROOT / "ultralytics/cfg/models/26/yolo26n-Paper1-S4-WPFormer-WDR-P3.yaml"
CHECKPOINT = ROOT / "yolo26n.pt"
REPORT = ROOT / "experiments/module_scan/paper1_s4_wpformer_wdr_audit.md"


def load_pilot_module():
    spec = importlib.util.spec_from_file_location("wpformer_pilot", ROOT / "scripts/train_module_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_mean(value):
    if isinstance(value, torch.Tensor):
        return value.float().mean()
    if isinstance(value, dict):
        return sum((tensor_mean(item) for item in value.values()), torch.tensor(0.0))
    if isinstance(value, (list, tuple)):
        return sum((tensor_mean(item) for item in value), torch.tensor(0.0))
    return torch.tensor(0.0)


def assert_finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    assert MODEL_YAML.exists() and CHECKPOINT.exists()
    config = yaml.safe_load(MODEL_YAML.read_text(encoding="utf-8"))
    assert isinstance(config, dict) and isinstance(config.get("pretrained_map"), dict)

    odd = torch.randn(2, 16, 7, 9, requires_grad=True)
    roundtrip = haar_reconstruct(*haar_decompose(odd))[..., :7, :9]
    roundtrip_error = float((roundtrip - odd).abs().max().detach())
    assert roundtrip_error < 1e-6
    even = torch.randn(2, 16, 8, 10)
    bands = haar_decompose(even)
    context = wavelet_context(*bands)
    signed_sum = sum(bands)
    aliased_sample = 2 * even[..., 1::2, 1::2]
    assert torch.allclose(signed_sum, aliased_sample, atol=1e-6)
    assert not torch.allclose(context, aliased_sample, atol=1e-4)
    for malformed in (
        (bands[0], bands[1][..., :-1], bands[2], bands[3]),
        (bands[0], bands[1].double(), bands[2], bands[3]),
        (bands[0], bands[1], bands[2], "malicious"),
    ):
        try:
            haar_reconstruct(*malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed Haar bands were accepted")

    for constructor in (
        lambda: WaveletDetailRefinement(16, 8),
        lambda: WaveletDetailRefinement(16, 16, 0),
        lambda: WaveletDetailRefinement(True),
    ):
        try:
            constructor()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid WaveletDetailRefinement configuration was accepted")

    module = WaveletDetailRefinement(16, 16, 4)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    initial = module(odd)
    identity_error = float((initial - odd).abs().max().detach())
    assert identity_error == 0.0
    initial.square().mean().backward()
    projection_grad = float(module.out_proj.weight.grad.abs().sum())
    assert projection_grad > 0 and torch.isfinite(module.out_proj.weight.grad).all()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module(odd).square().mean().backward()
    context_grads = [parameter.grad for name, parameter in module.named_parameters() if "context" in name]
    assert any(grad is not None and grad.abs().sum() > 0 for grad in context_grads)

    boundary_module = WaveletDetailRefinement(16, 16, 4).eval()
    state_before_failure = {name: value.clone() for name, value in boundary_module.state_dict().items()}
    singleton = torch.randn(1, 16, 1, 1)
    assert torch.equal(boundary_module(singleton), singleton)
    try:
        boundary_module(torch.randn(1, 8, 5, 7))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-channel input was accepted")
    for invalid in (torch.ones(1, 16, 5, 7, dtype=torch.int64), torch.empty(1, 16, 0, 7), "malicious"):
        try:
            boundary_module(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input was accepted")
    assert all(torch.equal(value, state_before_failure[name]) for name, value in boundary_module.state_dict().items())
    large_finite = boundary_module(torch.full((1, 16, 5, 7), 1e10))
    assert torch.isfinite(large_finite).all()
    concurrent_input = torch.randn(1, 16, 7, 9)
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_outputs = list(pool.map(lambda _: boundary_module(concurrent_input), range(8)))
    assert all(torch.equal(output, concurrent_input) for output in concurrent_outputs)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = WaveletDetailRefinement(16, 16, 4)(torch.randn(1, 16, 5, 7))
    assert_finite(mixed)

    candidate = YOLO(str(MODEL_YAML), task="detect")
    modules = list(candidate.model.modules())
    assert sum(isinstance(item, WaveletDetailRefinement) for item in modules) == 1
    assert candidate.model.stride.tolist() == [8.0, 16.0, 32.0]
    params = sum(parameter.numel() for parameter in candidate.model.parameters())

    candidate.model.train()
    full_input = torch.randn(1, 3, 640, 640, requires_grad=True)
    full_output = candidate.model(full_input)
    assert_finite(full_output)
    tensor_mean(full_output).backward()
    assert full_input.grad is not None and torch.isfinite(full_input.grad).all()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in candidate.model.parameters()
        if parameter.grad is not None
    )
    candidate.model.zero_grad(set_to_none=True)

    pilot = load_pilot_module()
    transfer = pilot.load_pretrained(candidate, CHECKPOINT, "auto", config["pretrained_map"])
    assert transfer["transfer_numel_ratio"] >= 0.995
    assert transfer["backbone_transfer_ratio"] == 1.0
    assert transfer["neck_transfer_ratio"] >= 0.99
    assert transfer["detect_transfer_ratio"] == 1.0

    baseline = YOLO(str(CHECKPOINT), task="detect")
    baseline.model.eval()
    candidate.model.eval()
    compare_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        baseline_prediction = baseline.model(compare_input)[0]
        candidate_prediction = candidate.model(compare_input)[0]
    baseline_error = float((candidate_prediction - baseline_prediction).abs().max())
    assert baseline_error < 1e-5

    fuse_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        before_fuse = candidate.model(fuse_input)[1]["one2one"]
    candidate.model.fuse(verbose=False)
    with torch.no_grad():
        after_fuse = candidate.model(fuse_input)[1]["one2one"]
    fuse_error = max(
        float((after_fuse[name] - before_fuse[name]).abs().max()) for name in ("boxes", "scores")
    )
    assert all(
        torch.allclose(after_fuse[name], before_fuse[name], atol=1e-4, rtol=1e-4)
        for name in ("boxes", "scores")
    )

    cuda_amp = "not available on this workstation; remote 1e smoke required"
    if torch.cuda.is_available():
        cuda_module = WaveletDetailRefinement(16, 16, 4).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            cuda_output = cuda_module(torch.randn(1, 16, 7, 9, device="cuda"))
        assert_finite(cuda_output)
        cuda_amp = "PASS"

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report_text = (
        "\n".join(
            [
                "# Paper 1 S4 WPFormer-WDR Audit",
                "",
                "This candidate independently adapts WPFormer's WCA frequency modulation to a YOLO feature map.",
                "It is not the full query-based WPFormer and does not include PCA.",
                "",
                "| check | result |",
                "| --- | --- |",
                f"| YAML safe load and model build | PASS |",
                f"| module instances | 1 |",
                f"| Detect strides | 8/16/32 |",
                f"| parameters (`nc=80` build) | {params} |",
                f"| Haar odd-size roundtrip max error | {roundtrip_error:.3e} |",
                f"| identity-init max error | {identity_error:.3e} |",
                f"| signed-subband cancellation regression | PASS |",
                f"| first-step output projection gradient L1 | {projection_grad:.6f} |",
                f"| second-step context gradient | PASS |",
                f"| invalid configuration/input rejection | PASS |",
                f"| failure leaves state unchanged | PASS |",
                f"| 1x1, large finite input, and concurrent re-entry | PASS |",
                f"| full 640x640 forward/backward | PASS |",
                f"| CPU bfloat16 | PASS |",
                f"| CUDA AMP | {cuda_amp} |",
                f"| parameter transfer | {transfer['transfer_numel_ratio']:.6%} |",
                f"| backbone transfer | {transfer['backbone_transfer_ratio']:.6%} |",
                f"| neck transfer | {transfer['neck_transfer_ratio']:.6%} |",
                f"| Detect transfer | {transfer['detect_transfer_ratio']:.6%} |",
                f"| pretrained full-model equivalence max error | {baseline_error:.3e} |",
                f"| fused prediction max error | {fuse_error:.3e} |",
                "",
                "Build and local numerical audits do not establish an accuracy gain. Run one remote CUDA AMP smoke before 30e.",
                "",
            ]
        )
    )
    temporary_report = REPORT.with_suffix(f".{os.getpid()}.tmp")
    temporary_report.write_text(report_text, encoding="utf-8")
    temporary_report.replace(REPORT)
    print(f"WROTE {REPORT}")
    print("PAPER1_S4_WPFORMER_WDR_AUDIT_OK")


if __name__ == "__main__":
    main()
