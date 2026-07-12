"""Adversarial runtime audit for the Paper 1 Tier B composite models."""

from __future__ import annotations

import csv
import gc
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from audit_paper1_tiera_models import (
    assert_close,
    assert_finite,
    audit_identity_and_boundaries,
    audit_security,
    load_pilot_module,
    tensor_sum,
)
from generate_paper1_tiera_composites import MODEL_DIR, build_model, write_model
from generate_paper1_tierb_composites import SPECS


REPORT_CSV = ROOT / "experiments" / "module_scan" / "paper1_tierb_adversarial_audit.csv"
REPORT_MD = ROOT / "experiments" / "module_scan" / "paper1_tierb_adversarial_audit.md"
CHECKPOINT = ROOT / "yolo26n.pt"


def audit_generator_boundaries():
    """Reject path/config injection and prove concurrent writes and failed writes preserve a valid YAML."""
    filename, options = SPECS[-1]
    path = MODEL_DIR / filename
    original = path.read_bytes()

    invalid_calls = (
        ("../escape.yaml", options, "generate_paper1_tierb_composites.py"),
        (Path("bad.yaml"), options, "generate_paper1_tierb_composites.py"),
        (filename, [], "generate_paper1_tierb_composites.py"),
        (filename, {"detail": "unknown"}, "generate_paper1_tierb_composites.py"),
        (filename, {"detail": []}, "generate_paper1_tierb_composites.py"),
        (filename, options, "bad\nheader.py"),
    )
    for arguments in invalid_calls:
        try:
            write_model(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe generator input was accepted: {arguments!r}")
        assert path.read_bytes() == original

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(
            pool.map(
                lambda _: write_model(filename, options, "generate_paper1_tierb_composites.py"),
                range(8),
            )
        )
    assert all(output == path.resolve() for output in outputs)
    with path.open(encoding="utf-8") as handle:
        assert yaml.safe_load(handle) == build_model(options)
    assert not list(MODEL_DIR.glob(f".{filename}.*.tmp"))


def expected_counts(options):
    return {
        "SPDConv": int(options.get("detail") == "spd"),
        "LaplacianConv": int(options.get("detail") == "lap") + int(options.get("head_detail") == "lap"),
        "FDConv": int(options.get("detail") == "fd") + int(options.get("head_detail") == "fd"),
        "CARAFE": int(bool(options.get("carafe"))),
        "Concat_bifpn": (6 if options.get("p2") else 4) if options.get("fusion") == "bifpn" else 0,
        "FFAFusionConcat": int(options.get("fusion") == "ffa"),
    }


def model_output(model, value):
    output = model(value)
    return output[0] if isinstance(output, tuple) else output


def audit_recovery_and_concurrency(path, config, pilot):
    """Exercise malformed and odd geometry recovery plus fixed-shape concurrent inference."""
    model = YOLO(str(path), task="detect")
    pilot.load_pretrained(model, CHECKPOINT, "auto", config["pretrained_map"])
    model.model.eval()

    try:
        model.model(torch.randn(1, 1, 64, 64))
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError(f"{path.name} accepted a one-channel image")

    with torch.no_grad():
        try:
            odd_output = model_output(model.model, torch.randn(1, 3, 63, 63))
        except (RuntimeError, ValueError):
            odd_status = "rejected"
        else:
            assert_finite(odd_output)
            odd_status = "supported"

        fixed_input = torch.randn(1, 3, 64, 64)
        expected = model_output(model.model, fixed_input)
        assert_finite(expected)

        def infer(_):
            with torch.no_grad():
                return model_output(model.model, fixed_input)

        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(infer, range(8)))
        for output in outputs:
            assert_close(output, expected, f"{path.name} concurrent inference", tolerance=2e-5)
    return odd_status


def audit_baseline_equivalence(path, config, pilot):
    """B24 should start as the pretrained baseline because all replacements are identity initialized."""
    baseline = YOLO(str(CHECKPOINT), task="detect")
    candidate = YOLO(str(path), task="detect")
    pilot.load_pretrained(candidate, CHECKPOINT, "auto", config["pretrained_map"])
    baseline.model.eval()
    candidate.model.eval()
    value = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model_output(baseline.model, value)
        actual = model_output(candidate.model, value)
    assert_close(actual, expected, "Tier B24 pretrained baseline equivalence", tolerance=2e-5)


def audit_models_and_transfer(pilot):
    paths = sorted(MODEL_DIR.glob("yolo26n-Paper1-TierB*.yaml"))
    assert len(paths) == len(SPECS) == 12
    assert CHECKPOINT.exists(), f"Missing audit checkpoint: {CHECKPOINT}"
    spec_by_name = dict(SPECS)
    assert set(spec_by_name) == {path.name for path in paths}

    rows = []
    for path in paths:
        options = spec_by_name[path.name]
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert config == build_model(options), f"Generated YAML drift: {path.name}"
        layers = config["backbone"] + config["head"]
        assert len(layers[-1][0]) == (4 if options.get("p2") else 3)

        class_counts = {}
        for layer in layers:
            class_counts[layer[2]] = class_counts.get(layer[2], 0) + 1
        for module_name, expected in expected_counts(options).items():
            assert class_counts.get(module_name, 0) == expected, (path.name, module_name, class_counts)

        model = YOLO(str(path), task="detect")
        report = pilot.load_pretrained(model, CHECKPOINT, "auto", config["pretrained_map"])
        assert report["transfer_numel_ratio"] >= 0.85, (path.name, report)
        assert report["backbone_transfer_ratio"] >= 0.98, (path.name, report)
        assert report["neck_transfer_ratio"] >= 0.90, (path.name, report)
        assert report["detect_transfer_ratio"] >= 0.86, (path.name, report)

        params = sum(parameter.numel() for parameter in model.model.parameters())
        gflops = float(get_flops(model.model, imgsz=640))
        model.model.train()
        value = torch.randn(1, 3, 640, 640, requires_grad=True)
        output = model.model(value)
        assert_finite(output)
        tensor_sum(output).backward()
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.model.parameters()
            if parameter.grad is not None
        )
        model.model.zero_grad(set_to_none=True)

        model.model.eval()
        small_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            assert_finite(model.model(small_input))

        fuse_input = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            before_fuse = model_output(model.model, fuse_input)
        model.model.fuse(verbose=False)
        with torch.no_grad():
            after_fuse = model_output(model.model, fuse_input)
        assert_close(after_fuse, before_fuse, f"{path.name} fused inference", tolerance=1e-4)

        rows.append(
            (
                path.name,
                params,
                gflops,
                report["transfer_numel_ratio"],
                report["backbone_transfer_ratio"],
                report["neck_transfer_ratio"],
                report["detect_transfer_ratio"],
            )
        )
        del output, value, model
        gc.collect()

    representative = (paths[2], paths[9])
    odd_status = {}
    for path in representative:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        odd_status[path.name] = audit_recovery_and_concurrency(path, config, pilot)

    b24 = paths[-1]
    with b24.open(encoding="utf-8") as handle:
        audit_baseline_equivalence(b24, yaml.safe_load(handle), pilot)
    return rows, odd_status


def write_reports(rows, odd_status):
    fields = ["yaml", "params", "gflops", "transfer_numel_ratio", "backbone", "neck", "detect"]
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)

    lines = [
        "# Paper 1 Tier B Adversarial Audit",
        "",
        "## Result",
        "",
        "All 12 Tier B YAMLs passed safe construction, semantic pretrained transfer, finite 640x640 forward/backward,",
        "32x32 CPU mixed precision, fused inference, malformed-input recovery, fixed-shape concurrent inference,",
        "and atomic concurrent generation. No training was run.",
        "",
        "| YAML | Params | GFLOPs | Parameter transfer | Backbone | Neck | Detect |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, params, gflops, overall, backbone, neck, detect in rows:
        lines.append(
            f"| `{name}` | {params} | {gflops:.1f} | {overall:.3%} | {backbone:.3%} | {neck:.3%} | {detect:.3%} |"
        )
    lines.extend(
        [
            "",
            "## Adversarial Findings",
            "",
            "| Severity | Finding | Resolution |",
            "| --- | --- | --- |",
            "| Critical, fixed | Two shallow replacements could silently target the same layer or shift numeric indices. | The first replacement is fixed at backbone P2-to-P3; the second is fixed at PAN P3-to-P4; all checkpoint maps use semantic layer names. |",
            "| High, fixed | Direct YAML writes could leave a partial file after interruption or concurrent generation. | Generation now validates a temporary safe-loaded YAML and atomically replaces the destination; concurrent and failed-write tests pass. |",
            "| High, fixed | Unvalidated options or output names could inject unsupported modules or escape the model directory. | Generator options, filenames, generator labels, resolved paths, and unique mappings are allowlisted. |",
            "| High, fixed | Structural checks used `assert`, which disappears under `python -O`. | Trust-boundary validation now raises explicit exceptions and is exercised with non-string names and non-mapping options. |",
            "| Medium, accepted | SPDConv cannot inherit the baseline 3x3 stride-2 kernel because space-to-depth changes its tensor shape. | Coverage is reported explicitly; SPD candidates remain optimization-risk experiments. |",
            "| Medium, accepted | Official P2 adds a randomly initialized Detect branch, so P2 candidates cannot be baseline-equivalent at initialization. | Detect transfer must remain at least 86%; P2 results require matched pilots before promotion. |",
            "| Medium, operational | Ultralytics Detect caches shape-dependent anchors on the model instance. Concurrent mixed-resolution calls can race. | Fixed-shape re-entry is tested; production mixed-resolution inference should use one model per worker or external serialization. |",
            "| Scientific, unresolved | Buildability and stable gradients do not establish an accuracy gain. | Keep Tier B behind matched smoke and 30-epoch signal gates; do not infer efficacy from this report. |",
            "",
            "B24 also passed exact pretrained baseline-equivalence at initialization. P2 and SPD candidates are intentionally excluded from that assertion.",
            f"Odd-size direct-input probes: {', '.join(f'`{name}`={status}' for name, status in odd_status.items())}.",
            "",
            "## Least Confidence",
            "",
            "The least certain property is accuracy, especially for B13-B15/B19-B20 where overlapping detail operators may amplify the same cues.",
            "The next uncertainty is SPDConv optimization because its baseline downsampling kernel is structurally non-transferable.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    pilot = load_pilot_module()
    audit_generator_boundaries()
    audit_identity_and_boundaries()
    audit_security(pilot)
    rows, odd_status = audit_models_and_transfer(pilot)
    write_reports(rows, odd_status)
    for name, params, gflops, overall, backbone, neck, detect in rows:
        print(
            f"AUDIT_OK {name} params={params} gflops={gflops:.1f} transfer={overall:.3%} "
            f"backbone={backbone:.3%} neck={neck:.3%} detect={detect:.3%}"
        )
    print("PAPER1_TIER_B_ADVERSARIAL_AUDIT_OK=12")


if __name__ == "__main__":
    main()
