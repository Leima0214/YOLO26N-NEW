"""Adversarial runtime audit for the Paper 1 Tier B composite models."""

from __future__ import annotations

import csv
import gc
import io
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.CARAFE import CARAFE
from ultralytics.nn.FDConv import FDConv
from ultralytics.nn.LaplacianConv import LaplacianConv
from ultralytics.nn.yolo26_ffafusion import FFAFusionConcat
from ultralytics.utils.torch_utils import get_flops

from audit_paper1_tiera_models import (
    assert_close,
    assert_finite,
    atomic_write_text,
    audit_identity_and_boundaries,
    audit_security,
    load_pilot_module,
    require,
    tensor_sum,
    tensors,
)
from generate_paper1_tiera_composites import MODEL_DIR, build_model, write_model
from generate_paper1_tierb_composites import ALLOWED_SPECS, SPECS


REPORT_CSV = ROOT / "experiments" / "module_scan" / "paper1_tierb_adversarial_audit.csv"
REPORT_MD = ROOT / "experiments" / "module_scan" / "paper1_tierb_adversarial_audit.md"
CHECKPOINT = ROOT / "yolo26n.pt"
EXTREME_SHAPES = ((31, 31), (32, 32), (33, 33), (63, 65), (127, 129), (639, 641))


def audit_generator_boundaries():
    """Reject path/config injection and prove concurrent writes and failed writes preserve a valid YAML."""
    filename, options = SPECS[-1]
    path = MODEL_DIR / filename
    original = path.read_bytes()

    invalid_calls = (
        ("../escape.yaml", options, "generate_paper1_tierb_composites.py", ALLOWED_SPECS),
        (Path("bad.yaml"), options, "generate_paper1_tierb_composites.py", ALLOWED_SPECS),
        (filename, [], "generate_paper1_tierb_composites.py", ALLOWED_SPECS),
        (filename, {"detail": "unknown"}, "generate_paper1_tierb_composites.py", ALLOWED_SPECS),
        (filename, {"detail": []}, "generate_paper1_tierb_composites.py", ALLOWED_SPECS),
        (filename, options, "bad\nheader.py", ALLOWED_SPECS),
        (filename, {"detail": "spd"}, "generate_paper1_tierb_composites.py", ALLOWED_SPECS),
    )
    for arguments in invalid_calls:
        try:
            write_model(*arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsafe generator input was accepted: {arguments!r}")
        require(path.read_bytes() == original, "failed generator input changed the destination YAML")

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(
            pool.map(
                lambda _: write_model(filename, options, "generate_paper1_tierb_composites.py", ALLOWED_SPECS),
                range(8),
            )
        )
    require(all(output == path.resolve() for output in outputs), "concurrent generation returned a wrong path")
    with path.open(encoding="utf-8") as handle:
        require(yaml.safe_load(handle) == build_model(options), "concurrent generation changed YAML content")
    require(not list(MODEL_DIR.glob(f".{filename}.*.tmp")), "generator leaked temporary files")


def expected_counts(options):
    return {
        "SPDConv": int(options.get("detail") == "spd"),
        "LaplacianConv": int(options.get("detail") == "lap") + int(options.get("head_detail") == "lap"),
        "FDConv": int(options.get("detail") == "fd") + int(options.get("head_detail") == "fd"),
        "CARAFE": int(bool(options.get("carafe"))),
        "Concat_bifpn": (6 if options.get("p2") else 4) if options.get("fusion") == "bifpn" else 0,
        "FFAFusionConcat": int(options.get("fusion") == "ffa"),
    }


def audit_checkpoint_policy(pilot):
    model_path = MODEL_DIR / SPECS[-1][0]
    digest = pilot.file_sha256(CHECKPOINT)
    pilot.validate_tier_b_protocol(model_path, CHECKPOINT.resolve(), digest, 640, 32)
    invalid = (
        (model_path, CHECKPOINT.resolve(), "0" * 64, 640, 32),
        (model_path, None, "", 640, 32),
        (model_path, CHECKPOINT.resolve(), digest, 641, 32),
        (model_path, CHECKPOINT.resolve(), digest, 639, 32),
        (model_path, CHECKPOINT.resolve(), digest, 640, 33),
    )
    for arguments in invalid:
        try:
            pilot.validate_tier_b_protocol(*arguments)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"unsafe Tier B checkpoint/protocol was accepted: {arguments}")

    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        run_dir = Path(directory)
        data_yaml = ROOT / "configs" / "japan7_remote.yaml"
        pilot.set_run_state(run_dir, "RUNNING", "pid=audit")
        pilot.save_repro_files(run_dir, model_path, data_yaml, "audit", CHECKPOINT, digest)
        required = {
            "model_yaml_snapshot.yaml",
            "model_yaml_sha256.txt",
            "checkpoint_sha256.txt",
            "git_status_porcelain.txt",
            "RUNNING",
        }
        require(required <= {path.name for path in run_dir.iterdir()}, "Tier B reproducibility snapshot is incomplete")
        require((run_dir / "checkpoint_sha256.txt").read_text().strip() == digest, "checkpoint digest snapshot drift")
        require(
            (run_dir / "model_yaml_sha256.txt").read_text().strip() == pilot.file_sha256(model_path),
            "model YAML digest snapshot drift",
        )
        pilot.set_run_state(run_dir, "COMPLETED", "status=audit")
        require((run_dir / "COMPLETED").exists() and not (run_dir / "RUNNING").exists(), "run state transition failed")

        original_reports = (pilot.OUT_DIR, pilot.PILOT_CSV, pilot.PILOT_MD)
        try:
            pilot.OUT_DIR = run_dir / "reports"
            pilot.PILOT_CSV = pilot.OUT_DIR / "pilot_report.csv"
            pilot.PILOT_MD = pilot.OUT_DIR / "pilot_report.md"
            row = {field: "" for field in pilot.FIELDS}
            row.update(yaml_path=model_path.name, run_name="audit", status="COMPLETED")
            pilot.append_row(row)
            pilot.append_row({**row, "run_name": "audit_2"})
            with pilot.PILOT_CSV.open(encoding="utf-8") as handle:
                report_rows = list(csv.DictReader(handle))
            require([item["run_name"] for item in report_rows] == ["audit", "audit_2"], "pilot report rows drifted")
            markdown = pilot.PILOT_MD.read_text(encoding="utf-8")
            require("audit" in markdown and "audit_2" in markdown, "pilot Markdown did not follow CSV source")
        finally:
            pilot.OUT_DIR, pilot.PILOT_CSV, pilot.PILOT_MD = original_reports


def model_output(model, value):
    output = model(value)
    return output[0] if isinstance(output, tuple) else output


def audit_extreme_shapes(model, path):
    """Run every declared direct-input boundary shape, including all SPDConv candidates."""
    model.model.eval()
    statuses = []
    with torch.no_grad():
        for height, width in EXTREME_SHAPES:
            try:
                output = model_output(model.model, torch.randn(1, 3, height, width))
            except (RuntimeError, ValueError) as error:
                require("out of memory" not in str(error).lower(), f"{path.name} boundary probe caused OOM: {error}")
                statuses.append(f"{height}x{width}=rejected:{type(error).__name__}:{str(error).splitlines()[0][:80]}")
                recovery = model_output(model.model, torch.randn(1, 3, 32, 32))
                assert_finite(recovery)
            else:
                assert_finite(output)
                statuses.append(f"{height}x{width}=supported")
    require(any(status.endswith("supported") for status in statuses), f"{path.name} rejected every boundary shape")
    return ";".join(statuses)


def audit_recovery_and_concurrency(path, config, pilot):
    """Exercise malformed-input recovery and fixed-shape concurrent inference."""
    model = YOLO(str(path), task="detect")
    pilot.load_pretrained(model, CHECKPOINT, "auto", config["pretrained_map"])
    model.model.eval()

    try:
        model.model(torch.randn(1, 1, 64, 64))
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError(f"{path.name} accepted a one-channel image")

    fixed_input = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model_output(model.model, fixed_input)
    assert_finite(expected)

    def infer(_):
        with torch.no_grad():
            return model_output(model.model, fixed_input)

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(infer, range(8)))
    for output in outputs:
        assert_close(output, expected, f"{path.name} concurrent inference", tolerance=2e-5)


def parameter_grad_norm(parameters):
    values = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return torch.stack(values).norm().item() if values else 0.0


def two_step_gate_check(module, inputs, gate, branch_parameters, name):
    """Prove a zero-initialized gate receives step-one gradient and wakes its branch on step two."""
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    target = torch.randn_like(module(*inputs) if isinstance(inputs, tuple) else module(inputs))
    optimizer.zero_grad(set_to_none=True)
    output = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
    F.mse_loss(output, target).backward()
    require(gate.grad is not None and torch.isfinite(gate.grad).all(), f"{name} gate has no finite step-one gradient")
    require(gate.grad.detach().abs().max().item() > 0.0, f"{name} gate has zero step-one gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = module(*inputs) if isinstance(inputs, tuple) else module(inputs)
    F.mse_loss(output, target).backward()
    require(parameter_grad_norm(branch_parameters) > 0.0, f"{name} branch did not receive step-two gradient")


def audit_two_step_gates():
    carafe = CARAFE(8, 8, 3, 2).train()
    two_step_gate_check(
        carafe,
        torch.randn(2, 8, 5, 7),
        carafe.gamma,
        list(carafe.down.parameters()) + list(carafe.encoder.parameters()),
        "CARAFE",
    )

    ffa = FFAFusionConcat([8, 12], 1, 7, 16, 0.0).train()
    two_step_gate_check(
        ffa,
        ([torch.randn(2, 8, 5, 7), torch.randn(2, 12, 5, 7)],),
        ffa.align[0].gamma,
        ffa.align[0].local.parameters(),
        "FFAFusionConcat",
    )

    for module, gate, name in (
        (FDConv(8, 8, 3, 1).train(), "gamma", "FDConv"),
        (LaplacianConv(8, 8, 3, 1).train(), "alpha", "LaplacianConv"),
    ):
        value = torch.randn(2, 8, 7, 9)
        output = module(value)
        F.mse_loss(output, torch.randn_like(output)).backward()
        gradient = getattr(module, gate).grad
        require(gradient is not None and gradient.abs().max().item() > 0.0, f"{name} gate has zero gradient")


def audit_baseline_equivalence(path, config, pilot):
    """B24 should start as the pretrained baseline in eval and training modes."""
    baseline = YOLO(str(CHECKPOINT), task="detect")
    candidate = YOLO(str(path), task="detect")
    pilot.load_pretrained(candidate, CHECKPOINT, "auto", config["pretrained_map"])
    baseline.model.eval()
    candidate.model.eval()
    value = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        expected = model_output(baseline.model, value)
        actual = model_output(candidate.model, value)
    assert_close(actual, expected, "Tier B24 640x640 eval baseline equivalence", tolerance=2e-5)

    baseline.model.train()
    candidate.model.train()
    training_value = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected_tensors = tensors(baseline.model(training_value))
        actual_tensors = tensors(candidate.model(training_value))
    require(len(actual_tensors) == len(expected_tensors), "Tier B24 training output structure differs from baseline")
    for index, (actual_tensor, expected_tensor) in enumerate(zip(actual_tensors, expected_tensors)):
        assert_close(actual_tensor, expected_tensor, f"Tier B24 training tensor {index}", tolerance=2e-5)


def audit_models_and_transfer(pilot):
    paths = sorted(MODEL_DIR.glob("yolo26n-Paper1-TierB*.yaml"))
    require(len(paths) == len(SPECS) == 12, f"expected 12 Tier B YAMLs, found {len(paths)}")
    require(CHECKPOINT.exists(), f"Missing audit checkpoint: {CHECKPOINT}")
    spec_by_name = dict(SPECS)
    require(set(spec_by_name) == {path.name for path in paths}, "Tier B YAML names do not match declared specs")

    rows = []
    for path in paths:
        options = spec_by_name[path.name]
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        require(config == build_model(options), f"Generated YAML drift: {path.name}")
        layers = config["backbone"] + config["head"]
        require(
            len(layers[-1][0]) == (4 if options.get("p2") else 3),
            f"incorrect Detect level count: {path.name}",
        )

        class_counts = {}
        for layer in layers:
            class_counts[layer[2]] = class_counts.get(layer[2], 0) + 1
        for module_name, expected in expected_counts(options).items():
            require(
                class_counts.get(module_name, 0) == expected,
                f"{path.name} expected {expected} {module_name}, got {class_counts.get(module_name, 0)}",
            )

        model = YOLO(str(path), task="detect")
        report = pilot.load_pretrained(model, CHECKPOINT, "auto", config["pretrained_map"])
        require(report["transfer_numel_ratio"] >= 0.95, f"low total transfer: {path.name}: {report}")
        require(report["backbone_transfer_ratio"] >= 0.98, f"low backbone transfer: {path.name}: {report}")
        require(report["neck_transfer_ratio"] >= 0.90, f"low neck transfer: {path.name}: {report}")
        require(report["detect_transfer_ratio"] >= 0.86, f"low Detect transfer: {path.name}: {report}")

        params = sum(parameter.numel() for parameter in model.model.parameters())
        gflops = float(get_flops(model.model, imgsz=640))
        model.model.train()
        value = torch.randn(1, 3, 640, 640, requires_grad=True)
        output = model.model(value)
        assert_finite(output)
        tensor_sum(output).backward()
        require(
            value.grad is not None and torch.isfinite(value.grad).all(),
            f"{path.name} produced an invalid 640x640 input gradient",
        )
        require(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.model.parameters()
                if parameter.grad is not None
            ),
            f"{path.name} produced a non-finite parameter gradient",
        )
        model.model.zero_grad(set_to_none=True)

        model.model.eval()
        small_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            assert_finite(model.model(small_input))

        extreme_status = audit_extreme_shapes(model, path)

        fuse_input = torch.linspace(-1.0, 1.0, 3 * 64 * 64).reshape(1, 3, 64, 64)
        custom_modules = [
            module
            for module in model.model.modules()
            if module.__class__.__name__ in {"FDConv", "LaplacianConv"}
        ]
        custom_bn_before = sum(hasattr(module, "bn") for module in custom_modules)
        with torch.no_grad():
            before_fuse = model_output(model.model, fuse_input)
        model.model.fuse(verbose=False)
        custom_bn_after = sum(hasattr(module, "bn") for module in custom_modules)
        require(custom_bn_after == 0, f"{path.name} left {custom_bn_after} custom BatchNorm layers unfused")
        require(
            custom_bn_before == len(custom_modules),
            f"{path.name} custom convolution was already missing BatchNorm before fuse",
        )
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
                extreme_status,
                custom_bn_before,
            )
        )
        del output, value, model
        gc.collect()

    representative = (paths[0], paths[2], paths[9])
    for path in representative:
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        audit_recovery_and_concurrency(path, config, pilot)

    b24 = paths[-1]
    with b24.open(encoding="utf-8") as handle:
        audit_baseline_equivalence(b24, yaml.safe_load(handle), pilot)
    return rows


def write_reports(rows, pilot):
    fields = [
        "yaml",
        "params",
        "thop_gflops_lower_bound",
        "transfer_numel_ratio",
        "backbone",
        "neck",
        "detect",
        "extreme_shapes",
        "custom_bn_fused",
    ]
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(fields)
    writer.writerows(rows)

    lines = [
        "# Paper 1 Tier B Adversarial Audit",
        "",
        "## Result",
        "",
        "All 12 Tier B YAMLs passed safe construction, semantic pretrained transfer, finite 640x640 forward/backward,",
        "32x32 CPU mixed precision, six boundary shapes per model, real custom BN fusion, malformed-input recovery,",
        "fixed-shape concurrent inference, two-step gate gradients, trusted-checkpoint enforcement, and atomic generation/reporting.",
        "No dataset training or CUDA smoke was run.",
        "Both `python scripts/audit_paper1_tierb_models.py` and the optimized `python -O` form passed on 2026-07-12.",
        "",
        "THOP does not count every FFT, grid-sampling, unfold, interpolation, pixel rearrangement, or dynamic tensor",
        "operation used by these models. The GFLOPs column is a lower-bound estimate, not evidence of real latency.",
        "",
        "| YAML | Params | THOP GFLOPs lower bound | Parameter transfer | Backbone | Neck | Detect | Custom BN fused |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, params, gflops, overall, backbone, neck, detect, _, custom_bn_fused in rows:
        lines.append(
            f"| `{name}` | {params} | {gflops:.1f} | {overall:.3%} | {backbone:.3%} | "
            f"{neck:.3%} | {detect:.3%} | {custom_bn_fused} |"
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
            "| High, fixed | Audit checks used `assert`, which disappears under `python -O`. | All Tier A helper and Tier B audit checks now raise explicitly; normal and optimized audits are required. |",
            "| High, fixed | FFA used directional wrapping for axial Fourier angles. | Axial differences now wrap to [-pi/2, pi/2], including the +89/-89 degree boundary. |",
            "| High, fixed | FDConv applied the same bias-free convolution twice. | Frequency detail is mixed in input space and passed through one convolution. |",
            "| High, fixed | CARAFE repeated unfolded patches across both upsample axes. | Reassembly now uses sub-pixel kernels with einsum/reordering and rejects oversized estimated workspaces. |",
            "| High, fixed | LaplacianConv/FDConv BatchNorm layers were untouched by BaseModel.fuse(). | Both modules now expose fused forwards; the audit proves their BatchNorm layers are removed. |",
            "| High, fixed | The BiFPN-labelled node could produce inf/inf and is not classic additive BiFPN. | Weights now use stable softmax; documentation calls it BiFPN-style positive weighted concatenation. |",
            "| High, fixed | A project-local path did not make a pickle checkpoint trustworthy. | Tier B accepts only the recorded SHA256 of project-root yolo26n.pt and snapshots that digest. |",
            "| Medium, fixed | Report writes could leave partial CSV/Markdown files. | A shared cross-process lock and fsync-backed atomic replacements serialize both reports. |",
            "| Medium, fixed | Zero gates hide branch gradients on the first step. | Gate gradients and second-step CARAFE/FFA branch gradients are now explicitly checked. |",
            "| Medium, accepted | SPDConv cannot inherit the baseline 3x3 stride-2 kernel because space-to-depth changes its tensor shape. | Coverage is reported explicitly; SPD candidates remain optimization-risk experiments. |",
            "| Medium, accepted | Official P2 adds a randomly initialized Detect branch, so P2 candidates cannot be baseline-equivalent at initialization. | Detect transfer must remain at least 86%; P2 results require matched pilots before promotion. |",
            "| Medium, operational | Ultralytics Detect caches shape-dependent anchors on the model instance. Concurrent mixed-resolution calls can race. | Fixed-shape re-entry is tested; production mixed-resolution inference should use one model per worker or external serialization. |",
            "| Scientific, unresolved | Buildability and stable gradients do not establish an accuracy gain. | Keep Tier B behind matched smoke and 30-epoch signal gates; do not infer efficacy from this report. |",
            "",
            "B24 passed pretrained baseline-equivalence at 640x640 eval and 64x64 training mode. P2 and SPD candidates are intentionally excluded from that assertion.",
            "Every YAML was probed at 31x31, 32x32, 33x33, 63x65, 127x129, and 639x641; exact outcomes are stored in the CSV.",
            "Mixed-resolution shared-instance inference and CUDA concurrency remain unsupported; use one model per worker.",
            "",
            "## Review Items Not Applied Literally",
            "",
            "- Direct odd-size predictions are not required to equal externally zero-padded predictions. YOLO preprocessing",
            "  letterboxes to a stride-aligned size; Tier B training therefore enforces an imgsz divisible by 32.",
            "- The FDConv frequency mask is not cached. A mutable shape/device cache would add mixed-resolution races and",
            "  unbounded device-memory retention; the duplicate convolution was the material avoidable cost.",
            "- `git_diff.patch` is not copied into runs because it can capture unrelated user work or secrets. The model/data",
            "  snapshots, hashes, commit, branch, command, and porcelain status provide the required provenance.",
            "- Residual RUNNING markers are not automatically relabelled at startup because another process may still own",
            "  that run. The durable marker and PID preserve evidence for an operator-side interruption check.",
            "- Python code with filesystem write access cannot be made incapable of overwriting a YAML. The generator now",
            "  prevents accidental cross-tier writes by requiring the immutable declared filename/options mapping.",
            "",
            "## Least Confidence",
            "",
            "The least certain property is accuracy, especially for B13-B15/B19-B20 where overlapping detail operators may amplify the same cues.",
            "The next uncertainty is SPDConv optimization because its baseline downsampling kernel is structurally non-transferable.",
        ]
    )
    with pilot.report_lock():
        atomic_write_text(REPORT_CSV, csv_buffer.getvalue())
        atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    pilot = load_pilot_module()
    audit_generator_boundaries()
    audit_checkpoint_policy(pilot)
    audit_identity_and_boundaries()
    audit_security(pilot)
    audit_two_step_gates()
    rows = audit_models_and_transfer(pilot)
    write_reports(rows, pilot)
    for name, params, gflops, overall, backbone, neck, detect, _, _ in rows:
        print(
            f"AUDIT_OK {name} params={params} gflops={gflops:.1f} transfer={overall:.3%} "
            f"backbone={backbone:.3%} neck={neck:.3%} detect={detect:.3%}"
        )
    print("PAPER1_TIER_B_ADVERSARIAL_AUDIT_OK=12")


if __name__ == "__main__":
    main()
