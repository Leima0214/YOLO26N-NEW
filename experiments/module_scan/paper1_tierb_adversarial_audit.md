# Paper 1 Tier B Adversarial Audit

## Result

All 12 Tier B YAMLs passed safe construction, semantic pretrained transfer, finite 640x640 forward/backward,
32x32 CPU mixed precision, six boundary shapes per model, real custom BN fusion, malformed-input recovery,
fixed-shape concurrent inference, two-step gate gradients, trusted-checkpoint enforcement, and atomic generation/reporting.
No dataset training or CUDA smoke was run.
Both `python scripts/audit_paper1_tierb_models.py` and the optimized `python -O` form passed on 2026-07-12.

THOP does not count every FFT, grid-sampling, unfold, interpolation, pixel rearrangement, or dynamic tensor
operation used by these models. The GFLOPs column is a lower-bound estimate, not evidence of real latency.

| YAML | Params | THOP GFLOPs lower bound | Parameter transfer | Backbone | Neck | Detect | Custom BN fused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo26n-Paper1-TierB13-P2-SPDConv-LaplacianConv.yaml` | 2641921 | 9.3 | 95.964% | 98.772% | 95.523% | 86.562% | 1 |
| `yolo26n-Paper1-TierB14-P2-SPDConv-FDConv.yaml` | 2641921 | 9.3 | 95.964% | 98.772% | 95.523% | 86.562% | 1 |
| `yolo26n-Paper1-TierB15-P2-LaplacianConv-FDConv.yaml` | 2662402 | 9.5 | 96.615% | 100.000% | 95.523% | 86.562% | 2 |
| `yolo26n-Paper1-TierB16-P2-SPDConv-BiFPN.yaml` | 2641932 | 9.3 | 95.963% | 98.772% | 95.522% | 86.562% | 0 |
| `yolo26n-Paper1-TierB17-P2-LaplacianConv-FFAFusion.yaml` | 2672673 | 9.7 | 96.244% | 100.000% | 94.490% | 86.562% | 1 |
| `yolo26n-Paper1-TierB18-P2-FDConv-FFAFusion.yaml` | 2672673 | 9.7 | 96.244% | 100.000% | 94.490% | 86.562% | 1 |
| `yolo26n-Paper1-TierB19-SPDConv-LaplacianConv-BiFPN.yaml` | 2551809 | 5.9 | 99.353% | 98.772% | 99.999% | 100.000% | 1 |
| `yolo26n-Paper1-TierB20-SPDConv-FDConv-FFAFusion.yaml` | 2562073 | 6.0 | 98.955% | 98.772% | 98.868% | 100.000% | 1 |
| `yolo26n-Paper1-TierB21-P2-CARAFE-BiFPN.yaml` | 2693329 | 9.8 | 95.506% | 100.000% | 92.478% | 86.562% | 0 |
| `yolo26n-Paper1-TierB22-P2-CARAFE-FFAFusion.yaml` | 2703589 | 9.9 | 95.143% | 100.000% | 91.510% | 86.562% | 0 |
| `yolo26n-Paper1-TierB23-SPDConv-CARAFE-BiFPN.yaml` | 2582725 | 6.1 | 98.163% | 98.772% | 96.668% | 100.000% | 0 |
| `yolo26n-Paper1-TierB24-LaplacianConv-CARAFE-FFAFusion.yaml` | 2613470 | 6.5 | 98.424% | 100.000% | 95.610% | 100.000% | 1 |

## Adversarial Findings

| Severity | Finding | Resolution |
| --- | --- | --- |
| Critical, fixed | Two shallow replacements could silently target the same layer or shift numeric indices. | The first replacement is fixed at backbone P2-to-P3; the second is fixed at PAN P3-to-P4; all checkpoint maps use semantic layer names. |
| High, fixed | Direct YAML writes could leave a partial file after interruption or concurrent generation. | Generation now validates a temporary safe-loaded YAML and atomically replaces the destination; concurrent and failed-write tests pass. |
| High, fixed | Unvalidated options or output names could inject unsupported modules or escape the model directory. | Generator options, filenames, generator labels, resolved paths, and unique mappings are allowlisted. |
| High, fixed | Audit checks used `assert`, which disappears under `python -O`. | All Tier A helper and Tier B audit checks now raise explicitly; normal and optimized audits are required. |
| High, fixed | FFA used directional wrapping for axial Fourier angles. | Axial differences now wrap to [-pi/2, pi/2], including the +89/-89 degree boundary. |
| High, fixed | FDConv applied the same bias-free convolution twice. | Frequency detail is mixed in input space and passed through one convolution. |
| High, fixed | CARAFE repeated unfolded patches across both upsample axes. | Reassembly now uses sub-pixel kernels with einsum/reordering and rejects oversized estimated workspaces. |
| High, fixed | LaplacianConv/FDConv BatchNorm layers were untouched by BaseModel.fuse(). | Both modules now expose fused forwards; the audit proves their BatchNorm layers are removed. |
| High, fixed | The BiFPN-labelled node could produce inf/inf and is not classic additive BiFPN. | Weights now use stable softmax; documentation calls it BiFPN-style positive weighted concatenation. |
| High, fixed | A project-local path did not make a pickle checkpoint trustworthy. | Tier B accepts only the recorded SHA256 of project-root yolo26n.pt and snapshots that digest. |
| Medium, fixed | Report writes could leave partial CSV/Markdown files. | A shared cross-process lock and fsync-backed atomic replacements serialize both reports. |
| Medium, fixed | Zero gates hide branch gradients on the first step. | Gate gradients and second-step CARAFE/FFA branch gradients are now explicitly checked. |
| Medium, accepted | SPDConv cannot inherit the baseline 3x3 stride-2 kernel because space-to-depth changes its tensor shape. | Coverage is reported explicitly; SPD candidates remain optimization-risk experiments. |
| Medium, accepted | Official P2 adds a randomly initialized Detect branch, so P2 candidates cannot be baseline-equivalent at initialization. | Detect transfer must remain at least 86%; P2 results require matched pilots before promotion. |
| Medium, operational | Ultralytics Detect caches shape-dependent anchors on the model instance. Concurrent mixed-resolution calls can race. | Fixed-shape re-entry is tested; production mixed-resolution inference should use one model per worker or external serialization. |
| Scientific, unresolved | Buildability and stable gradients do not establish an accuracy gain. | Keep Tier B behind matched smoke and 30-epoch signal gates; do not infer efficacy from this report. |

B24 passed pretrained baseline-equivalence at 640x640 eval and 64x64 training mode. P2 and SPD candidates are intentionally excluded from that assertion.
Every YAML was probed at 31x31, 32x32, 33x33, 63x65, 127x129, and 639x641; exact outcomes are stored in the CSV.
Mixed-resolution shared-instance inference and CUDA concurrency remain unsupported; use one model per worker.

## Review Items Not Applied Literally

- Direct odd-size predictions are not required to equal externally zero-padded predictions. YOLO preprocessing
  letterboxes to a stride-aligned size; Tier B training therefore enforces an imgsz divisible by 32.
- The FDConv frequency mask is not cached. A mutable shape/device cache would add mixed-resolution races and
  unbounded device-memory retention; the duplicate convolution was the material avoidable cost.
- `git_diff.patch` is not copied into runs because it can capture unrelated user work or secrets. The model/data
  snapshots, hashes, commit, branch, command, and porcelain status provide the required provenance.
- Residual RUNNING markers are not automatically relabelled at startup because another process may still own
  that run. The durable marker and PID preserve evidence for an operator-side interruption check.
- Python code with filesystem write access cannot be made incapable of overwriting a YAML. The generator now
  prevents accidental cross-tier writes by requiring the immutable declared filename/options mapping.

## Least Confidence

The least certain property is accuracy, especially for B13-B15/B19-B20 where overlapping detail operators may amplify the same cues.
The next uncertainty is SPDConv optimization because its baseline downsampling kernel is structurally non-transferable.
