# Paper 1 Tier B Adversarial Audit

## Result

All 12 Tier B YAMLs passed safe construction, semantic pretrained transfer, finite 640x640 forward/backward,
32x32 CPU mixed precision, fused inference, malformed-input recovery, fixed-shape concurrent inference,
and atomic concurrent generation. No training was run.

| YAML | Params | GFLOPs | Parameter transfer | Backbone | Neck | Detect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo26n-Paper1-TierB13-P2-SPDConv-LaplacianConv.yaml` | 2641921 | 9.3 | 95.964% | 98.772% | 95.523% | 86.562% |
| `yolo26n-Paper1-TierB14-P2-SPDConv-FDConv.yaml` | 2641921 | 9.4 | 95.964% | 98.772% | 95.523% | 86.562% |
| `yolo26n-Paper1-TierB15-P2-LaplacianConv-FDConv.yaml` | 2662402 | 9.6 | 96.615% | 100.000% | 95.523% | 86.562% |
| `yolo26n-Paper1-TierB16-P2-SPDConv-BiFPN.yaml` | 2641932 | 9.3 | 95.963% | 98.772% | 95.522% | 86.562% |
| `yolo26n-Paper1-TierB17-P2-LaplacianConv-FFAFusion.yaml` | 2672673 | 9.7 | 96.244% | 100.000% | 94.490% | 86.562% |
| `yolo26n-Paper1-TierB18-P2-FDConv-FFAFusion.yaml` | 2672673 | 10.1 | 96.244% | 100.000% | 94.490% | 86.562% |
| `yolo26n-Paper1-TierB19-SPDConv-LaplacianConv-BiFPN.yaml` | 2551809 | 5.9 | 99.353% | 98.772% | 99.999% | 100.000% |
| `yolo26n-Paper1-TierB20-SPDConv-FDConv-FFAFusion.yaml` | 2562073 | 6.1 | 98.955% | 98.772% | 98.868% | 100.000% |
| `yolo26n-Paper1-TierB21-P2-CARAFE-BiFPN.yaml` | 2693329 | 9.8 | 95.506% | 100.000% | 92.478% | 86.562% |
| `yolo26n-Paper1-TierB22-P2-CARAFE-FFAFusion.yaml` | 2703589 | 9.9 | 95.143% | 100.000% | 91.510% | 86.562% |
| `yolo26n-Paper1-TierB23-SPDConv-CARAFE-BiFPN.yaml` | 2582725 | 6.1 | 98.163% | 98.772% | 96.668% | 100.000% |
| `yolo26n-Paper1-TierB24-LaplacianConv-CARAFE-FFAFusion.yaml` | 2613470 | 6.5 | 98.424% | 100.000% | 95.610% | 100.000% |

## Adversarial Findings

| Severity | Finding | Resolution |
| --- | --- | --- |
| Critical, fixed | Two shallow replacements could silently target the same layer or shift numeric indices. | The first replacement is fixed at backbone P2-to-P3; the second is fixed at PAN P3-to-P4; all checkpoint maps use semantic layer names. |
| High, fixed | Direct YAML writes could leave a partial file after interruption or concurrent generation. | Generation now validates a temporary safe-loaded YAML and atomically replaces the destination; concurrent and failed-write tests pass. |
| High, fixed | Unvalidated options or output names could inject unsupported modules or escape the model directory. | Generator options, filenames, generator labels, resolved paths, and unique mappings are allowlisted. |
| High, fixed | Structural checks used `assert`, which disappears under `python -O`. | Trust-boundary validation now raises explicit exceptions and is exercised with non-string names and non-mapping options. |
| Medium, accepted | SPDConv cannot inherit the baseline 3x3 stride-2 kernel because space-to-depth changes its tensor shape. | Coverage is reported explicitly; SPD candidates remain optimization-risk experiments. |
| Medium, accepted | Official P2 adds a randomly initialized Detect branch, so P2 candidates cannot be baseline-equivalent at initialization. | Detect transfer must remain at least 86%; P2 results require matched pilots before promotion. |
| Medium, operational | Ultralytics Detect caches shape-dependent anchors on the model instance. Concurrent mixed-resolution calls can race. | Fixed-shape re-entry is tested; production mixed-resolution inference should use one model per worker or external serialization. |
| Scientific, unresolved | Buildability and stable gradients do not establish an accuracy gain. | Keep Tier B behind matched smoke and 30-epoch signal gates; do not infer efficacy from this report. |

B24 also passed exact pretrained baseline-equivalence at initialization. P2 and SPD candidates are intentionally excluded from that assertion.
Odd-size direct-input probes: `yolo26n-Paper1-TierB15-P2-LaplacianConv-FDConv.yaml`=supported, `yolo26n-Paper1-TierB22-P2-CARAFE-FFAFusion.yaml`=supported.

## Least Confidence

The least certain property is accuracy, especially for B13-B15/B19-B20 where overlapping detail operators may amplify the same cues.
The next uncertainty is SPDConv optimization because its baseline downsampling kernel is structurally non-transferable.
