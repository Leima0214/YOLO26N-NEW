# Paper 1 S4 WPFormer-WDR Audit

This candidate independently adapts WPFormer's WCA frequency modulation to a YOLO feature map.
It is not the full query-based WPFormer and does not include PCA.

| check | result |
| --- | --- |
| YAML safe load and model build | PASS |
| module instances | 1 |
| Detect strides | 8/16/32 |
| parameters (`nc=80` build) | 2578568 |
| Haar odd-size roundtrip max error | 2.384e-07 |
| identity-init max error | 0.000e+00 |
| signed-subband cancellation regression | PASS |
| first-step output projection gradient L1 | 1.390139 |
| second-step context gradient | PASS |
| invalid configuration/input rejection | PASS |
| failure leaves state unchanged | PASS |
| 1x1, large finite input, and concurrent re-entry | PASS |
| full 640x640 forward/backward | PASS |
| CPU bfloat16 | PASS |
| CUDA AMP | not available on this workstation; remote 1e smoke required |
| parameter transfer | 99.756144% |
| backbone transfer | 100.000000% |
| neck transfer | 99.303994% |
| Detect transfer | 100.000000% |
| pretrained full-model equivalence max error | 0.000e+00 |
| fused prediction max error | 1.907e-04 |

Build and local numerical audits do not establish an accuracy gain. Run one remote CUDA AMP smoke before 30e.
