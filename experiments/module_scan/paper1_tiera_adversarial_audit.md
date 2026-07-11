# Paper 1 Tier A Adversarial Runtime Audit

All rows passed safe YAML loading, construction, finite 64x64 forward/backward, identity-init module checks,
odd-size module checks, CPU mixed precision, fused inference, concurrent re-entry, semantic transfer, and malicious-input rejection.

| YAML | Params | Parameter transfer | Backbone | Neck | Detect | FFA instances |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo26n-Paper1-TierA01-P2-SPDConv-EMA-P3f8.yaml` | 2642593 | 95.939% | 98.772% | 95.455% | 86.562% | 0 |
| `yolo26n-Paper1-TierA02-P2-LaplacianConv-EMA-P3f8.yaml` | 2663074 | 96.591% | 100.000% | 95.455% | 86.562% | 0 |
| `yolo26n-Paper1-TierA03-P2-EMA-P3f8-BiFPN.yaml` | 2663085 | 96.590% | 100.000% | 95.453% | 86.562% | 0 |
| `yolo26n-Paper1-TierA04-SPDConv-EMA-P3f8-FFAFusion.yaml` | 2562745 | 98.929% | 98.772% | 98.795% | 100.000% | 1 |
| `yolo26n-Paper1-TierA05-LaplacianConv-EMA-P3f8-BiFPN.yaml` | 2572962 | 99.973% | 100.000% | 99.924% | 100.000% | 0 |
| `yolo26n-Paper1-TierA06-P2-SEAttention-P3-FFAFusion.yaml` | 2673185 | 96.225% | 100.000% | 94.439% | 86.562% | 1 |
| `yolo26n-Paper1-TierA07-P2-CBAM-P3-BiFPN.yaml` | 2663023 | 96.592% | 100.000% | 95.460% | 86.562% | 0 |
| `yolo26n-Paper1-TierA08-P2-EMA-P3f8-slimneck.yaml` | 2467105 | 85.492% | 100.000% | 58.347% | 86.562% | 0 |
| `yolo26n-Paper1-TierA09-SPDConv-EMA-P3f8-slimneck.yaml` | 2356505 | 87.935% | 98.772% | 61.843% | 100.000% | 0 |
| `yolo26n-Paper1-TierA10-FDConv-EMA-P3f8-FFAFusion.yaml` | 2583226 | 99.576% | 100.000% | 98.795% | 100.000% | 1 |
| `yolo26n-Paper1-TierA11-P2-LaplacianConv-CARAFE.yaml` | 2693318 | 95.506% | 100.000% | 92.479% | 86.562% | 0 |
| `yolo26n-Paper1-TierA12-FDConv-GSConv-CARAFE.yaml` | 2601294 | 98.705% | 99.797% | 96.669% | 100.000% | 0 |
