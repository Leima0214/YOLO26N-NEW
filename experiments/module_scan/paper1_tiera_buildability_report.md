# Paper 1 Tier A Composite Buildability Report

Date: 2026-07-11

Scope: build only. No dataset was loaded and no training was started.

## Result

All 12 generated Tier A YAMLs completed `YOLO(yaml, task="detect")` construction and the initialization forward pass.

| priority | YAML | Params | Detect strides | build |
| ---: | --- | ---: | --- | --- |
| 1 | `yolo26n-Paper1-TierA01-P2-SPDConv-EMA-P3f8.yaml` | 2.643M | 4/8/16/32 | OK |
| 2 | `yolo26n-Paper1-TierA02-P2-LaplacianConv-EMA-P3f8.yaml` | 2.663M | 4/8/16/32 | OK |
| 3 | `yolo26n-Paper1-TierA03-P2-EMA-P3f8-BiFPN.yaml` | 2.663M | 4/8/16/32 | OK |
| 4 | `yolo26n-Paper1-TierA04-SPDConv-EMA-P3f8-FFAFusion.yaml` | 2.563M | 8/16/32 | OK |
| 5 | `yolo26n-Paper1-TierA05-LaplacianConv-EMA-P3f8-BiFPN.yaml` | 2.573M | 8/16/32 | OK |
| 6 | `yolo26n-Paper1-TierA06-P2-SEAttention-P3-FFAFusion.yaml` | 2.673M | 4/8/16/32 | OK |
| 7 | `yolo26n-Paper1-TierA07-P2-CBAM-P3-BiFPN.yaml` | 2.663M | 4/8/16/32 | OK |
| 8 | `yolo26n-Paper1-TierA08-P2-EMA-P3f8-slimneck.yaml` | 2.467M | 4/8/16/32 | OK |
| 9 | `yolo26n-Paper1-TierA09-SPDConv-EMA-P3f8-slimneck.yaml` | 2.357M | 8/16/32 | OK |
| 10 | `yolo26n-Paper1-TierA10-FDConv-EMA-P3f8-FFAFusion.yaml` | 2.583M | 8/16/32 | OK |
| 11 | `yolo26n-Paper1-TierA11-P2-LaplacianConv-CARAFE.yaml` | 2.693M | 4/8/16/32 | OK |
| 12 | `yolo26n-Paper1-TierA12-FDConv-GSConv-CARAFE.yaml` | 2.601M | 8/16/32 | OK |

FLOPs are intentionally not reported from this workstation. `ultralytics-thop` is not installed, so the local `get_flops()` result is `0.0`; that value is an unavailable measurement, not a real model cost.

## Audit Notes

- Official P2 uses four Detect inputs and preserves strides 4/8/16/32.
- EMA, SE, and CBAM are applied to the final P3 feature immediately before Detect and start as exact identity residuals.
- SPD downsampling is one atomic padded `pixel_unshuffle` plus projection layer, preserving downstream semantic layer numbering and defining odd-size behavior.
- FDConv is Conv-BN-SiLU compatible, uses a dynamic frequency mask, computes FFT work in float32, and starts as the inherited standard Conv.
- FFA candidates contain one fusion node at the P3 top-down merge instead of replacing every concat.
- CARAFE preserves channels and starts as nearest-neighbor upsampling; learned content-aware reassembly is introduced through a zero residual gain.
- BiFPN candidates use a minimal weighted concat initialized to ordinary concat; core parsing no longer imports the unrelated `v9.py` module graph.
- `slimneck.py` is a minimal clean port containing only GSConv, GSBottleneck, and VoVGSCSP.
- The SE constructor now accepts the parser-supplied output-channel argument; its reduction remains 16 instead of accidentally receiving the scaled channel count.

## Boundary

Build OK does not establish accuracy or publishable improvement. The separate adversarial audit now establishes finite forward/backward behavior and parameter-weighted pretrained coverage, but a 1 epoch smoke is still required before any 30 epoch run. Official P2 has no baseline P2 branch, so only that new Detect branch is intentionally random.
