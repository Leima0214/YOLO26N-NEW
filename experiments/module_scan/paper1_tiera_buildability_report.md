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
| 4 | `yolo26n-Paper1-TierA04-SPDConv-EMA-P3f8-FFAFusion.yaml` | 2.598M | 8/16/32 | OK |
| 5 | `yolo26n-Paper1-TierA05-LaplacianConv-EMA-P3f8-BiFPN.yaml` | 2.573M | 8/16/32 | OK |
| 6 | `yolo26n-Paper1-TierA06-P2-SEAttention-P3-FFAFusion.yaml` | 2.718M | 4/8/16/32 | OK |
| 7 | `yolo26n-Paper1-TierA07-P2-CBAM-P3-BiFPN.yaml` | 2.663M | 4/8/16/32 | OK |
| 8 | `yolo26n-Paper1-TierA08-P2-EMA-P3f8-slimneck.yaml` | 2.467M | 4/8/16/32 | OK |
| 9 | `yolo26n-Paper1-TierA09-SPDConv-EMA-P3f8-slimneck.yaml` | 2.357M | 8/16/32 | OK |
| 10 | `yolo26n-Paper1-TierA10-FDConv-EMA-P3f8-FFAFusion.yaml` | 2.627M | 8/16/32 | OK |
| 11 | `yolo26n-Paper1-TierA11-P2-LaplacianConv-CARAFE.yaml` | 2.677M | 4/8/16/32 | OK |
| 12 | `yolo26n-Paper1-TierA12-FDConv-GSConv-CARAFE.yaml` | 2.594M | 8/16/32 | OK |

FLOPs are intentionally not reported from this workstation. `ultralytics-thop` is not installed, so the local `get_flops()` result is `0.0`; that value is an unavailable measurement, not a real model cost.

## Audit Notes

- Official P2 uses four Detect inputs and preserves strides 4/8/16/32.
- EMA, SE, and CBAM are applied to the final P3 feature immediately before Detect, not stacked on the backbone output.
- SPD downsampling is expressed as `space_to_depth` followed by a stride-1 channel projection, preserving spatial information before P3.
- FDConv is placed at the P2-to-P3 downsampling layer where both input and output channels exceed its activation threshold. Putting it on the RGB stem silently falls back to ordinary `nn.Conv2d`.
- FDConv requires explicit `padding=1`. The first A12 build exposed the missing padding through a 31/32 feature-size mismatch; the generator was corrected and A12 then built successfully.
- CARAFE and FDConv no longer import unused `mmcv` APIs. This is dependency cleanup, not an algorithm change.
- `slimneck.py` is a minimal clean port containing only GSConv, GSBottleneck, and VoVGSCSP.
- The SE constructor now accepts the parser-supplied output-channel argument; its reduction remains 16 instead of accidentally receiving the scaled channel count.

## Boundary

Build OK does not establish accuracy, stability, pretrained compatibility, or publishable improvement. Before any 30 epoch run, record `yolo26n.pt` transferred/total items, install THOP for real FLOPs, and complete a 1 epoch smoke. Official P2 has no directly comparable pretrained P2 checkpoint in this project.
