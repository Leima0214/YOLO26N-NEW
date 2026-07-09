# YOLO26 Paper Module Matrix

This matrix is the current Paper 1/Paper 2 module screening source of truth. `BUILD OK` means model construction only, not training stability or accuracy.

| module | yaml_path | position | paper_target | research_role | build_status | params | risk | pilot_priority | notes |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| CPUBoneNano-P2Lite | `ultralytics/cfg/models/26/yolo26-CPUBoneNano-P2Lite.yaml` | backbone/head shallow path | Paper 1 | P2-like shallow detail / small-target enhancement | BUILD OK | 3,930,380 | high params increase | second batch | Keep as P2-like candidate; do not equate with official P2 without noting pretrained-weight fairness. |
| BiFPN | `ultralytics/cfg/models/26/yolo26-BiFPN.yaml` | neck | Paper 1 | multi-scale fusion | BUILD OK | 2,572,292 | classic module, less lightweight than BiFPN1 | backup | Use after BiFPN1 unless BiFPN1 is unstable. |
| BiFPN1 | `ultralytics/cfg/models/26/yolo26-BiFPN1.yaml` | neck | Paper 1 | lightweight multi-scale fusion | BUILD OK | 2,555,508 | moderate novelty | first batch | Preferred multi-scale pilot because it is lighter. |
| EMA_attention | `ultralytics/cfg/models/26/yolo26-EMA_attention.yaml` | attention | Paper 1 | lightweight attention for damage response | BUILD OK | 2,572,952 | may overlap with YOLO26 attention blocks | first batch | First attention candidate; compare against D00/D10 AP, not only total mAP. |
| SEAttention | `ultralytics/cfg/models/26/yolo26-SEAttention.yaml` | attention | Paper 1 | channel attention backup | BUILD OK | 2,572,792 | weak novelty | backup | Use as EMA backup or attention ablation, not main innovation. |
| CBAM | `ultralytics/cfg/models/26/yolo26-CBAM.yaml` | attention | Paper 1 | traditional attention control | BUILD OK | 2,580,843 | weak novelty | control | Useful control only; do not position as primary contribution. |
| LaplacianConv | `ultralytics/cfg/models/26/yolo26-LaplacianConv.yaml` | convolution/detail | Paper 1 | edge/detail enhancement for cracks | BUILD OK | 2,572,281 | may over-emphasize noise | first batch | Strong Paper 1 candidate for D00/D10 crack boundaries. |
| SPDConv | `ultralytics/cfg/models/26/yolo26-SPDConv.yaml` | downsampling/conv | Paper 1 | detail-preserving small-target feature extraction | BUILD OK | 2,795,480 | larger params than first batch | second batch | Good story for thin cracks and small defects; promote only after pilot stability. |
| FFAFusion-Neck | `ultralytics/cfg/models/26/yolo26-FFAFusion-Neck.yaml` | neck | Paper 1 / Paper 2 | feature fusion | BUILD OK | 2,617,656 | fusion benefit may be dataset-specific | second batch | Backup fusion candidate; avoid stacking before single-module evidence. |
| HVIEnhanceStem | `ultralytics/cfg/models/26/yolo26-HVIEnhanceStem.yaml` | stem | Paper 2 | color/light robustness for cross-domain scenes | BUILD OK | 2,575,008 | Paper 2 protocol dependent | Paper 2 first batch | Current first Paper 2 candidate. |
| CARAFE | `ultralytics/cfg/models/26/yolo26-CARAFE.yaml` | upsampling/neck | Paper 1 | content-aware upsampling | BUILD FAIL | n/a | mmcv API compatibility | frozen | Do not fix here; use `fix/mmcv-compat-module-build` later. |
| FDConv | `ultralytics/cfg/models/26/yolo26-FDConv.yaml` | convolution/detail | Paper 1 / Paper 2 deferred | frequency/detail convolution | BUILD FAIL | n/a | mmcv API compatibility | frozen | Do not run until standalone compatibility scan passes. |
| ContextAggregation | `ultralytics/cfg/models/26/yolo26-ContextAggregation.yaml` | context/neck | Paper 2 deferred | context aggregation for domain shift | BUILD FAIL | n/a | mmcv API compatibility | frozen | Paper 2 idea only after separate fix branch. |

## Pilot Queue

First Paper 1 pilot batch:

- `yolo26-EMA_attention.yaml`
- `yolo26-LaplacianConv.yaml`
- `yolo26-BiFPN1.yaml`

Second Paper 1 pilot batch:

- `yolo26-SPDConv.yaml`
- `yolo26-CPUBoneNano-P2Lite.yaml`
- `yolo26-FFAFusion-Neck.yaml`

Paper 2 first pilot:

- `yolo26-HVIEnhanceStem.yaml`

Frozen:

- `yolo26-CARAFE.yaml`
- `yolo26-FDConv.yaml`
- `yolo26-ContextAggregation.yaml`
