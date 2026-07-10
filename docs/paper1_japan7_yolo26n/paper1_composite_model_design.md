# Paper 1 Composite Model Design

## 2026-07-10 Status

The initial EMA and P2Lite 100 epoch runs were scratch experiments, not pretrained YOLO26n ablations. Both peaked near `mAP50=0.587` and `mAP50-95=0.314`, below the pretrained baseline. This does not justify either composite YAML, and it does not establish that either single module is ineffective after compatible weight transfer.

Both composite YAMLs remain frozen. Reconsider them only after a pretrained, AMP-enabled single-module signal improves the baseline or produces a defensible D00/D10 gain.

## Historical Scratch Signal

The Paper 1 scratch ranking identified EMA_attention as the strongest attention candidate and CPUBoneNano-P2Lite as the strongest shallow-detail candidate. It does not authorize another formal run under the old initialization protocol.

30 epoch single-module signal:

| module | mAP50 | mAP50-95 | params | FLOPs | current decision |
| --- | ---: | ---: | ---: | ---: | --- |
| EMA_attention | 0.202 | 0.0884 | 2.377M | 5.2G | rerun pretrained+AMP signal |
| CPUBoneNano-P2Lite | 0.153 | 0.0674 | 3.672M | 6.6G | CPUBone pretraining required |
| SPDConv | 0.129 | 0.0516 | 2.600M | 1.5G | paused |

P2Lite remains a shallow-detail research idea, but not an active formal candidate until its own pretrained backbone can be converted and validated. EMA is the only active single-module candidate.

## Why P2Lite Stays

Paper 1 targets Japan7 damage detection, especially D00/D10 thin cracks and low-contrast small defects. P2Lite keeps a P2/4 detection path, so the model preserves more high-resolution spatial detail than a P3-P5-only detector.

The parameter and FLOPs increase is acceptable for the current paper story because the model remains in the lightweight range while adding a specific mechanism for small road damage.

## Why Add SPDConv

D10 remains weak after P2Lite, and SPDConv has a positive single-module signal. However, the current composite pilot shows that adding SPDConv in the present bottom-up position does not improve over P2Lite alone.

The composite YAML expresses this as `space_to_depth + stride1 Conv` at each bottom-up downsampling step. This keeps the four-scale Detect inputs at P2, P3, P4, and P5.

## Why Add EMA Attention

EMA_attention is retained as the lightweight attention branch because it is currently the strongest single-module result. In the existing composite YAMLs it is placed on the final P5 fused feature before `Detect`, matching the earlier standalone EMA insertion style.

The 3 epoch composite result suggests this placement may not be optimal when combined with P2Lite. Future composite work should test alternative EMA insertion points instead of promoting the current composite directly.

## Current Composite Candidates

YAMLs:

`ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-EMA.yaml`

`ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-SPDConv-EMA.yaml`

3 epoch composite pilot:

| model | Recall | mAP50 | mAP50-95 | params | FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| P2Lite | 0.0714 | 0.000306 | 0.0000811 | 3.672M | 6.6G |
| P2Lite + EMA | 0.0306 | 0.000128 | 0.0000280 | 3.673M | 6.6G |
| P2Lite + SPDConv + EMA | 0.0601 | 0.000230 | 0.0000569 | 4.254M | 7.7G |

Decision:

- Both composite YAMLs can train.
- The three-module composite is better than `P2Lite + EMA`, but still below P2Lite alone.
- Neither composite should enter 30 epoch or 100 epoch formal training in its current form.
- Keep both YAMLs as exploration candidates for future insertion-position research.

The next Paper 1 path is an EMA_attention pretrained+AMP smoke followed by one 30 epoch pretrained signal. P2Lite is paused because only 8/881 `yolo26n.pt` state items match its replacement backbone; it needs a separately validated CPUBone checkpoint conversion. No further 100 epoch or composite run is authorized by the current scratch results.
