# Paper 1 Composite Model Design

## Current Signal

The Paper 1 signal ranking has changed after additional 30 epoch single-module experiments. `EMA_attention` is now the strongest single-module candidate, while `CPUBoneNano-P2Lite` remains the strongest shallow-detail/P2 candidate.

30 epoch single-module signal:

| module | mAP50 | mAP50-95 | params | FLOPs | current decision |
| --- | ---: | ---: | ---: | ---: | --- |
| EMA_attention | 0.202 | 0.0884 | 2.377M | 5.2G | 100e formal |
| CPUBoneNano-P2Lite | 0.153 | 0.0674 | 3.672M | 6.6G | 100e formal |
| SPDConv | 0.129 | 0.0516 | 2.600M | 1.5G | optional 100e |

P2Lite still locks the shallow-detail direction into the Paper 1 candidate set. EMA now becomes the main single-module formal candidate.

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

The current Paper 1 formal path should be `EMA_attention 100e`, `P2Lite 100e`, and optional `SPDConv 100e`.
