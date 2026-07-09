# Paper 1 Composite Model Design

## Current Signal

`CPUBoneNano-P2Lite` has completed the 30 epoch signal experiment and is the current strongest Paper 1 single-module candidate.

Observed signal:

- mAP50 = 0.153
- mAP50-95 = 0.0674
- D10 mAP50 improved from 0 at 3 epochs to 0.0261 at 30 epochs
- D43/D44/D20 show stronger behavior than the early pilot
- Params = 3.672M
- FLOPs = 6.6G

This is enough to lock the P2/shallow-detail direction into the Paper 1 final candidate set.

## Why P2Lite Stays

Paper 1 targets Japan7 damage detection, especially D00/D10 thin cracks and low-contrast small defects. P2Lite keeps a P2/4 detection path, so the model preserves more high-resolution spatial detail than a P3-P5-only detector.

The parameter and FLOPs increase is acceptable for the current paper story because the model remains in the lightweight range while adding a specific mechanism for small road damage.

## Why Add SPDConv

D10 is still weak after P2Lite, even though the 30 epoch signal is positive. SPDConv-style downsampling is added to the bottom-up neck path to preserve local spatial detail when moving from P2 to P3/P4/P5.

The composite YAML expresses this as `space_to_depth + stride1 Conv` at each bottom-up downsampling step. This keeps the four-scale Detect inputs at P2, P3, P4, and P5.

## Why Add EMA Attention

EMA_attention is retained as the lightweight attention branch to strengthen damage-region response and suppress background texture. It is placed on the final P5 fused feature before `Detect`, matching the existing standalone EMA candidate style while keeping the combination minimal.

## Current Composite Candidate

YAML:

`ultralytics/cfg/models/26/yolo26-Paper1-P2Lite-SPDConv-EMA.yaml`

Design:

- Base: CPUBoneNano-P2Lite four-scale P2/P3/P4/P5 structure
- SPDConv merge: `space_to_depth + Conv(stride=1)` in the bottom-up neck
- EMA merge: one `EMA_attention` block before the P5 Detect input
- Detect inputs: P2, P3, P4, P5

This YAML is a buildability and smoke-test candidate only. It must not replace the single-module evidence chain until EMA_attention and SPDConv complete their own 30 epoch signal experiments.
