# Module Scan Decision Memory

Project: YOLO26-probe

Branch: `codex/yolo26-module-scan-cleanup`

Stage: module cleanup and Paper 1/Paper 2 candidate screening.

## Decision

Use the current buildability scan as the execution baseline:

- Candidates: 13
- Build OK: 10
- Build failed: 3

Do not restore deleted module-zoo files into this branch. Do not repair failed modules here. Do not train from stacked module combinations.

## Build OK Candidates

| module | paper | params | execution decision |
| --- | --- | ---: | --- |
| CPUBoneNano-P2Lite | Paper 1 | 3,930,380 | second pilot batch |
| BiFPN | Paper 1 | 2,572,292 | backup after BiFPN1 |
| BiFPN1 | Paper 1 | 2,555,508 | first pilot batch |
| EMA_attention | Paper 1 | 2,572,952 | first pilot batch |
| SEAttention | Paper 1 | 2,572,792 | EMA backup |
| CBAM | Paper 1 | 2,580,843 | attention control |
| LaplacianConv | Paper 1 | 2,572,281 | first pilot batch |
| SPDConv | Paper 1 | 2,795,480 | second pilot batch |
| FFAFusion-Neck | Paper 1 / Paper 2 | 2,617,656 | second pilot batch |
| HVIEnhanceStem | Paper 2 | 2,575,008 | Paper 2 first pilot |

## Frozen Candidates

| module | reason | decision |
| --- | --- | --- |
| CARAFE | `mmcv` API compatibility | freeze until fix branch |
| FDConv | `mmcv` API compatibility | freeze until fix branch |
| ContextAggregation | `mmcv` API compatibility | freeze until fix branch |

## Paper 1

The long-term story remains `P2-like shallow detail + SPDConv + EMA`, but the current execution must be single-module first.

First 3 epoch pilot batch:

- `EMA_attention`
- `LaplacianConv`
- `BiFPN1`

Second 3 epoch pilot batch:

- `SPDConv`
- `CPUBoneNano-P2Lite`
- `FFAFusion-Neck`

Do not combine modules until individual pilots and signal experiments justify it.

## Paper 2

The current first candidate is `HVIEnhanceStem` because it matches color/light robustness under domain shift.

`ContextAggregation`, `FDConv`, and `hwd / wavelet-frequency enhancement` are deferred until buildability or compatibility is separately validated.

Strict DG must use:

- train = Japan_train
- val = Japan_val
- test = target domain

Loose transfer may use:

- train = Japan_train
- val = target_val

If target-domain validation selects `best.pt`, the experiment is not strict domain generalization.

## Guardrails

- Do not modify `ultralytics` core model code on this branch.
- Do not train on this documentation update.
- Do not submit `runs`, `datasets`, `.pt` weights, or large images.
- Do not treat `BUILD OK` as proof of effectiveness.
- Do not mix Paper 1 Japan7 and Paper 2 Common4 in one main result table.
