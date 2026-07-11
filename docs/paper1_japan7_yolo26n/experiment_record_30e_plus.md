# Experiment Record: Paper 1 Runs Above 30 Epochs

This file is the ongoing ledger for the Japan7 baseline and every Paper 1 run above `30` epochs. Add new `30e`, `100e`, or other long-run results here as they are produced so later decisions do not depend on scattered chat history. Its purpose is fast lookup and fair comparison, not launch authorization.

## 1. Historical Japan7 Baseline (100 epochs)

These are the stored baseline results for the Japan7 protocol.

| model | Params | FLOPs | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | 3.007M | 8.1G | 0.642 | 0.353 |
| YOLO11n | 2.584M | 6.3G | 0.642 | 0.349 |
| YOLO26s | 9.468M | 20.5G | 0.630 | 0.347 |
| YOLO26n | 2.376M | 5.2G | 0.623 | 0.341 |

Paper 1 uses `YOLO26n` as the direct baseline. The practical target for any module run is therefore:

- overall: beat `mAP50-95 = 0.341`
- key classes: improve `D00` and `D10`
- efficiency: keep Params/FLOPs growth scientifically explainable

## 2. Protocol-Matched Control (30 epochs)

This is the clean 30 epoch control used to anchor later module reruns under the same budget.

| run_name | initialization | AMP | epochs | Params | FLOPs | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_Paper1_YOLO26n_japan7_e30_img640_b32_pretrained_amp_seed42_20260710` | `yolo26n.pt` | True | 30 | 2.376M | 5.2G | 0.572 | 0.318 |

Per-class anchor for the key categories:

| model | D00 AP50 | D00 AP50-95 | D10 AP50 | D10 AP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLO26n 30e control | 0.406 | 0.190 | 0.315 | 0.123 |

This control is the correct comparison target for any 30 epoch pretrained module signal.

## 3. Historical 30 Epoch Module Signals

These results were kept because they showed module direction, but they are not clean pretrained-vs-pretrained comparisons.

| run | protocol note | mAP50 | mAP50-95 | Params | FLOPs | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `signal_Paper1_EMA_attention_japan7_e30_img640_b32_seed42` | historical scratch signal | 0.202 | 0.0884 | 2.377M | 5.2G | archive only |
| `signal_Paper1_P2Lite_japan7_e30_img640_b32_seed42` | historical scratch signal | 0.153 | 0.0674 | 3.672M | 6.6G | archive only |
| `signal_Paper1_SPDConv_japan7_e30_img640_b32_seed42` | historical scratch signal | 0.129 | 0.0516 | 2.600M | 1.5G | archive only |
| earlier EMA 30e remap-attempt | Detect inheritance incomplete at that time | 0.499 | 0.276 | n/a | n/a | invalid for formal comparison |

Interpretation:

- `EMA_attention` was the strongest scratch signal.
- `P2Lite` was the second strongest and best matched the shallow-detail/P2 story.
- the earlier `0.499 / 0.276` EMA result must not be used as the Paper 1 decision point because the Detect-layer mapping had not yet been cleaned at that stage.

### 2026-07-11 Protocol-Matched EMA P5 Rerun

| run | initialization | transferred | AMP | Params | mAP50 | mAP50-95 | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `signal_Paper1_EMA_P5_remap_japan7_e30_img640_b32_pretrained_amp_seed42_20260711_v2` | `yolo26n.pt` | 708/714 | True | 2.377M | 0.523 | 0.290 | reject current P5 placement |

Comparison against the protocol-matched YOLO26n 30e control:

| model | D00 AP50 | D00 AP50-95 | D10 AP50 | D10 AP50-95 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26n 30e control | 0.406 | 0.190 | 0.315 | 0.123 | 0.572 | 0.318 |
| EMA P5 remap 30e | 0.369 | 0.176 | 0.236 | 0.098 | 0.523 | 0.290 |
| delta (EMA - control) | -0.037 | -0.014 | -0.079 | -0.025 | -0.049 | -0.028 |

The run trained normally and produced valid artifacts, so this is not an AMP or runtime failure. Under the matched 30 epoch protocol, the current EMA P5 placement weakens both overall accuracy and the key classes; it must not advance to 100 epochs in this form. The deterministic `adaptive_avg_pool2d_backward_cuda` warning is non-fatal because deterministic mode is configured as `warn_only=True`.

### 2026-07-11 EMA P3 Factor-8 Rerun

| run | initialization | transferred | AMP | Params | mAP50 | mAP50-95 | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `signal_Paper1_EMA_P3_f8_japan7_e30_img640_b32_pretrained_amp_seed42_20260711` | `yolo26n.pt` | 708/714 | True | 2.377M | 0.518 | 0.292 | reject as a single-module signal; retain only for composite exploration |

Comparison against the protocol-matched YOLO26n 30e control:

| model | P | R | D00 AP50 | D00 AP50-95 | D10 AP50 | D10 AP50-95 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26n 30e control | 0.584 | 0.552 | 0.406 | 0.190 | 0.315 | 0.123 | 0.572 | 0.318 |
| EMA P3 factor-8 30e | 0.552 | 0.499 | 0.355 | 0.172 | 0.240 | 0.0979 | 0.518 | 0.292 |
| delta (EMA - control) | -0.032 | -0.053 | -0.051 | -0.018 | -0.075 | -0.0251 | -0.054 | -0.026 |

The factor-8 correction did not recover the EMA signal. It also reduced Recall and both key-class metrics, so EMA-P3-factor8 is not justified as a standalone accuracy or recall module. It remains in the exploratory three-module YAML queue only because the project is screening composites before ablation; any final claim still requires the composite to exceed the baseline.

## 4. 2026-07-10 Formal 100 Epoch Module Runs

These are the two completed 100 epoch module runs discussed yesterday. Both trained successfully, but neither beat the historical YOLO26n baseline.

| model | initialization | AMP | best epoch | Params | FLOPs | mAP50 | mAP50-95 | disposition |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO26n baseline | `yolo26n.pt` | True | 100 | 2.376M | 5.2G | 0.623 | 0.341 | stored baseline |
| EMA_attention | scratch | False | 65 | 2.377M | 5.2G | 0.587 | 0.314 | archive as exploratory |
| CPUBoneNano-P2Lite | scratch | True | 82 | 3.672M | 6.6G | 0.587 | 0.314 | archive as exploratory |

Key-class comparison:

| model | D00 AP50-95 | D10 AP50-95 |
| --- | ---: | ---: |
| YOLO26n baseline 100e | 0.183 | 0.148 |
| EMA_attention 100e | 0.184 | 0.125 |
| P2Lite 100e | 0.177 | 0.130 |

Per-class summaries preserved from the completed validation logs:

| model | D20 AP50-95 | D40 AP50-95 | D43 AP50-95 | D44 AP50-95 | D50 AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMA_attention 100e | 0.333 | 0.212 | 0.510 | 0.439 | 0.395 |
| P2Lite 100e | 0.334 | 0.212 | 0.511 | 0.432 | 0.399 |

## 5. Why Yesterday's Formal Results Were Not Good Enough

The current evidence points to protocol and representation issues, not just "insufficient training."

1. The baseline is a pretrained fine-tune, while both completed module 100 epoch runs were scratch runs. That alone prevents a fair scientific conclusion.
2. `EMA_attention` did not show a large enough gain to justify its extra structure. At the current insertion position it likely overlaps with features YOLO26 already models reasonably well.
3. `P2Lite` increases cost from `2.376M / 5.2G` to `3.672M / 6.6G`, but the added shallow-detail path did not convert into a better final `mAP50-95`.
4. The hardest classes remain `D00` and `D10`. Neither module produced the kind of key-class lift needed for a convincing Paper 1 story.
5. The near-identical final scores of EMA and P2Lite suggest the branch was still dominated by initialization/protocol effects rather than a strong module-specific advantage.

## 6. Current Safe Interpretation

- `YOLO26n baseline` remains the official Paper 1 reference line.
- the completed `EMA_attention 100e` and `P2Lite 100e` runs are useful as exploratory archive evidence, not as formal ablation wins.
- any new `30e` or `100e` module conclusion must be compared first against the clean `YOLO26n 30e control` and then, if promoted, against the historical `YOLO26n 100e` baseline.
- do not describe yesterday's module runs as having surpassed the baseline; they did not.
