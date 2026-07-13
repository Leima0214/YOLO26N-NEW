# Experiment Record: Paper 1 Runs At 30 Epochs And Above

This file is the ongoing ledger for the Japan7 baseline and every Paper 1 run with `epochs >= 30`. Add new long-run results here as they are produced so later decisions do not depend on scattered chat history. Its purpose is fast lookup and fair comparison, not launch authorization.

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

| run_name | commit | initialization | AMP | epochs | Params | FLOPs | mAP50 | mAP50-95 | role |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `control_Paper1_YOLO26n_japan7_e30_img640_b32_pretrained_amp_seed42_20260710` | historical | `yolo26n.pt` | True | 30 | 2.376M | 5.2G | 0.572 | 0.318 | historical control |
| `control_Paper1_YOLO26n_commit6c34d74_japan7_e30_img640_b32_pretrained_amp_seed42_20260713_115118` | `6c34d74` | `yolo26n.pt` | True | 30 | 2.376M | 5.2G | 0.574 | 0.319 | current canonical control |

Per-class anchor for the key categories:

| model | D00 AP50 | D00 AP50-95 | D10 AP50 | D10 AP50-95 |
| --- | ---: | ---: | ---: | ---: |
| historical YOLO26n 30e control | 0.406 | 0.190 | 0.315 | 0.123 |
| current `6c34d74` YOLO26n 30e control | 0.397 | 0.193 | 0.324 | 0.130 |

The `6c34d74` run used full `708/708` tensor and `100%` regional/parameter transfer with AMP enabled. It exactly reproduces the corrected `80bdad9` control at `0.319 mAP50-95`, so it is the comparison target for subsequent 30 epoch pretrained signals on the current GPU environment.

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

### 2026-07-11 Tier A05 Pre-Correction Composite

| run | initialization | AMP | Params | FLOPs | P | R | mAP50 | mAP50-95 | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `signal_Paper1_TierA05_Lap_EMA_BiFPN_japan7_e30_img640_b32_pretrained_amp_seed42_20260711_144442` | reported pretrained run | True | 2.377M | 5.3G | 0.543 | 0.527 | 0.526 | 0.295 | archive; old implementation underperformed |

| model | D00 AP50 | D00 AP50-95 | D10 AP50 | D10 AP50-95 | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26n 30e control | 0.406 | 0.190 | 0.315 | 0.123 | 0.572 | 0.318 |
| old Tier A05 30e | 0.360 | 0.172 | 0.245 | 0.0997 | 0.526 | 0.295 |
| delta (A05 - control) | -0.046 | -0.018 | -0.070 | -0.0233 | -0.046 | -0.023 |

The run completed normally, but every primary/key-class metric was below the matched control. Its YAML and modules were subsequently superseded by the 2026-07-11 adversarial correction: identity initialization, semantic parameter transfer, and corrected BiFPN/Laplacian behavior materially change the model definition. This row is evidence for the old `be61dc3` implementation only and cannot be reused as a result for corrected A05.

### 2026-07-11 Commit-Matched Corrected Series

All three runs below used commit `80bdad9270519967c1e9cfdcef6814b53e4820ae`, `yolo26n.pt`, AMP, Japan7, 30 epochs, image size 640, batch 32, and seed 42.

| model | P | R | mAP50 | mAP50-95 | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| B0 YOLO26n | 0.587 | 0.561 | 0.574 | 0.319 | matched control |
| corrected A05: Laplacian + EMA + weighted concat | 0.552 | 0.517 | 0.526 | 0.294 | reject; no 100e |
| corrected A10: FDRConv + EMA + one-node FFA | 0.539 | 0.509 | 0.513 | 0.289 | reject; no 100e |

| model | D00 AP50-95 | D10 AP50-95 | D20 | D40 | D43 | D44 | D50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.193 | 0.130 | 0.348 | 0.189 | 0.537 | 0.439 | 0.397 |
| corrected A05 | 0.173 | 0.100 | 0.328 | 0.162 | 0.506 | 0.404 | 0.384 |
| corrected A10 | 0.164 | 0.0942 | 0.324 | 0.154 | 0.499 | 0.416 | 0.373 |

The corrected B0 reproduces the historical 30e control within `+0.001 mAP50-95`, so commit `80bdad9` did not regress the baseline. Every class declined in both corrected composites. Learned residual strengths were also small: A05 Laplacian gain `-0.00915`, A05 EMA gamma `0.02426`, A10 FDRConv gain `-0.00133`, A10 FFA mean absolute scale `0.00494`, and A10 EMA gamma `0.00779`. A05/A10 are formally eliminated and existing EMA composites are paused. New work returns to single-module diagnosis.

### 2026-07-13 S4 WPFormer-WCA-Inspired WDR

Run `signal_Paper1_S4_WDR_P3_japan7_e30_img640_b32_pretrained_amp_seed42_20260713` used commit `92c18ef12dc5b97823e83ff3652f1e716c2b3728`, the trusted `yolo26n.pt` checkpoint (`SHA256 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`), AMP, Japan7, 30 epochs, image size 640, batch 32, and seed 42. Parameter-weighted transfer was `99.7561%`; backbone, neck, and Detect transfer were `100%`, `99.3040%`, and `100%`.

| model | Params | FLOPs | P | R | mAP50 | mAP50-95 | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current canonical B0 (`6c34d74`) | 2.376M | 5.2G | 0.587 | 0.561 | 0.574 | 0.319 | control |
| S4 WDR P3 | 2.382M | 5.3G | 0.542 | 0.530 | 0.531 | 0.296 | reject at 30e; no combination or 100e promotion |
| delta (S4 - B0) | +0.006M | +0.1G | -0.045 | -0.031 | -0.043 | -0.023 | negative signal |

| model | D00 | D10 | D20 | D40 | D43 | D44 | D50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 AP50-95 | 0.193 | 0.130 | 0.348 | 0.189 | 0.537 | 0.439 | 0.397 |
| S4 AP50-95 | 0.169 | 0.103 | 0.318 | 0.174 | 0.506 | 0.412 | 0.387 |
| delta | -0.024 | -0.027 | -0.030 | -0.015 | -0.031 | -0.027 | -0.010 |

The best checkpoint occurred at epoch 30, but the matched B0 was also still improving at epoch 30. The mAP50-95 gap narrowed from `-0.0352` at epoch 10 to `-0.0232` at epoch 30, which does not override the predefined rejection gate. The learned WDR output projection was nonzero (`mean_abs=0.003324`, `max_abs=0.024628`, `L2=0.285238`), so failure cannot be attributed to an inactive module. S4 is closed as a valid negative result; W1-W3 are not authorized, and S4 must not enter a pair or three-module model. A fresh 100e run is allowed only as a low-priority convergence diagnostic after higher-value single-module screening, never as automatic promotion evidence.

The remote run contains `results.csv`, `args.yaml`, both weights, model/data snapshots, commit/branch/status, command, checkpoint/model hashes, transfer metadata, and environment snapshots. `best_validation_log.txt` preserves a repeat validation with complete per-class output. The original training stdout was not redirected to a file and cannot be reconstructed; `results.csv` is the authoritative epoch curve.

### 2026-07-13 A1 Standard Shape-IoU

Run `signal_Paper1_ShapeIoU_s1_japan7_e30_img640_b32_pretrained_amp_seed42_20260713_134524` used commit `3d9d448`, `yolo26n.pt`, full `708/708` transfer, AMP, the cleaned Japan7 dataset, 30 epochs, image size 640, batch 32, and seed 42. A1 changes only the training-time box regression from CIoU to standard Shape-IoU with scale 1; architecture, classification loss, assigner, Params, FLOPs, and inference remain baseline-identical.

| model | Params | FLOPs | P | R | mAP50 | mAP50-95 | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current canonical B0 | 2.376M | 5.2G | 0.587 | 0.561 | 0.574 | 0.319 | control |
| A1 standard Shape-IoU | 2.376M | 5.2G | 0.603 | 0.548 | 0.573 | 0.318 | retain as neutral comparison; no 100e |
| delta (A1 - B0) | 0 | 0 | +0.016 | -0.013 | -0.001 | -0.001 | fails promotion gate |

| model | D00 | D10 | D20 | D40 | D43 | D44 | D50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 AP50/AP50-95 | 0.397/0.193 | 0.324/0.130 | 0.657/0.348 | 0.434/0.189 | 0.760/0.537 | 0.697/0.439 | 0.750/0.397 |
| A1 AP50/AP50-95 | 0.405/0.193 | 0.308/0.124 | 0.660/0.350 | 0.440/0.189 | 0.756/0.528 | 0.698/0.439 | 0.741/0.404 |

A1 raises precision but reduces recall, overall mAP50-95, and the target D10 AP50/AP50-95. It is reproducible evidence that direct Shape-IoU replacement does not solve the measured D10 failure. Keep its YAML, code path, and run artifacts as a method-selection comparison, but do not include it in the final composite or promote it to 100e. A2 retains baseline CIoU and adds only a bounded log-aspect term. Its corrected adversarial audit proves full loss and every parameter gradient are bitwise baseline-equivalent at weight zero; A2 remains pre-smoke and has no experimental result yet.

## 4. 2026-07-10 Formal 100 Epoch Module Runs

These are the two completed 100 epoch module runs discussed yesterday. Both trained successfully, but neither beat the historical YOLO26n baseline.

| model | initialization | AMP | best epoch | Params | FLOPs | P | R | mAP50 | mAP50-95 | disposition |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO26n baseline | `yolo26n.pt` | True | 100 | 2.376M | 5.2G | 0.644 | 0.597 | 0.623 | 0.341 | stored baseline |
| EMA_attention | scratch | False | 65 | 2.377M | 5.2G | 0.614 | 0.563 | 0.587 | 0.314 | archive as exploratory |
| CPUBoneNano-P2Lite | scratch | True | 82 | 3.672M | 6.6G | 0.597 | 0.560 | 0.587 | 0.314 | archive as exploratory |

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

Full per-class best-checkpoint validation:

| model | D00 AP50/AP50-95 | D10 | D20 | D40 | D43 | D44 | D50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EMA_attention 100e | 0.386/0.184 | 0.331/0.125 | 0.661/0.333 | 0.479/0.212 | 0.775/0.510 | 0.713/0.439 | 0.766/0.395 |
| P2Lite 100e | 0.378/0.177 | 0.347/0.130 | 0.652/0.334 | 0.488/0.212 | 0.780/0.511 | 0.708/0.432 | 0.758/0.399 |

Completed artifact directories are `formal_Paper1_EMA_attention_japan7_e100_img640_b32_noamp_202607102` and `formal_Paper1_P2Lite_japan7_e100_img640_b32_scratch_amp_seed42_202607102`. Their `results.csv`, `args.yaml`, and `best.pt` are backed up. The older launcher wrote command/commit metadata into the sibling names without the trailing `2`; both halves are retained in `paper1_20260710_results_docs.tar.gz` and `paper1_20260710_best_weights.tar.gz` and must be paired during reproduction review.

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

## 7. Artifact Completeness Audit (2026-07-12)

Numerical summaries and reproducibility artifacts are tracked separately. A row is `complete` only when metrics, `results.csv`, `args.yaml`, `best.pt`, command, commit, pretrained metadata, and environment snapshot are all preserved or intentionally inapplicable.

| experiment group | numerical detail | local artifact evidence | status |
| --- | --- | --- | --- |
| Four Japan7 baselines, 100e | overall P/R/mAP, full per-class metrics, 100-row curves, commands, run manifest, environment | `experiments/japan7_baseline_20260707` and `experiments/baseline_japan7_summary`; its README explicitly says `args.yaml` was not retrieved, and no baseline `best.pt` was found in inspected backups | metrics complete; reproduction artifacts incomplete |
| EMA and P2Lite formal, 100e | overall, best epoch, full per-class AP50/AP50-95 | results/args/commands/commits and both `best.pt` files in the 2026-07-10 archives; metadata split across sibling run names as documented above | recoverable but not self-contained |
| Historical scratch EMA/P2Lite/SPDConv, 30e | overall signal metrics | results/args/best weights and logs in the 2026-07-09 archives | historical artifacts retained |
| Protocol-matched B0 and EMA 30e from 2026-07-10 | overall/results curves and key classes | results/args/commands/commits/pretrained metadata plus best weights in 2026-07-10 archives | complete for their historical protocol |
| EMA-P5-remap and EMA-P3-factor8, 30e | overall and per-class metrics recorded in this ledger | no matching local `results.csv`, `args.yaml`, or `best.pt` found in inspected archives | artifact gap |
| S4 WDR P3, 30e | overall, full per-class AP, 30-row curve, transfer and learned-projection statistics | self-contained remote run plus repeat best-checkpoint validation log; original training stdout was not captured | complete except original stdout log |
| Pre-correction Tier A05, 30e | overall and per-class metrics recorded | no matching local run artifacts found | artifact gap |
| Corrected commit-`80bdad9` B0/A05/A10, 30e | overall, full per-class metrics, learned-module diagnostics | each has results/args/best/command/commit/pretrained/environment files in `paper1_80bdad9_e30_results_20260711.tar.gz`; `last.pt` intentionally excluded | complete |

Therefore, it is not accurate to claim that every `30e+` experiment is fully archived. The scientific conclusions are recorded, but the missing historical run artifacts cannot be reconstructed after the corresponding remote GPU was shut down. Future `30e+` runs must be marked complete only after archive checksum verification.
