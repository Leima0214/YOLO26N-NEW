# Module Scan Decision Memory

Project: YOLO26-probe

Branch: `codex/yolo26-module-scan-cleanup`

Stage: module cleanup and Paper 1/Paper 2 candidate screening.

## 2026-07-10 Update

The candidate list remains valid, but its historical scratch rankings are not directly comparable to the pretrained YOLO26n baseline. EMA_attention transfers `468/714` items from `yolo26n.pt` and remains active for a protocol-matched signal. CPUBoneNano-P2Lite transfers only `8/881`; it is paused pending architecture-native CPUBone weight conversion and build validation.

## Decision

Use the current buildability scan as the execution baseline:

- Candidates: 13
- Build OK: 10
- Build failed: 3

Do not restore deleted module-zoo files into this branch. Do not repair unrelated failed modules here. A generated composite must pass its audit and a matched smoke before any signal training.

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

## 2026-07-11 Three-Module Paper 1 Strategy

Paper 1 now targets a final YOLO26n model with exactly three named modules. The final model must exceed the historical YOLO26n Japan7 baseline (`mAP50 = 0.623`, `mAP50-95 = 0.341`). Composite screening is exploratory; the later single-removal ablation table must report every metric rather than selecting only favorable columns.

Each module must have a role declared before its experiment:

- accuracy/detail role: expected to improve `mAP50-95`, especially D00/D10 AP;
- recall role: may be retained for a reproducible Recall increase even if its standalone mAP gain is neutral;
- efficiency role: may be retained for lower Params/FLOPs or higher measured FPS despite a small accuracy trade-off;
- fusion role: expected to improve multi-scale feature use or stabilize the final three-module model.

Recall alone must be described as improved detection coverage or sensitivity, not as proof of robustness. Robustness requires separate perturbation or cross-domain evidence. The final three-module model still has to beat the primary baseline metric; secondary-metric explanations cannot excuse a weaker final model.

### Audited Combination Pools

The pools below define the low-risk standard search space. They are not a hard requirement: combinations may use multiple modules from the same pool when the expected accuracy gain justifies the overlap.

| slot | candidates | status |
| --- | --- | --- |
| detail | official YOLO26 P2, SPDConv, LaplacianConv, FDConv | P2 uses the official YOLO26 backbone; the local FDConv port is transfer-compatible and audited |
| attention | EMA-P3-factor8, SEAttention-P3, CBAM-P3 | EMA-P5 is rejected; P3 variants require fresh build checks |
| neck/efficiency | BiFPN-labelled weighted concat, FFAFusion-Neck, CARAFE, slimneck, GSConv | `Concat_bifpn` is BiFPN-style positive weighted concatenation, not classic additive BiFPN; FFA/CARAFE are audited locally |

The one-per-pool grid defines 57 low-risk triples (`4 x 3 x 5`, excluding the three LaplacianConv + GSConv direct collisions because both replace the same shallow layer). The broader search space also includes repeated-role, serial same-stage, multi-attention, and multi-fusion triples. Rank expected gain first; use complementarity, insertion conflicts, and independent interpretability only to order experiments within similar expected gain. Do not materialize the full combinatorial space.

### Tier A: Complementary and Directly Interpretable

| priority | three-module composite | intended roles | current decision |
| ---: | --- | --- | --- |
| 1 | official P2 + SPDConv + EMA-P3-factor8 | high-resolution detail + downsampling preservation + key-class attention | main Paper 1 hypothesis after clean single-module checks |
| 2 | official P2 + LaplacianConv + EMA-P3-factor8 | high-resolution detail + edge response + key-class attention | strongest interpretable fallback |
| 3 | official P2 + EMA-P3-factor8 + BiFPN | small-target coverage + attention + weighted fusion | active fusion alternative |
| 4 | SPDConv + EMA-P3-factor8 + FFAFusion-Neck | detail preservation + attention + adaptive fusion | no-P2 accuracy alternative |
| 5 | LaplacianConv + EMA-P3-factor8 + BiFPN | edge response + attention + weighted fusion | lower-complexity accuracy alternative |
| 6 | official P2 + SEAttention-P3 + FFAFusion-Neck | high-resolution detail + lightweight attention control + adaptive fusion | attention-control composite |
| 7 | official P2 + CBAM-P3 + BiFPN | high-resolution detail + traditional attention control + weighted fusion | traditional-control composite |
| 8 | official P2 + EMA-P3-factor8 + slimneck | accuracy + recall/attention + efficiency | speed-oriented candidate; port review required |
| 9 | SPDConv + EMA-P3-factor8 + slimneck | detail + attention + efficiency | speed-oriented no-P2 candidate |
| 10 | FDConv + EMA-P3-factor8 + FFAFusion-Neck | frequency-aware stem + attention + adaptive fusion | frozen until FDConv builds cleanly |
| 11 | official P2 + LaplacianConv + CARAFE | high-resolution detail + edge response + content-aware upsampling | accuracy-heavy candidate; CARAFE blocked |
| 12 | FDConv + GSConv + CARAFE | frequency/detail + efficient shallow convolution + upsampling | high-risk efficiency candidate; not in first queue |

### Tier B: Overlapping Roles but Plausible Accuracy Synergy

These combinations do not satisfy the one-module-per-role rule, but remain valid candidates when the objective is final composite accuracy.

| priority | three-module composite | overlap/risk | current decision |
| ---: | --- | --- | --- |
| 13 | official P2 + SPDConv + LaplacianConv | three shallow-detail mechanisms | strong accuracy candidate if initialization remains transferable |
| 14 | official P2 + SPDConv + FDConv | P2 plus two downsampling/stem changes | materialized; SPD backbone plus FDConv PAN downsampling |
| 15 | official P2 + LaplacianConv + FDConv | two shallow filtering mechanisms | materialized; both Conv-compatible replacement paths inherit baseline weights |
| 16 | official P2 + SPDConv + BiFPN | two detail mechanisms plus fusion | plausible recall-heavy composite |
| 17 | official P2 + LaplacianConv + FFAFusion-Neck | detail duplication plus adaptive fusion | plausible accuracy-heavy composite |
| 18 | official P2 + FDConv + FFAFusion-Neck | P2 and frequency-aware stem plus fusion | materialized; build/runtime audit passed |
| 19 | SPDConv + LaplacianConv + BiFPN | two shallow-detail mechanisms plus fusion | no-P2 fallback |
| 20 | SPDConv + FDConv + FFAFusion-Neck | two convolution replacements plus fusion | dependency and optimization risk |
| 21 | official P2 + CARAFE + BiFPN | two neck/fusion mechanisms | potentially strong multi-scale model, but expensive |
| 22 | official P2 + CARAFE + FFAFusion-Neck | two adaptive fusion mechanisms | potentially strong but hard to attribute |
| 23 | SPDConv + CARAFE + BiFPN | detail preservation plus two neck changes | high-FLOPs exploratory candidate |
| 24 | LaplacianConv + CARAFE + FFAFusion-Neck | edge stem plus two fusion mechanisms | materialized; exact pretrained baseline-equivalence passed |

### 2026-07-12 Tier B Materialization

All 12 Tier B rows (B13-B24) now have generated YAMLs. Repeated shallow replacements use explicit, non-conflicting placements: the first operator replaces the backbone P2-to-P3 downsampling and the second replaces PAN P3-to-P4 downsampling. This preserves the semantic checkpoint map and lets LaplacianConv/FDConv inherit the corresponding baseline Conv weights.

All 12 passed safe construction, parameter-transfer checks, finite 640x640 forward/backward, 32x32 CPU mixed precision, real custom BatchNorm fusion, malformed-input recovery, fixed-shape concurrent inference, and atomic concurrent generation/reporting. Parameters span `2.552M-2.704M`; THOP reports `5.9-9.9 GFLOPs` after the FDConv correction, but this is a lower bound because functional FFT, grid sampling, unfold, interpolation, and rearrangement operations may be omitted. Detailed evidence and severity-ranked residual risks are in `experiments/module_scan/paper1_tierb_adversarial_audit.md`.

The follow-up adversarial review corrected axial angle wrapping in FFA, removed FDConv's duplicate convolution, replaced CARAFE patch replication with bounded einsum reassembly, stabilized weighted concatenation with softmax, implemented LaplacianConv/FDConv BN fusion, and made the audit effective under `python -O`. Every model is now probed at six boundary shapes; non-stride-aligned direct inputs may be rejected by FPN fusion, so Tier B training enforces an `imgsz` divisible by 32.

Tier B training additionally requires the SHA256-pinned project-root `yolo26n.pt`, `imgsz <= 640`, `batch <= 32`, and the audited regional transfer minima. Each run stores model/data snapshots, their hashes, checkpoint hash, Git status, and a durable state marker. CUDA AMP, measured latency/FPS, peak VRAM, and training iteration time remain remote smoke requirements and are not established by this structural audit.

Materialization count after this update:

- original Tier A/B/C list: 24 of 30 have YAMLs; the 6 Tier C rows remain hypotheses;
- including the conditional WDR queue: 24 of 33 have YAMLs; Tier C plus W1-W3 leave 9 hypotheses;
- before this update, the corresponding missing counts were 18 and 21.

This is structural evidence only. No Tier B model was trained, and none is promoted by buildability alone.

### Tier C: Stacked or Conflicting Modules

These are not forbidden. They are ranked last because direct YAML merging is insufficient: a serial order or relocation must be specified, pretrained mapping must be re-audited, and each resulting YAML needs a fresh build report.

| priority | three-module composite | conflict | current decision |
| ---: | --- | --- | --- |
| 25 | EMA-P3-factor8 + SEAttention-P3 + CBAM-P3 | three attention gates on one scale | test only after single-attention variants; highest over-suppression risk |
| 26 | EMA-P3-factor8 + SEAttention-P4 + CBAM-P5 | three attention modules across scales | more buildable than same-scale stacking, still difficult to attribute |
| 27 | CARAFE + BiFPN + FFAFusion-Neck | three neck/fusion mechanisms | large optimization and latency risk |
| 28 | SPDConv + LaplacianConv + FDConv | three shallow convolution/detail mechanisms | requires explicit layer assignment |
| 29 | LaplacianConv + GSConv + EMA-P3-factor8 | LaplacianConv and GSConv target the same shallow layer | relocate or serialize before build |
| 30 | official P2 + CPUBone-P2Lite + EMA-P3-factor8 | duplicate P2 paths and backbone replacement | lowest priority; invalid as a direct merge |

Tier labels express engineering and attribution risk, not an assumption that Tier B/C cannot improve accuracy. A lower-tier composite may be promoted when its matched pilot result is stronger.

EMA-P3-factor8 completed its matched 30 epoch signal on 2026-07-11 with `mAP50 = 0.518`, `mAP50-95 = 0.292`, and `Recall = 0.499`, below the 30 epoch control (`0.572`, `0.318`, `0.552`). It is rejected as a standalone module and retained only in the user-directed composite-first exploration queue.

All 12 Tier A composites were materialized and passed build-only checks on 2026-07-11. CARAFE and FDConv are no longer dependency-blocked in these combinations; unused `mmcv` imports were removed. This is build evidence only. Pretrained transfer coverage, THOP FLOPs, smoke stability, and accuracy remain unverified.

## 2026-07-11 Adversarial Correction

The first buildable A01-A12 queue was superseded after a source-level and adversarial review. The corrected definitions use identity-initialized attention/fusion, a Conv-compatible FDConv, bounded Laplacian enhancement, one FFA node per FFA composite, channel-preserving CARAFE, minimal weighted concat, atomic SPDConv, and semantic pretrained maps.

All 12 corrected models passed finite full-model forward/backward and fused-inference checks. Parameter-weighted pretrained coverage is `95.5-100%` for A01-A07/A10-A12, except that P2 Detect coverage is intentionally `86.562%` because the new P2 branch has no source. A08/A09 have only `58.347%`/`61.843%` neck coverage and remain lower priority. Detailed evidence is in `experiments/module_scan/paper1_tiera_adversarial_audit.md`.

The old A05 30 epoch result (`0.526` mAP50, `0.295` mAP50-95) was produced with the superseded implementation. It proves that old design did not beat the control, not that corrected A05 fails. Fresh runs must use unique names and `--checkpoint-remap auto`.

## 2026-07-11 Corrected Composite Results And S4

Commit-matched B0 reproduced at `0.574/0.319` mAP50/mAP50-95. Corrected A05 reached `0.526/0.294`; corrected A10 reached `0.513/0.289`. Both reduced all seven class AP50-95 values and are rejected without 100e promotion. Current EMA composites are paused, and the workflow returns to single-module diagnosis.

WPFormer is a query-based pixel-level segmentation system, not a YOLO plug-in. Its WCA idea is relevant because it was designed for weak elongated defects and evaluated on CrackSeg9k. S4 therefore adapts only WCA's context-modulated Haar detail mechanism as `WaveletDetailRefinement` on the P3 Detect input. PCA is excluded because YOLO26 Detect has no mask-query state. S4 must be described as WPFormer-WCA-inspired WDR, not as full WPFormer.

S4 passed exact pretrained baseline equivalence, 640x640 forward/backward, mixed-precision, fusion, and semantic-transfer audits. It remains an untrained candidate. Run one CUDA AMP smoke, then a matched 30e signal only; require at least `0.323` mAP50-95 without a material D00/D10 decline before any combination.

### Conditional WDR Three-Module Queue

WDR is technically shape-compatible with the remaining S1/S2/S3 single-module ideas, but no combination is authorized yet. The three possible three-module candidates formed by WDR plus two of those modules are:

| priority | conditional three-module model | placement and roles | current risk |
| ---: | --- | --- | --- |
| W1 | WDR + SEAttention + single-node FFAFusion | FFA at P3 top-down fusion; WDR then SE on the final P3 Detect feature | best role separation, but WDR and FFA both use frequency cues |
| W2 | FDRConv + WDR + SEAttention | FDRConv at P2-to-P3 downsampling; WDR then SE at final P3 Detect | two detail/frequency mechanisms may overlap; FDRConv was nearly unused in A10 |
| W3 | FDRConv + WDR + single-node FFAFusion | FDRConv shallow downsampling; FFA P3 fusion; WDR P3 Detect refinement | three frequency-oriented mechanisms, highest redundancy and lowest priority |

These rows are hypotheses, not YAMLs. If S4 is below `0.319`, delete all three. A constituent must first reach at least `0.323` alone; the corresponding two-module pair must then retain a positive signal before adding the third module. Do not force a three-module paper model when the single and pair evidence is negative.

### Decision Rules

- A final composite is successful only if `mAP50-95 > 0.341`; target at least `0.346` to avoid treating noise as a paper result.
- An accuracy module should target at least `+0.003 mAP50-95` under a matched protocol.
- A recall-role module should target at least `+0.02 Recall` with no more than `-0.003 mAP50-95` in the final composite.
- An efficiency-role module should target at least `10%` measured FPS improvement or a clear Params/FLOPs reduction with no more than `-0.003 mAP50-95` in the final composite.
- Report `P`, `R`, `mAP50`, `mAP50-95`, D00/D10 AP, Params, FLOPs, and FPS for every ablation row.
- The old Baidu Netdisk source tree contains 105 YOLO26 YAMLs but is not a Git repository and currently fails import on an unrelated `timm.models.layers.weight_init` incompatibility. Use it only as a read-only module source; port selected modules into the clean branch and build-check them before training.
