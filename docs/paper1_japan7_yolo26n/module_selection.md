# Module Selection for Paper 1 — YOLO26n Lightweight Detail Enhancement

## Strategy

**Do NOT stack modules.** Each candidate is evaluated as a single-variable change against YOLO26n baseline.

## Phase 1: Buildability Scan

Run `scripts/scan_yolo26_module_buildability.py` to filter candidates that actually build.

Only YAMLs that produce `BUILD OK` proceed to pilot.

## Phase 2: 3-Epoch Pilot

Run `scripts/train_module_pilot.py --model-yaml <YAML>` for each BUILD OK candidate.

Promotion criteria to 20-epoch:
- No OOM
- No NaN
- mAP50 > 0.3 after 3 epochs
- mAP50-95 > 0.15 after 3 epochs
- Loss decreasing monotonically

## Phase 3: 20-Epoch Signal Test

Only top 3 pilots. Compare against YOLO26n baseline (same 20 epochs).

## Phase 4: 100-Epoch Formal

Only 1–2 modules, in this order:

| B# | Model | Description |
| --- | --- | --- |
| B0 | YOLO26n | Baseline (already done) |
| B1 | YOLO26n + P2/4D | Shallow detail branch |
| B2 | YOLO26n + EMA | Lightweight attention |
| B3 | YOLO26n + FDConv | Frequency detail enhancement |
| B4 | YOLO26n + P2 + best module | Combination |
| B5 | Ours-YOLO26n | Final |

## Candidate Priority

### Tier 1 — Shallow Detail (P2)

| YAML | Status | Notes |
| --- | --- | --- |
| `yolo26-4D.yaml` | Clean, standard-only modules | P2 detection head. Most fair comparison. |
| `yolo26-CPUBoneNano-P2Lite.yaml` | Custom backbone | P2 + CPU-optimized backbone. Less fair. |

### Tier 2 — Lightweight Attention

| YAML | Priority | Notes |
| --- | --- | --- |
| `yolo26-EMA_attention.yaml` | ⭐⭐⭐ | EMA: cross-channel + spatial encoding |
| `yolo26-SEAttention.yaml` | ⭐⭐ | SE: pure channel attention, 0 params |
| `yolo26-CBAM.yaml` | ⭐ | Classic spatial+channel, widely known |

### Tier 3 — Edge / Frequency Detail

| YAML | Priority | Notes |
| --- | --- | --- |
| `yolo26-LaplacianConv.yaml` | ⭐⭐⭐ | Edge-aware convolution |
| `yolo26-FDConv.yaml` | ⭐⭐ | Frequency-domain convolution |

### Tier 4 — Multi-Scale Fusion

| YAML | Priority | Notes |
| --- | --- | --- |
| `yolo26-CARAFE.yaml` | ⭐⭐⭐ | Content-aware upsampling |
| `yolo26-BiFPN.yaml` | ⭐⭐ | Weighted bidirectional FPN |
| `yolo26-SPDConv.yaml` | ⭐⭐ | Space-to-depth, detail-preserving downsampling |

## Skipped (not in scope)

- All Mamba variants — too new for domestic journals
- SwinTransformer — too heavy for lightweight narrative
- MoE / NAS / SAM — experimental, not interpretable
- FFAFusion series — unknown module registration
- ECCV2026 / CVPR2026 series — author submission experiments
