# WPFormer Wavelet Adaptation For Paper 1

## Verdict

WPFormer is relevant to Paper 1, but it is not a plug-in object-detection model. The CVPR 2025 method uses PVTv2, FPN, mask queries, and a transformer decoder for pixel-level segmentation. Its strongest transferable idea is Wavelet-enhanced Cross-Attention (WCA): Haar low/high-frequency decomposition followed by local/global modulation of high-frequency details. This is directly motivated by weak, elongated defects and was evaluated on CrackSeg9k.

Prototype-guided Cross-Attention (PCA) is not transferred. PCA updates a fixed set of segmentation queries through learned prototypes; YOLO26's convolutional Detect head has no equivalent mask-query state. Adding PCA would create a new detector rather than a defensible single module.

Primary sources:

- Paper: [Wavelet and Prototype Augmented Query-based Transformer for Pixel-level Surface Defect Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.html)
- Official repository: [fengyan-cv/WPFormer](https://github.com/fengyan-cv/WPFormer), reviewed at commit `83a33bbf5ed96dff069e9d58f5f3e0c464bae446`

The implementation in this project is an independent detection adaptation of the paper equations. It must be called **WPFormer-WCA-inspired Wavelet Detail Refinement (WDR)**, not WPFormer or the original WCA.

## S4 Design

YAML: `ultralytics/cfg/models/26/yolo26n-Paper1-S4-WPFormer-WDR-P3.yaml`

Placement: one `WaveletDetailRefinement` instance on the final P3 feature used by Detect. P4/P5 and the bottom-up neck remain unchanged.

Processing:

1. Fixed orthonormal Haar decomposition produces LL, LH, HL, and HH bands.
2. Local and global context from `LL + mean(|LH|, |HL|, |HH|)` generates a channel/spatial gate.
3. The same gate modulates the three directional detail bands, suppressing noisy high-frequency responses.
4. Inverse Haar reconstruction returns to the original P3 resolution.
5. A zero-initialized 1x1 output projection adds only the learned frequency correction to the baseline feature.

This differs from the failed FDRConv diagnostic: WDR retains directional Haar bands, conditions high-frequency suppression on low/high context, and does not replace a pretrained downsampling convolution.

## Audit Evidence

The local audit is recorded in `experiments/module_scan/paper1_s4_wpformer_wdr_audit.md`.

| property | result |
| --- | ---: |
| Added parameters | 6,288 |
| Expected Japan7 parameters | 2,382,489 (baseline 2,376,201 + 6,288) |
| Estimated extra compute at 640 | less than 0.07G arithmetic; functional Haar operations may be absent from THOP output, so measured FPS is authoritative |
| Parameter transfer | 99.756144% |
| Backbone transfer | 100% |
| Neck transfer | 99.303994% |
| Detect transfer | 100% |
| Pretrained full-model equivalence | exact (`max_error=0`) |
| Full 640x640 CPU forward/backward | PASS |
| CPU bfloat16 | PASS |
| Local CUDA AMP | unavailable; remote smoke required |

## Remote Commands

First pull the commit containing this model, verify the dataset, and run the audit. Then run only the 1 epoch smoke:

```bash
cd ~/YOLO26-probe
python scripts/check_dataset.py --data configs/japan7_remote.yaml
python scripts/audit_paper1_wpformer_wavelet.py

python scripts/train_module_pilot.py \
  --model-yaml ultralytics/cfg/models/26/yolo26n-Paper1-S4-WPFormer-WDR-P3.yaml \
  --data configs/japan7_remote.yaml \
  --epochs 1 --imgsz 640 --batch 32 --device 0 --workers 8 \
  --pretrained yolo26n.pt --checkpoint-remap auto \
  --name "smoke_Paper1_S4_WPFormerWDR_P3_japan7_e1_img640_b32_pretrained_amp_seed42_$(date +%Y%m%d_%H%M%S)"
```

Run 30 epochs only if the smoke completes without OOM/NaN and `pretrained.txt` confirms the audited regional coverage:

```bash
python scripts/train_module_pilot.py \
  --model-yaml ultralytics/cfg/models/26/yolo26n-Paper1-S4-WPFormer-WDR-P3.yaml \
  --data configs/japan7_remote.yaml \
  --epochs 30 --imgsz 640 --batch 32 --device 0 --workers 8 \
  --pretrained yolo26n.pt --checkpoint-remap auto \
  --name "signal_Paper1_S4_WPFormerWDR_P3_japan7_e30_img640_b32_pretrained_amp_seed42_$(date +%Y%m%d_%H%M%S)"
```

Decision against the matched `80bdad9` B0 (`mAP50-95=0.319`):

- below `0.319`: reject WDR;
- `0.319-0.322`: treat as neutral, do not combine;
- at least `0.323` with no material D00/D10 decline: positive signal;
- at least `0.328` with a D00 or D10 gain: strong signal and eligible for a two-module experiment.

Do not run a WPFormer-WDR three-module combination before this single-module result.

## Adversarial Review (2026-07-12)

| severity | finding | disposition |
| --- | --- | --- |
| High | Signed `LL + LH + HL + HH` algebraically collapsed to twice one polyphase sample for the selected Haar signs, so the context gate did not receive genuine multi-band context. | Fixed by combining signed LL structure with mean absolute directional detail. A regression check proves the old cancellation and rejects its return. |
| High | Pixel-level CrackSeg9k evidence may not transfer to Japan7 bounding-box loss; a box can be large even when its crack line is thin. | Not fixable in code. S4 remains a single-module diagnostic with a `0.323` promotion gate. |
| Medium | Zero-initialized output projection could leave the context path under-trained. | First-step projection and second-step context gradients are explicitly checked; learned projection/gate statistics must still be inspected after 30e. |
| Medium | Empty spatial dimensions, integer tensors, wrong channels, malformed Haar bands, and invalid constructor values previously failed indirectly. | Explicit validation added; failed calls are verified not to mutate module state. |
| Medium | THOP may omit functional slicing, arithmetic, padding, and Haar reconstruction. | Do not claim efficiency from reported GFLOPs alone; measure end-to-end FPS with the same hardware and protocol. |
| Medium | A `.pt` checkpoint remains a trusted binary input; repository path confinement does not make a hostile pickle safe. | Use only the official/project-controlled `yolo26n.pt` and preserve its checksum in formal experiment metadata. |
| Low | A hostile in-repository YAML could still request excessive model work even though YAML loading and module resolution do not execute arbitrary expressions. | Run only the committed S4 YAML at a verified Git hash; the pilot launcher already bounds image size, batch, epochs, and paths. |
| Low | Concurrent audit processes could leave a partial report. | Report publication now uses atomic temporary-file replacement; module forward itself is stateless and passed concurrent re-entry. |
| Residual | CUDA AMP, DDP, ONNX, and TensorRT behavior are not established locally. Non-finite inputs are propagated rather than hidden. | CUDA AMP is gated by remote 1e smoke. Export/DDP checks are required only if S4 is promoted. Training remains responsible for NaN detection. |

The least certain issue is scientific, not mechanical: whether a box-supervised YOLO loss supplies enough signal for wavelet detail modulation to improve D00/D10. Buildability and exact baseline initialization cannot answer that question.
