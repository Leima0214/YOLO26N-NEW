# Paper 1 A2 Adversarial Audit

- status: `PASS`
- git commit: `33be9d86c4ea2541f24c145f75a706d0e0d66b4e`
- checks passed: `251`
- checkpoint SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- loss.py SHA256: `21ad7fa3d1fcfcea78d77df35ba7b234d84b9529a841f2bd56de0675cc1baf1d`
- A2 YAML SHA256: `4a588c7f8e9c42edfa573c3eadb1d5b51252868e0fb905fd5f3cb483f9b9081a`
- torch/CUDA: `2.8.0+cu128` / `12.8`
- CUDA device: `NVIDIA GeForce RTX 2060`
- DDP: `not verified; formal protocol is single GPU`

## Numerical Details

```json
{
  "a2_one2many_assigned_positive_probe": {
    "base_ciou_grad_norm": 0.042182184755802155,
    "base_ciou_mean": 0.6228004097938538,
    "base_ciou_weighted": 0.5352506637573242,
    "bounded_penalty_mean": 0.5284682512283325,
    "grad_norm_ratio": 0.08748739212751389,
    "penalty_grad_norm": 0.0036904094740748405,
    "penalty_mean_by_ar": {
      "2to3": null,
      "3to5": 0.458636999130249,
      "5to8": 0.775534451007843,
      "ge8": 0.918727457523346,
      "lt2": 0.0
    },
    "positive_count": 39,
    "weighted_penalty": 0.03092050366103649,
    "weighted_penalty_to_ciou": 0.05776826664805412
  },
  "a2_one2one_assigned_positive_probe": {
    "base_ciou_grad_norm": 0.11992879956960678,
    "base_ciou_mean": 0.5095163583755493,
    "base_ciou_weighted": 0.4414278268814087,
    "bounded_penalty_mean": 0.4238513708114624,
    "grad_norm_ratio": 0.09915390610694885,
    "penalty_grad_norm": 0.011891408823430538,
    "penalty_mean_by_ar": {
      "2to3": null,
      "3to5": 0.15470343828201294,
      "5to8": 0.6388086080551147,
      "ge8": 0.9018934965133667,
      "lt2": 0.0
    },
    "positive_count": 4,
    "weighted_penalty": 0.03360963240265846,
    "weighted_penalty_to_ciou": 0.0761384516954422
  },
  "fp16_extreme_matrix": {
    "penalty_max": 0.8823529481887817,
    "penalty_min": 0.0,
    "skipped_unrepresentable": 16,
    "tested_orientations": 124
  },
  "synthetic_penalty_contribution": {
    "aspect_error_mean": 0.2006707489490509,
    "base_ciou_mean": 0.17677146196365356,
    "base_grad_norm": 0.3616466820240021,
    "bounded_penalty_mean": 0.14663271605968475,
    "elongation_gate_mean": 0.7404950261116028,
    "gate_by_ar": {
      "1": 0.0,
      "10": 0.9801980257034302,
      "3": 0.800000011920929,
      "5": 0.9230769276618958,
      "50": 0.9992003440856934
    },
    "grad_norm_ratio": 0.13003212895356175,
    "penalty_grad_norm": 0.047025687992572784,
    "weighted_penalty_mean": 0.014663273468613625,
    "weighted_penalty_to_ciou": 0.08295045651446034
  }
}
```

## Checks

- PASS: trusted yolo26n checkpoint exists
- PASS: trusted yolo26n checkpoint SHA256 matches
- PASS: invalid input rejected: broadcast shape
- PASS: invalid input rejected: rank/last dimension
- PASS: invalid input rejected: integer dtype
- PASS: invalid input rejected: bool dtype
- PASS: invalid input rejected: pred NaN
- PASS: invalid input rejected: pred +Inf
- PASS: invalid input rejected: pred -Inf
- PASS: invalid input rejected: target NaN
- PASS: invalid input rejected: target +Inf
- PASS: invalid input rejected: target -Inf
- PASS: invalid input rejected: zero pred width
- PASS: invalid input rejected: negative target height
- PASS: mixed floating dtypes promote to float32
- PASS: mixed floating dtype gradients are finite
- PASS: invalid input rejected: device mismatch
- PASS: invalid Shape-IoU scale rejected: -1.0
- PASS: invalid Shape-IoU scale rejected: nan
- PASS: invalid Shape-IoU scale rejected: True
- PASS: Shape-IoU xyxy/xywh match
- PASS: Shape-IoU gradients are finite
- PASS: invalid A2 weight rejected: -0.1
- PASS: invalid A2 weight rejected: 1.1
- PASS: invalid A2 weight rejected: nan
- PASS: invalid A2 weight rejected: True
- PASS: A1 and A2 cannot be enabled together
- PASS: CUDA is available for adversarial FP16 audit
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 penalty is finite
- PASS: extreme FP16 horizontal/vertical symmetry
- PASS: extreme FP16 gradients finite
- PASS: extreme FP16 matrix executed
- PASS: baseline remains CIoU with zero elongation penalty
- PASS: A1: state names match baseline
- PASS: A1: state shapes match baseline
- PASS: A1: checkpoint has 708 matching items
- PASS: A1: all transferred tensors are bitwise equal
- PASS: lambda0 control explicitly configures both E2E branches
- PASS: lambda0 CPU single: one2many box/cls/dfl exact
- PASS: lambda0 CPU single: one2one box/cls/dfl exact
- PASS: lambda0 CPU single: total loss exact
- PASS: lambda0 CPU single: every parameter gradient exact
- PASS: lambda0 CPU empty: one2many box/cls/dfl exact
- PASS: lambda0 CPU empty: one2one box/cls/dfl exact
- PASS: lambda0 CPU empty: total loss exact
- PASS: lambda0 CPU empty: every parameter gradient exact
- PASS: lambda0 CPU mixed: one2many box/cls/dfl exact
- PASS: lambda0 CPU mixed: one2one box/cls/dfl exact
- PASS: lambda0 CPU mixed: total loss exact
- PASS: lambda0 CPU mixed: every parameter gradient exact
- PASS: lambda0 CUDA AMP 640 batch2 mixed: one2many box/cls/dfl exact
- PASS: lambda0 CUDA AMP 640 batch2 mixed: one2one box/cls/dfl exact
- PASS: lambda0 CUDA AMP 640 batch2 mixed: total loss exact
- PASS: lambda0 CUDA AMP 640 batch2 mixed: every parameter gradient exact
- PASS: A2: state names match baseline
- PASS: A2: state shapes match baseline
- PASS: A2: checkpoint has 708 matching items
- PASS: A2: all transferred tensors are bitwise equal
- PASS: A2 CUDA AMP 640 batch2 mixed loss finite
- PASS: A2 CUDA AMP 640 batch2 mixed gradients finite
- PASS: A2 empty-target loss finite
- PASS: A2 empty-target backward gradients finite
- PASS: A2 one2many and one2one both use CIoU plus weight 0.1
- PASS: A2 diagnostics captured for both E2E branches
- PASS: A2 one2many diagnostic has assigned positives
- PASS: A2 one2many diagnostic values are finite
- PASS: A2 one2one diagnostic has assigned positives
- PASS: A2 one2one diagnostic values are finite
