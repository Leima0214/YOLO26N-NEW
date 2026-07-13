# Paper 1 B2 Adversarial Audit

- status: `PASS`
- git commit: `cffaacbe279af66579d3c0bf8345c1baa90bc61f`
- checks passed: `65`
- checkpoint SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- loss.py SHA256: `b4ca2a18f4ba2eb6597462a099c4e49b89bb38572a2336b46401cfe480d08200`
- B2 YAML SHA256: `040d0ff873f18fddac14e42d68060985c9d377edcb012692e9fd9c116dc3cc71`
- torch/CUDA: `2.5.1+cu124` / `12.4`
- CUDA device: `NVIDIA GeForce RTX 4090`
- CUDA AMP verified: `True`
- DDP: `not verified; formal protocol is single GPU`

## Numerical Details

```json
{
  "b2_one2many_assigned_positive_probe": {
    "added_bce": 1.3423553705215454,
    "added_grad_norm": 0.03338835388422012,
    "added_to_base_ratio": 0.08630610257387161,
    "base_bce": 15.553422927856445,
    "base_grad_norm": 0.47241997718811035,
    "boost_max": 0.19374559819698334,
    "boost_mean": 0.09406619518995285,
    "correct_confidence_mean": 4.194639404886402e-05,
    "grad_norm_ratio": 0.07067514955997467,
    "positive_count": 39,
    "quality_mean": 0.37628400325775146
  },
  "b2_one2one_assigned_positive_probe": {
    "added_bce": 1.9804863929748535,
    "added_grad_norm": 0.08684053272008896,
    "added_to_base_ratio": 0.12121295928955078,
    "base_bce": 16.338899612426758,
    "base_grad_norm": 0.6936584115028381,
    "boost_max": 0.19570598006248474,
    "boost_mean": 0.12275685369968414,
    "correct_confidence_mean": 6.696969649055973e-05,
    "grad_norm_ratio": 0.1251920759677887,
    "positive_count": 4,
    "quality_mean": 0.49105748534202576
  },
  "cpu_synthetic_probe": {
    "boost": [
      0.19999995827674866,
      0.049999985843896866,
      0.021362556144595146
    ],
    "confidence": [
      0.20000000298023224,
      0.800000011920929,
      0.20000000298023224
    ],
    "positive_weights": [
      1.1999999284744263,
      1.0499999523162842,
      1.021362543106079
    ],
    "quality": [
      0.9999997615814209,
      0.9999997615814209,
      0.10681277513504028
    ]
  },
  "cuda_synthetic_probe": {
    "boost": [
      0.19999995827674866,
      0.049999985843896866,
      0.021362556144595146
    ],
    "confidence": [
      0.20000000298023224,
      0.800000011920929,
      0.20000000298023224
    ],
    "positive_weights": [
      1.1999999284744263,
      1.0499999523162842,
      1.021362543106079
    ],
    "quality": [
      0.9999997615814209,
      0.9999997615814209,
      0.10681277513504028
    ]
  }
}
```

## Checks

- PASS: trusted yolo26n checkpoint exists
- PASS: trusted yolo26n checkpoint SHA256 matches
- PASS: loss.py: no eval or exec calls
- PASS: train_module_pilot.py: no eval or exec calls
- PASS: audit_paper1_b2_hard_positive.py: no eval or exec calls
- PASS: B2 YAML pins hard-positive weight 0.25
- PASS: B2 YAML architecture and all baseline settings are unchanged
- PASS: invalid hard-positive strength rejected: True
- PASS: invalid hard-positive strength rejected: -0.1
- PASS: invalid hard-positive strength rejected: 1.1
- PASS: invalid hard-positive strength rejected: nan
- PASS: invalid hard-positive strength rejected: inf
- PASS: cpu: adaptive weights are detached
- PASS: cpu: negative classes unchanged
- PASS: cpu: weights bounded to [1, 1.25]
- PASS: cpu: lower-confidence match receives more weight
- PASS: cpu: poor localization suppresses the boost
- PASS: cpu: factors are finite
- PASS: cpu: weighted BCE gradients are finite
- PASS: cpu: empty foreground is baseline BCE
- PASS: cpu: empty foreground diagnostics are empty
- PASS: cuda: adaptive weights are detached
- PASS: cuda: negative classes unchanged
- PASS: cuda: weights bounded to [1, 1.25]
- PASS: cuda: lower-confidence match receives more weight
- PASS: cuda: poor localization suppresses the boost
- PASS: cuda: factors are finite
- PASS: cuda: weighted BCE gradients are finite
- PASS: cuda: empty foreground is baseline BCE
- PASS: cuda: empty foreground diagnostics are empty
- PASS: weight0 config reaches both E2E branches
- PASS: weight0 CPU single: one2many exact
- PASS: weight0 CPU single: one2one exact
- PASS: weight0 CPU single: total exact
- PASS: weight0 CPU single: every parameter gradient exact
- PASS: weight0 CPU empty: one2many exact
- PASS: weight0 CPU empty: one2one exact
- PASS: weight0 CPU empty: total exact
- PASS: weight0 CPU empty: every parameter gradient exact
- PASS: weight0 CPU mixed: one2many exact
- PASS: weight0 CPU mixed: one2one exact
- PASS: weight0 CPU mixed: total exact
- PASS: weight0 CPU mixed: every parameter gradient exact
- PASS: weight0 CUDA AMP 640 batch2 mixed: one2many exact
- PASS: weight0 CUDA AMP 640 batch2 mixed: one2one exact
- PASS: weight0 CUDA AMP 640 batch2 mixed: total exact
- PASS: weight0 CUDA AMP 640 batch2 mixed: every parameter gradient exact
- PASS: baseline remains unweighted BCE in both E2E branches
- PASS: B2 state names match baseline
- PASS: B2 state shapes match baseline
- PASS: B2 checkpoint has 708 matching items
- PASS: B2 transferred tensors are bitwise equal
- PASS: B2 weight reaches both E2E branches
- PASS: B2 mixed-target loss is finite
- PASS: B2 mixed-target gradients are finite
- PASS: B2 one2many diagnostics captured
- PASS: B2 one2many diagnostics contain positives
- PASS: B2 one2many boost is active and bounded
- PASS: B2 one2many diagnostics are finite
- PASS: B2 one2one diagnostics captured
- PASS: B2 one2one diagnostics contain positives
- PASS: B2 one2one boost is active and bounded
- PASS: B2 one2one diagnostics are finite
- PASS: B2 empty-target loss is finite
- PASS: B2 empty-target gradients are finite
