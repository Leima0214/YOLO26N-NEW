# Paper 1 Baseline Breakthrough Plan

## Problem Statement

The protocol-matched YOLO26n control reaches `0.574` mAP50 and `0.319` mAP50-95 at 30 epochs. A1 Shape-IoU, A2 bounded aspect localization, B1 assignment changes, and B2 hard-positive classification have been rejected. These results close the loss/assignment family; they do not prove that stronger feature supervision or representation cannot work.

The historical 100-epoch YOLO26s result (`0.347` mAP50-95) already exceeds YOLO26n (`0.341`) while remaining in the YOLO26 family. It is therefore the first evidence-backed teacher candidate.

## Ranked Route

### G0: Verify Teacher Headroom

Train official COCO-pretrained YOLO26s for 30 epochs under the exact B0 Japan7 protocol. This is a gate, not a Paper 1 contribution.

Promote YOLO26s as a teacher only when all conditions hold:

- full architecture-native checkpoint transfer;
- AMP, seed 42, image size 640, and the clean Japan7 split;
- mAP50-95 at least `0.325`, giving at least `+0.006` over the 30e YOLO26n control;
- D10 AP50 and AP50-95 no lower than B0 (`0.324/0.130`).

If this gate fails, do not implement distillation. A teacher without measured headroom cannot improve the student reliably.

### K1: Multi-Scale Pearson Feature Distillation

Use the frozen Japan7 YOLO26s teacher to supervise YOLO26n P3/P4/P5 neck features. Channel adapters are training-only. Features are standardized before Pearson-correlation loss so teacher magnitude and wider channels do not dominate the student. Student inference architecture, Params, FLOPs, and FPS remain unchanged.

This is preferred over localization distribution distillation because YOLO26 uses `reg_max=1`; there is no multi-bin DFL distribution to transfer.

K1 receives one smoke and one matched 30e signal. Promotion requires at least `0.322` mAP50-95 with no D10 regression. It is not promoted because a paper reported gains elsewhere.

### K2: Valuable-Region Prediction Distillation

Only after K1 has a positive signal, add teacher prediction supervision on spatially aligned P3/P4/P5 outputs. The mask must use teacher confidence and GT-overlapping regions to avoid overwhelming the student with dense background. K2 remains training-only and independently switchable.

### C1: Directional Strip Representation

Only after teacher-guided YOLO26n is stronger than B0, test one residual P3 strip-receptive-field block. It uses horizontal and vertical depthwise kernels to model elongated crack continuity. It is not combined with P2Lite: Japan7 geometry shows D00/D10 are not primarily small objects, while D10 is predominantly high-aspect-ratio.

## Paper Structure If The Gates Pass

The final three attributable components become:

1. `K1`: normalized multi-scale feature distillation;
2. `K2`: valuable-region prediction distillation;
3. `C1`: lightweight directional strip representation.

The YOLO26s teacher is used only during training. The final detector remains YOLO26n-based, with only C1 affecting inference cost.

## Formal Success Rule

The final 100e model must exceed the historical YOLO26n mAP50-95 of `0.341`; target `0.346` or higher to avoid presenting random variation as a contribution. Every ablation must use the same initialization, split, AMP mode, seed, image size, batch size, and epoch budget. Three components are retained only when their individual roles are supported by measured accuracy, recall, or efficiency evidence.

## Primary Evidence

- Localization Distillation for Dense Object Detection, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html
- Focal and Global Knowledge Distillation for Detectors, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html
- PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html
- CrossKD: Cross-Head Knowledge Distillation for Object Detection, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf
- The Devil is in the Crack Orientation, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Chen_The_Devil_is_in_the_Crack_Orientation_A_New_Perspective_ICCV_2023_paper.html
- StripRFNet, road-damage-specific preprint: https://arxiv.org/abs/2510.16115
