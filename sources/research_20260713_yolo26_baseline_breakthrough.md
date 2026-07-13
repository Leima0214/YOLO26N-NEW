# YOLO26 Paper 1 Baseline Breakthrough Research Notes

The repository's research-lookup plugin was attempted first, but neither `PARALLEL_API_KEY` nor `OPENROUTER_API_KEY` was configured. The sources below were therefore checked through their primary proceedings or preprint pages.

## Findings

- YOLO26 has `reg_max=1`, so multi-bin localization-distribution distillation from LD cannot be copied directly.
- PKD normalizes teacher and student FPN features and optimizes their Pearson correlation, explicitly addressing feature-magnitude and channel/stage imbalance.
- FGD shows that foreground/background imbalance makes uniform feature imitation harmful and motivates region-aware supervision.
- CrossKD separates detection and distillation heads to reduce target conflict in dense detectors.
- Crack-orientation research shows that horizontal boxes create scale and intra-class variation for curved cracks.
- StripRFNet is directly evaluated on RDD2022 and reports gains from strip receptive fields, but its P2/small-scale component conflicts with the measured Japan7 geometry and should not be copied wholesale.

## Selected Hypothesis

First verify a Japan7-fine-tuned YOLO26s teacher. If it has sufficient matched-protocol headroom, implement training-only multi-scale Pearson feature distillation into YOLO26n. Add valuable-region prediction distillation and one P3 directional strip block only after positive single-mechanism evidence.

## Sources

- https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html
- https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html
- https://proceedings.neurips.cc/paper_files/paper/2022/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html
- https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf
- https://openaccess.thecvf.com/content/ICCV2023/html/Chen_The_Devil_is_in_the_Crack_Orientation_A_New_Perspective_ICCV_2023_paper.html
- https://arxiv.org/abs/2510.16115
