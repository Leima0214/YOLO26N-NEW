# Paper 1 Writing Outline

## 1. Introduction
- Road crack detection importance
- Challenges: thin cracks, low contrast, multi-scale
- Existing methods: manual inspection → CNN → YOLO
- YOLO26n: lightweight but detail-deficient
- Our contribution: lightweight detail enhancement for YOLO26n

## 2. Related Work
- 2.1 Road crack detection (traditional CV → deep learning)
- 2.2 YOLO series evolution (v8 → v11 → YOLO26)
- 2.3 Lightweight detection improvements
- 2.4 Detail enhancement methods (P2, attention, wavelet)

## 3. Method
- 3.1 Overview of YOLO26n architecture
- 3.2 Problem analysis: why D00/D10 are hard
- 3.3 Proposed module(s) — TBD after ablation
- 3.4 Ours-YOLO26n architecture

## 4. Experiments
- 4.1 Dataset: Japan7 protocol
- 4.2 Implementation details (training protocol)
- 4.3 Baseline comparison (4 models)
- 4.4 Ablation study (B0–B5)
- 4.5 Per-class analysis
- 4.6 Visualization and discussion

## 5. Discussion
- Why proposed method works
- Limitations: single domain, single seed, no test set
- Comparison with SOTA (if domestic journal, moderate requirement)

## 6. Conclusion
- Summary of contributions
- Future work: Paper 2 cross-domain

## Writing tips

- Don't claim "SOTA"
- Don't claim "YOLO26 is better than YOLOv8"
- Focus on: "We identified that lightweight backbones lose shallow crack detail; we propose X to address this"
- Evidence-driven: every claim backed by table or figure
- Acknowledge limitations honestly
