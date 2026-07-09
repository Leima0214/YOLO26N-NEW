# Method Candidates

## Candidate 1: P2 Shallow Detail Branch

**What**: Add a P2-level (4× downsampled) detection head to preserve high-resolution crack details.

**Why**: YOLO26n's smallest feature map is P3 (8× downsampled). Thin cracks (<5px) lose spatial information at this resolution. P2 (4× downsampled) preserves finer detail.

**Implementation**:
- Add P2 output from backbone (after first C3k2 block)
- Add Conv + C3k2 + Detect head at P2 scale
- Lightweight: ~0.1–0.3M additional params

**Expected effect**: Better D00/D10 detection, small mAP gain.

**Risk**: P2 increases FLOPs due to higher-resolution feature processing. Must keep lightweight.

## Candidate 2: Lightweight Channel/Spatial Attention

**What**: Insert lightweight attention module (e.g., ECA, CBAM-lite) into shallow feature layers (P3/P4).

**Why**: Enhance crack-relevant features while suppressing road texture noise.

**Implementation**:
- ECA-Net (1D conv, ~0 params) at C3k2 outputs
- Or light CBAM at P3/P4 fusion points
- Only in shallow layers (deep layers already have C2PSA)

**Expected effect**: Modest mAP gain, very low parameter cost.

**Risk**: Attention on already-compressed features may have limited effect.

## Candidate 3: Wavelet Frequency Enhancement (WFA)

**What**: Decompose shallow features via DWT, enhance high-frequency subbands (crack edges), reconstruct via IDWT.

**Why**: Cracks are high-frequency edge features. Standard convolution loses high-frequency detail through downsampling. Wavelet decomposition explicitly preserves frequency information.

**Implementation**:
- Apply DWT to P3/P4 features
- Lightweight enhancement block on HF subbands
- IDWT reconstruction, fused with original features

**Expected effect**: Good D00/D10 improvement, interpretable (frequency domain).

**Risk**: Implementation complexity, may require careful tuning.

## Candidate 4: Multi-Scale Feature Fusion

**What**: Better fusion of shallow (P2/P3) and deep (P5) features, e.g., BiFPN-lite or improved PANet.

**Why**: Standard FPN may not effectively route fine crack details from shallow to deep layers.

**Expected effect**: General mAP improvement, less targeted to D00/D10.

**Risk**: Increases FLOPs, may not help bottleneck classes specifically.

## Recommended order

1. **First try**: P2 (B1) — most direct fix for D00/D10 resolution issue
2. **Add**: Lightweight attention (B2) — cheap parameter cost
3. **Try separately**: WFA (B3) — if P2 not sufficient
4. **Combine best**: P2 + best auxiliary module (B4/B5)
