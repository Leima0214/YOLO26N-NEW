# Paper 1 Tier A Three-Module Composite Design

These are the 12 prioritized three-module candidates. Ordering follows the project rule: complementary, non-conflicting, independently interpretable combinations come first; higher-risk or efficiency-oriented combinations come later.

| priority | combination | concrete placement | intended role |
| ---: | --- | --- | --- |
| 1 | official P2 + SPDConv + EMA-P3-factor8 | P2 Detect head; atomic SPD P2-to-P3 downsample; identity-initialized EMA before P3 Detect | shallow detail + lossless downsampling + attention |
| 2 | official P2 + LaplacianConv + EMA-P3-factor8 | P2 Detect head; Laplacian P2-to-P3 downsample; EMA before P3 Detect | shallow detail + edge response + attention |
| 3 | official P2 + EMA-P3-factor8 + BiFPN label | P2 Detect head; EMA before P3 Detect; neck merges use identity-initialized positive weighted concat | detail + attention + weighted fusion |
| 4 | SPDConv + EMA-P3-factor8 + FFAFusion-Neck | SPD P2-to-P3 downsample; EMA before P3 Detect; one identity-initialized FFA node at P3 top-down fusion | detail + attention + adaptive fusion |
| 5 | LaplacianConv + EMA-P3-factor8 + BiFPN label | Laplacian P2-to-P3 downsample; EMA before P3 Detect; positive weighted-concat neck | edge + attention + weighted fusion |
| 6 | official P2 + SEAttention-P3 + FFAFusion-Neck | P2 Detect head; SE before P3 Detect; one FFA node at P3 top-down fusion | detail + attention control + adaptive fusion |
| 7 | official P2 + CBAM-P3 + BiFPN label | P2 Detect head; CBAM before P3 Detect; positive weighted-concat neck | detail + traditional attention control + fusion |
| 8 | official P2 + EMA-P3-factor8 + slimneck | P2 Detect head; EMA before P3 Detect; VoVGSCSP at final P5 | detail + attention + efficiency |
| 9 | SPDConv + EMA-P3-factor8 + slimneck | SPD P2-to-P3 downsample; EMA before P3 Detect; VoVGSCSP at final P5 | detail + attention + efficiency |
| 10 | FDConv + EMA-P3-factor8 + FFAFusion-Neck | Conv-compatible FDConv P2-to-P3 downsample; EMA before P3 Detect; one FFA node at P3 top-down fusion | frequency detail + attention + adaptive fusion |
| 11 | official P2 + LaplacianConv + CARAFE | P2 Detect head; Laplacian P2-to-P3 downsample; channel-preserving identity-initialized CARAFE P4-to-P3 upsample | detail + edge + content-aware upsampling |
| 12 | FDConv + GSConv + CARAFE | GSConv shallow downsample; FDConv P2-to-P3 downsample; CARAFE P4-to-P3 | frequency detail + efficiency + upsampling |

## Modules Involved

The 12 composites use 12 distinct named mechanisms:

1. official YOLO26 P2 detection head
2. SPDConv (atomic padded `pixel_unshuffle` downsampling and projection)
3. LaplacianConv
4. FDConv
5. EMA_attention with factor 8 at P3
6. SEAttention at P3
7. CBAM at P3
8. BiFPN-labelled positive weighted concatenation (`Concat_bifpn`, not classic additive BiFPN)
9. FFAFusion-Neck (`FFAFusionConcat`, one P3 fusion node)
10. CARAFE
11. slimneck (`VoVGSCSP`)
12. GSConv

## Scientific Constraint

EMA-P3-factor8 was weaker than the matched 30 epoch control as a single module (`0.292` vs `0.318` mAP50-95). It remains in this queue only to test interaction effects. A final three-module model must still exceed the YOLO26n baseline; speed or Recall explanations may justify an individual ablation row, but cannot excuse a weaker final composite.

Generated YAMLs are reproducible from `scripts/generate_paper1_tiera_composites.py`. Each YAML embeds a semantic pretrained mapping rather than relying on numeric layer shifts. The corresponding build evidence is in `experiments/module_scan/paper1_tiera_buildability_report.md`; the stronger runtime and transfer audit is in `experiments/module_scan/paper1_tiera_adversarial_audit.md`.

## Corrected Initialization Contract

Every newly inserted attention or fusion path must preserve the inherited baseline function at initialization. EMA, SE, CBAM, LaplacianConv, FDConv, FFAFusion, CARAFE, and weighted concat therefore start with zero residual gain or baseline-equivalent weights. This does not guarantee a gain, but it removes the avoidable optimization shock present in the first A01-A12 implementation.

Pretrained coverage is measured by parameter count and by model region. For P2 models, YOLO26n Detect branches P3/P4/P5 are mapped to target branches 1/2/3; only the new P2 branch is initialized from scratch. A08/A09 remain higher-risk because their slimneck replacements intentionally inherit only about 58-62% of neck parameters.
