# Paper 1 Candidate Recommendations

Generated from 46 YAML files. 46 parsed OK, 0 failed.

## 1. Clean Baselines (recommended for formal comparison)

| YAML | Detect Layers | Scales | Custom Modules | Risk |
| --- | ---: | --- | --- | --- |
| yolo26-2D.yaml | 2 | n/s/m/l/x | none | low |
| yolo26-4D.yaml | 4 | n/s/m/l/x | none | low |
| yolo26.yaml | 3 | n/s/m/l/x | none | low |

## 2. P2 / Small Object Candidates

| YAML | Detect Layers | P2 | Custom Modules | Risk |
| --- | ---: | --- | --- | --- |
| yolo26-4D.yaml | 4 | yes | none |  |

## 3. Attention Module Candidates

| YAML | Attention Module | Other Custom | Risk |
| --- | --- | --- | --- |
| yolo26-BiFPN1.yaml | Concat_bifpn, EMA_attention, GhostConv | 3 heads | medium |
| yolo26-CBAM.yaml | CBAM | 3 heads | medium |
| yolo26-EMA_attention.yaml | EMA_attention | 3 heads | medium |
| yolo26-SEAttention.yaml | SEAttention | 3 heads | medium |
| yolo26-SimA.yaml | SimAM | 3 heads | medium |
| yolo26-StokenAttention.yaml | StokenAttention | 3 heads | medium |

## 4. Not Recommended for Formal Baseline

| YAML | Reason |
| --- | --- |
| yolo26-AConv.yaml | custom modules: AConv |
| yolo26-ADown.yaml | custom modules: ADown |
| yolo26-AKConv.yaml | custom modules: AKConv |
| yolo26-BiForm.yaml | custom modules: BiLevelRoutingAttention |
| yolo26-BiFPN.yaml | custom modules: Concat_bifpn |
| yolo26-BiFPN1.yaml | custom modules: Concat_bifpn, EMA_attention, GhostConv |
| yolo26-BoTNet.yaml | custom modules: BoTNet |
| yolo26-C2f_DySnakeConv.yaml | custom modules: C2f_DySnakeConv |
| yolo26-C2f_Faster.yaml | custom modules: C2f_Faster |
| yolo26-C3_Faster.yaml | custom modules: C3_Faster |
| yolo26-CARAFE.yaml | custom modules: CARAFE |
| yolo26-CBAM.yaml | custom modules: CBAM |
| yolo26-ContextAggregation.yaml | custom modules: ContextAggregation |
| yolo26-DASI.yaml | custom modules: DASI |
| yolo26-DSConv.yaml | custom modules: DSConv |
| yolo26-DualConv.yaml | custom modules: DualConv |
| yolo26-DWConv.yaml | custom modules: DWConv |
| yolo26-EfficientNetv2.yaml | custom modules: FusedMBConv, MBConv, stem |
| yolo26-EMA_attention.yaml | custom modules: EMA_attention |
| yolo26-GhostConv.yaml | custom modules: GhostConv |
| yolo26-Glod.yaml | custom modules: AdvPoolFusion, IFM, InjectionMultiSum_Auto_pool, PyramidPoolAgg, SimFusion_3in, SimFusion_4in, TopBasicLayer |
| yolo26-GSConv.yaml | custom modules: GSConv |
| yolo26-HorBlock,.yaml | custom modules: HorBlock |
| yolo26-Involution.yaml | custom modules: Involution |
| yolo26-MDCR.yaml | custom modules: MDCR |
| yolo26-Moblileone.yaml | custom modules: MobileOneBlock |
| yolo26-MSFN.yaml | custom modules: MSFN |
| yolo26-OREPA.yaml | custom modules: OREPA |
| yolo26-PatchExpand.yaml | custom modules: PatchExpand |
| yolo26-RepConv.yaml | custom modules: RepConv |
| yolo26-RepLKNet.yaml | custom modules: RepLKNet_Stem, RepLKNet_stage1, RepLKNet_stage2, RepLKNet_stage3, RepLKNet_stage4 |
| yolo26-RepViT.yaml | custom modules: RFAConv, RepViTblock |
| yolo26-RFAConv.yaml | custom modules: RFAConv |
| yolo26-SEAttention.yaml | custom modules: SEAttention |
| yolo26-ShuffleNetV2.yaml | custom modules: Conv_maxpool, ShuffleNetV2 |
| yolo26-SimA.yaml | custom modules: SimAM |
| yolo26-slimneck.yaml | custom modules: VoVGSCSP |
| yolo26-SPDConv.yaml | custom modules: space_to_depth |
| yolo26-StokenAttention.yaml | custom modules: StokenAttention |
| yolo26-SwinTransformer.yaml | custom modules: SwinTransformer |
| yolo26-v10D.yaml | custom modules: v10Detect |
| yolo26-vanillanet.yaml | custom modules: vanillanetBlock |
| yolo26LDConv.yaml | custom modules: LDConv |

## 5. Top 3 Recommended YAMLs for Paper 1

1. **yolo26.yaml** — Official clean baseline. P3/P4/P5, scales n/s/m/l/x, zero custom modules.
2. **yolo26-2D.yaml** — Clean variant, standard structure, no custom code needed.
3. **yolo26-4D.yaml** — Another clean baseline option.
