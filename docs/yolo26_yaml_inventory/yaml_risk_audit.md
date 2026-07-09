# YAML Risk Audit

| YAML | Parse OK | Custom Modules | Scales | Num Detect | Can Parse | Fair Compare | Notes |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| yolo26-2D.yaml | ✅ | none | yes | 2 | ✅ | ✅ |  |
| yolo26-4D.yaml | ✅ | none | yes | 4 | ✅ | ✅ |  |
| yolo26-AConv.yaml | ✅ | AConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-ADown.yaml | ✅ | ADown | yes | 3 | ✅ | ⚠️ |  |
| yolo26-AKConv.yaml | ✅ | AKConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-BiForm.yaml | ✅ | BiLevelRoutingAttention | yes | 3 | ✅ | ⚠️ |  |
| yolo26-BiFPN.yaml | ✅ | Concat_bifpn | yes | 3 | ✅ | ⚠️ |  |
| yolo26-BiFPN1.yaml | ✅ | Concat_bifpn, EMA_attention, GhostConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-BoTNet.yaml | ✅ | BoTNet | yes | 3 | ✅ | ⚠️ |  |
| yolo26-C2f_DySnakeConv.yaml | ✅ | C2f_DySnakeConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-C2f_Faster.yaml | ✅ | C2f_Faster | yes | 3 | ✅ | ⚠️ |  |
| yolo26-C3_Faster.yaml | ✅ | C3_Faster | yes | 3 | ✅ | ⚠️ |  |
| yolo26-CARAFE.yaml | ✅ | CARAFE | yes | 3 | ✅ | ⚠️ |  |
| yolo26-CBAM.yaml | ✅ | CBAM | yes | 3 | ✅ | ⚠️ |  |
| yolo26-ContextAggregation.yaml | ✅ | ContextAggregation | yes | 3 | ✅ | ⚠️ |  |
| yolo26-DASI.yaml | ✅ | DASI | yes | 3 | ✅ | ⚠️ |  |
| yolo26-DSConv.yaml | ✅ | DSConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-DualConv.yaml | ✅ | DualConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-DWConv.yaml | ✅ | DWConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-EfficientNetv2.yaml | ✅ | FusedMBConv, MBConv, stem | yes | 3 | ✅ | ⚠️ |  |
| yolo26-EMA_attention.yaml | ✅ | EMA_attention | yes | 3 | ✅ | ⚠️ |  |
| yolo26-GhostConv.yaml | ✅ | GhostConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-Glod.yaml | ✅ | AdvPoolFusion, IFM, InjectionMultiSum_Auto_pool, PyramidPoolAgg, SimFusion_3in, SimFusion_4in, TopBasicLayer | yes | 3 | ✅ | ⚠️ |  |
| yolo26-GSConv.yaml | ✅ | GSConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-HorBlock,.yaml | ✅ | HorBlock | yes | 3 | ✅ | ⚠️ |  |
| yolo26-Involution.yaml | ✅ | Involution | yes | 3 | ✅ | ⚠️ |  |
| yolo26-MDCR.yaml | ✅ | MDCR | yes | 3 | ✅ | ⚠️ |  |
| yolo26-Moblileone.yaml | ✅ | MobileOneBlock | yes | 3 | ✅ | ⚠️ |  |
| yolo26-MSFN.yaml | ✅ | MSFN | yes | 3 | ✅ | ⚠️ |  |
| yolo26-OREPA.yaml | ✅ | OREPA | yes | 3 | ✅ | ⚠️ |  |
| yolo26-PatchExpand.yaml | ✅ | PatchExpand | yes | 3 | ✅ | ⚠️ |  |
| yolo26-RepConv.yaml | ✅ | RepConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-RepLKNet.yaml | ✅ | RepLKNet_Stem, RepLKNet_stage1, RepLKNet_stage2, RepLKNet_stage3, RepLKNet_stage4 | yes | 3 | ✅ | ⚠️ |  |
| yolo26-RepViT.yaml | ✅ | RFAConv, RepViTblock | yes | 3 | ✅ | ⚠️ |  |
| yolo26-RFAConv.yaml | ✅ | RFAConv | yes | 3 | ✅ | ⚠️ |  |
| yolo26-SEAttention.yaml | ✅ | SEAttention | yes | 3 | ✅ | ⚠️ |  |
| yolo26-ShuffleNetV2.yaml | ✅ | Conv_maxpool, ShuffleNetV2 | yes | 3 | ✅ | ⚠️ |  |
| yolo26-SimA.yaml | ✅ | SimAM | yes | 3 | ✅ | ⚠️ |  |
| yolo26-slimneck.yaml | ✅ | VoVGSCSP | yes | 3 | ✅ | ⚠️ |  |
| yolo26-SPDConv.yaml | ✅ | space_to_depth | yes | 3 | ✅ | ⚠️ |  |
| yolo26-StokenAttention.yaml | ✅ | StokenAttention | yes | 3 | ✅ | ⚠️ |  |
| yolo26-SwinTransformer.yaml | ✅ | SwinTransformer | yes | 3 | ✅ | ⚠️ |  |
| yolo26-v10D.yaml | ✅ | v10Detect | yes | 0 | ✅ | ⚠️ |  |
| yolo26-vanillanet.yaml | ✅ | vanillanetBlock | yes | 3 | ✅ | ⚠️ |  |
| yolo26.yaml | ✅ | none | yes | 3 | ✅ | ✅ |  |
| yolo26LDConv.yaml | ✅ | LDConv | yes | 3 | ✅ | ⚠️ |  |

## Key Risks
- **43** YAMLs require custom Python modules
- **0** YAMLs have no scales field (cannot use n/s/m/l/x)
- **0** YAMLs failed to parse