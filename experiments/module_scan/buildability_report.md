# YOLO26 Module Buildability Report

Generated: 2026-07-09T15:54:36
Candidates: 13
Build OK: 10
Build failed: 3

| yaml_path | exists | build_ok | error_type | error_message_short | params | flops | recommended_next_step |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| ultralytics/cfg/models/26/yolo26-CPUBoneNano-P2Lite.yaml | True | True |  |  | 3930380 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-CARAFE.yaml | True | False | ModuleNotFoundError | No module named 'mmcv' |  |  | skip until build error is fixed |
| ultralytics/cfg/models/26/yolo26-BiFPN.yaml | True | True |  |  | 2572292 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-BiFPN1.yaml | True | True |  |  | 2555508 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-EMA_attention.yaml | True | True |  |  | 2572952 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-SEAttention.yaml | True | True |  |  | 2572792 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-CBAM.yaml | True | True |  |  | 2580843 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-LaplacianConv.yaml | True | True |  |  | 2572281 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-FDConv.yaml | True | False | ModuleNotFoundError | No module named 'mmcv' |  |  | skip until build error is fixed |
| ultralytics/cfg/models/26/yolo26-SPDConv.yaml | True | True |  |  | 2795480 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-FFAFusion-Neck.yaml | True | True |  |  | 2617656 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-HVIEnhanceStem.yaml | True | True |  |  | 2575008 |  | pilot: run 3 epoch single-module test |
| ultralytics/cfg/models/26/yolo26-ContextAggregation.yaml | True | False | ModuleNotFoundError | No module named 'mmcv' |  |  | skip until build error is fixed |
