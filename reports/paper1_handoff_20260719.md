# Paper 1 Japan7 实验交接（2026-07-19）

## 一句话结论

今天完成了 OverLoCK ProjectFit 系列的六个 30 epoch 受控实验。当前最好结构是 **P4 Gate-1e-3**：

- 训练曲线 best/final：mAP50 `0.56465`，mAP50-95 `0.31801`。
- 独立 batch=32 复验：P `0.57529`，R `0.55551`，mAP50 `0.56493`，mAP50-95 `0.31790`。
- D10：mAP50 `0.30386`，mAP50-95 `0.12102`。

它是后续 OverLoCK 改进的架构基线，但仍未明确超过 canonical B0 的 `0.31900`。明天应在保留 P4 Gate-1e-3 不变的前提下，只增加一个可归因的小模块。

## 今日统一实验协议

- 数据：`configs/japan7_remote.yaml`
- 数据规模：8,387 张训练图、2,119 张验证图、7 类
- 预训练：`model.load("yolo26n.pt")`
- epochs：30
- imgsz：640
- batch：32
- device：0
- workers：8
- seed：42
- deterministic：True
- AMP：True
- optimizer：auto
- lr0 / lrf：0.01 / 0.01
- momentum：0.937
- warmup_epochs：3.0
- weight_decay：0.0005
- mosaic / mixup / copy_paste：1.0 / 0.0 / 0.0
- close_mosaic：10
- iou / max_det：0.7 / 300

所有结构结论都只在上述协议一致时成立。

## 今日 30e 实验总表

表中总体指标采用各自 `results.csv` 的 best/final epoch 30；P4/P3 Gate 的独立复验指标在后文单列。

| 实验 | mAP50 | mAP50-95 | 相对 P4 Gate | 决策 |
|---|---:|---:|---:|---|
| P4 ZeroInit Linear | 0.56418 | 0.31528 | -0.00273 | 否决；分支启动过弱 |
| P4 ZeroInit Cosine | 0.56352 | 0.31370 | -0.00431 | 否决；余弦没有改善最终结果 |
| **P4 Gate-1e-3 Linear** | **0.56465** | **0.31801** | 基线 | 保留为明天唯一 OverLoCK 基础 |
| P3 Gate-1e-3 Linear | 0.56195 | 0.31499 | -0.00302 | 否决 P3 位置 |
| P4 AdaptiveMix | 0.56180 | 0.31590 | -0.00211 | 否决；全局混合仍有类别冲突 |
| P4 ChannelSpatial | 0.56125 | 0.31372 | -0.00429 | 否决；细粒度路由基本退化为等权混合 |

## 关键诊断

### 1. P4 ZeroInit 与余弦调度

- Linear：mAP50-95 `0.31528`，D10 约 `0.116`。
- Cosine：mAP50-95 `0.31370`，D10 约 `0.114`。
- 余弦相对线性下降 `-0.00158`，因此后续恢复线性调度。
- ZeroInit 会让新增分支第一步难以得到有效更新，因此改为 `LayerScale init=1e-3`。

### 2. P4 Gate-1e-3

- 运行名：`paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_30e_seed42`
- 独立复验：mAP50-95 `0.3178994`，D10 `0.1210184`。
- 最终门控：mean_abs `0.00829915`，min `-0.01794434`，max `0.02633667`。
- 同权重将 P4 门控置零后：总体 `0.3183168`，D10 `0.1197013`。
- P4 分支直接贡献：总体 `-0.0004174`，D10 `+0.0013171`。

结论：小非零初始化改善了优化路径，P4 位置和结构值得保留；但现有分支仍存在类别间正负贡献冲突，因此它是“最好 OverLoCK 候选”，不是已经战胜 B0 的最终模型。

### 3. P3 Gate-1e-3

- 独立复验：mAP50-95 `0.31518`，D10 `0.11944`。
- 同权重门控贡献：总体 `-0.0003353`，D10 `-0.0009193`。

结论：移动到更高分辨率 P3 没有解决 D10/细裂缝问题，停止 P3 位置搜索。

### 4. P4 AdaptiveMix

- best/final：mAP50-95 `0.31590`。
- 独立复验约为：P `0.598`，R `0.534`，mAP50 `0.561`，mAP50-95 `0.316`。
- D10 约 `0.123`，但总体低于 P4 Gate `-0.00211`。
- 最终 local/overview 权重约 `0.5022/0.4978`。

结论：一个全局 local/context 权重没有学出有意义的偏好，不能解决不同类别和位置的冲突。

### 5. P4 ChannelSpatial

- best/final：P `0.58161`，R `0.54359`，mAP50 `0.56125`，mAP50-95 `0.31372`。
- 独立复验：mAP50-95 约 `0.314`，D10 约 `0.122`。
- 参数量 `2,501,033`，GFLOPs `5.8`；比普通 P4 Gate 多 49,280 个参数。
- 32 张验证图的路由输出：mean `0.50041`，std `0.00516`，1%–99% 为 `0.48747–0.51453`。
- 外层 LayerScale mean_abs `0.00895`，说明分支在训练；但内部路由绝大多数仍接近 50:50。

结论：ChannelSpatial 没学出明显的通道/空间选择，反而比 P4 Gate 低 `0.00429`。不追加到 50 epoch，不继续 CfC/Liquid 路由。

## 冻结的 P4 Gate-1e-3 参考资产

### 服务器原始文件

- 模型 YAML：`ultralytics/cfg/models/26/yolo26n-OverLoCK-ProjectFit-P4-Gate1e3.yaml`
- 实现源码：`ultralytics/nn/yolo26_2025_backbones/modules.py`
- 注册位置：`ultralytics/nn/tasks.py`
- 数据配置：`configs/japan7_remote.yaml`
- 基础预训练：`yolo26n.pt`
- 运行目录：`runs/paper1/paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_30e_seed42`
- 最佳权重：上述运行目录下的 `weights/best.pt`
- 最终权重：上述运行目录下的 `weights/last.pt`
- 精确参数：上述运行目录下的 `args.yaml`
- 曲线数据：上述运行目录下的 `results.csv`

### 权重哈希

- best.pt：`efea3942b463e8d206805d27ed5234ae397653ee7c77519bf9ae077c32d779d9`
- last.pt：`e8813b5b22349e420f24200d11eaca20e89df9b2bbc2bae32a219ceb2e172736`

这些文件必须保持不变。后续实验使用新的 YAML、类名和运行名，不能覆盖原始 P4 Gate 资产。

## 明天的正确起点

### 原则

1. **保留原 P4 Gate YAML 和 `OverLoCKStage` 行为不变。**
2. 新模块使用新的明确类名和新的模型 YAML；一次只改一个变量。
3. `train.py` 继续只改顶部 `MODEL`、`RUN_NAME` 两行。
4. 公平主实验仍从 `yolo26n.pt` 初始化，不能直接续训 P4 Gate 的 `best.pt`。
5. P4 Gate 的 `best.pt` 用于复验、消融和对照；若以后想做增量微调，应单独命名为 secondary fine-tune，不能与公平 30e 主实验混在一起。

### 为什么不能直接从 Gate best.pt 续训新模块

P4 Gate best.pt 已经接受完整 30 epoch 训练。若新增模块后再从它继续训练，新模型同时获得“30e 旧模型训练 + 新一轮训练”的额外优势，无法判断提升来自新模块还是更长训练。正式对照必须让 P4 Gate 和新候选具有相同初始化来源与相同训练预算。

### 明天开始前核对

```bash
cd /root/YOLO26N-NEW

sha256sum \
  runs/paper1/paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_30e_seed42/weights/best.pt \
  runs/paper1/paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_30e_seed42/weights/last.pt

test -f ultralytics/cfg/models/26/yolo26n-OverLoCK-ProjectFit-P4-Gate1e3.yaml
test -f yolo26n.pt
```

哈希必须与上节完全一致。

### 新候选的最小验证顺序

1. 复制 P4 Gate YAML 为新文件，不修改原文件。
2. 只实现一个新增模块；若 YAML 使用新类，同步完成 `tasks.py` 导入和 `parse_model` 注册。
3. build 检查。
4. `model.load("yolo26n.pt")` 并审计原有同构层迁移覆盖。
5. FP32 forward。
6. AMP forward。
7. single-batch backward，确认新模块梯度非零。
8. 1 epoch smoke。
9. smoke 正常后才运行匹配协议的 30 epoch。

### 明天的评价门槛

- P4 Gate 参考：results.csv `0.31801`；独立复验 `0.31790`。
- canonical B0：约 `0.31900`。
- 单 seed 相对 P4 Gate 小于 `0.002` 的差异不宣布有效。
- 新候选达到约 `0.320`：保留但仍需复验。
- 新候选达到 `>=0.323`：进入多 seed 正式验证。
- 总体不升、只有 D10 小于 `0.002` 的波动：不保留。

## 当前 Git 和运行状态

- 远程分支：`codex/paper1-japan7-remote`
- 分支提交：`c04d1d8`
- 当前分支提交与 `origin/codex/paper1-japan7-remote` 一致。
- 今天的 OverLoCK 源码、YAML、运行结果均未 commit、未 push。
- 当前 `train.py` 指向 ChannelSpatial，明天不能直接重复运行；开始新实验前必须更新 `MODEL` 和 `RUN_NAME`。
- 本交接不授权上传 GitHub。确认新候选有效前继续只保留远程/本地备份。

## 明天不要继续的路线

- P3 Gate 位置搜索
- P4 ZeroInit
- 余弦调度复跑
- AdaptiveMix
- ChannelSpatial
- CfC/Liquid/轴向递归
- 多模块同时堆叠
- MobileMamba
- 蒸馏

## 明天首要任务

从冻结的 P4 Gate-1e-3 YAML 复制一个新候选，只添加一个与 Japan7 小目标/细裂缝特征直接相关的轻量模块。先完成构建、迁移覆盖、梯度和 1e smoke；拿到诊断证据后再决定是否启动 30e。
