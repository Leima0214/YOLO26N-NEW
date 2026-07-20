# OverLoCK ProjectFit-P4 审计与实验记录（2026-07-19）

## 结论

当前 `OverLoCK` 是面向本项目的自定义改版，不以复现官方 XT/T/S/B 结构为目标。首个“整套自定义 backbone”方案因大量随机初始化，30 epoch 内收敛明显落后，已在第 24 epoch 主动止损。当前采用最小决定性改动：保留 YOLO26n 的预训练 backbone、neck 和 head，只在 P4/16 的原 C3k2 后追加一个零初始化门控的 OverLoCK 增强分支。

## 外部建议取舍

| 建议 | 决定 | 依据 |
|---|---|---|
| 检查实现文件是否缺失、是否受 Git 管理 | 已核查，问题不成立 | `ultralytics/nn/yolo26_2025_backbones` 是 Python 包，实现在 `modules.py`，远端分支可见 |
| 必须改成官方 OverLoCK XT/T/S/B | 不采用 | 用户明确要求优化当前项目特定改版，而非复刻官方版本 |
| 给自定义 tiny 直接加载官方 XT 权重 | 不采用 | 两者拓扑和参数名不匹配，不能形成可靠的 name-and-shape 迁移 |
| 对 YOLO26n 层号重映射，保留可兼容预训练区域 | 已采用并审计 | 旧全 backbone 方案可迁移共享 neck/head；新 ProjectFit 方案直接保留原层号 |
| 明确允许缺失区和异常缺失区 | 已采用 | 自动审计要求 `unexpected_missing=0` |

## 旧全 backbone 方案：否决

- 模型：`yolo26-OverLoCK-Backbone1.yaml`
- 规模：约 3.41M 参数、12.8 GFLOPs
- 初始迁移：YOLO26n 到 80 类 YAML 为 `522/732`
- Japan7 重建后迁移：`630/732`
- 允许缺失：102 个检测分类分支参数
- 异常缺失：0
- 停止点：epoch 24
- epoch 24：mAP50 `0.32921`，mAP50-95 `0.17546`
- best.pt / last.pt SHA256：`addc3e037940cc137c9be37f6f32dde045eca7a891a2a3f136091153dee3cf2f`
- 判定：大量随机初始化的 backbone 使 30 epoch 公平预算被用于重新学习基础表征，不适合当前 Paper 1 小改版路线。

## 当前 ProjectFit-P4 方案

- YAML：`ultralytics/cfg/models/26/yolo26n-OverLoCK-ProjectFit-P4.yaml`
- 唯一结构改动：YOLO26n backbone 第 6 层 P4/16 `C3k2` 替换为兼容其全部原参数的 `OverLoCKStage`
- 新增分支：75,680 参数，约为基线参数量的 2.9%
- Japan7 训练模型：2,582,210 参数、6.3 GFLOPs
- 设计理由：P4 具备足够上下文分辨率，同时能通过 PAN/FPN 回流到小目标 P3；单点插入便于归因
- 安全启动：OverLoCK 残差门控 `init_scale=0.0`，加载权重后的初始输出与 YOLO26n 完全相等
- 可学习性：第一次反向传播门控梯度非零；更新门控后第二次反向传播增强分支梯度非零

## 权重迁移审计

审计命令：

```bash
/opt/conda/bin/python scripts/audit_overlock_transfer.py
```

当前 ProjectFit 结果：

- YOLO26n -> 80 类候选：`708/735`
- 新增 P4 项：27
- 80 类候选 -> Japan7：`633/735`
- 允许的 Japan7 分类分支缺失：102
- 异常缺失：0

含义：除了新增 OverLoCK 分支和类别数变化引起的分类头重建，其余 name-and-shape 兼容权重全部继承。

## 当前训练

- 远端 PID：`13365`
- 运行名：`paper1_overlock_projectfit_p4_zeroinit_pretrained_auto_japan7_30e_seed42`
- 结果目录：`/root/YOLO26N-NEW/runs/paper1/paper1_overlock_projectfit_p4_zeroinit_pretrained_auto_japan7_30e_seed42`
- 数据：Japan7，train 8387 / val 2119，损坏样本 0
- 协议：30 epochs、imgsz 640、batch 32、seed 42、deterministic、AMP、optimizer auto
- 实际优化器：MuSGD，初始 lr `0.000909`，momentum `0.9`
- 30e 结果：mAP50 `0.56418`，mAP50-95 `0.31528`，best epoch 30
- D10：mAP50 `0.28585`，mAP50-95 `0.11605`
- best.pt SHA256：`15309043a2eefa27fe41d22406da7847ddd34132fb7b930a9584e0ef84b9df6a`
- last.pt SHA256：`bd094e38a0b04a9182e64fb560224909ed7a0f1d25823cefc384cc3a21d42e32`
- 状态：首轮训练已完成；整体比 B0 低 `0.00280`

### 门控消融

使用同一个 best.pt、batch 32 和相同验证参数，临时将 `model.6.enhance.scale.gamma` 置零：

- 完整 P4：mAP50-95 `0.3153211`，D10 `0.1160544`
- 门控置零：mAP50-95 `0.3152620`，D10 `0.1159792`
- OverLoCK 直接贡献：整体约 `+0.0000591`，D10 约 `+0.0000752`

结论：P4 分支并未明显拖累，但贡献接近零；最终门控平均绝对值仅 `0.00737`。由于曲线在 epoch 30 仍上升且只差 B0 `0.00280`，按用户要求保留为一次受控调参候选，不立即永久否决。

### 调参建议取舍

| 建议 | 决定 | 依据 |
|---|---|---|
| 3 epoch 预热 | 已经启用 | 当前 `warmup_epochs=3.0` |
| 余弦退火 | 采用为唯一训练变量 | 首轮为线性衰减；下一轮只增加 `cos_lr=True` |
| 梯度裁剪 | 已经启用 | trainer 每步执行全局范数裁剪，`max_norm=10.0` |
| 贝叶斯/大规模超参搜索 | 暂不采用 | 单次完整试验约 49 分钟；先做一个有证据的单变量试验，避免验证集过拟合 |
| 叠加 Mamba SSM | 不采用 | 会改变架构归因，且“参数减半、精度提高 3%-5%”不能视为本数据集保证 |
| 代价感知动态损失 | 保留为后续独立课题 | 可能改善 D10，但必须与 B0 使用同一损失重新对照 |
| 推理提前退场 | 不用于当前精度修复 | 它主要优化平均延迟，通常需要额外预测头和退出阈值，不直接提升 mAP |

下一轮运行名：

`paper1_overlock_projectfit_p4_zeroinit_pretrained_auto_cos_japan7_30e_seed42`

只把 `cos_lr` 从 `False` 改为 `True`，其余结构、数据、预训练、seed、增广和 30e 预算不变。若结果有提升，再补跑 B0 cosine 以分离“调度器收益”和“OverLoCK 收益”。

## P4 余弦退火单变量结果

- 运行名：`paper1_overlock_projectfit_p4_zeroinit_pretrained_auto_cos_japan7_30e_seed42`
- best / final epoch：30
- mAP50：`0.56352`
- mAP50-95：`0.31370`
- D00 mAP50-95：`0.187`
- D10 mAP50-95：`0.114`
- best.pt SHA256：`7479bc4b562b649bc8c855f118ae79a4daf2391c8e325040bf8d08dc06d52b6f`
- last.pt SHA256：`08f187e4fab2801d8720aa1c58613035030a0e329c3734b78253fcaa2dda24bd`
- 最终门控：mean_abs `0.00906117`，min `-0.02989197`，max `0.02534485`

与线性调度比较：

| 运行 | mAP50 | mAP50-95 | D10 mAP50-95 |
|---|---:|---:|---:|
| P4-linear | 0.56418 | 0.31528 | 0.116 |
| P4-cos | 0.56352 | 0.31370 | 0.114 |
| cosine - linear | -0.00066 | -0.00158 | -0.002 |

判定：余弦在 epoch 10–26 多数阶段略高，但最终低于线性版；它主要改变收敛过程，没有提高最终 best，因此不继续做 B0-cos，也不进入贝叶斯学习率搜索。后续 P4 实验恢复线性调度。

## 下一步

P4 暂不永久否决，但只再做一个直接针对根因的受控试验：

1. 保持 P4 位置、线性调度和全部训练协议。
2. 将 LayerScale 初值从完全关闭的 `0.0` 改为很小的 `0.001`。
3. 先验证预训练输出扰动有界、迁移仍为 `708/735 -> 633/735`、增强分支从第一步即可获得梯度。
4. 验证通过才跑 30e。

原因：两次零初始化训练的门控都很小，且线性 best 的门控置零消融只有约 `+0.000059` 的直接贡献。继续调全局学习率没有击中根因；小非零 LayerScale 是让分支从第一批开始学习的最小改动。

停止条件：

- 若 Gate-1e-3 达到 mAP50-95 `>=0.319` 且 D10 改善，保留 P4 调优线。
- 若达到 `>=0.323`，进入正式复验。
- 若仍 `<0.319` 或门控消融仍近零，停止 P4 位置调参，转到已经准备好的 P3/8 单点布局。

Gate-1e-3 启动前验证：

- YOLO26n -> 80 类候选：`708/735`
- 80 类候选 -> Japan7：`633/735`
- 异常缺失：0
- 初始输出相对差：`0.00145523`
- 第一批门控梯度绝对和：`447.43444824`
- 第一批增强分支梯度绝对和：`253.96241489`
- 判定：预训练输出扰动约 0.15%，增强分支从第一批即可学习，允许进入 30e。

## 验收门槛

本次保持与 B0 相同的 30 epoch 预算，不在看到结果后临时追加 epoch。

- 严格匹配 B0 control：epoch 30，mAP50 `0.57152`，mAP50-95 `0.31808`
- mAP50-95 `< 0.319`：否决
- `0.319 <=` mAP50-95 `< 0.323`：中性/仅允许一次受控微调
- mAP50-95 `>= 0.323`：进入后续验证

训练结束后还需记录：

1. 最佳 epoch、mAP50、mAP50-95。
2. D10 类的独立 mAP50-95。
3. `best.pt` 与 `last.pt` 的 SHA256。
4. 与严格匹配 B0 的参数量、GFLOPs 和精度差。

## P4 Gate-1e-3 30e 最终结果

- 运行名：`paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_30e_seed42`
- best / final epoch：30
- 独立 batch-32 复验：P `0.57529`，R `0.55551`，mAP50 `0.56493`，mAP50-95 `0.31790`
- D10：mAP50 `0.30386`，mAP50-95 `0.12102`
- best.pt SHA256：`efea3942b463e8d206805d27ed5234ae397653ee7c77519bf9ae077c32d779d9`
- last.pt SHA256：`e8813b5b22349e420f24200d11eaca20e89df9b2bbc2bae32a219ceb2e172736`
- 最终门控：mean_abs `0.00829915`，min `-0.01794434`，max `0.02633667`

与零初始化线性版相比，Gate-1e-3 的 mAP50-95 提高 `+0.00258`，D10 提高 `+0.00496`。但它仍比当前 canonical B0 的 `0.319` 低 `0.00110`，D10 仍比 B0 的 `0.130` 低 `0.00898`。

### 同权重门控消融

- 开启已训练 P4 分支：整体 `0.3178994`，D10 `0.1210184`
- 将该分支门控置零：整体 `0.3183168`，D10 `0.1197013`
- P4 分支直接贡献：整体 `-0.0004174`，D10 `+0.0013171`

结论：`0.001` 只是初值，不是固定强度；门控训练后平均绝对值已增长到约初值的 8.3 倍，因此不存在“数值太小导致学不动”。小非零初值确实改善了优化路径，但 P4 最终特征对不同类别有冲突：略微帮助 D10，却轻微拖累整体指标。按预设停止条件结束 P4 位置调参；下一次只移动同一个 Gate-1e-3 模块到 P3/8，保持其余协议不变。

## P3 Gate-1e-3 30e 最终结果

- 运行名：`paper1_overlock_projectfit_p3_gate1e3_pretrained_auto_linear_japan7_30e_seed42`
- best / final epoch：30
- 独立 batch-32 复验：P `0.56964`，R `0.55364`，mAP50 `0.56132`，mAP50-95 `0.31518`
- D10：mAP50 `0.30284`，mAP50-95 `0.11944`
- best.pt SHA256：`107bb2f8765a3baf1ef80ebdd45ccbc2c60a8baa2e1687fe74a5431b428dcc13`
- last.pt SHA256：`e190aa1867a5ed7b4e5766e11232e1beee32a8ab7caff18c2651e6f9f22c3857`
- 最终门控：mean_abs `0.00684186`，min `-0.02877808`，max `0.02096558`
- 训练结构：2.582M 参数、6.6 GFLOPs；融合验证结构为 2.452M 参数、6.0 GFLOPs

与对照相比：

- 相对 P4 Gate-1e-3：整体 `-0.00272`，D10 `-0.00158`
- 相对 canonical B0：整体 `-0.00382`，D10 `-0.01056`

### 同权重P3门控消融

- 开启已训练P3分支：整体 `0.3151833`，D10 `0.1194403`
- 将layer 4门控置零：整体 `0.3155186`，D10 `0.1203596`
- P3分支直接贡献：整体 `-0.0003353`，D10 `-0.0009193`

结论：P3门控已经充分学习，但开启分支后整体和D10都下降；更高分辨率位置没有解决细裂缝问题。该位置不进入50轮训练，也不再继续门控初值搜索。
