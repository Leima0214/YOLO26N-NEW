# LiteRG-YOLO26 完整版 30 epoch signal run 审计

## 结论

本次运行固定定义为 **B5 full LiteRG signal run on c11da7b**。

30 epoch 的默认 NMS-free one-to-one 结果与同配方 B0 基本打平，而不是已经证明有效：mAP50-95 仅从 0.317994 变为 0.318255（+0.000262），D00 基本不变，D10 反而下降。另一方面，同一个 B5 检查点的 one-to-many 分支达到 0.345235，且所有 LiteRG 子模块均存在非零梯度。这说明模型具备正向容量信号，但瓶颈主要在默认 one-to-one 分支的召回和定位，不能把结果写成 LiteRG 已经超过 baseline。

建议只执行一个主方案：保持完整结构和当前损失公式，从官方 YOLO26n 预训练权重重新训练 100 epoch。不得从当前 30e `last.pt` resume。100e 的验收门槛是默认 one-to-one 至少超过同配方 B0 100e 的最佳 mAP50-95=0.35310（epoch 70），并同时改善 D00/D10。

## 1. 运行保护与来源

- 原始提交：`c11da7b322a91596c33664c813c70c17adb99de8`，提交时间 2026-07-22 11:18:18 +0800。
- 远程工作树：`/root/YOLO26N-NEW-LiteRG-9e94269`。
- 原始运行目录：`/root/YOLO26N-NEW-LiteRG-9e94269/runs/paper1_literg/literg_b5_japan7_30e_seed42`。
- 运行时间约为 11:38 至 12:29；审计前工作树 HEAD 为 c11da7b。
- 审计前除仓库历史中已追踪的 `__pycache__/*.pyc` 运行时变化外，没有 `.py`、`.yaml` 等源码差异。模型内嵌 YAML 与 c11da7b 的完整 LiteRG 配置一致。
- 限制：框架写入检查点的 `git.commit` 为 `null`，所以检查点本身不提供可独立验证的 commit 哈希。结合审计前 HEAD、非缓存源码状态、时间和内嵌 YAML，来源证据与 clean c11da7b 一致，但不是由检查点元数据单独完成的密码学证明。
- 原运行目录、`best.pt` 和 `last.pt` 未被覆盖、删除、重命名。新增文件只写入用户要求的 `diagnostics/` 子目录。

检查点哈希：

- `best.pt`: `282bc67675c2a52e839f425d2f11d36b82befab8fcb5362f0d7d0eee9cd294c0`
- `last.pt`: `7d326059023e71a6128ebf73ff03173117fa0a7e424a106b601698698d2cc0c9`

两个文件哈希不同是 strip 后元数据序列化差异；758 个模型 tensor key 逐项比较全部一致，最大绝对差为 0。两者均为 5,603,076 bytes。

环境：Python 3.11.10、PyTorch 2.5.1+cu124、CUDA runtime 12.4、NVIDIA GeForce RTX 4090 24 GB、Ultralytics 8.4.2。

完整训练参数保存在原运行的 `args.yaml`，完整逐 epoch 曲线保存在 `results.csv`。关键公平配置为 epochs=30、batch=32、imgsz=640、seed=42、deterministic=True、AMP、optimizer=auto、lr0=0.01、lrf=0.01、momentum=0.937、warmup=3、weight_decay=0.0005、mosaic=1、mixup=0、copy_paste=0、close_mosaic=10、IoU=0.7、max_det=300。

## 2. 30e 曲线与 B0

严格核对的 B0 和 B5 在上述关键训练、增强和验证参数上全部一致；两个 YAML 最终都解析到 `/yolo26-probe/derived/japan7`。两者使用真实 YOLO26n 预训练权重。当前源码重新验证得到：

| 模型 | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| B0 one-to-one | 0.583411 | 0.553305 | 0.571525 | 0.317994 |
| B5 one-to-one | 0.587704 | 0.554153 | 0.572990 | 0.318255 |
| B5-B0 | +0.004293 | +0.000848 | +0.001464 | +0.000262 |

这属于工程误差范围内的平局。逐类上，B5 的 D00 AP50 由 0.406693 降至 0.398767，D10 由 0.314498 降至 0.291838；D20、D44 有增长，但没有实现最初针对 D00/D10 的核心目标。

最佳 mAP50 和 mAP50-95 都在 epoch 30。最后 10 轮的线性斜率（每 epoch）和 epoch21→30 净变化为：

| 指标 | 斜率 | 净变化 |
|---|---:|---:|
| Precision | +0.003906 | +0.03710 |
| Recall | +0.002721 | +0.01570 |
| mAP50 | +0.003540 | +0.02988 |
| mAP50-95 | +0.002468 | +0.02107 |

同期 train box/cls loss 斜率为 -0.003129/-0.017311，val box/cls loss 为 -0.000558/-0.009037。验证损失没有反弹，P、R 和 AP 同时上升，因此没有明显过拟合迹象，仍属于欠收敛。

结论：曲线支持进行一次 100e 完整验证，但不是简单“续训 70e”。`optimizer=auto` 在 30e 时选择 LiteRG/主干基础 LR=0.000909，而 100e 迭代数跨过阈值后会选择 0.01。100e 是新的、与 B0 100e 匹配的完整训练配方。

- early stopping：固定 100e 对比应继续实质关闭，避免不同模型停在不同轮。
- close_mosaic：保持最后 10 轮关闭；当前没有证据要求改变。
- cosine LR：主方案不改变，保持与现有 B0 100e 相同的 linear 配方。
- 新配置每 10 epoch 保存一次检查点，以补做 D00/D10 逐 epoch/阶段趋势；旧 `results.csv` 没有逐类列，无法从现有 30e 结果恢复逐类历史。

## 3. 检查点、fuse 与接口

未 fuse 的独立 `best.pt` 加载结果：

- 总参数 2,598,317；LiteRG 参数 91,787。
- `lite_rg.*` state key 共 50 个。
- prior、drg3、drg4、rff3、rff4、gamma3/4、eta3/4 全部存在。
- Detect 同时存在 one-to-many 和 one-to-one。
- `yaml.lite_rg.enabled=True`，完整配置保存在检查点。
- 推理产生有限的 `region_logits`，形状为 1×1×160×160。

fuse 后：

- one-to-many 的 `cv2/cv3` 被删除，这是 YOLO26 正常 NMS-free 部署行为。
- LiteRG 模块及 50 个 state key 全部保留。
- 参数量严格满足 `2,376,201 fused B0 + 91,787 LiteRG = 2,467,988`。
- 同一随机输入 fuse 前后 one-to-one 输出最大/平均绝对差为 0.001053/0.0000404，符合半精度检查点与 BN fuse 的数值误差量级。
- fused one-to-one 验证成功；TorchScript 导出成功，输出为 1×300×6，文件约 10.58 MB。

为兼容旧 B0 检查点，`DetectionModel` 对缺失的 `lite_rg` 属性改为 `getattr`。这是加载旧模型对象所必需的向后兼容修复，不改变 LiteRG 或 B0 的计算路径。

## 4. 残差标量与梯度

`best.pt` 与 `last.pt` 标量完全相同：

| 标量 | 数值 | 方向/量级 |
|---|---:|---|
| gamma3 | -0.0079956055 | 负，约 8.0e-3 |
| gamma4 | -0.0045738220 | 负，约 4.6e-3 |
| eta3 | +0.0024185181 | 正，约 2.4e-3 |
| eta4 | -0.0041351318 | 负，约 4.1e-3 |

没有异常大值，也没有一路严格停在 0；但全部仍很小，说明直接残差注入较保守。临时将四个门控清零、不改权重的验证中，mAP50-95 从 0.318255 变为 0.318642，mAP50 从 0.572990 变为 0.572544。即目前门控对 AP50 有极小正贡献，对 mAP50-95 没有净正贡献。

真实 Japan7 batch、只 forward/backward、不执行 optimizer step 的梯度范数：

| 组 | 完整损失梯度范数 |
|---|---:|
| prior | 1.2873 |
| drg3 / drg4 | 0.1808 / 0.0979 |
| rff3 / rff4 | 0.1376 / 0.0764 |
| gamma3 / gamma4 | 0.0330 / 0.0902 |
| eta3 / eta4 | 1.3663 / 0.6303 |
| backbone / neck | 48.4097 / 35.7150 |
| one-to-many / one-to-one head | 21.0205 / 6.9588 |

单独 one-to-one loss 只给 one-to-one head 梯度（34.7927），共享 backbone、neck 和 LiteRG 均为 None；detach 语义正确。单独 region loss 给 prior 1.1494、backbone 1.6582 的梯度，不给 DRG/RFF 梯度；完整检测损失再把梯度传入 DRG/RFF。这证明所有机制确实参与优化，但不证明它们已带来精度净增益。

## 5. 区域先验

2119 张验证图的均值：

| 量 | 均值 |
|---|---:|
| raw BCE | 0.15539 |
| raw Dice loss | 0.79705 |
| raw BCE+Dice | 0.95244 |
| epoch30 effective lambda | 0.10000 |
| weighted region loss | 0.09524 |
| logit mean / std | -5.0722 / 2.7977 |
| probability mean | 0.06978 |
| probability >0.5 | 4.137% |
| GT proxy >0.5 | 3.471% |
| binary IoU | 0.14975 |
| soft Dice | 0.20295 |
| GT box 内/背景响应 | 0.28968 / 0.03967 |

前景响应约为背景 7.3 倍，且输出比例接近 GT proxy 的高响应比例，所以 prior 不是全背景、全前景或完全饱和；它确实学到了损伤相关响应。问题是空间重合度低，Dice/IoU 仍弱。

可视化显示 prior 会响应裂缝和破损纹理，也会响应车道线、道路边缘、接缝和大面积粗糙纹理；不是只学框中心，但存在正常背景污染。D00 示例中 prior/DRG 已在目标附近激活而最终 O2O 仍漏检；D10 示例中大面积 D20 被检测，细长 D10 被漏掉。这进一步把问题指向检测头召回，而不是 prior 完全失效。

## 6. D00/D10 诊断

验证集几何分布：

| 类别 | 实例 | 正样本图 | 每正样本图实例均值 | area 中位数 | area<1% | 宽高比中位数 | 长短边比中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D00 | 811 | 568 | 1.428 | 1.286% | 42.7% | 0.713 | 1.581 |
| D10 | 787 | 453 | 1.737 | 0.793% | 60.0% | 5.032 | 5.032 |

D10 明显更小、更细长，是更困难的类别。D00/D10 的 directional branch 响应没有塌缩：D00 图像上的 P3 横/纵均值为 0.00994/0.01009，P4 为 0.01228/0.01283；D10 为 0.01122/0.01141 和 0.01379/0.01443。纵向仅高约 1.5%–5%，没有足够证据认为 kernel=9 或某个方向分支已经失配。

在 conf=0.25、IoU=0.5 的固定 operating point：

- D00：250/811 TP；486 miss（59.9%）、44 localization error（5.4%）、31 classification confusion（3.8%）；另有 163 个 FP，其中 90 个背景 FP、52 个定位 FP、21 个类别混淆 FP。
- D10：154/787 TP；593 miss（75.3%）、33 localization error（4.2%）、7 classification confusion（0.9%）；另有 147 个 FP，其中 104 个背景 FP、35 个定位 FP、8 个类别混淆 FP。
- D00 最常被混为 D20（22），D10 的类别混淆总量很小。
- O2O 的 D00 AP50/AP75=0.3988/0.1593，D10=0.2918/0.0698。主因是漏检，同时 AP50→AP75 的大幅下降说明定位也是第二瓶颈。

`diagnostics/d00_d10_failures.csv` 包含 1504 条 GT failure/FP 记录；远程 run 的 diagnostics 保存了 20 个按遍历顺序选取、非挑优的失败可视化。Japan7 没有“阴影/车道线/修补区”语义标签，因此脚本不伪造自动语义标签；人工复核确认部分高响应来自车道线、接缝和粗糙修补纹理。

soft target 的 sigma=0.25 会让细长框两端的监督逐渐衰减；它可能对 D10 不利，但当前 prior 已能广泛响应纹理，尚无证据把 sigma 定为首要根因。mosaic/scale 也可能进一步缩小细线，但当前没有单变量实验，不能归因。

## 7. Progressive Region Guidance

源码确认 `criterion.update()` 位于 epoch batch 循环之后，每 epoch 调用一次。训练 epoch 使用更新前的权重：

| epoch | o2m | o2o | lambda_region | logged weighted | 推导 raw | region/检测日志损失 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8000 | 0.2000 | 0.4000 | 0.63380 | 1.58450 | 6.84% |
| 10 | 0.5828 | 0.4172 | 0.2914 | 0.29535 | 1.01363 | 7.37% |
| 20 | 0.3414 | 0.6586 | 0.1707 | 0.16594 | 0.97217 | 4.57% |
| 30 | 0.1000 | 0.9000 | 0.1000 | 0.09502 | 0.95020 | 2.86% |

因此 0.095 并不只来自 lambda 下降：raw loss 也从约 1.585 降到 0.950，但 4 倍权重衰减是日志大幅下降的重要原因。后期 region 对共享特征的相对权重不算过大。

另一个日志细节是验证使用 EMA 模型新建 criterion，o2m 从 0.8 开始，因此 `val/region_loss≈0.382` 按 lambda=0.4 记录，不能与 epoch30 train 的 0.095 直接比较。这不影响训练梯度，只影响日志解释。

理论选项：A 当前 `0.5*max(o2m,0.2)` 从 0.4 降至 0.1；B fixed lambda=0.5；C 端点归一化从 0.4 平滑降至 0.1。现有 raw loss、梯度比例和 prior 分离度没有证明 A 限制效果，主方案保持 A。

## 8. Optimizer

所有 LiteRG 参数都进入 optimizer，未冻结：prior 9,121、DRG3/4 各 27,443、RFF3 9,728、RFF4 18,048、gamma/eta 共 4，总计 91,787。

- 30e `auto`：LiteRG、backbone、neck 基础 LR=0.000909；Detect 分类参数按 MuSGD 规则为 3×=0.002727。最后一行实际基础/高 LR 为 0.00003909/0.00011726。
- 100e `auto`：迭代数超过阈值，基础 LR=0.01，Detect 分类参数 3×=0.03；linear 配方预计 epoch100 日志 LR 为基础约 0.000199、高组约 0.000597。
- LiteRG 随机初始化模块当前与预训练主干使用同一基础 LR；gamma/eta 位于普通 weight-decay 组，wd=0.0005。GN/bias 组 wd=0，二维卷积权重进入 Muon 组。

30e 的小 LR 可能造成新模块学习偏慢，但 100e 主配方已经把基础 LR 提高约 11 倍，因此不应再未经小实验叠加 2×/3× LiteRG LR。

## 9. EMA、保存和 resume

- 训练时 `ModelEMA` 深复制完整模型，验证结果以及最终保存模型均含 50 个 `lite_rg.*` key。
- 独立加载 best/last 后 `lite_rg` 非 None，内嵌 YAML 完整。
- 训练结束的 `final_eval()` 对 last 和 best 都执行 `strip_optimizer()`：两者 `epoch=-1`、optimizer/scaler/EMA/updates/criterion 均为空。
- 因此当前 best/last 都不能严格 resume；框架会直接判定训练已完成。
- 即使保留未 strip 的中途检查点，当前 trainer 保存 EMA updates，但没有保存 `E2ELoss.updates/o2m/o2o`，Progressive 状态不能保证连续恢复。

结论：100e 必须从真实 YOLO26n 预训练权重重新开始。专用入口固定 `resume=False` 并拒绝 Git LFS pointer。

## 10. 同检查点双头结果与候选

| 分支 | P | R | mAP50 | mAP75 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| one-to-one, NMS-free | 0.5877 | 0.5542 | 0.5730 | 0.3091 | 0.3183 |
| one-to-many + NMS | 0.6423 | 0.5796 | 0.6254 | 0.3334 | 0.3452 |

O2M 同权重优势为 +0.0270 mAP50-95、+0.0524 AP50；D10 AP50 优势达 +0.0940。这是当前最强的性能证据：共享特征有能力，但默认 O2O 分支没有充分兑现。

完整 B5 候选不超过三个：

1. **唯一主方案：当前完整结构从头 100e。** 不改模块、sigma、kernel、region schedule、增强或 LR multiplier；仅增加训练时长和每 10e 保存。目标是解决欠收敛和 O2O/O2M 差距。参数 2,467,988 fused、6.0 GFLOPs 不变。风险是 100e 仍只追平 B0；验收看默认 O2O，不以 O2M 代替。
2. **备选 A：sigma_scale 0.25→0.30，先做同配方 30e。** 只针对 D10 细长框端部监督衰减和 prior IoU 低；参数/FLOPs 不变。风险是进一步覆盖车道线和正常背景。只有主方案 D10 仍弱时再做，获胜后最终 100e 必须从头跑。
3. **备选 B：LiteRG 专属 2× LR，先做同配方 30e。** 针对随机初始化模块与预训练主干同 LR、门控仍小；参数/FLOPs 不变。风险是 100e auto 的基础 LR 已高，叠加后可能不稳定。需先实现参数组隔离并小跑，不能直接用于主 100e；不建议 3×。

暂不改变 directional kernel：横纵响应均衡，没有 kernel=9 失配证据。暂不改 gamma/eta 初始化：当前所有路径已有梯度，且零门控反事实没有损害 mAP50-95，放大初值存在更高风险。暂不改 region_gain/floor：后期 region 占比仅约 2.9%。

## 11. 交付物和命令

- `scripts/audit_literg_full_checkpoint.py`：检查点、fuse、梯度、detach 和 optimizer 分组审计。
- `scripts/analyze_literg_region_prior.py`：全验证集 prior、方向响应、几何、D00/D10 failure/FP 与可视化。
- `scripts/eval_literg_full.py`：同一检查点 O2O/O2M 评估和隔离目录中的 fused export 检查。
- `scripts/train_literg_full.py`：不依赖 B0-B7 stage 的完整 B5 专用入口。
- `configs/literg_full_japan7_100e.yaml`：当前结构的 100e 固定配方。
- `diagnostics/literg_full_30e_signal/`：机器可读摘要和比较 CSV。
- 远程原 run 的 `diagnostics/`：完整 gradient JSON、2119 图 prior CSV、1504 failure CSV、双头 JSON 和 27 张可视化。

推荐命令（仅交付，不由审计自动执行）：

```bash
cd /root/YOLO26N-NEW-LiteRG-9e94269
git fetch origin codex/literg-yolo26-migration
git merge --ff-only origin/codex/literg-yolo26-migration
PYTHONPATH=. /opt/conda/bin/python scripts/train_literg_full.py \
  --config configs/literg_full_japan7_100e.yaml \
  --data configs/japan7_remote.yaml \
  --weights /root/YOLO26N-NEW/yolo26n.pt \
  --device 0 \
  --name literg_full_japan7_100e_seed42
```

运行前应确认 `nvidia-smi` 无其他训练进程、权重不是 LFS pointer，并记录实际 branch HEAD。运行完成后以默认 O2O 对比已有 B0 100e best=0.35310，同时报告同检查点 O2M 作为诊断上限。
