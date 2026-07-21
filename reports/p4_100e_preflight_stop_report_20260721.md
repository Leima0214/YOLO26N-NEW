# Paper1 P4 Single 100e 预检与强制停止报告

日期：2026-07-21
实验分支基线：`codex/p4-single-gate1e3`
基线提交：`09a63f65d637cf3b6fe739ad4c986cf866f841c7`

## 最终决定

本轮没有启动 B0 或 P4 Single 的正式 100 epoch 训练。原因不是环境、代码或 GPU 故障，而是数据审计确认当前 Japan7 train/val 存在跨集合的近重复场景。用户任务明确规定：发现明显 train/val 泄漏时停止正式训练并报告。因此继续训练会系统性抬高验证指标，并使 B0 与 P4 的微小差异无法可靠归因。

当前唯一合理的下一候选是 `Japan7 Scene/Sequence Group Split v1`。只有新切分通过零重叠审计后，才运行已准备好的 B0/P4 配对 100e。

## 已完成项目

1. 固化 B0 与 P4 Single 的 30e、seed 42/0/3447 独立验证结果。
2. 审计远端 CUDA、PyTorch、Ultralytics、GPU、磁盘、分支、权重和配置哈希。
3. 运行已有标签/图像完整性检查。
4. 新增跨 split 的文件名、SHA256、pHash、序列邻近度和人工图廊审计。
5. 搜索现存 B0 100e；结论为 `RERUN_REQUIRED`，没有公平可复用结果。
6. 创建共享、参数锁定的 B0/P4 100e 入口与数据审计硬门槛。
7. 验证硬门槛：当前审计状态下，入口在模型构建前退出，未产生训练 run。
8. 完成道路损伤检测、定位不确定性、微小目标度量、任务对齐、误差分解和梯度冲突的前沿资料审查。
9. 输出唯一下一候选及最小、可证伪实验设计。

## 30e 三 seed 不可变结论

| 模型 | seed42 | seed0 | seed3447 | 三 seed 均值 |
|---|---:|---:|---:|---:|
| B0 | 0.31799 | 0.31932 | 0.31709 | 0.31813 |
| P4 Single | 0.31790 | 0.32059 | 0.31769 | 0.31872 |
| 配对差值 | -0.00009 | +0.00127 | +0.00059 | +0.00059 |

P4 Single 是弱正、但没有达到可宣称有效的幅度。三 seed 类别均值显示 D40、D43、D50 上升，而 D10、D20、D44 下降；其中 D10/D20 在三个配对 seed 中均下降。详细曲线、类别均值、标准差和开销见 `reports/p4_single_30e_three_seed_summary.md`。

P4 Single 相比 B0 增加约 3.18% 参数和 9.62% FLOPs。Gate 和分支可以收到梯度，不支持“完全梯度饥饿”的解释；但总体净收益仍小于当前数据切分不确定性。

## 数据审计结论

基础完整性检查通过：

- train 8387 张，val 2119 张；
- train 19752 个框，val 5000 个框；
- 无缺失标签、坏行、越界框、非法类别或损坏软链接；
- 空标签图像为有效负样本：train 638，val 156。

跨 split 审计失败：

- 文件名交集：0；
- 字节级完全重复：0；
- pHash 距离不大于 2 的最近配对：8；
- 人工复核确认这些配对为相同道路场景的近邻视角，具有相同建筑、车辆、道路标线或路面结构；
- 2119 张 val 图像中有 1869 张的最近 train 序列编号距离为 1，进一步表明逐帧/邻帧分割风险。

最终状态：`FAIL_CONFIRMED_NEAR_DUPLICATE_LEAKAGE`。证据见：

- `reports/dataset_integrity_and_leakage_audit.md`
- `reports/dataset_integrity_and_leakage_audit.json`
- `reports/dataset_leakage_gallery.jpg`

## 未执行项目及原因

以下项目不是遗漏，而是由泄漏停止条件主动取消：

- B0 seed42 100e 正式训练；
- P4 Single seed42 100e 正式训练；
- 100e 曲线、平台期、close-mosaic 前后与提前停止分析；
- 基于 100e best.pt 的 TIDE/错误样本配对；
- 基于 100e checkpoints 的梯度冲突分析；
- 任何新的结构模块、超参数搜索或多候选长训练。

在无有效 100e checkpoints 的情况下生成上述分析会制造伪证据，因此均标记为 `N/A — blocked by confirmed split leakage`。

## 已准备的正式训练入口

数据重划分并审计为严格 `PASS` 后，可运行：

```bash
cd /root/YOLO26N-NEW
/opt/conda/bin/python scripts/train_b0_japan7_100e_seed42.py \
  2>&1 | tee logs/paper1_b0_pretrained_auto_linear_japan7_100e_seed42.log

/opt/conda/bin/python scripts/train_p4single_japan7_100e_seed42.py \
  2>&1 | tee logs/paper1_overlock_projectfit_p4_gate1e3_pretrained_auto_linear_japan7_100e_seed42.log
```

两个入口共享同一个参数源，锁定 100e、imgsz 640、batch 32、seed 42、deterministic、AMP、optimizer auto、momentum 0.937、warmup 3、mosaic 1、close_mosaic 10 等协议，并都从同一 `yolo26n.pt` 加载。入口会读取数据审计 JSON；状态不是严格 `PASS` 时立即停止。

## 下一阶段最小方案

1. 保留当前数据不动，创建新的派生 manifest/软链接目录。
2. 用 SHA256、人工复核的 pHash 近重复关系和连续序列窗口形成场景组。
3. 以场景组为不可拆分单位分配 train/val，并尽量维持 80/20 和类别分布。
4. 要求文件名、SHA256、人工确认近重复、场景组跨 split 交集全部为 0；所有类别仍存在；标签检查通过。
5. 审计状态写成严格 `PASS` 后，只运行上述 B0/P4 seed42 配对 100e。
6. 若配对趋势仍值得验证，再补 seed0/3447；否则停止 P4 路线。

该候选不改变参数量、FLOPs、导出或部署路径，却能恢复指标的可解释性，优先级高于任何新模型模块。

## 相关文档

- `reports/environment_100e_seed42.md`
- `reports/b0_100e_reuse_audit.md`
- `reports/p4_single_30e_three_seed_summary.md`
- `docs/frontier_review_road_damage_2026.md`
- `docs/next_candidate_decision_matrix.md`
