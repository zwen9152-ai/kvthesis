# 论文章节与实验对照表

正文由作者执笔，这里只列出每章需要的证据从哪里来、目前的状态。
"P1b"指第 1b 阶段的数据（MATH-500 50 题，3090，bf16）；"主实验"指 `make_queue.py main ablation 7b` 生成的那批任务。

## 第 1 章 绪论

- 问题：推理模型的思维链很长，KV Cache 在解码期线性增长；现有压缩方法在小预算下会掉点。
- 贡献：
  1. 实证分析：压缩引起的是"不收敛"，而不是简单的信息丢失。
  2. RewindKV：检测到退化后召回被淘汰的 KV、回滚，并按需升级预算。
  3. 在 6 种基础策略和 KV-Rescue 简化版上做系统对比。

## 第 2 章 背景与相关工作

| 小节 | 要点与文献 |
|---|---|
| KV Cache 与解码 | Transformer、GQA、RoPE |
| 预填充阶段的压缩 | SnapKV、PyramidKV |
| 解码阶段的淘汰 | StreamingLLM、H2O、R-KV、LazyEviction（ACL 2026）、VaSE（2606.03928）、AvoKV-E（2610.03007）、ThinKV（ICLR 2026，无开源代码，只做文字对比）、TAM |
| 压缩带来的副作用 | SkipKV（2512.07993）、Random Attention（2609.03430）：淘汰会让输出变长；VaSE：大幅值 value 被淘汰后陷入循环 |
| 恢复类方法 | KV-Rescue（2608.15797，辅助模型）、RestoreKV、LOUVER / HGCA（每步检索 CPU 上的 KV）、VECTOR（2605.23258） |
| 区别 | 我们的方法平时只用很小的预算，**只在检测到退化时**才召回并回滚，召回的是**模型自己的** KV，不需要辅助模型 |

写作前按计划文件里记录的关键词重新检索一遍。

## 第 3 章 压缩的隐性代价：不收敛

| 图表 | 内容 | 产出命令 | 状态 |
|---|---|---|---|
| 表 3.1 | 各方法、各预算下的准确率、长度、截断率、重复步骤率、"Wait"频率、终端循环 | `analysis.degeneration_stats` 的 summary | P1b 已有（只有 SnapKV），等主实验 |
| 图 3.1 | 损失归因：答错的题分为重复步骤 / 截断无重复 / 其他错误 | `fig_loss.pdf`、`loss.md` | 同上 |
| 图 3.2 | 输出长度分布：Full 与各预算对比 | `fig_length.pdf` | 同上 |
| 表 3.2 | 触发位置相对第一次淘汰的距离，以及周期 | `loops.md` | 同上 |
| 案例 3.1–3.3 | 逐字重做之前的推导；把正常枚举误判为重复（假阳性） | `analysis.loop_examples` | P1b 已有 5 个例子 |
| 图 3.3 | 注意力的步骤局部性和锚点（O1、O2） | `analysis.attn_stats` | 需要 GPU，在主实验 full 跑完后执行 |

**P1b 的结论（写作时以主实验为准）**：
- SnapKV@128 的截断率从 16% 升到 36%，"Wait"从每千 token 2.6 次升到 4.2 次，输出长度是 Full 的 1.63 倍。
- 截断输出最后 1024 个 token 中，不同 8-gram 占 80%–90%，说明几乎不是逐字死循环。
- 失败案例中常见的现象是模型逐字重做几千 token 之前已经做过的推导。我们的推测是：之前那段推导的 KV 被淘汰后，模型"忘了自己算过"。

## 第 4 章 RewindKV

| 图表 | 内容 | 产出 | 状态 |
|---|---|---|---|
| 算法 4.1 | 卸载、检测、回滚到步骤起点、召回、re-feed、重选、升级 | `kvcomp/batch.py` 的 `after_token` | 已完成 |
| 表 4.1 主结果 | 1.5B 模型，MATH / AIME / GSM8K × 预算 128、256 × 6 种基础策略 ± RewindKV | `degeneration_stats` 加 `paired.md` | 等主实验 |
| 表 4.2 显著性 | 配对 bootstrap 95% 置信区间和 McNemar 检验 | `paired.md` | P1b：+11.1 分，p=0.125，尚不显著 |
| 图 4.1 | 准确率与平均峰值 KV 的 Pareto 图（RewindKV 与静态预算 512 对比） | `fig_pareto.pdf` | 等主实验 |
| 表 4.3 消融 | recall=0、不升级、升级 1.5 和 4、`max_rewind=0`、检测阈值、`step_repeats`、`min_step_len` | `make_queue.py ablation` | 等主实验 |
| 表 4.4 | 7B 模型：Full、SnapKV ± RewindKV、R-KV ± RewindKV、KV-Rescue 简化版 | `make_queue.py 7b` | 等主实验 |

## 第 5 章 系统开销

| 图表 | 内容 | 产出 |
|---|---|---|
| 表 5.1 | 吞吐、相对权重的峰值显存、CPU 卸载量、PCIe 传输量 | `scripts/bench_efficiency.py` |
| 图 5.1 | batch size 与吞吐的关系 | `results/efficiency/throughput_1.5b.jsonl`（`run_verify.sh` 生成） |
| 表 5.2 | 回滚的代价：被丢弃的 token（computed − len）、回滚耗时（`profile=1`）、瞬时缓存长度 | 结果 jsonl 中的 `rollbacks` 字段 |
| 讨论 | 和 KV-Rescue 比较：辅助模型权重约 3.5GB，另加它自己的 full KV；RewindKV 只占用 CPU 内存 | `helper_max_cache_len` |

## 第 6 章 总结与展望

- 局限：检测器存在假阳性；只在数学推理上做了评测；KV-Rescue 是简化复现。
- 展望：基于语义（而不是逐字）的不收敛检测；和 vLLM 或 PagedAttention 集成。
