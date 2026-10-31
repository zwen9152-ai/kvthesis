# kvthesis：推理模型长思维链的 KV Cache 压缩

硕士论文的实验代码。研究对象是 R1 类推理模型：它们在解码阶段会生成很长的思维链，KV Cache 随之不断增长。

## 核心方法：RewindKV

1. **基础淘汰策略**（`kvcomp/methods.py`）：StreamingLLM、H2O、SnapKV、R-KV、LazyEviction、VaSE，以及 StepKV（项目早期的设计，现作为对照）。所有方法共用同一个调度器：prompt 永不淘汰；解码部分达到 `budget + buffer` 时压缩回 `budget`。
2. **卸载而不是丢弃**（`kvcomp/compressor.py`，`offload=True`）：被淘汰的 KV 移到 CPU 内存保存。
3. **退化检测**（`kvcomp/degeneration.py`）：只看 token id，有两类信号。一是推理步骤和之前某一步逐字相同（第 3 次出现），二是周期性重复。
4. **召回、回滚与预算升级**（`kvcomp/decode.py`、`kvcomp/batch.py`）：
   - 检测到退化后，回到出问题的那个推理步骤的开头，把 CPU 上的 KV 全部召回。
   - 重新编码最后 `refeed` 个 token，用新的 query 重新选出 `budget` 个条目，然后继续生成。
   - 每回滚一次，预算乘以 `escalate`，上限为 `max_budget`。

第 1b 阶段的发现（MATH-500，50 题）：压缩后答错的题，很少是逐字死循环，主要表现是**不收敛**：反复验证（"Wait"增多）、逐字重做之前做过的推导、输出越来越长，最后被截断。检测器捕捉的是"重复推理步骤"这个触发信号。分析脚本把这几种现象分开统计。

## 对比基线

| 写法 | 含义 |
|---|---|
| `full` | 不压缩 |
| `streaming` / `h2o` / `snapkv` / `rkv` | 经典淘汰方法（R-KV 与官方 `R1KV` 对齐） |
| `lazy` | LazyEviction（ACL 2026），移植自官方代码。需要逐步观测注意力，只能走批量解码路径 |
| `vase` | VaSE-AttnV（arXiv 2606.03928），官方模式为 `range_sink_sample_attn` |
| `X@rollback=1,escalate=2,max_budget=1024` | 在 X 上叠加 RewindKV |
| `X@rescue=1`（需要 `--helper_model`） | KV-Rescue 简化版：触发条件和 RewindKV 相同，但由 full-KV 的小模型重写出问题的步骤 |

其他可写参数：`recall=0`（不召回，用作消融）、`max_rewind=0`（只回到循环本身）、`refeed`、`max_rollbacks`、检测器参数（`min_span`、`step_repeats`、`min_step_len` 等）、`shard=k/m`（把一个实验按题目切给多个进程）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `kvcomp/decode.py` | 逐题解码循环。显式传入真实位置，因为淘汰之后不能再用缓存长度当位置 |
| `kvcomp/batch.py` | **连续批量解码**：每个槽位有自己的长度和填充掩码，一道题解完立即补位；支持回滚、升级、LazyEviction 观测器和 helper 接手。测试中与逐题解码逐 token 一致 |
| `kvcomp/query_capture.py` | 用 hook 重算最近 w 个 token 的 query（含 RoPE），供打分使用 |
| `kvcomp/rescue.py` | KV-Rescue 简化版 |
| `eval/run_eval.py` | 跑"方法 × 预算"网格。每题写一行 jsonl，支持断点续跑、`--batch_size`、`--helper_model` |
| `scripts/run_verify.sh` | **第一次租 GPU 时用**：验证批量解码、测吞吐，并给出全部实验所需的 GPU 小时 |
| `scripts/make_queue.py` + `scripts/run_queue.sh` | 生成全部论文实验的任务队列，多进程按优先级执行 |
| `scripts/bench_efficiency.py` | 第 5 章：吞吐、显存、CPU 卸载量、PCIe 传输量、回滚耗时 |
| `analysis/degeneration_stats.py` | 第 3、4 章：准确率、截断、重复步骤、"Wait"频率、损失归因、配对 bootstrap 加 McNemar 检验，生成图和 LaTeX 表 |
| `analysis/loop_examples.py` | 案例研究：触发检测器的原文，以及 RewindKV 回滚后的原文 |
| `analysis/attn_stats.py` | 第 3 章：注意力的步骤局部性、锚点、废弃分支 |
| `analysis/gate.py` | 闸门 G2′ |
| `analysis/verify_batching.py` | 对比批量解码和逐题解码的结果 |

## 在 GPU 机器上运行

详细步骤见 [`docs/RUNBOOK.md`](docs/RUNBOOK.md)。最简流程：

```bash
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com   # 国内网络
bash scripts/run_verify.sh                  # 第一次：约 2 小时
python scripts/make_queue.py main ablation 7b > logs/queue.txt
WORKERS=2 bash scripts/run_queue.sh logs/queue.txt
```

T4 和 2080 Ti 不支持 bf16，请加 `DTYPE=float16`。
