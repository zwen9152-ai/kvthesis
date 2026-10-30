# kvthesis：推理模型长思维链的 KV Cache 压缩

硕士论文的实验代码。研究对象是 R1 类推理模型：它们在解码阶段生成很长的思维链，KV Cache 随之不断增长。

## 核心思路（RewindKV）

1. **基础淘汰策略**（`kvcomp/methods.py`）：StreamingLLM、H2O、SnapKV、R-KV（复现版），以及 StepKV（本项目早期设计，现作为对照）。所有方法共用同一个调度器：prompt 永不淘汰；解码部分达到 `budget + buffer` 时，压缩回 `budget`。
2. **卸载而不是丢弃**（`kvcomp/compressor.py`，`offload=True`）：被淘汰的 KV 移到 CPU 内存。
3. **退化检测**（`kvcomp/degeneration.py`）：只看 token id，检测两类信号：周期性重复，以及与之前完全相同的推理步骤。
4. **召回 + 回滚**（`kvcomp/decode.py`）：检测到循环后，回到循环开始的那个推理步骤，把 CPU 上的 KV 全部召回；重新编码最后 `refeed` 个 token 得到新的 query，用它重新选出 `budget` 个条目，然后继续生成。

## 文件说明

| 文件 | 作用 |
|---|---|
| `kvcomp/query_capture.py` | 用 hook 重算最近 w 个 token 的 query（含 RoPE），给 KV 打分；`window_attention` 已和 eager 注意力逐元素对拍 |
| `kvcomp/decode.py` | 自写解码循环：显式传入真实位置（淘汰之后不能用缓存长度当位置） |
| `kvcomp/segment.py` | 按 `\n\n` 在线切分推理步骤 |
| `kvcomp/quant.py` | 模拟量化（后续扩展用） |
| `eval/run_eval.py` | 跑"方法 × 预算"网格，每题一行 jsonl，支持断点续跑 |
| `eval/grader.py` | 抽取 `\boxed{}` 答案并判等 |
| `analysis/attn_stats.py` | 第 3 章：注意力的步骤局部性、锚点集中度、废弃分支 |
| `analysis/gate.py` | 闸门 G2′：退化在准确率损失中占多大比例，回滚能追回多少 |

## 在 GPU 机器上运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 按你的 CUDA 版本选
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com  # 国内网络：模型和数据集改从镜像下载（建议写进 ~/.bashrc）
python -m pytest -q                      # 单元测试，约 1 分钟
bash scripts/run_gpu_pilot.sh            # 先跑探针：压缩后退化有多常见（约 4–6 小时）
bash scripts/run_gpu_small.sh            # 第 1 阶段验证，约 8–10 小时，中断后重跑会接着跑
```

跑完后把 `results/` 和 `logs/` 两个目录发给我。

方法写法示例：`snapkv`、`snapkv@rollback=1`、`stepkv@anchors=4,beta=2,step_trigger=1`。
