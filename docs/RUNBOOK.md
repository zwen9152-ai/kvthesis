# GPU 运行手册

适用于 AutoDL 这类按小时计费的单卡机器，推荐 4090 或 3090（24GB）。全部命令都在仓库根目录执行。

## 0. 环境（每台新机器做一次，约 15 分钟）

```bash
cd /root/autodl-tmp
source /etc/network_turbo            # AutoDL 学术加速，用于访问 GitHub
git clone https://github.com/zwen9152-ai/kvthesis.git && cd kvthesis
unset http_proxy https_proxy          # 下载模型走镜像，不需要走加速
pip install -r requirements.txt       # 机器自带的 torch 就够用
cat >> ~/.bashrc <<'EOF'
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export LC_ALL=C.UTF-8 LANG=C.UTF-8    # 不加这行，conda 的 python 一 import readline 就段错误
EOF
source ~/.bashrc
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   # 第 2 次租卡再下也可以
python -m pytest -q                   # 67 个测试应全部通过
```

如果机器是迁移过来的，数据盘 `/root/autodl-tmp` 上的代码、模型和结果都在，直接 `git pull` 即可。

## 1. 第一次租卡：验证批量解码并测吞吐（约 2 小时）

```bash
nohup bash scripts/run_verify.sh > logs/verify_nohup.log 2>&1 &
tail -f logs/verify.log
```

脚本依次执行：

1. 单元测试。
2. 吞吐和显存扫描：batch size 取 1、16、32、64，结果写到 `results/efficiency/throughput_1.5b.jsonl`。
3. 在 16 道题上用贪心解码，分别跑 batch 1 和 batch 16，对比两者的准确率、长度和公共前缀。bf16 下批次形状不同会导致数值略有差异，长输出出现分叉是正常的，但准确率和长度应当接近。
4. 用实测吞吐重新估算全部实验所需的 GPU 小时。

`logs/verify.log` 最后打印出 `VERIFY_DONE` 就表示完成。把 `logs/` 和 `results/` 发回来即可。

## 2. 主实验（按上一步的估算，一次性租一整段）

```bash
python scripts/make_queue.py main ablation 7b > logs/queue.txt
python scripts/make_queue.py main ablation 7b --estimate      # 再看一眼预计耗时
WORKERS=2 nohup bash scripts/run_queue.sh logs/queue.txt > logs/queue_nohup.log 2>&1 &
```

- 队列是纯文本文件，每行一个任务，**排在前面的先跑**。运行中可以直接编辑 `logs/queue.txt` 调整顺序，或者追加任务。
- 失败的任务写到 `logs/queue.txt.failed`。
- 断电或关机后重新执行同一条命令即可。已完成的（方法，题目）会自动跳过。
- `WORKERS` 设多少：先用 2，看 `nvidia-smi`。显存还剩一半以上，而且 GPU 利用率低于 90%，再加到 3。
- 7B 任务会自动把 batch 调小。如果显存不够（OOM），把那一行的 `--batch_size` 减半，重新放回队列。

查看进度：

```bash
tail -n 3 logs/queue_w*.log
wc -l results/main/*.jsonl
python -m analysis.gate --runs results/main/1.5b_math500_s0.jsonl
```

## 3. 效率实验（第 5 章，约 3 小时）

```bash
python scripts/bench_efficiency.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --specs "full,snapkv,snapkv@rollback=1,escalate=2,max_budget=1024,profile=1,lazy,vase,rkv" \
    --budgets 128,256 --batch_sizes 1,16,32 --n 32 --max_new_tokens 4096 \
    --out results/efficiency/bench_1.5b.jsonl
```

`profile=1` 会在压缩和回滚处同步 GPU 来计时，吞吐会因此略降，所以只在需要统计耗时占比的那组配置里打开。

## 4. 出图出表（不需要 GPU）

```bash
python -m analysis.degeneration_stats --runs results/main/1.5b_math500_s0.jsonl --out results/analysis/math500
python -m analysis.loop_examples --runs results/main/1.5b_math500_s0.jsonl \
    --base "snapkv|128" --variant "snapkv@rollback=1,escalate=2,max_budget=1024|128" \
    --out results/analysis/math500/loop_examples.md
```

## 注意事项

- **关机**：实验跑完后到 AutoDL 控制台关机，否则会一直计费。数据盘上的结果关机后不会丢。
- **精度**：T4 和 2080 Ti 不支持 bf16，需要 `DTYPE=float16`。同一张结果表不要混用两种精度，输出文件名里已经带了 dtype。
- **磁盘**：`--save_ids` 会保存全部 token，主实验大约需要 1–2GB，50GB 的数据盘足够。
