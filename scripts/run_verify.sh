#!/usr/bin/env bash
# First GPU session (~2 h): is batched decoding right on the real model, and how fast?
#   1. unit tests
#   2. throughput / memory sweep (chapter-5 style) for the batch-size choice
#   3. batch-1 vs batch-16 greedy runs on 16 MATH problems, compared
#   4. the GPU-hour estimate of the full plan at the measured throughput
# DTYPE=float16 on T4 / 2080 Ti.
set -uo pipefail
cd "$(dirname "$0")/.."
M15=${M15:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
DTYPE=${DTYPE:-bfloat16}
mkdir -p results/verify results/efficiency logs

python -m pytest -q 2>&1 | tail -3 | tee logs/verify.log

python scripts/bench_efficiency.py --model "$M15" --dtype "$DTYPE" \
    --specs "full,snapkv,snapkv@rollback=1,escalate=2,max_budget=1024,lazy" \
    --budgets 128 --batch_sizes 1,16,32,64 --n 64 --max_new_tokens 2048 \
    --out results/efficiency/throughput_1.5b.jsonl 2>&1 | grep -v -i warn | tee -a logs/verify.log

python scripts/make_queue.py verify > logs/queue_verify.txt
WORKERS=1 DTYPE="$DTYPE" bash scripts/run_queue.sh logs/queue_verify.txt
python -m analysis.verify_batching results/verify/math500_batching_bs1.jsonl \
    results/verify/math500_batching_bs16.jsonl | tee -a logs/verify.log

# measured aggregate throughput of SnapKV at the best batch size -> plan estimate
TOK=$(python -c "
import json
rs=[json.loads(l) for l in open('results/efficiency/throughput_1.5b.jsonl')]
print(int(max(r['tok_per_s'] for r in rs if r.get('spec')=='snapkv' and not r.get('oom'))))")
echo "measured 1.5B throughput: $TOK tok/s" | tee -a logs/verify.log
python scripts/make_queue.py main ablation 7b --estimate --tok_s_15b "$TOK" --tok_s_7b $((TOK * 2 / 5)) 2>&1 | tee -a logs/verify.log
echo VERIFY_DONE | tee -a logs/verify.log
