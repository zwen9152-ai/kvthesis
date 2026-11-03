#!/usr/bin/env bash
# Chapter-5 efficiency, second round (after the time-averaged KV analysis):
#   A. realistic-length throughput (MATH-500, natural EOS, 8k cap), batch 32, budget 128/256
#   B. the same RewindKV runs with profile=1 (synced timers: compression / rollback share)
#   C. 16k forced-length stress: peak GPU memory and largest batch that fits, per method
cd "$(dirname "$0")/.."
M15=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DTYPE=${DTYPE:-bfloat16}
R=rollback=1,escalate=2,max_budget=1024
mkdir -p results/efficiency logs
B="python scripts/bench_efficiency.py --model $M15 --dtype $DTYPE"
$B --specs "full,snapkv,snapkv@$R,snapkv@$R,recall=0,offload=0,rkv,rkv@$R,rkv@$R,recall=0,offload=0" \
   --budgets 128,256 --batch_sizes 32 --n 64 --max_new_tokens 8192 --out results/efficiency/eff2_real.jsonl
$B --specs "snapkv@$R,profile=1,snapkv@$R,recall=0,offload=0,profile=1,snapkv@profile=1" \
   --budgets 128 --batch_sizes 32 --n 64 --max_new_tokens 8192 --out results/efficiency/eff2_profile.jsonl
$B --specs "full" --batch_sizes 16,32,48 --n_per_bs --ignore_eos --n 64 --max_new_tokens 16384 \
   --out results/efficiency/eff2_16k.jsonl
$B --specs "snapkv,snapkv@$R,snapkv@$R,recall=0,offload=0" --budgets 128 --batch_sizes 64,128 \
   --n_per_bs --ignore_eos --n 64 --max_new_tokens 16384 --out results/efficiency/eff2_16k.jsonl
echo EFF2_DONE
