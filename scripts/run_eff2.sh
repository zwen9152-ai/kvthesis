#!/usr/bin/env bash
# Chapter-5 efficiency, second round (after the time-averaged KV analysis), ~70 min on a 3090:
#   A. realistic-length throughput (MATH-500, natural EOS, 8k cap), batch 32, budget 128
#   B. RewindKV with profile=1 (synced timers: compression / rollback share)
#   C. 16k forced-length stress: peak GPU memory at one batch size per method
#      (KV memory is linear in batch size; the largest batch that fits is extrapolated)
cd "$(dirname "$0")/.."
# timing must not overlap accuracy jobs
while pgrep -f "python -m eval.run_eval" > /dev/null; do sleep 60; done
M15=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DTYPE=${DTYPE:-bfloat16}
R=rollback=1,escalate=2,max_budget=1024
mkdir -p results/efficiency logs
B="python scripts/bench_efficiency.py --model $M15 --dtype $DTYPE"
$B --specs "full,snapkv,snapkv@$R,snapkv@$R,recall=0,offload=0,rkv,rkv@$R,rkv@$R,recall=0,offload=0" \
   --budgets 128 --batch_sizes 32 --n 64 --max_new_tokens 8192 --out results/efficiency/eff2_real.jsonl
$B --specs "snapkv@$R,profile=1,snapkv@$R,recall=0,offload=0,profile=1" \
   --budgets 128 --batch_sizes 32 --n 32 --max_new_tokens 8192 --out results/efficiency/eff2_profile.jsonl
$B --specs "full" --batch_sizes 16 --n_per_bs --ignore_eos --n 64 --max_new_tokens 16384 \
   --out results/efficiency/eff2_16k.jsonl
$B --specs "snapkv,snapkv@$R,snapkv@$R,recall=0,offload=0" --budgets 128 --batch_sizes 32 \
   --n_per_bs --ignore_eos --n 64 --max_new_tokens 16384 --out results/efficiency/eff2_16k.jsonl
echo EFF2_DONE
