#!/usr/bin/env bash
# Phase-1 validation on a single 24GB GPU (R1-Distill-Qwen-1.5B), ~8-10 h in total.
#   A: full-KV baseline on 40 MATH-500 problems, saving token ids
#      (sanity check: accuracy should be near the published ~80%).
#   B: attention analysis of the full-KV traces -> results/attn_stats/.
#   C: eviction baselines at budget 256, with and without rollback (gate G2').
# Everything resumes if interrupted: just re-run the script.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
MAXTOK=${MAXTOK:-8192}
DTYPE=${DTYPE:-bfloat16}   # use float16 on T4 (no bf16 support)
N=${N:-40}
BUDGETS=${BUDGETS:-256}
OUT=results/math500_n${N}.jsonl
mkdir -p results logs

echo "== unit tests =="
python -m pytest -q

echo "== A: full-KV baseline =="
python -m eval.run_eval --model "$MODEL" --dataset math500 --n "$N" --methods full \
    --max_new_tokens "$MAXTOK" --dtype "$DTYPE" --save_ids --out "$OUT" 2>&1 | tee -a logs/A.log

echo "== B: attention analysis =="
python -m analysis.attn_stats --model "$MODEL" --runs "$OUT" --n 20 --dtype "$DTYPE" --out results/attn_stats 2>&1 | tee logs/B.log

echo "== C: eviction baselines, with and without rollback =="
python -m eval.run_eval --model "$MODEL" --dataset math500 --n "$N" --budgets "$BUDGETS" \
    --methods streaming,h2o,snapkv,rkv,snapkv@rollback=1,rkv@rollback=1 \
    --max_new_tokens "$MAXTOK" --dtype "$DTYPE" --out "$OUT" 2>&1 | tee -a logs/C.log

python -m analysis.gate --runs "$OUT" | tee logs/gate.log
echo "done. Send back the results/ and logs/ folders."
