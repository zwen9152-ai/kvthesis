#!/usr/bin/env bash
# Pilot (~4-6 h on one 24GB GPU): how often does eviction cause degeneration?
# 20 MATH-500 problems, full KV vs. SnapKV / R-KV at budgets 128/256/512, no rollback.
# analysis.gate then prints, per run: accuracy, problems lost vs. full KV, and the share
# of those losses whose trace contains a repetition loop. Resumes if interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
MAXTOK=${MAXTOK:-8192}
DTYPE=${DTYPE:-bfloat16}   # use float16 on T4 (no bf16 support)
N=${N:-20}
OUT=results/pilot_math500_n${N}.jsonl
mkdir -p results logs

python -m pytest -q
python -m eval.run_eval --model "$MODEL" --dataset math500 --n "$N" \
    --methods full,snapkv,rkv --budgets 128,256,512 \
    --max_new_tokens "$MAXTOK" --dtype "$DTYPE" --save_ids --out "$OUT" 2>&1 | tee -a logs/pilot.log
python -m analysis.gate --runs "$OUT" | tee logs/pilot_gate.log
echo "done. Send back $OUT and logs/pilot_gate.log"
