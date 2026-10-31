#!/usr/bin/env bash
# Phase 1b: gate G2' on 50 MATH-500 problems, plus RewindKV with budget escalation.
# Batch-1 decoding leaves the GPU mostly idle, so the configs run in parallel lanes
# (one process each, ~4 GB). Every lane runs its "method|budget" items in order and
# the gate is refreshed after each item. Resumes if interrupted: just re-run.
#   DTYPE=float16 on T4 / 2080 Ti (no bf16), bfloat16 on 30/40-series.
cd "$(dirname "$0")/.."

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
MAXTOK=${MAXTOK:-8192}
DTYPE=${DTYPE:-bfloat16}
N=${N:-50}
OUT=${OUT:-results/phase1b_math500_n${N}_${DTYPE}.jsonl}
ESC="rollback=1,escalate=2,max_budget=1024"
mkdir -p results logs

LANES=(
    "full|0 snapkv@$ESC|128"
    "snapkv|128 snapkv|256 rkv@rollback=1|256"
    "snapkv@rollback=1|128 snapkv@rollback=1|256"
    "rkv|128 rkv@$ESC|128 rkv|256"
)

lane() {  # $1 = lane index, $2 = space-separated "spec|budget" items
    for item in $2; do
        spec=${item%|*}
        budget=${item##*|}
        python -m eval.run_eval --model "$MODEL" --dataset math500 --n "$N" --methods "$spec" \
            --budgets "$budget" --max_new_tokens "$MAXTOK" --dtype "$DTYPE" --save_ids --out "$OUT"
        python -m analysis.gate --runs "$OUT" > logs/phase1b_gate.log 2>&1
    done
}

for i in "${!LANES[@]}"; do
    lane "$i" "${LANES[$i]}" > "logs/phase1b_lane$i.log" 2>&1 &
    sleep 60  # stagger model loading
done
wait
echo PHASE1B_DONE >> logs/phase1b_gate.log
