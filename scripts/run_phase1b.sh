#!/usr/bin/env bash
# Phase 1b: gate G2' on 50 MATH-500 problems, plus RewindKV with budget escalation.
# Batch-1 decoding leaves the GPU mostly idle, so WORKERS processes (~4 GB each) pull
# "method|budget" items from a shared queue in priority order; the gate is refreshed
# after each item. Resumes if interrupted: just re-run.
#   DTYPE=float16 on T4 / 2080 Ti (no bf16), bfloat16 on 30/40-series.
cd "$(dirname "$0")/.."

MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
MAXTOK=${MAXTOK:-8192}
DTYPE=${DTYPE:-bfloat16}
N=${N:-50}
WORKERS=${WORKERS:-5}
OUT=${OUT:-results/phase1b_math500_n${N}_${DTYPE}.jsonl}
ESC="rollback=1,escalate=2,max_budget=1024"
QUEUE=logs/phase1b_queue.txt
mkdir -p results logs

# highest priority first: the SnapKV story, then R-KV at 128, then R-KV at 256
cat > "$QUEUE" <<EOF
full|0
snapkv|128
snapkv@rollback=1|128
snapkv@$ESC|128
snapkv|256
snapkv@rollback=1|256
rkv|128
rkv@$ESC|128
rkv|256
rkv@rollback=1|256
EOF

pop() {  # print and remove the first queue line, atomically across workers
    flock "$QUEUE.lock" bash -c "head -n1 '$QUEUE'; sed -i 1d '$QUEUE'"
}

worker() {
    while item=$(pop) && [ -n "$item" ]; do
        echo "== $item"
        python -m eval.run_eval --model "$MODEL" --dataset math500 --n "$N" --methods "${item%|*}" \
            --budgets "${item##*|}" --max_new_tokens "$MAXTOK" --dtype "$DTYPE" --save_ids --out "$OUT"
        python -m analysis.gate --runs "$OUT" > logs/phase1b_gate.log 2>&1
    done
}

for i in $(seq 0 $((WORKERS - 1))); do
    worker > "logs/phase1b_w$i.log" 2>&1 &
    sleep 30  # stagger model loading
done
wait
echo PHASE1B_DONE >> logs/phase1b_gate.log
