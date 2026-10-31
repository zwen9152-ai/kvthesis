#!/usr/bin/env bash
# Run a run_eval job queue (from scripts/make_queue.py) with WORKERS parallel workers.
# Workers pop the first line atomically, so higher lines run first and lines appended or
# reordered while the queue runs are picked up. Resumes if interrupted: re-run it.
#   WORKERS=2 DTYPE=bfloat16 bash scripts/run_queue.sh logs/queue.txt
# DTYPE=float16 on T4 / 2080 Ti (no bf16). With batched decoding one or two workers
# already fill a 24 GB GPU; check nvidia-smi and raise WORKERS only if memory allows.
cd "$(dirname "$0")/.."
QUEUE=${1:-logs/queue.txt}
WORKERS=${WORKERS:-2}
DTYPE=${DTYPE:-bfloat16}
mkdir -p results/main results/verify logs

pop() {
    flock "$QUEUE.lock" bash -c "head -n1 '$QUEUE'; sed -i 1d '$QUEUE'"
}

worker() {
    while line=$(pop) && [ -n "$line" ]; do
        echo "== $(date +%H:%M) $line"
        # shellcheck disable=SC2086
        if ! python -m eval.run_eval $line --dtype "$DTYPE"; then
            echo "!! failed: $line"
            echo "$line" >> "$QUEUE.failed"
        fi
    done
}

for i in $(seq 0 $((WORKERS - 1))); do
    worker > "logs/queue_w$i.log" 2>&1 &
    sleep 30  # stagger model loading
done
wait
echo "QUEUE_DONE $(date)" >> logs/queue_status.log
