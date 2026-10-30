#!/usr/bin/env bash
# CPU smoke test with a real (small) chat model: every method must run end to end.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
.venv/bin/python -m eval.run_eval --model "$MODEL" --dataset gsm8k --n 2 --dtype bfloat16 \
    --methods full,streaming,h2o,snapkv,rkv,stepkv,snapkv@rollback=1 --budgets 64 --buffer 32 --window 16 \
    --max_new_tokens 256 --temperature 0 --out results/smoke_cpu.jsonl
