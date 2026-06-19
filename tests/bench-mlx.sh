#!/bin/bash
# Quick perf benchmark for mlx_lm.server.
# Sends fixed-size prompts, parses usage from the (non-streaming) response,
# and measures wall time. Decode tok/s = completion_tokens / (wall - prefill_estimate).

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8083}"
MAX_TOKENS="${MAX_TOKENS:-256}"

# Build a filler prompt of approximately N tokens (~4 chars/token rule of thumb).
make_prompt() {
    local target_tokens=$1
    local chars=$((target_tokens * 4))
    python3 -c "
import random, string
random.seed(42 + $target_tokens)
words = ['the','quick','brown','fox','jumps','over','lazy','dog','code','model','token','test','prompt','data','func','class','return','value','result','run','perf','bench','memory','cache','context','byte','line','file','user','agent']
out = []
n = 0
while n < $chars:
    w = random.choice(words)
    out.append(w)
    n += len(w) + 1
print(' '.join(out))
"
}

run_one() {
    local label=$1
    local target=$2
    local prompt
    prompt=$(make_prompt "$target")
    local body
    body=$(python3 -c "
import json, sys
prompt = sys.stdin.read()
print(json.dumps({
    'model': 'FakeRockert543/gemma-4-31b-it-MLX-8bit',
    'messages': [
        {'role': 'system', 'content': 'You are a benchmark target. Reply concisely.'},
        {'role': 'user', 'content': prompt + '\n\nSummarize the above text in one sentence and then count from 1 to 50.'}
    ],
    'max_tokens': $MAX_TOKENS,
    'temperature': 1.0,
    'top_p': 0.95,
    'stream': False
}))
" <<<"$prompt")

    echo "=== $label (target ~${target} tokens input) ==="
    local t0 t1 resp
    t0=$(python3 -c "import time; print(time.time())")
    resp=$(curl -sS -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body")
    t1=$(python3 -c "import time; print(time.time())")
    local wall
    wall=$(python3 -c "print(round($t1 - $t0, 2))")
    local pt ct
    pt=$(echo "$resp" | jq -r '.usage.prompt_tokens // "?"')
    ct=$(echo "$resp" | jq -r '.usage.completion_tokens // "?"')
    echo "  prompt_tokens=$pt  completion_tokens=$ct  wall=${wall}s"
    if [[ "$pt" != "?" && "$ct" != "?" ]]; then
        python3 -c "
pt=$pt; ct=$ct; wall=$wall
# Crude split: assume prefill ~161 t/s (from observed), rest is decode.
pf = pt / 161.0
dec_time = max(wall - pf, 0.001)
print(f'  est. decode rate: {ct/dec_time:.1f} tok/s  (decode_time≈{dec_time:.1f}s, prefill≈{pf:.1f}s)')
print(f'  e2e tok/s: {(pt+ct)/wall:.1f}')
"
    fi
    echo
}

# Warmup with new system prompt so cache cold for our sequence
run_one "warmup" 200
run_one "1k input"  900
run_one "8k input"  7600
run_one "16k input" 15500
