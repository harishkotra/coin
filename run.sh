#!/usr/bin/env bash
# Coin — one command bring-up.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8077}"
export COIN_DEVICE="${COIN_DEVICE:-mps}"
export COIN_DTYPE="${COIN_DTYPE:-bfloat16}"

exec .venv/bin/python -m uvicorn app.server:app --host 0.0.0.0 --port "$PORT" --log-level warning
