#!/bin/bash
# Production startup script for CA RAG Assistant
# Usage: ./run_production.sh

# Number of workers (typically 2-4 x CPU cores, but limited by RAM and DB connections)
WORKERS=${WORKERS:-4}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-9000}

echo "=== CA RAG Assistant - Production Mode ==="
echo "Workers: $WORKERS"
echo "Host: $HOST:$PORT"
echo ""

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Starting with uv..."
    uv run uvicorn api:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        --loop uvloop \
        --http httptools \
        --limit-concurrency 100 \
        --timeout-keep-alive 30
else
    echo "Starting with python..."
    python -m uvicorn api:app \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        --loop uvloop \
        --http httptools \
        --limit-concurrency 100 \
        --timeout-keep-alive 30
fi
