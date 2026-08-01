#!/bin/bash
set -u
cd "$(dirname "$0")"

LOGFILE="run.log"
: > "$LOGFILE"

export PATH="$HOME/.local/bin:$PATH"
uv run python -u train_grpo.py >> "$LOGFILE" 2>&1 &
PID=$!

START=$(date +%s)
LAST_SIZE=-1
LAST_CHANGE=$START
STALL_LIMIT=360   # 6 min with no new output => treat as hung
HARD_TIMEOUT=1800 # 30 min absolute cap
CHECK_INTERVAL=15

STATUS="unknown"
while kill -0 "$PID" 2>/dev/null; do
    sleep "$CHECK_INTERVAL"
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    SIZE=$(wc -c < "$LOGFILE" 2>/dev/null || echo 0)

    if [ "$SIZE" != "$LAST_SIZE" ]; then
        LAST_SIZE=$SIZE
        LAST_CHANGE=$NOW
    fi
    STALL=$((NOW - LAST_CHANGE))

    if [ "$STALL" -ge "$STALL_LIMIT" ]; then
        echo "WATCHDOG: no log output growth for ${STALL}s (limit ${STALL_LIMIT}s) at ${ELAPSED}s elapsed — killing PID $PID as hung" >> "$LOGFILE"
        kill -9 "$PID" 2>/dev/null
        STATUS="killed_stall"
        break
    fi
    if [ "$ELAPSED" -ge "$HARD_TIMEOUT" ]; then
        echo "WATCHDOG: hard timeout ${HARD_TIMEOUT}s reached — killing PID $PID" >> "$LOGFILE"
        kill -9 "$PID" 2>/dev/null
        STATUS="killed_timeout"
        break
    fi
done

wait "$PID" 2>/dev/null
EXIT_CODE=$?
TOTAL=$(( $(date +%s) - START ))
if [ "$STATUS" = "unknown" ]; then
    STATUS="finished"
fi
echo "WATCHDOG: run ${STATUS}, exit_code=${EXIT_CODE}, total_elapsed=${TOTAL}s" >> "$LOGFILE"
