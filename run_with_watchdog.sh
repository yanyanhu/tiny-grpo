#!/bin/bash
# Usage: ./run_with_watchdog.sh --profile {smoke,debug,longer} --hardware {mps_16gb,cuda_4gb} [other train_grpo.py args...]
# All arguments are forwarded to train_grpo.py as-is (e.g. --hardware is required there).
# Override the watchdog's own timeouts via env vars, e.g. for a longer experiment:
#   STALL_LIMIT=600 HARD_TIMEOUT=3600 ./run_with_watchdog.sh --profile debug --hardware mps_16gb
set -u
cd "$(dirname "$0")"

LOGFILE="run.log"
: > "$LOGFILE"

export PATH="$HOME/.local/bin:$PATH"
uv run python -u train_grpo.py "$@" >> "$LOGFILE" 2>&1 &
PID=$!

START=$(date +%s)
LAST_SIZE=-1
LAST_CHANGE=$START
STALL_LIMIT="${STALL_LIMIT:-360}"     # 6 min with no new output => treat as hung
HARD_TIMEOUT="${HARD_TIMEOUT:-1800}"  # 30 min absolute cap
CHECK_INTERVAL=15
TERM_GRACE="${TERM_GRACE:-30}"       # SIGTERM cleanup window before SIGKILL

terminate_child() {
    REASON=$1
    echo "WATCHDOG: ${REASON} — sending SIGTERM to PID $PID (${TERM_GRACE}s grace)" >> "$LOGFILE"
    kill -TERM "$PID" 2>/dev/null || return

    GRACE_START=$(date +%s)
    while kill -0 "$PID" 2>/dev/null; do
        NOW=$(date +%s)
        if [ $((NOW - GRACE_START)) -ge "$TERM_GRACE" ]; then
            echo "WATCHDOG: PID $PID still running after ${TERM_GRACE}s grace — sending SIGKILL" >> "$LOGFILE"
            kill -KILL "$PID" 2>/dev/null || true
            return
        fi
        sleep 1
    done
}

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
        terminate_child "no log output growth for ${STALL}s (limit ${STALL_LIMIT}s) at ${ELAPSED}s elapsed"
        STATUS="killed_stall"
        break
    fi
    if [ "$ELAPSED" -ge "$HARD_TIMEOUT" ]; then
        terminate_child "hard timeout ${HARD_TIMEOUT}s reached"
        STATUS="killed_timeout"
        break
    fi
done

wait "$PID" 2>/dev/null
EXIT_CODE=$?
TOTAL=$(( $(date +%s) - START ))
if [ "$STATUS" = "unknown" ]; then
    STATUS="finished"
else
    # Use GNU timeout's conventional status so a child that handles SIGTERM
    # and exits cleanly cannot make a watchdog-triggered run look successful.
    EXIT_CODE=124
fi
echo "WATCHDOG: run ${STATUS}, exit_code=${EXIT_CODE}, total_elapsed=${TOTAL}s" >> "$LOGFILE"
exit "$EXIT_CODE"
