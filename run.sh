#!/usr/bin/env bash
# Bridge runner: sync code, execute, and drop the full output where the agent
# can read it from the Windows side.
#
#   bash /mnt/e/AI/AI_Agent/workplace/hk-labour-triage/run.sh "python train.py --eval-only --no-quant"
#   bash /mnt/e/AI/AI_Agent/workplace/hk-labour-triage/run.sh "python check_dataset.py && python train.py --epochs 3"
#
# Everything (stdout + stderr) is teed to hk-labour-triage/logs/<timestamp>.log
# on the E: drive, which is the same directory the agent reads.

set -uo pipefail

# Python block-buffers stdout when it is not a tty — and behind the tee below it
# never is. Without this, the log stays empty until the process exits, so a run
# that is working fine looks hung.
export PYTHONUNBUFFERED=1

WIN=/mnt/e/AI/AI_Agent/workplace/hk-labour-triage
WSL=~/projects/hk-labour-triage

if [ $# -eq 0 ]; then
    echo "usage: bash run.sh \"<shell command>\""
    echo "example: bash run.sh \"python train.py --eval-only --no-adapter --no-quant --base Qwen/Qwen3-1.7B\""
    exit 2
fi

CMD="$*"
mkdir -p "$WIN/logs"
LOG="$WIN/logs/$(date +%m%d-%H%M%S).log"
LATEST="$WIN/logs/latest.log"

{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "\$ $CMD"
    echo

    # Sync code from the E: drive into the WSL working copy. data/ and models/
    # live only on the WSL side, and `cp` merges rather than replaces, so they
    # are left untouched.
    mkdir -p "$WSL"
    cp -r "$WIN/." "$WSL/" 2>/dev/null

    cd "$WSL" || exit 1
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "[run.sh] python: $(command -v python)  ($(python --version 2>&1))"
    echo "[run.sh] cwd: $(pwd)"
    echo

    eval "$CMD"
    rc=$?
    echo
    echo "=== exit code: $rc ==="
    exit $rc
} 2>&1 | tee "$LOG"

# Stable name so the reader always knows where to look (the glob tool does not
# index this logs/ directory, so "newest file" is not discoverable).
cp -f "$LOG" "$LATEST" 2>/dev/null || true

echo
echo "log written -> $LOG"
echo "latest      -> $LATEST"
