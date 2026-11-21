#!/usr/bin/env bash
# Live dry run: Alice x calendar test tasks, arms C0-C4 (oracle demos).
set -u
cd "$(dirname "$0")/.."
T="calendar_5 calendar_6 calendar_7"
A=adapters/oracle/C3/alice/calendar/seed0
run() { uv run python -m agent.run --persona alice --tasks $T --base http://127.0.0.1:8918 --max-steps 24 "$@" 2>&1 | grep -v -i warn; }
run --out results/online/oracle/C0
run --context-file results/contexts/oracle_alice_calendar_C1.md --out results/online/oracle/C1
run --context-file results/contexts/oracle_alice_calendar_C2.md --out results/online/oracle/C2
run --adapter $A --out results/online/oracle/C3
run --adapter $A --context-file results/contexts/oracle_alice_calendar_C2.md --out results/online/oracle/C4
