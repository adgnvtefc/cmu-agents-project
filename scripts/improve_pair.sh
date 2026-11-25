#!/usr/bin/env bash
# Train and evaluate the five C3 improvements (A1-A5) plus baselines on one dev pair.
#   bash scripts/improve_pair.sh alice calendar
# Needs: servers on :8917/:8918, oracle demos incl. dev split, C3 adapter for the pair.
# Each stage skips work that already exists, so the script can be re-run after a failure.
set -u
cd "$(dirname "$0")/.."
P=$1; S=$2; ROOT=data/oracle; M=models/UI-TARS-1.5-7B-mlx-bf16
PAIR=${P}_${S}
A=adapters/oracle
DEV=$(uv run python -c "from apps.tasks import task_ids; print(' '.join(t for t in task_ids('dev') if t.startswith('$S')))")
SYN=$(uv run python -c "from apps.tasks import synthetic_ids; print(' '.join(synthetic_ids('$S')))")
CTX=results/contexts/oracle_${PAIR}_C1.md
mkdir -p results/contexts
[ -f $CTX ] || uv run python -m agent.context $ROOT $P $S > $CTX
log() { echo "[$(date +%H:%M)] $*"; }

log "== $PAIR: prerequisites (C3, C5)"
uv run python -m train.train_adapters --condition C3 --demos-root $ROOT --personas $P --skills $S --epochs 4 2>&1 | grep -v -i warn
uv run python -m train.train_adapters --condition C5 --demos-root $ROOT --personas $P --skills $S --epochs 4 2>&1 | grep -v -i warn

log "== A1 teacher rollouts (C3 + demos in prompt) on synthetic tasks"
TDIR=results/teacher/oracle/$P/$S
if [ ! -d $TDIR ]; then
  uv run python -m agent.run --persona $P --tasks $SYN --base http://127.0.0.1:8918 --max-steps 24 \
    --adapter $A/C3/$P/$S/seed0 --context-file $CTX --out $TDIR 2>&1 | grep -v -i warn
fi

for X in A1 A2 A3 A4 A5; do
  log "== train $X"
  uv run python -m train.improve --approach $X --persona $P --skill $S --demos-root $ROOT 2>&1 | grep -v -i warn
done

log "== evaluate on dev (offline + online)"
for C in C0 C1 C3 A1 A2 A3 A4 A5; do
  case $C in
    C0) ARGS=(); RARGS=() ;;
    C1) ARGS=(--context-from $ROOT); RARGS=(--context-file $CTX) ;;
    *)  [ -f $A/$C/$P/$S/seed0/adapters.safetensors ] || { log "skip $C (no adapter)"; continue; }
        ARGS=(--adapter $A/$C/$P/$S/seed0); RARGS=(--adapter $A/$C/$P/$S/seed0) ;;
  esac
  OFF=results/offline_dev/oracle/$C/$PAIR.jsonl
  [ -s $OFF ] || uv run python -m eval.offline --model $M --demos $ROOT/$P --skill $S --split dev ${ARGS[@]+"${ARGS[@]}"} --out $OFF 2>&1 | grep -v -i warn | tail -1
  ON=results/online_dev/oracle/$PAIR/$C
  [ -d $ON ] || uv run python -m agent.run --persona $P --tasks $DEV --base http://127.0.0.1:8918 --max-steps 24 ${RARGS[@]+"${RARGS[@]}"} --out $ON 2>&1 | grep -v -i warn
  log "done $C"
done

log "== summary"
uv run python -m eval.summarize results/offline_dev/oracle --demos-root $ROOT
uv run python -m eval.summarize_online results/online_dev/oracle/$PAIR
