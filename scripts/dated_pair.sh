#!/usr/bin/env bash
# Rerun of the C3 study with today's date in every prompt (PROMPT_DATE=1), plus A2b (A2 + SFT anchor).
#   bash scripts/dated_pair.sh alice calendar
# Retrains C3 with the dated prompt so train and test prompts match, trains A2b from it, then
# evaluates C0, C1, C3, A2b on the dev split. Uses the same oracle demos through the alias
# data/oracle_dated, so adapters/results land under oracle_dated/ and never overwrite earlier runs.
set -u
cd "$(dirname "$0")/.."
export PROMPT_DATE=1
P=$1; S=$2; M=models/UI-TARS-1.5-7B-mlx-bf16
[ -e data/oracle_dated ] || ln -s oracle data/oracle_dated
ROOT=data/oracle_dated; PAIR=${P}_${S}; A=adapters/oracle_dated
DEV=$(uv run python -c "from apps.tasks import task_ids; print(' '.join(t for t in task_ids('dev') if t.startswith('$S')))")
CTX=results/contexts/oracle_${PAIR}_C1.md
mkdir -p results/contexts
[ -f $CTX ] || uv run python -m agent.context data/oracle $P $S > $CTX
log() { echo "[$(date +%H:%M)] $*"; }

log "== $PAIR: train C3 (dated prompt)"
uv run python -m train.train_adapters --condition C3 --demos-root $ROOT --personas $P --skills $S --epochs 4 2>&1 | grep -v -i warn
log "== train A2b (ORPO from dated C3)"
uv run python -m train.improve --approach A2b --persona $P --skill $S --demos-root $ROOT 2>&1 | grep -v -i warn

log "== evaluate on dev (offline + online)"
for C in C0 C1 C3 A2b; do
  case $C in
    C0) ARGS=(); RARGS=() ;;
    C1) ARGS=(--context-from data/oracle); RARGS=(--context-file $CTX) ;;
    *)  [ -f $A/$C/$P/$S/seed0/adapters.safetensors ] || { log "skip $C (no adapter)"; continue; }
        ARGS=(--adapter $A/$C/$P/$S/seed0); RARGS=(--adapter $A/$C/$P/$S/seed0) ;;
  esac
  OFF=results/offline_dev/oracle_dated/$C/$PAIR.jsonl
  [ -s $OFF ] || uv run python -m eval.offline --model $M --demos data/oracle/$P --skill $S --split dev ${ARGS[@]+"${ARGS[@]}"} --out $OFF 2>&1 | grep -v -i warn | tail -1
  ON=results/online_dev/oracle_dated/$PAIR/$C
  [ -d $ON ] || uv run python -m agent.run --persona $P --tasks $DEV --base http://127.0.0.1:8918 --max-steps 24 ${RARGS[@]+"${RARGS[@]}"} --out $ON 2>&1 | grep -v -i warn
  log "done $C"
done

log "== summary"
uv run python -m eval.summarize results/offline_dev/oracle_dated --demos-root data/oracle
uv run python -m eval.summarize_online results/online_dev/oracle_dated/$PAIR
