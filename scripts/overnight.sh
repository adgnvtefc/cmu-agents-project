#!/usr/bin/env bash
# Full study batch on one demo set (default: human demos in data/raw). Not started automatically.
#   bash scripts/overnight.sh [data/raw]
# Requires: server on :8917 (for online eval), models/UI-TARS-1.5-7B-mlx-bf16.
# Rough time on M5 Ultra: training ~10 h (C3 12x25 min, C5 12x45 min, C6 ~1.5 h), offline ~2 h,
# online ~6 h. Run stages separately if needed: each stage skips work that already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${1:-data/raw}
SET=$(basename "$ROOT")
M=models/UI-TARS-1.5-7B-mlx-bf16
PERSONAS="alice bob carol"
SKILLS="send_email file_email calendar organize_files"

echo "== 0. sanity: recordings consistent"
uv run python -m scripts.check_demos "$ROOT" | tail -1

echo "== 1. C2 skill files"
uv run python -m agent.skillfile --demos-root "$ROOT"

echo "== 2. adapters"
for C in C3 C5 C6; do uv run python -m train.train_adapters --condition $C --demos-root "$ROOT" --epochs 4; done

echo "== 3. offline eval (test demos)"
for P in $PERSONAS; do for S in $SKILLS; do
  out=results/offline/$SET
  uv run python -m eval.offline --model $M --demos $ROOT/$P --skill $S --split test --out $out/C0/${P}_$S.jsonl
  uv run python -m eval.offline --model $M --demos $ROOT/$P --skill $S --split test --context-from $ROOT --out $out/C1/${P}_$S.jsonl
  uv run python -m eval.offline --model $M --demos $ROOT/$P --skill $S --split test --skills-dir skills/$SET --out $out/C2/${P}_$S.jsonl
  uv run python -m eval.offline --model $M --demos $ROOT/$P --skill $S --split test --adapter adapters/$SET/C3/$P/$S/seed0 --out $out/C3/${P}_$S.jsonl
  uv run python -m eval.offline --model $M --demos $ROOT/$P --skill $S --split test --adapter adapters/$SET/C3/$P/$S/seed0 --skills-dir skills/$SET --out $out/C4/${P}_$S.jsonl
  uv run python -m eval.offline --model $M --demos $ROOT/$P --skill $S --split test --adapter adapters/$SET/C5/$P/$S/seed0 --out $out/C5/${P}_$S.jsonl
  uv run python -m eval.offline --model $M --demos $ROOT/$P --skill $S --split test --adapter adapters/$SET/C6/ALL/ALL/seed0 --out $out/C6/${P}_$S.jsonl
done; done
uv run python -m eval.summarize results/offline/$SET --demos-root "$ROOT"

echo "== 4. online eval: see scripts/run_online_dryrun.sh for the per-arm pattern (C7 swap and"
echo "   per-arm contexts need persona-specific files; run after reviewing offline results)"
