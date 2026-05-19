#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/market_live_strategy.json}"
MODE="${1:-infer}"
AS_OF_DATE="${2:-$(date +%F)}"
FOLD_YEAR="${3:-}"

if [[ -z "${FOLD_YEAR}" ]]; then
  FOLD_YEAR="$(date -d "${AS_OF_DATE}" +%Y 2>/dev/null || python - <<'PY'
from datetime import datetime
print(datetime.utcnow().year)
PY
)"
fi

case "$MODE" in
  train)
    "$PYTHON_BIN" scripts/market_daily/prod_train.py \
      --config "$CONFIG_PATH" \
      --fold_years "$FOLD_YEAR"
    ;;
  infer)
    "$PYTHON_BIN" scripts/market_daily/prod_infer.py \
      --config "$CONFIG_PATH" \
      --as_of_date "$AS_OF_DATE" \
      --fold_year "$FOLD_YEAR"
    "$PYTHON_BIN" scripts/market_daily/prod_select_strategy.py \
      --config "$CONFIG_PATH" \
      --as_of_date "$AS_OF_DATE" \
      --fold_year "$FOLD_YEAR"
    ;;
  replay)
    "$PYTHON_BIN" scripts/market_daily/prod_replay.py \
      --config "$CONFIG_PATH" \
      --start_date "${4:-${FOLD_YEAR}-01-01}" \
      --end_date "$AS_OF_DATE" \
      --fold_years "$FOLD_YEAR"
    ;;
  all)
    "$PYTHON_BIN" scripts/market_daily/prod_train.py \
      --config "$CONFIG_PATH" \
      --fold_years "$FOLD_YEAR"
    "$PYTHON_BIN" scripts/market_daily/prod_infer.py \
      --config "$CONFIG_PATH" \
      --as_of_date "$AS_OF_DATE" \
      --fold_year "$FOLD_YEAR"
    "$PYTHON_BIN" scripts/market_daily/prod_select_strategy.py \
      --config "$CONFIG_PATH" \
      --as_of_date "$AS_OF_DATE" \
      --fold_year "$FOLD_YEAR"
    ;;
  *)
    echo "Usage: $0 [train|infer|replay|all] [AS_OF_DATE] [FOLD_YEAR]"
    exit 1
    ;;
esac
