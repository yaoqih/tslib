#!/usr/bin/env bash
set -euo pipefail
cd /huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2
PY=/huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python
CFG=configs/market_live_strategy.json
OUT=/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/logs/fold_rerun_balanced_4gpu_run
mkdir -p "$OUT"
python - <<'PY2'
import json
from pathlib import Path
p=Path('/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/logs/fold_rerun_balanced_4gpu_run/job_manifest.json')
items=json.loads(p.read_text())
for gpu in sorted(set(item['gpu'] for item in items)):
    gpu_items=[it for it in items if it['gpu']==gpu]
    gpu_path=Path(f'/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/logs/fold_rerun_balanced_4gpu_run/gpu{gpu}_jobs.json')
    gpu_path.write_text(json.dumps(gpu_items, ensure_ascii=False, indent=2))
    print(gpu_path)
PY2
