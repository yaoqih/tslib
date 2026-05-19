import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.market_daily.prod_common import build_run_command, checkpoint_path, load_live_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train production live models while keeping research flow unchanged")
    parser.add_argument("--config", type=str, default="configs/market_live_strategy.json")
    parser.add_argument("--fold_years", type=str, default="")
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--output_manifest", type=str, default="logs/market_live_prod/train_manifest.json")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_live_config(args.config)
    strategy_models = set()
    strategy_models.update(config["strategies"][config["primary_strategy"]]["models"])
    for name in config.get("backup_strategies", []):
        strategy_models.update(config["strategies"][name]["models"])
    models = [item for item in (args.models.split(",") if args.models else sorted(strategy_models)) if item]
    fold_years = [int(item) for item in args.fold_years.split(",") if item] if args.fold_years else [pd.Timestamp.today().year]

    manifest = []
    for fold_year in fold_years:
        for model_key in models:
            ckpt = checkpoint_path(config, model_key, fold_year)
            command = build_run_command(
                config=config,
                model_key=model_key,
                fold_year=fold_year,
                is_training=True,
                python_bin=args.python,
                gpu=args.gpu,
            )
            item = {
                "fold_year": fold_year,
                "model_key": model_key,
                "checkpoint_path": str(ckpt),
                "command": command,
                "status": "pending",
            }
            if ckpt.exists() and not args.force:
                item["status"] = "skipped_existing"
            elif not args.dry_run:
                subprocess.run(command, cwd=str(ROOT), check=True)
                item["status"] = "done"
            manifest.append(item)

    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"output_manifest": str(output_path), "num_jobs": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    import pandas as pd

    main()
