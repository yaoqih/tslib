# Stage2 Topk Headreg Matrix

## Goal

This stage keeps the current backbone and only studies the loss behavior around the current aggressive mainline:

- mainline family: `stage2topheavy_topk`
- fixed scope: `2021-2024`
- final target: tradable `top1` next-two-open return

The purpose is not to find a single-year peak.

The purpose is to find a parameter region that:

- improves or at least preserves `top1_mean`
- does not damage `top1_sharpe`
- does not create worse head concentration
- remains acceptable across `2021-2024`

## What Is Fixed

These parts are held fixed:

- backbone structure
- feature cache
- rolling fold protocol
- dataset filtering
- stage1 objective family
- cross-section batch mode

Base training family:

- stage1:
  - regression
  - pairwise rank
- stage2:
  - pairwise rank
  - top-k listwise
  - optional head concentration penalty
  - optional static-bias surrogate penalty

## Matrix

Years:

- `2021`
- `2022`
- `2023`
- `2024`

Grid:

- `stage2_topk_weight`: `0.03`, `0.05`, `0.08`
- `stage2_head_concentration_weight`: `0.00`, `0.01`, `0.02`
- `stage2_static_bias_weight`: `0.00`, `0.01`, `0.02`

Default total grid size:

- `3 * 3 * 3 * 4 = 108` runs

## Evaluation Priority

Primary:

- `top1_mean`
- `top1_sharpe`

Secondary:

- `top3_mean`
- `top5_mean`

Risk diagnostics:

- `unique`
- `max_rep`
- `debias015_top1_mean`

Interpretation diagnostics:

- `ic`
- `rank_ic`

## Decision Rule

A configuration should not be promoted just because it wins in one year.

The first pass filter should be:

1. average `top1_mean` on `2021-2024` is not below current mainline
2. average `top1_sharpe` on `2021-2024` is not below current mainline
3. `2022` does not collapse materially versus current mainline
4. `max_rep` does not become materially worse

If no configuration passes all four, do not promote any new loss combination.

## Script

Script:

- [scripts/market_daily/run_stage2_topk_headreg_matrix.py](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/scripts/market_daily/run_stage2_topk_headreg_matrix.py)

Supported subcommands:

- `manifest`
- `launch`
- `summarize`

## Typical Usage

Generate the default matrix manifest:

```bash
python scripts/market_daily/run_stage2_topk_headreg_matrix.py manifest
```

Launch on `gpu0,gpu1,gpu2,gpu3`:

```bash
python scripts/market_daily/run_stage2_topk_headreg_matrix.py launch --gpus 0,1,2,3
```

Summarize completed runs:

```bash
python scripts/market_daily/run_stage2_topk_headreg_matrix.py summarize
```

## Outputs

Default output directory:

- `logs/stage2_topk_headreg_matrix`

Main files:

- `manifest.csv`
- `manifest.json`
- `run_logs/*.log`
- `summary_raw.csv`
- `summary_vs_mainline.csv`
- `summary_vs_headreg.csv`
- `summary_aggregate.csv`

## Notes

- The script compares each run against existing corrected-cache baselines when those artifacts already exist:
  - `stage2topheavy_topk_lossstop_fullyear`
  - `headreg_lossstop_fullyear`
- The script does not require changing the backbone.
- The script is intended for loss research only.
