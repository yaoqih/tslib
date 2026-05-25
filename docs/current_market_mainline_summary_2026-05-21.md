# Current Market Mainline Summary

## Scope

This document summarizes the corrected-cache full-year market experiments after the cache audit.

Evaluation basis:

- cache: `cache/market_daily_features_full2010.parquet`
- cache coverage: `2010-07-02` to `2026-05-12`
- protocol: intended `5-year train -> 1-year test`
- annual test folds: `2015-2024`
- final online-style metric: `top1` next-two-open tradable return

## Audit Result

The old conclusion was invalid.

Root cause:

- the old cache at `cache/market_daily_features.parquet` only covered from `2019-07-02`
- earlier so-called full-year comparisons therefore did not have the intended training history
- some previous cross-year comparisons were effectively using truncated train windows

What was fixed:

- file: [utils/market_research.py](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/utils/market_research.py)
- cache validation now rejects cached market features unless coverage reaches the requested `start_date`
- corrected-cache full-year reruns were completed for all `2015-2024` folds

## Compared Variants

- `headreg_lossstop_fullyear`
- `stage2topheavy_lossstop_fullyear`
- `stage2topheavy_topk_lossstop_fullyear`

## Full-Year Summary

Average over `2015-2024`:

| variant | top1_mean | top1_sharpe | top3_mean | top5_mean | ic | rank_ic | unique | max_rep | debias@0.15 top1_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `headreg_lossstop_fullyear` | 0.003622 | 1.050684 | 0.003267 | 0.002816 | 0.028838 | 0.030395 | 103.4 | 32.1 | 0.002242 |
| `stage2topheavy_lossstop_fullyear` | 0.003563 | 0.645016 | 0.003529 | 0.003489 | 0.034534 | 0.039344 | 120.5 | 31.2 | 0.004167 |
| `stage2topheavy_topk_lossstop_fullyear` | **0.006131** | **1.101297** | **0.006040** | **0.005321** | 0.029296 | 0.023968 | **88.0** | **46.2** | **0.006302** |

Average over recent folds `2021-2024`:

| variant | top1_mean | top1_sharpe | top3_mean | top5_mean | ic | rank_ic | unique | max_rep | debias@0.15 top1_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `headreg_lossstop_fullyear` | 0.002892 | 0.673004 | 0.003987 | 0.003370 | 0.030657 | 0.030742 | 114.25 | 25.25 | 0.000092 |
| `stage2topheavy_lossstop_fullyear` | 0.000555 | -0.028534 | 0.001092 | 0.001651 | 0.035165 | 0.039926 | 159.00 | 10.75 | 0.002509 |
| `stage2topheavy_topk_lossstop_fullyear` | **0.006008** | **0.734359** | **0.006339** | **0.005689** | 0.029897 | 0.024365 | 96.00 | 40.25 | **0.005821** |

Positive `top1_mean` years over `2015-2024`:

- `headreg_lossstop_fullyear`: `8 / 10`
- `stage2topheavy_lossstop_fullyear`: `6 / 10`
- `stage2topheavy_topk_lossstop_fullyear`: `8 / 10`

Year winners by `top1_mean`:

- `2015`: `stage2topheavy_topk_lossstop_fullyear`
- `2016`: `stage2topheavy_topk_lossstop_fullyear`
- `2017`: `headreg_lossstop_fullyear`
- `2018`: `stage2topheavy_topk_lossstop_fullyear`
- `2019`: `stage2topheavy_topk_lossstop_fullyear`
- `2020`: `stage2topheavy_lossstop_fullyear`
- `2021`: `stage2topheavy_topk_lossstop_fullyear`
- `2022`: `headreg_lossstop_fullyear`
- `2023`: `stage2topheavy_lossstop_fullyear`
- `2024`: `stage2topheavy_topk_lossstop_fullyear`

## Diagnosis

### 1. `stage2topheavy_topk` is the strongest top1 line

It is currently the best option if the objective is explicitly:

- maximize `top1` tradable return
- keep a top-focused training target
- accept some concentration risk in exchange for better head capture

Why:

- highest full-period average `top1_mean`
- highest full-period average `top1_sharpe`
- strongest `top3/top5` averages at the same time
- still `8 / 10` positive years on corrected-cache full-year folds

### 2. `headreg` is still the cleaner stability baseline

`headreg` remains the better control line for robustness diagnostics:

- better diversification than `stage2topheavy_topk`
- lower `max_rep`
- better recent `unique/max_rep` profile
- less dependence on aggressive head concentration

This makes `headreg` the right fallback baseline, but not the best final top1-seeking line.

### 3. `stage2topheavy` is not enough by itself

`stage2topheavy_lossstop_fullyear` has decent `ic/rank_ic`, but the conversion from ranking signal to final `top1` trade selection is not stable enough.

Interpretation:

- it often improves broad ranking quality
- it does not reliably transform that into the single-name decision required by the final metric

### 4. Main remaining problem is head concentration / static bias

`stage2topheavy_topk` wins on top1, but its diagnostics are still aggressive:

- lower `unique`
- higher `max_rep`
- debiasing at evaluation time still helps in several years

Interpretation:

- the model has learned some real top-focused signal
- but it still tends to over-allocate score mass to a smaller set of names
- part of that behavior is consistent with static preference / repeated over-promotion

## Formal Recommendation

Current formal recommendation:

- **current top1 mainline**: `stage2topheavy_topk_lossstop_fullyear`
- **stable baseline / regression guardrail**: `headreg_lossstop_fullyear`
- **do not promote** `stage2topheavy_lossstop_fullyear` as standalone mainline

This is the best current split:

- use `stage2topheavy_topk` as the active research and top1-optimization line
- keep `headreg` as the stability reference during future modifications

## Next Mainline Change

The next minimal training-side improvement should stay on the current `stage2topheavy_topk` line and only modify the loss:

1. `anti-concentration`
   - penalize excessive same-day softmax concentration on the top score
2. `anti-static-bias surrogate`
   - penalize an overly large `top1 score - head mean score` gap

These are intentionally:

- same-day only
- stateless
- no cross-batch memory
- no leakage risk
- simple to audit

## 2021 Matrix Update

We also ran a 2021-only sweep over `stage2_topk_weight / head_concentration_weight / static_bias_weight`.

Current 2021 interpretation:

- **2021 top1 candidate**: `s2hg_tw080_hc000_sb020_fy`
- **2021 balanced reference**: `s2hg_tw030_hc000_sb000_fy`
- **2021 high-return upper bound**: `s2hg_tw080_hc020_sb000_fy`

Why:

- `s2hg_tw080_hc000_sb020_fy` keeps strong `top1` while avoiding the most extreme concentration profile
- `s2hg_tw080_hc020_sb000_fy` has the highest `top1`, but it is more concentrated
- the old mainline is still materially weaker on 2021 `top1`

This 2021 result is a local reference only. Final promotion still needs cross-year confirmation.

## Result Sources

- logs: [logs/full_year_matrix](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/logs/full_year_matrix)
- outputs: [test_results](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/test_results)
- cache logic: [utils/market_research.py](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/utils/market_research.py)

## 2021-2024 Backbone Baseline Matrix

We also ran the same corrected-cache baseline on `PatchTST / iTransformer / TimeMixer / TimesNet / WPMixer` across `2021-2024`.

### Year winners

- 2021: `TimesNet` `top1_mean=0.041299`
- 2022: `TimesNet` `top1_mean=0.002906`
- 2023: `WPMixer` `top1_mean=0.010352`
- 2024: `WPMixer` `top1_mean=0.078408`

### Cross-year summary

- best average `top1_mean`: `WPMixer` `0.029809`
- best average `top1_sharpe`: `TimesNet` `1.362940`
- best average `ic`: `TimesNet` `0.015269`
- most balanced `unique/max_rep`: `iTransformer` is relatively stable, but not the top return winner

### Takeaway

- backbone choice matters, but no single backbone dominates all years
- `TimesNet` is the strongest stability reference
- `WPMixer` is the strongest top1-return reference
- these baselines are useful guardrails, but they do not replace the current `stage2topheavy_topk` mainline
