# Loss Upgrade Experiment Plan

## Goal

在不改动主干时序模型结构的前提下，为 A 股 `Top1` 选股任务加入 `Huber + BCE` 多任务训练目标，验证是否能提升正式交易口径下的：

- `Top1` 日均收益
- `Sharpe`
- 跨年份稳定性

## Design

### Model Strategy

- 保持 backbone 不变：
  - `DLinear seq_len=120`
  - `iTransformer seq_len=20`
- 在训练层外加一个通用分类头：
  - 输入：模型预测出的完整 `pred_len x channel` 输出
  - 输出：`pred_len x 1` 的分类 logit
- 回归头仍使用原预测值，不改测试与收益评估逻辑

### Loss Strategy

- 回归目标：`label = open_{t+2}/open_{t+1} - 1`
- 分类目标：`label_cls = 1(label > 0)`
- 总损失：

```text
loss = huber_loss + alpha * bce_loss
```

- 待搜索 `alpha`：
  - `0.2`
  - `0.5`
  - `1.0`

### Experiment Scope

优先覆盖主模型与副模型的强区/弱区：

- `DLinear sl120`:
  - `2019`
  - `2020`
  - `2021`
  - `2023`
- `iTransformer sl20`:
  - `2015`
  - `2021`
  - `2022`
  - `2024`

总实验数：

- `DLinear`: `4 folds x 3 alpha = 12`
- `iTransformer`: `4 folds x 2 alpha = 8`
- 合计：`20`

说明：

- `iTransformer` 第一轮只跑 `alpha=0.2, 0.5`
- 若 `0.5` 明显优于基线，再补 `alpha=1.0`

## GPU Schedule

### Wave-L1

| GPU | 任务 |
| --- | --- |
| `GPU0` | `DLinear sl120, alpha=0.2` |
| `GPU1` | `DLinear sl120, alpha=0.5` |
| `GPU2` | `DLinear sl120, alpha=1.0` |
| `GPU3` | `iTransformer sl20, alpha=0.2` |

### Wave-L2

| GPU | 任务 |
| --- | --- |
| `GPU0` | `iTransformer sl20, alpha=0.5` |
| `GPU1-3` | 预留给补跑 / smoke / 第二波扩展 |

## Success Criteria

一组配置被视为值得扩大到全折，至少满足：

- 相比原基线同年份 `mean_return` 提升
- `Sharpe` 不显著恶化
- 至少在 `4` 个代表 fold 中有 `3` 个方向不差于基线

## Output Paths

- 代码改动：
  - `exp/exp_long_term_forecasting.py`
  - `data_provider/data_loader.py`
  - `run.py`
  - `scripts/market_daily/launch_round1.py`
  - `tests/test_market_research.py`
- 日志：
  - `logs/market_loss_dlinear120_hbce_a02/`
  - `logs/market_loss_dlinear120_hbce_a05/`
  - `logs/market_loss_dlinear120_hbce_a10/`
  - `logs/market_loss_itransformer20_hbce_a02/`
  - `logs/market_loss_itransformer20_hbce_a05/`

## Final Results

截至 `2026-05-17 06:51 UTC`，全部 `20` 个实验已完成，`0` 失败。

### Per-fold Result Table

| Model | Alpha | Year | mean_return | mean_delta vs baseline | sharpe | sharpe_delta vs baseline |
| --- | --- | --- | --- | --- | --- | --- |
| `DLinear` | `0.2` | `2019` | `0.013542` | `+0.000835` | `1.5153` | `+0.1015` |
| `DLinear` | `0.2` | `2020` | `0.011627` | `-0.002410` | `1.4537` | `-0.2970` |
| `DLinear` | `0.2` | `2021` | `0.016154` | `+0.011880` | `1.5922` | `+0.8649` |
| `DLinear` | `0.2` | `2023` | `0.003838` | `-0.000774` | `1.4053` | `-0.2113` |
| `DLinear` | `0.5` | `2019` | `0.011165` | `-0.001543` | `1.2446` | `-0.1693` |
| `DLinear` | `0.5` | `2020` | `0.011620` | `-0.002418` | `1.3976` | `-0.3531` |
| `DLinear` | `0.5` | `2021` | `0.023407` | `+0.019133` | `2.0917` | `+1.3644` |
| `DLinear` | `0.5` | `2023` | `0.003327` | `-0.001285` | `1.0814` | `-0.5352` |
| `DLinear` | `1.0` | `2019` | `0.002837` | `-0.009871` | `0.3190` | `-1.0949` |
| `DLinear` | `1.0` | `2020` | `0.009242` | `-0.004795` | `1.1119` | `-0.6388` |
| `DLinear` | `1.0` | `2021` | `0.018610` | `+0.014337` | `1.6027` | `+0.8754` |
| `DLinear` | `1.0` | `2023` | `0.006706` | `+0.002093` | `1.3197` | `-0.2969` |
| `iTransformer` | `0.2` | `2015` | `0.002228` | `-0.004764` | `0.4807` | `-0.9712` |
| `iTransformer` | `0.2` | `2021` | `0.002902` | `-0.011330` | `0.3410` | `-1.4785` |
| `iTransformer` | `0.2` | `2022` | `-0.002004` | `-0.005130` | `-0.4281` | `-1.0961` |
| `iTransformer` | `0.2` | `2024` | `0.005960` | `-0.004546` | `0.7462` | `-0.5744` |
| `iTransformer` | `0.5` | `2015` | `0.004435` | `-0.002557` | `0.9026` | `-0.5492` |
| `iTransformer` | `0.5` | `2021` | `0.003410` | `-0.010822` | `0.4130` | `-1.4064` |
| `iTransformer` | `0.5` | `2022` | `-0.005747` | `-0.008873` | `-1.2749` | `-1.9429` |
| `iTransformer` | `0.5` | `2024` | `0.012820` | `+0.002314` | `1.6435` | `+0.3229` |

### Aggregated Summary

| Model | Alpha | Folds | Avg mean_return | Avg mean_delta | Avg sharpe | Avg sharpe_delta | Positive mean_delta folds | Positive sharpe_delta folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DLinear` | `0.2` | `4` | `0.011290` | `+0.002383` | `1.4916` | `+0.1145` | `2/4` | `2/4` |
| `DLinear` | `0.5` | `4` | `0.012380` | `+0.003472` | `1.4538` | `+0.0767` | `1/4` | `1/4` |
| `DLinear` | `1.0` | `4` | `0.009349` | `+0.000441` | `1.0883` | `-0.2888` | `2/4` | `1/4` |
| `iTransformer` | `0.2` | `4` | `0.002272` | `-0.006442` | `0.2849` | `-1.0300` | `0/4` | `0/4` |
| `iTransformer` | `0.5` | `4` | `0.003729` | `-0.004984` | `0.4211` | `-0.8939` | `1/4` | `1/4` |

## Decision

### What Worked

- `DLinear + Huber + BCE(alpha=0.2)` 是这一轮唯一值得保留并继续扩大的 loss 改造方向
- 它在 `2019` 和 `2021` 上都明显增强，尤其 `2021` 提升很大

### What Failed

- `iTransformer + Huber + BCE` 在代表 fold 上几乎全面退化
- `DLinear alpha=1.0` 明显过强，已经开始破坏主任务排序
- `DLinear alpha=0.5` 虽然均值看起来不差，但改善几乎全部集中在 `2021`，跨年份稳定性不够

### Recommended Next Wave

1. 只保留 `DLinear sl120 + alpha=0.2`
2. 把它扩大到 `2015-2024` 全部 folds
3. 在 `DLinear` 上再试：
   - `alpha=0.1`
   - `alpha=0.2`
   - `alpha=0.3`
4. 暂停 `iTransformer` 的 BCE 多任务扩展

## Full-Fold Expansion Results

随后已按正式 tradable-only 口径完成 `DLinear sl120` 的全折扩展：

- `alpha=0.1`
- `alpha=0.2`
- `alpha=0.3`

共 `30` 个 fold，全部完成，`0` 失败。

### Full-fold Summary

| Alpha | Folds | Days | Mean Return | Cumulative Return | Sharpe | Win Rate | Max Drawdown | Positive Years | Avg mean delta vs DLinear baseline | Avg sharpe delta vs DLinear baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.1` | `10` | `2340` | `0.006059` | `45.2618` | `0.8831` | `0.4868` | `-0.9838` | `6/10` | `+0.001608` | `+0.0541` |
| `0.2` | `10` | `2340` | `0.004440` | `0.4513` | `0.6863` | `0.4859` | `-0.9936` | `4/10` | `-0.000470` | `-0.2226` |
| `0.3` | `10` | `2340` | `0.005225` | `5.5790` | `0.7932` | `0.4838` | `-0.9891` | `4/10` | `+0.000105` | `-0.1073` |

对照当前正式主 baseline：

- `DLinear sl120 baseline`：
  - `mean_return = 0.004779`
  - `cumulative_return = 4.6605`
  - `sharpe = 0.7905`

### Final Decision Update

全折结果推翻了“优先保留 `alpha=0.2`”这个中期判断。

最终结论应更新为：

1. `DLinear sl120 + Huber + BCE(alpha=0.1)` 是当前最优 loss 改造版本
2. `alpha=0.2` 在代表折上看起来不错，但扩到全折后整体退化
3. `alpha=0.3` 有一定收益弹性，但 Sharpe 提升不如 `alpha=0.1` 稳定

因此，后续若继续走 loss 改造线，只保留：

- `alpha=0.1` 作为主线
- `alpha=0.3` 作为次级对照

不再继续：

- `alpha=0.2`
- `iTransformer + BCE`
