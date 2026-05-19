# 市场实盘最小闭环说明

## 目标

这一层代码的目标是：

- 冻结一套可实盘的策略配置
- 保留现有研究脚本和研究路径不变
- 在研究层之外，补一套最小可运行的生产入口

也就是说：

- 研究脚本继续用于试模型、扫参数、做组合实验
- 生产脚本只负责训练、每日推理、策略切换和回放审计

## 冻结策略集合

配置文件：

- `configs/market_live_strategy.json`

当前冻结的实盘策略为：

- `primary_live_v1`
  - 组合：`wpmixer + timemixer + timesnet`
  - 方法：`zscore_mean`
- `backup_live_v1`
  - 组合：`wpmixer + timesnet`
  - 方法：`mean`
- `legacy_backup_v1`
  - 组合：`itransformer + patchtst`
  - 方法：`gated_q50_left`

## 配置文件说明

主配置文件：

- `configs/market_live_strategy.json`

建议把它理解成“实盘冻结层”的唯一事实来源。

### 顶层字段

- `parquet_path`
  - 原始市场数据路径
- `cache_path`
  - 特征缓存路径
- `checkpoints_dir`
  - 模型 checkpoint 保存目录
- `test_results_dir`
  - 预测结果目录，里面会有 `top1_predictions.csv`
- `logs_dir`
  - 生产层日志与状态输出目录
- `market_start_year`
  - 市场数据起始年份
- `market_min_history`
  - 最少上市交易日过滤
- `market_min_avg_amount`
  - 最少 20 日平均成交额过滤
- `train_epochs`
  - 训练轮数上限
- `patience`
  - early stop patience
- `batch_size`
  - 训练 batch size
- `num_workers`
  - dataloader worker 数
- `learning_rate`
  - 学习率
- `loss`
  - 当前冻结训练损失，默认 `Huber`
- `huber_delta`
  - `Huber` 损失的 delta

### 策略字段

- `primary_strategy`
  - 当前主策略名
- `backup_strategies`
  - 备份策略列表
- `strategies`
  - 每条策略的具体定义

每条策略至少有：

- `models`
  - 用到的模型 key 列表
- `kind`
  - 可选：
    - `combo`
    - `selector`
    - `gated`

不同 `kind` 的附加字段：

- `combo`
  - `method`
    - 可选如 `mean / rank_mean / zscore_mean`
- `selector`
  - `method`
    - 当前常用 `top1_gap`
- `gated`
  - `quantile`
    - 置信度门控分位数
  - `fallback_source`
    - `left` 或 `right`

### 模型字段

位置：

- `models`

每个模型配置包含：

- `model`
  - 模型名，如 `WPMixer / TimeMixer / TimesNet`
- `seq_len`
  - 输入窗口长度
- `feature_set`
  - 特征集合编号
- `d_model / d_ff / e_layers / d_layers / n_heads`
  - 主网络结构参数
- `factor`
  - 模型内部 factor
- `dropout`
  - dropout
- `embed`
  - 时间编码方式
- `des`
  - 冻结实验版本标识，影响 checkpoint / result 路径匹配
- `use_amp`
  - 是否启用 AMP
- `extra_args`
  - 某些模型的附加参数，如 `TimeMixer` 的下采样参数

### 成本字段

位置：

- `scenarios`

当前定义了三种场景：

- `low`
- `base`
- `high`

每个场景都有：

- `buy_cost_bps`
- `sell_cost_bps`

### 自动切换字段

位置：

- `switch_rules`

关键字段：

- `scenario`
  - 用哪种成本场景决定主备切换
- `lookback_days`
  - 最近多少天作为切换判断窗口
- `min_hold_days`
  - 最小持有天数，防止频繁切换
- `min_excess_mean_return`
  - challenger 超过当前策略的最小收益门槛
- `min_excess_sharpe`
  - challenger 超过当前策略的最小 Sharpe 门槛
- `state_path`
  - 当前激活策略状态文件路径

## 入口脚本

### 1. 训练入口

```bash
python scripts/market_daily/prod_train.py \
  --config configs/market_live_strategy.json \
  --fold_years 2024
```

作用：

- 按冻结配置训练主策略和备份策略涉及的模型
- 复用现有 `run.py`
- 不修改研究脚本逻辑

常用参数：

- `--config`
  - 指定生产配置文件
- `--fold_years`
  - 指定训练哪些滚动 fold，例如 `2023,2024`
  - 这里的 `fold_year` 表示“测试年/交易年”
  - 例如 `fold_year=2024` 表示：
    - 训练窗口使用 `2019-2023` 这 5 年
    - 在 `2024` 年上做预测、选股和收益评估
  - 对应当前研究口径：`5 年训练 -> 1 年测试` 的滚动循环
- `--models`
  - 只训练指定模型，例如 `wpmixer,timemixer`
- `--gpu`
  - 指定 GPU
- `--python`
  - 指定 Python 解释器
- `--force`
  - 即使 checkpoint 已存在也强制重训
- `--dry_run`
  - 只生成 manifest，不实际训练
- `--output_manifest`
  - 训练任务清单输出位置

### 2. 每日推理入口

```bash
python scripts/market_daily/prod_infer.py \
  --config configs/market_live_strategy.json \
  --as_of_date 2024-09-02 \
  --fold_year 2024
```

作用：

- 生成指定日期可用的日信号
- 输出主策略和备份策略的候选股
- 输出 `top20` 排名文件和最终 `daily_signal.json`

常用参数：

- `--config`
  - 指定生产配置文件
- `--as_of_date`
  - 以哪一天为截止日期做推理
- `--fold_year`
  - 使用哪个年度 fold 的模型
  - 含义同上，`2024` 就是“用 2019-2023 训练得到、专门服务 2024 年交易日”的那套模型
- `--strategy_names`
  - 只跑指定策略，逗号分隔
- `--gpu`
  - 指定 GPU
- `--python`
  - 指定 Python 解释器
- `--force_refresh`
  - 强制刷新预测，不复用已有 `top1_predictions.csv`
- `--output_dir`
  - 每日信号输出目录

说明：

- 当前最小闭环版本会在已有预测结果基础上，找 `as_of_date` 之前最后一个所有模型都有共同覆盖的交易日
- 如果后续接上真正的最新日数据推理，这里可以进一步升级为“当天直接出信号”

### 3. 自动策略切换入口

```bash
python scripts/market_daily/prod_select_strategy.py \
  --config configs/market_live_strategy.json \
  --as_of_date 2024-09-02 \
  --fold_year 2024
```

作用：

- 根据最近窗口的 `base cost live proxy` 表现
- 决定当前激活哪条策略
- 输出当日策略决策文件

常用参数：

- `--config`
  - 指定生产配置文件
- `--as_of_date`
  - 用哪一天作为切换判断日
- `--fold_year`
  - 读取哪一年的预测结果
  - 即读取对应测试年那一套滚动结果；例如 `2024` 表示读取 `2024` 年交易日对应的预测与回放记录
- `--strategy_names`
  - 只在指定策略集合里做切换
- `--output_json`
  - 决策文件输出路径

### 4. 准实盘回放入口

```bash
python scripts/market_daily/prod_replay.py \
  --config configs/market_live_strategy.json \
  --start_date 2024-01-01 \
  --end_date 2024-09-02 \
  --fold_years 2024
```

作用：

- 基于已保存的 `top1_predictions.csv`
- 对主策略与备份策略做顺序回放
- 输出不同成本场景下的回放汇总

常用参数：

- `--config`
  - 指定生产配置文件
- `--start_date`
  - 回放起始日期
- `--end_date`
  - 回放结束日期
- `--strategy_names`
  - 指定回放哪些策略
- `--fold_years`
  - 指定用哪些 fold 结果做拼接
  - 适合跨多年回放时使用，例如 `2021,2022,2023,2024`
  - 程序会按年份去读取各自那一年的滚动模型结果，再顺序拼成连续的准实盘轨迹
- `--output_dir`
  - 回放输出目录

## 一键执行

一键入口脚本：

- `scripts/market_daily/run_live_pipeline.sh`

用法：

```bash
bash scripts/market_daily/run_live_pipeline.sh infer 2024-09-02 2024
bash scripts/market_daily/run_live_pipeline.sh train 2024-09-02 2024
bash scripts/market_daily/run_live_pipeline.sh replay 2024-09-02 2024
bash scripts/market_daily/run_live_pipeline.sh all 2024-09-02 2024
```

含义：

- `train`
  - 训练冻结模型
- `infer`
  - 生成每日信号
  - 再执行自动策略切换
- `replay`
  - 执行准实盘回放
- `all`
  - 训练 + 推理 + 自动切换

脚本参数说明：

- 第 1 个参数：模式
  - `train / infer / replay / all`
- 第 2 个参数：`AS_OF_DATE`
  - 例如 `2024-09-02`
- 第 3 个参数：`FOLD_YEAR`
  - 例如 `2024`
  - 含义是“当前操作的测试年”，不是训练起始年
  - 对应训练窗口会由脚本按研究规则自动回推 5 年
- `replay` 模式下第 4 个参数可以额外指定回放起始日期

补充理解：

- `fold_year`
  - 单个年度滚动窗口
  - 常用于 `infer / select_strategy`
- `fold_years`
  - 多个年度滚动窗口列表
  - 常用于 `train / replay`
- 当前研究从 `2010` 年之后开始看，因此如果严格按 `5 年训练 -> 1 年测试`，最早可用的完整 `fold_year` 通常是 `2015`

## 输出文件

### 每日推理输出

- `logs/market_live_prod/daily/...json`
- `logs/market_live_prod/daily/...top20.csv`

### 策略切换输出

- `logs/market_live_prod/strategy_decision_YYYY-MM-DD.json`
- `logs/market_live_prod/active_strategy_state.json`

### 回放输出

- `logs/market_live_prod/replay/...csv`
- `logs/market_live_prod/replay/...json`

### 训练输出

- `logs/market_live_prod/train_manifest.json`

## 自动切换规则

配置位置：

- `configs/market_live_strategy.json` 中的 `switch_rules`

当前逻辑：

1. 取最近 `lookback_days` 窗口
2. 固定使用 `base` 成本场景做策略比较
3. 默认保持当前激活策略不变
4. 只有 challenger 同时满足以下条件，才允许切换：
   - `mean_return` 超过当前策略至少 `min_excess_mean_return`
   - `sharpe` 超过当前策略至少 `min_excess_sharpe`
5. 同时满足最小持有期：
   - `min_hold_days`

这样做的目的，是避免因为短期噪声导致频繁切主策略。

如果你想让切换更保守：

- 提高 `min_hold_days`
- 提高 `min_excess_mean_return`
- 提高 `min_excess_sharpe`

如果你想让切换更灵敏：

- 降低上述三个阈值
- 缩短 `lookback_days`

## 建议的实盘运行节奏

推荐最小节奏：

1. 每周重训一次
2. 每天收盘后跑一次推理
3. 每天跑一次策略切换决策
4. 每周做一次准实盘回放审计

## 定时调度模板

模板文件：

- `docs/market_live_cron.template`

包含：

- 每日收盘后推理
- 每周重训
- 每周回放审计

## 当前边界

这是一套“最小可运行实盘闭环”，不是完整券商交易系统。

当前还没有直接覆盖：

- 券商下单接口
- 实时持仓同步
- 生产级报警与重试
- 多数据源容错

当前这套闭环更适合：

- 固化研究结论
- 每日生成交易候选
- 做准实盘验证
- 在上线前把流程跑顺

## 后续高价值方向

如果继续往前推进，最值得补的是：

1. 真实每日数据更新流程接入
2. 失败报警和结果校验
3. 空信号/异常信号保护
4. 券商执行层对接
5. 真正的滚动重训调度

## 核心原则

这套设计刻意保持：

- 研究层可继续快速试验
- 生产层配置冻结、流程清晰
- 两层之间共享底层推理能力，但互不污染

这样后续你既可以继续研究，也可以逐步把当前最佳策略推向可上线状态。
