# A股日频 Top1 选股研究实施方案

## 0. 当前修正后结论

### 0.1 重要审计结论

- 早期 `market_daily` 结果存在未来信息泄漏，旧版 P0/P1 结果不能再用于正式决策
- 泄漏修复后，已按同一口径完成 corrected rerun
- 当前所有结论都只基于“可交易收益率”
- 已剔除 `t+1` 即涨停、无法实际买入的样本

### 0.2 corrected rerun 主矩阵

已完成重跑的核心模型：

- `DLinear`
- `TimesNet`
- `PatchTST`
- `iTransformer`
- `TimeMixer`

按 2015-2024 共 10 个 fold 的单模平均 `mean_return` 排名：

1. `iTransformer`: `0.012732`
2. `TimeMixer`: `0.012263`
3. `PatchTST`: `0.012234`
4. `TimesNet`: `0.006596`
5. `DLinear`: `0.001132`

### 0.3 当前最优组合结论

目前 corrected 结果下，最佳主方案不是单模，而是：

- `iTransformer + PatchTST`
- 融合方式：`rank_mean`

聚合后核心指标：

- `mean_return = 0.014044`
- `cumulative_return = 12.154646`
- `sharpe = 0.756673`

### 0.4 selector 与 gating 结论

对 `iTransformer + PatchTST` 做 selector 审计后：

- `top1_gap` selector 的毛收益更高：`mean_return = 0.014516`
- 但 live-proxy 后弱于 `rank_mean`

进一步做 threshold gating 后，最佳阈值版本为：

- `selector = top1_gap`
- `quantile = 0.5`
- `fallback = right`
- 此处 `right = PatchTST`

该版本毛收益指标：

- `mean_return = 0.015524`
- `cumulative_return = 0.754533`
- `sharpe = 0.778470`

但在 live-proxy 下，仍未超过 `rank_mean` 的稳健性。

### 0.5 live-proxy 结论

以 `low / base / high` 三档交易成本做代理验证：

`iTransformer + PatchTST / rank_mean`

- `base` 场景：
- `mean_return = 0.013352`
- `cumulative_return = 1.675380`
- `sharpe = 0.720090`

`iTransformer + PatchTST / top1_gap`

- `base` 场景：
- `mean_return = 0.013781`
- `cumulative_return = -0.980066`
- `sharpe = 0.688935`

`iTransformer + PatchTST / gated_q50_right`

- `base` 场景：
- `mean_return = 0.014956`
- `cumulative_return = -0.529108`
- `sharpe = 0.750171`

因此当前建议：

- 主实盘候选：`iTransformer + PatchTST / rank_mean`
- 研究备选：`iTransformer + PatchTST / top1_gap + q50 + fallback=PatchTST`

### 0.6 当前研究收敛判断

到目前为止，本轮“修正后可交易口径”的基础建模层面已经基本收敛：

- 单模 rerun 已完成
- 核心 pair 融合已验证
- selector / gating / live-proxy 已完成首轮收口

后续高价值工作，不再是继续大范围重跑基础模型，而是：

- 做更严格的稳定性审计
- 做组合持仓与换手约束优化
- 做更贴近实盘的成交与滑点建模
- 做上线前灰度验证与实时监控

## 1. 研究目标

### 1.1 任务定义
在交易日 `t` 收盘后，使用截至 `t` 的历史数据，对当日全市场股票打分。

最终交易收益标签定义为：

```text
y_t = open_{t+2} / open_{t+1} - 1
```

交易动作：

- `t` 日收盘后生成预测分数
- `t+1` 开盘买入当日预测分数最高的 1 只股票
- `t+2` 开盘卖出
- 每天只持有 `top1`

### 1.2 数据范围
仅使用：

- `2010-01-01` 之后的数据
- 数据文件：`market_daily.parquet`

### 1.3 验证方式
严格使用滚动 fold：

- `5年训练 -> 1年测试`
- 每个训练窗内再切一段验证集
- 不能随机切分
- 不能跨期标准化
- 不能使用未来信息

## 2. 最终优化目标

### 2.1 主目标
最终模型选择只看以下核心指标：

- `Top1` 日收益均值
- 年化收益
- Sharpe
- 最大回撤
- Fold 间稳定性

### 2.2 辅助目标
辅助筛模型时同时看：

- 日度截面 `Rank IC`
- 日度截面 `IC`
- Top1 命中率
- Top5 / Top10 收益单调性
- 预测误差 `MSE/MAE/Huber`

### 2.3 选模原则
优先选择：

- 收益高
- 不依赖少数极端行情日
- 在多个 fold 都有效
- 不依赖过重参数和超长训练

## 阶段1：数据基线与验证框架搭建

### 目标
先把数据、标签、交易评估、fold 切分全部固定下来。这个阶段不追求复杂模型，只追求评估口径完全正确。

### 产出

- 可复用的数据预处理产物
- 滚动 fold 配置文件
- 基础因子 baseline
- 统一评估器
- 候选池过滤规则

### 具体实施

#### 1. 数据清洗
从 `market_daily.parquet` 读取后，按 `code, date` 排序，保留字段：

- `code`
- `date`
- `open`
- `close`
- `high`
- `low`
- `volume`
- `amount`
- `amplitude`
- `pct_chg`
- `change`
- `turnover_rate`

处理规则：

- 去掉 `open <= 0`、`high <= 0`、`low <= 0`、`close <= 0`
- 去掉关键字段缺失
- 每只股票内按时间升序
- 仅保留 `date >= 2010-01-01`

#### 2. 标签构造
对每只股票单独构造：

```text
ret_1o = open_{t+1} / open_t - 1
ret_2o = open_{t+2} / open_{t+1} - 1
label = ret_2o
```

同时保留辅助标签：

- `label_cls = 1(label > 0)`
- `label_rank_bucket`：按每日截面分位分桶，如 5 桶

#### 3. 可交易样本过滤
建议先定义基础交易池：

- 上市满 `120` 个交易日
- 过去 `20` 日平均 `amount` 大于阈值
- 最近 `20` 日有效价格记录充足
- `t+1`、`t+2` 开盘价必须存在
- 样本窗口长度足够构造 `seq_len`

建议第一版过滤阈值：

- `rolling_20d_avg_amount >= 2e7`，或按分位过滤底部 `20%`
- `rolling_20d_avg_turnover_rate > 0`

#### 4. 特征工程
第一阶段只做轻量但有效的特征，不做过深金融因子扩展。

建议特征集合 `FeatureSet_A`：

价格收益特征：

- `co = close/open - 1`
- `ho = high/open - 1`
- `lo = low/open - 1`
- `cc = close/prev_close - 1`
- `oo = open/prev_open - 1`

量能特征：

- `log_volume_diff`
- `log_amount_diff`
- `turnover_rate`
- `amplitude`

滚动统计特征：

- `ret_5, ret_10, ret_20, ret_60`
- `vol_5, vol_10, vol_20, vol_60`
- `avg_turnover_5, avg_turnover_20`
- `avg_amplitude_5, avg_amplitude_20`

截面标准化特征：

- 每个交易日做 z-score
- 或 rank / pct-rank

建议同时保留两套：

- 时序原值特征
- 截面 rank 特征

#### 5. Fold 切分
从 `2010` 后开始做：

- Fold1: train `2010-2014`, test `2015`
- Fold2: train `2011-2015`, test `2016`
- Fold3: train `2012-2016`, test `2017`
- Fold4: train `2013-2017`, test `2018`
- Fold5: train `2014-2018`, test `2019`
- Fold6: train `2015-2019`, test `2020`
- Fold7: train `2016-2020`, test `2021`
- Fold8: train `2017-2021`, test `2022`
- Fold9: train `2018-2022`, test `2023`
- Fold10: train `2019-2023`, test `2024-09-02`

每个 fold 的训练集内部再切：

- 前 `4年` 真训练
- 第 `5年` 做验证

或：

- 训练窗内最后 `20%` 日期做验证

建议第一版直接按日期切最后一年验证，简单稳健。

#### 6. 建立 baseline
必须先跑非深度 baseline，验证口径。

建议 baseline：

- `momentum_20`：过去20日收益排序
- `reversal_5`：过去5日反转
- `turnover_factor`：换手率排序
- `volatility_penalized_momentum`：`ret_20 / vol_20`
- `linear_regression`
- `lightgbm/xgboost` 可选，作为 tabular baseline

如果简单因子和线性模型完全不赚钱，说明数据口径或评估实现有问题，不能直接进入深度模型。

#### 7. 评估器
统一评估逻辑：

每天：

- 取候选池全部股票预测分数
- 排序
- 选 `top1`
- 用真实 `open_{t+2}/open_{t+1}-1` 计收益

输出：

- 日收益序列
- 累计净值
- 年化收益
- 年化波动
- Sharpe
- Sortino
- Calmar
- 最大回撤
- 胜率
- Fold 统计表

#### 8. 阶段1结束标准
满足以下条件才进入阶段2：

- baseline 全部跑通
- fold 评估稳定
- 无未来函数
- 每天 top1 选择逻辑正确
- 训练/验证/测试边界清晰
- 评估结果可重复

## 阶段2：快速模型筛选

### 目标
用最少时间筛出最有潜力的模型族，不做大规模精调。

### 核心原则
这一阶段要的是广度覆盖加快速淘汰，不是极限性能。

### 候选模型池

第一优先级：

- `DLinear`
- `TSMixer`
- `PatchTST`
- `TimesNet`
- `iTransformer`
- `TimeMixer`

第二优先级：

- `TimeXer`
- `FreTS`
- `Koopa`
- `Nonstationary_Transformer`

暂缓：

- `Mamba`
- `Chronos`
- `TimesFM`
- `Moirai`
- `TimeMoE`

原因是前者部署成本低、速度快、对高噪声日频更现实。

### 统一实验模板

基础标签：

- 主任务：回归 `label = open_{t+2}/open_{t+1} - 1`

基础损失：

- 首选 `Huber`
- 对照 `MSE`

基础序列长度：

- `seq_len = 20`
- `seq_len = 60`
- `seq_len = 120`

基础特征：

- 固定 `FeatureSet_A`

训练配置：

- `batch_size` 按 GPU 调整
- `epochs = 8-15`
- `early_stopping = patience 2~3`
- 开启 `amp`
- 固定随机种子 2 个即可

### 筛选策略

#### Round 1：单折预筛
先只用两个代表性 fold：

- `2018 -> 2019`
- `2020 -> 2021`

每个模型只跑：

- 3 个 `seq_len`
- 1 套特征
- 1 个损失

总目标是 1 天内筛出前 3 到 4 名。

#### Round 2：扩大 fold 验证
保留 Round 1 前 `3-4` 个模型，跑全部 folds。

根据以下打分综合排序：

```text
score = 0.45 * annual_return_rank
      + 0.25 * sharpe_rank
      + 0.20 * rank_ic_rank
      + 0.10 * stability_rank
```

### 阶段2结束标准
得到：

- 2 到 3 个主力模型
- 1 个稳健基线模型
- 1 个备用轻量模型

推荐目标结果形态：

- 主力候选：`TimeMixer / iTransformer / PatchTST`
- 稳健基线：`DLinear`
- 备用：`TSMixer` 或 `TimesNet`

## 阶段3：重点模型深挖与超参搜索

### 目标
在少数优胜模型上集中算力，快速逼近最优单模型。

### 范围
只对阶段2前 `2-3` 个模型做精调。

### 搜索维度

#### 1. 序列长度
建议搜索：

- `20`
- `40`
- `60`
- `120`
- `240`

A股短周期噪声大，很多时候 `20-60` 更好，但多尺度模型可能在 `120-240` 受益。

#### 2. 特征集搜索
定义 4 套特征：

- `FeatureSet_A`：基础量价 + rolling
- `FeatureSet_B`：A + 截面 rank 特征强化
- `FeatureSet_C`：A + 波动/成交额归一化增强
- `FeatureSet_D`：只保留收益率化和量价核心特征，做轻量输入

#### 3. 损失函数搜索
建议只搜索这几种：

- `Huber`
- `MSE`
- `BCE` 方向分类
- `Huber + BCE` 多任务
- `pairwise rank loss` 可作为最后增强项

最短时间路线建议：

- 先 `Huber`
- 再 `Huber + BCE`

#### 4. 正则化

- `dropout`: `0.0 / 0.1 / 0.2 / 0.3`
- `weight_decay`: `0 / 1e-5 / 1e-4`
- `learning_rate`: `1e-4 / 3e-4 / 5e-4`

#### 5. 训练策略
对优胜模型只尝试两种：

- 每个 fold 完全独立重训
- 每年 warm-start 微调

如果时间极其紧，先只做完全独立重训，逻辑更干净。

### 搜索方法

#### 首选：分层搜索，不做全组合
顺序：

1. 固定损失和特征，找最好 `seq_len`
2. 固定 `seq_len`，找最好特征集
3. 固定前两项，找最好损失
4. 最后调小范围学习率和 dropout

#### 不建议

- 全网格搜索
- 所有模型同等算力
- 先试太多 loss

### 多卡与并行策略

#### 原则
这一阶段不建议大量单任务多卡训练，建议多实验并行优先于单实验 `DataParallel`。

优先级：

1. `一卡多实例` 跑轻量模型
2. `多卡多实例` 跑不同模型/不同 fold
3. 仅对超大模型才考虑单任务多卡

#### 推荐调度方式
假设有 `4` 张卡：

- `GPU0`: `DLinear/TSMixer` 多开 2-3 实例
- `GPU1`: `PatchTST` 多开 2 实例
- `GPU2`: `iTransformer/TimeMixer` 各 1-2 实例
- `GPU3`: 保留给大 batch 或复核实验

#### 同卡多实例原则
如果显存允许，同一卡上并发 2 个实验：

- 小模型如 `DLinear/TSMixer`：可 2-4 开
- 中模型如 `PatchTST/TimesNet`：可 2 开
- 大模型如 `iTransformer/TimeMixer`：先单开，再视显存决定

#### 任务优先级队列
第一批：

- 不同模型，同一 fold，同一配置

第二批：

- 优胜模型，不同 `seq_len`

第三批：

- 优胜模型，不同特征集和损失

这样可以最快得到方向性结论。

### 阶段3结束标准
输出：

- 每个主力模型的最优配置
- 单模型排行榜
- 每 fold 详细收益表
- 确认 1 个最优单模型和 1 个最稳单模型

## 阶段4：集成与交易层增强

### 目标
把单模型最优提升为最终收益最优。通常 Top1 任务里，集成比单模型更稳。

### 4.1 集成方式

方式1：Rank Average

对多个模型的每日截面分数先转 rank，再平均。

```text
ensemble_score = mean(rank(score_model_i))
```

优点：

- 最稳
- 不怕不同模型分数量纲不一致
- 很适合 Top1 任务

这是第一优先级。

方式2：Z-score Average

先对每日截面分数标准化，再平均。

适合模型输出比较稳定时使用。

方式3：Stacking

用训练窗验证集上的 out-of-fold 预测训练二层模型：

- 线性回归
- ridge
- lightgbm ranker

只有在前两种集成确认有效后再做。

### 4.2 集成池构造
建议只从以下模型中选：

- 最优单模型
- 最稳单模型
- 一个风格差异较大模型

例如：

- `TimeMixer`
- `iTransformer`
- `DLinear`

不要把所有模型都塞进去，噪声会变大。

### 4.3 交易层增强
在不改变最终评估定义的前提下，只加最基础过滤：

- 当日候选池去掉低流动性股票
- 去掉近期波动极端异常股票
- 去掉连续异常缺口样本
- 可选加入分数置信度阈值

置信度规则示例：

- 若 top1 与 top2 分数差太小，则放弃交易
- 或仅在 top1 分数高于当日截面分位阈值时交易

如果要求每天必须选 1 只，这一条只作为研究对照，不作为主策略。

### 4.4 市场状态分层
如果阶段3结果显示不同年份风格差异极大，可增加 regime 分层：

- 高波动市场
- 低波动市场
- 反弹市场
- 震荡市场

然后：

- 不同 regime 用不同模型
- 或不同 regime 用不同集成权重

这个增强很可能有效，但必须排在基础集成之后。

### 阶段4结束标准
输出：

- 最优集成方法
- 集成前后收益提升对比
- 交易过滤对收益/回撤的影响
- 最终候选主策略 1-2 套

## 阶段5：严谨复核、压力测试与最终定版

### 目标
确认收益不是假象，不是泄漏，不是过拟合，不是少数极端样本贡献。

### 5.1 未来函数检查
逐项确认：

- 标签只使用 `t+1`、`t+2` 开盘价
- 特征只使用 `t` 及以前
- 每日截面标准化只用当日已知信息
- scaler 仅用训练期拟合
- 验证集和测试集绝不参与训练

### 5.2 收益来源拆解
检查：

- 收益是否高度依赖某几年
- 收益是否由极少数涨停/异常跳空股票贡献
- top1 选择是否过于集中在少数股票
- 某几个 fold 是否完全失效

建议输出：

- 年度收益表
- 每 fold 收益表
- 前 20 大贡献股票
- 前 20 大亏损交易

### 5.3 敏感性测试
对最终主策略做以下扰动：

- `seq_len` 小幅变化
- 特征轻微删减
- 交易池阈值小幅变化
- 损失函数替换
- seed 替换

如果结果大幅崩掉，说明策略太脆弱。

### 5.4 真实交易近似检查
虽然当前评估不含手续费和滑点，但建议做对照：

- 单边 `10bp`
- 单边 `20bp`
- 双边合计 `30-50bp`

因为 `top1` 高频切换很容易被交易成本吃掉。最终至少要确认加入轻微成本后是否仍有优势。

### 5.5 最终定版标准
最终模型或策略必须同时满足：

- 全 folds 平均收益领先
- fold 间稳定
- 回撤可控
- 对轻微参数扰动不敏感
- 非依赖单一极端年份
- 工程复杂度可接受

## 最短时间求最优的执行策略

下面是实际执行时的最短路径。

### Day 1-2
完成阶段1：

- 数据清洗
- 标签构造
- fold 切分
- baseline 因子
- 评估器

### Day 3-4
完成阶段2第一轮：

- `DLinear`
- `TSMixer`
- `PatchTST`
- `TimesNet`
- `iTransformer`
- `TimeMixer`

只跑两个代表 fold，只测 `seq_len = 20/60/120`

### Day 5-6
完成阶段2第二轮和阶段3开始：

- 选前 3 模型
- 跑全部 fold
- 锁定优胜模型
- 开始 `seq_len + feature set + loss` 分层搜索

### Day 7-8
完成阶段3：

- 找出最优单模型
- 找出最稳模型
- 保存所有 OOF/test 预测

### Day 9
完成阶段4：

- rank ensemble
- z-score ensemble
- 基础交易过滤

### Day 10
完成阶段5：

- 泄漏检查
- 压力测试
- 成本敏感性
- 最终定版

## 并行资源调度建议

### 优先原则
优先使用：

- `多卡多实验`
- `同卡多实例`
- 不优先使用单任务多卡

因为当前目标是找最优模型，不是训练一个超大模型。

### 推荐队列

队列A：快速筛选

每张卡并行多个轻量实验：

- `DLinear`
- `TSMixer`
- `PatchTST`

队列B：中等模型

单卡 1-2 实例：

- `TimesNet`
- `iTransformer`
- `TimeMixer`

队列C：复核队列

保留 1 张卡用于：

- 最优模型复跑
- fold 补跑
- 异常结果复核

### 调度建议
实验优先级按这个顺序：

1. 不同模型、相同配置
2. 优胜模型、不同 `seq_len`
3. 优胜模型、不同特征集
4. 优胜模型、不同 loss
5. 集成与复核

这样最短时间内最容易锁定方向。

## 最终推荐路线

如果追求最快找到最优决策，建议直接执行这条路线：

### 第一轮主战模型

- `DLinear`
- `PatchTST`
- `iTransformer`
- `TimeMixer`

### 第一轮固定配置

- 标签：`open_{t+2}/open_{t+1} - 1`
- 损失：`Huber`
- 特征：`FeatureSet_A`
- `seq_len`：`20/60/120`

## 实验矩阵

### Round 1 基线矩阵

| 类别 | 名称 | Fold | 说明 |
| --- | --- | --- | --- |
| Baseline | `momentum_20` | `2019`, `2021` | 过去20日动量直选股 |
| Baseline | `reversal_5` | `2019`, `2021` | 过去5日反转 |
| Baseline | `turnover` | `2019`, `2021` | 换手率排序 |
| Baseline | `risk_adjusted_momentum` | `2019`, `2021` | `ret_20 / vol_20` |

### Round 1 主模型矩阵

| 模型 | Fold | `seq_len` | 特征集 | 损失 | 目标 |
| --- | --- | --- | --- | --- | --- |
| `DLinear` | `2019`, `2021` | `20`, `60`, `120` | `A` | `Huber` | 快速线性强基线 |
| `PatchTST` | `2019`, `2021` | `20`, `60`, `120` | `A` | `Huber` | Patch 局部结构 |
| `iTransformer` | `2019`, `2021` | `20`, `60`, `120` | `A` | `Huber` | 多变量截面表达 |
| `TimeMixer` | `2019`, `2021` | `20`, `60`, `120` | `A` | `Huber` | 多尺度混合 |

### Round 2 扩展矩阵

Round 1 筛出前 2 到 3 个模型后，再扩展：

- Fold 扩展到 `2015-2024`
- 特征集扩展到 `B/C/D`
- 损失扩展到 `MSE`、`Huber+BCE`
- `seq_len` 扩展到 `40/240`

## GPU 调度表

当前机器资源：

- `GPU0-3`: `NVIDIA H100 80GB`

### 推荐首轮调度

| GPU | 并发数 | 任务 |
| --- | --- | --- |
| `GPU0` | `1-2` | `DLinear` 全部首轮任务 |
| `GPU1` | `1-2` | `PatchTST` 全部首轮任务 |
| `GPU2` | `1` | `iTransformer` 全部首轮任务 |
| `GPU3` | `1` | `TimeMixer` 全部首轮任务，空闲时补 baseline / smoke test |

### 同卡多实例建议

| 模型 | 单实例建议 batch | 同卡建议实例数 |
| --- | --- | --- |
| `DLinear` | `512-1024` | `2-4` |
| `PatchTST` | `256-512` | `2` |
| `iTransformer` | `256-512` | `1-2` |
| `TimeMixer` | `256-512` | `1-2` |

### 最短时间执行顺序

1. 先跑 baseline，确认收益评估口径。
2. 同时启动 `DLinear/PatchTST/iTransformer/TimeMixer` 在 `fold=2019`、`seq_len=20/60/120`。
3. 只要 `fold=2019` 有结果，立即补 `fold=2021`。
4. 先比较 `Rank IC + Top1 cumulative return`，筛掉明显弱模型。
5. 再把优胜模型扩展到全 folds 和更多特征集。

### 第二轮深化
保留前 2-3 名，搜索：

- `FeatureSet_B/C/D`
- `Huber + BCE`
- `dropout/lr/weight_decay`

### 最终产物

- 1 个最优单模型
- 1 个最稳单模型
- 1 个 rank ensemble 主策略

## 首轮实验矩阵

这一节定义可以直接启动的第一批实验，目标是在最短时间内拿到可靠的横向比较结果，而不是一次性跑完全量搜索。

### Round 0：管线校验

目的：

- 验证 `market_daily.parquet -> feature cache -> dataset -> train/val/test -> top1 evaluator` 全链路可运行
- 验证日志、结果文件、checkpoint 写入路径正确

任务：

- baseline 因子：`2019`、`2021`
- smoke train：`DLinear + fold2019 + seq_len=20 + 1 epoch`

通过标准：

- baseline 能输出 `baseline_summary.json`
- smoke train 能产出：
  - `checkpoints/.../checkpoint.pth`
  - `test_results/.../top1_predictions.csv`
  - `test_results/.../market_metrics.txt`

### Round 1：主战模型预筛

目标：

- 快速比较 4 个主战模型在两个代表性 fold 上的表现
- 控制实验总量，优先保证结果返回速度

fold：

- `2019`
- `2021`

模型：

- `DLinear`
- `PatchTST`
- `iTransformer`
- `TimeMixer`

序列长度：

- `20`
- `60`
- `120`

特征：

- `FeatureSet_A`

损失：

- `Huber`

训练配置：

- `epochs = 10`
- `patience = 2`
- `batch_size = 4096`
- `amp = on`
- `num_workers = 8`

总任务数：

```text
4 models * 2 folds * 3 seq_lens = 24 jobs
```

### Round 2：优胜模型深化

触发条件：

- Round 1 完成，拿到完整收益和排序指标

保留模型：

- Round 1 前 2 到 3 名

扩展维度：

- `FeatureSet_B`
- `FeatureSet_C`
- `FeatureSet_D`
- `seq_len = 40 / 240`
- `Huber + BCE` 或双头训练

执行原则：

- 只在优胜模型上扩展
- 不回头给明显落后的模型继续烧卡

## 首轮 GPU 调度表

当前硬件：

- `GPU0-3`
- `NVIDIA H100 80GB HBM3`

调度原则：

- 优先 `多卡多实验`
- 次优 `同卡多实例`
- 不优先 `单任务多卡`
- 轻量模型和中等模型分开排，避免单卡负载失衡

### Round 0 调度

| 任务 | GPU | 并发 | 说明 |
| --- | --- | --- | --- |
| Baseline 因子 | CPU | 1 | 优先构建特征缓存，不占 GPU |
| DLinear smoke | GPU0 | 1 | 校验训练链路 |
| 其余 GPU | GPU1-3 | 空闲 | 留作正式任务 |

### Round 1 调度

推荐首轮并发上限：

- `DLinear`: 每卡 2 实例
- `PatchTST`: 每卡 1-2 实例
- `iTransformer`: 每卡 1-2 实例
- `TimeMixer`: 每卡 1 实例起步

推荐排布：

| GPU | 任务类型 | 并发建议 | 备注 |
| --- | --- | --- | --- |
| `GPU0` | `DLinear` | 2 | 最轻，优先多开 |
| `GPU1` | `PatchTST` | 2 | 中等负载 |
| `GPU2` | `iTransformer` | 2 | 中等负载 |
| `GPU3` | `TimeMixer` | 1-2 | 先 1，再视显存加开 |

如果首轮显存压力低于预期，可切换为：

- `max_jobs_per_gpu = 2`
- `TimeMixer` 若稳定，可加入第二实例

如果某模型显存异常或速度偏慢：

- 降低该模型并发到 1
- 其他卡继续保持 2 并发

### 任务发射顺序

第一批：

- `DLinear` 全部 6 任务
- `PatchTST` 的 `seq_len = 20/60`
- `iTransformer` 的 `seq_len = 20/60`
- `TimeMixer` 的 `seq_len = 20`

第二批：

- 剩余 `seq_len = 120`
- 剩余 fold

这样可以最快得到前几个结果，提前判断是否需要调整 batch 或并发。

## 落盘与监控规范

### 日志目录

- 启动日志：`logs/market_round1/`
- baseline 结果：`results/market_baselines/`
- 模型预测：`test_results/<setting>/`
- 模型数值结果：`results/<setting>/`
- checkpoint：`checkpoints/<setting>/`

### 每个任务最少输出

- 标准输出日志
- `market_metrics.txt`
- `top1_predictions.csv`
- `metrics.npy`
- `pred.npy`
- `true.npy`

### 首轮停止条件

出现以下情况之一，暂停继续扩任务：

- 任一模型重复 OOM
- 数据集样本数异常为 0
- `market_metrics.txt` 缺失
- `top1_predictions.csv` 写出为空
- baseline 和模型收益口径明显冲突

## 实施顺序

### 第一步

- 跑 baseline
- 跑 `DLinear` smoke

### 第二步

- 修正启动脚本中的环境继承、日志和任务清单
- 确保子进程使用 `tslib` 环境的 Python

### 第三步

- 启动 Round 1 的 24 个任务
- 先使用 `max_jobs_per_gpu = 1`
- 观察 10-20 分钟后再决定是否切到 2 并发

### 第四步

- 汇总 Round 1 各任务：
  - `mean_return`
  - `cumulative_return`
  - `sharpe`
  - `rank_ic`
  - `ic`

### 第五步

- 选出前 2 到 3 个模型进入 Round 2

## 当前执行状态

日期：

- `2026-05-16`

当前真实状态：

- baseline 已完成，结果文件在 `results/market_baselines/baseline_summary.json`
- `DLinear / PatchTST / iTransformer` 的首轮有效结果已覆盖：
  - `2019/2021`
  - `seq_len = 20 / 60`
  - `DLinear` 额外完成 `seq_len = 120`
  - `PatchTST` 额外完成 `seq_len = 120`
- `DLinear FeatureSet_B seq_len=60` 补跑已完成，但没有优于 `FeatureSet_A`
- `TimeMixer` 已确认不是简单调参问题，而是 `market_daily` 数据接口兼容问题，暂不继续烧卡
- 4 张 `H100 80GB` 当前空闲，可直接启动下一波

### 当前已完成首轮结果

| 模型 | Fold | `seq_len` | `mean_return` | `cumulative_return` | `sharpe` | `ic` | `rank_ic` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `DLinear` | `2019` | `20` | `0.012467` | `0.900893` | `1.2490` | `0.003446` | `0.008542` |
| `DLinear` | `2019` | `60` | `0.012931` | `2.301582` | `1.4178` | `0.001360` | `0.006837` |
| `DLinear` | `2019` | `120` | `0.015896` | `4.429777` | `1.6191` | `0.002096` | `0.006347` |
| `DLinear` | `2021` | `20` | `0.009349` | `-0.566657` | `0.8297` | `0.005567` | `0.018556` |
| `DLinear` | `2021` | `60` | `0.010568` | `3.526138` | `1.6782` | `0.013850` | `0.023695` |
| `DLinear` | `2021` | `120` | `0.007467` | `1.057459` | `1.1595` | `0.012583` | `0.022429` |
| `PatchTST` | `2019` | `20` | `0.000638` | `-0.452474` | `0.1291` | `-0.009248` | `-0.008932` |
| `PatchTST` | `2021` | `20` | `0.013315` | `3.552775` | `1.6653` | `0.000775` | `-0.005840` |
| `PatchTST` | `2019` | `60` | `0.010555` | `6.019093` | `2.1724` | `-0.002757` | `0.001847` |
| `PatchTST` | `2021` | `60` | `0.002336` | `-0.282025` | `0.4329` | `-0.004728` | `-0.001850` |
| `PatchTST` | `2019` | `120` | `-0.010664` | `-0.994816` | `-1.5062` | `-0.007580` | `-0.001093` |
| `PatchTST` | `2021` | `120` | `0.006926` | `1.278014` | `1.2080` | `-0.001807` | `0.008807` |
| `iTransformer` | `2019` | `20` | `0.003217` | `-0.747966` | `0.4542` | `0.003159` | `0.010738` |
| `iTransformer` | `2021` | `20` | `0.018434` | `11.162454` | `2.1917` | `0.001203` | `0.001949` |
| `iTransformer` | `2019` | `60` | `-0.001071` | `-0.853340` | `-0.1531` | `0.001761` | `0.006254` |
| `iTransformer` | `2021` | `60` | `0.011061` | `0.632137` | `1.2700` | `0.001643` | `0.004965` |
| `DLinear FeatureSet_B` | `2019` | `60` | `0.012232` | `1.754066` | `1.3378` | `0.001360` | `0.006838` |
| `DLinear FeatureSet_B` | `2021` | `60` | `0.010568` | `3.526138` | `1.6782` | `0.013851` | `0.023695` |

### 已完成的离线组合/selector 检验

基于现有 `top1_predictions.csv` 已做两类快速复核：

- `2-model rank-average ensemble`
- `year selector`

结果：

- `2019` 最强单模型仍是 `PatchTST seq_len=60`
- `2021` 最强单模型仍是 `iTransformer seq_len=20`
- 简单 `2-model rank-average` 没有稳定超过对应年份最优单模型
- 但 `year selector` 很强：
  - `2019 -> PatchTST_sl60`
  - `2021 -> iTransformer_sl20`
  - 合并后 `mean_return = 0.014487`
  - 合并后 `sharpe = 2.1089`

这说明当前最值得推进的方向不是“无脑 pair ensemble”，而是：

- 继续强化各 regime 的最优单模型
- 最终做年份 / regime 条件切换器

阶段性结论：

- `DLinear` 在 `2019` 上呈现 `120 > 60 > 20`
- `DLinear` 在 `2021` 上呈现 `60 > 120 > 20`
- `DLinear FeatureSet_B` 相比 `FeatureSet_A` 没有带来增益：
  - `2019 sl60` 持平
  - `2021 sl60` 持平
  - 因此 `DLinear` 暂不继续大规模扩展特征集
- `PatchTST` 出现明显 regime 分化：
  - `2019 + seq_len=20` 明显失效
  - `2021 + seq_len=20` 与最强 `DLinear` 基本同级
  - `2019 + seq_len=60` 非常强，但 `2021 + seq_len=60` 明显失效
  - `seq_len=120` 没有成为有效补充，尤其 `2019` 明显失败
- `iTransformer` 出现更强的 regime 分化：
  - `2019 + seq_len=20` 失效
  - `2021 + seq_len=20` 当前是全局最强单点
- `iTransformer` 扩展到 `seq_len=60` 后没有改善稳定性：
  - `2019` 继续失效，且比 `sl20` 更差
  - `2021` 明显弱于 `sl20`
- 当前全局最优已知配置仍是 `iTransformer + 2021 + seq_len=20`
- 当前最优跨年份决策不是单一模型，而是：
  - `2019` 使用 `PatchTST_sl60`
  - `2021` 使用 `iTransformer_sl20`
- `TimeMixer` 当前不能直接纳入主搜索：
  - 已确认需要专门适配市场数据接口
  - 目前在 `market_daily` 上不是“调参问题”，而是模型兼容性问题
- 下一步必须验证：
  - `PatchTST_sl60` 在 `FeatureSet_B/C` 下能否进一步放大 `2019` 优势且不明显损伤 `2021`
  - `iTransformer_sl20` 在 `FeatureSet_B/C` 下能否维持 `2021` 强势并改善 `2019`
  - 后续是否需要从“按年份 selector”推进到“按市场状态 selector”

## 实施命令与队列策略

### 环境

- Python 环境：`/huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python`

### 已修复的调度器行为

脚本：

- `scripts/market_daily/launch_round1.py`

当前支持：

- `--launch_mode queued`
- 每个 GPU 按 `max_jobs_per_gpu` 串行补位
- `job_manifest.json` 持续写入 `pending/running/done/failed`

这意味着现在可以安全后台运行 launcher，而不是一次性瞬间发射全部任务。

### 结果汇总命令

```bash
/huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
scripts/market_daily/summarize_round1.py \
--manifest logs/market_round1/job_manifest.json \
--output_csv logs/market_round1/round1_summary.csv
```

### 推荐后台启动命令

首轮剩余任务建议拆成 3 波，每波最多占满 4 张卡，优先拿到跨模型结果。

注意：

- 不要用 `--models DLinear,PatchTST --seq_lens 120,20` 这种写法直接交叉展开
- 这会产生 `DLinear 20` 和 `PatchTST 120` 的额外任务，不符合最短时间策略

第一波已按下面两条独立命令启动：

- `DLinear 2019/2021 seq_len=120`，日志目录：`logs/market_dlinear120_wave/`
- `PatchTST 2019/2021 seq_len=20`，日志目录：`logs/market_patchtst20_wave/`

```bash
mkdir -p logs/market_dlinear120_wave
setsid /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
scripts/market_daily/launch_round1.py \
--python /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
--models DLinear \
--fold_years 2019,2021 \
--seq_lens 120 \
--gpus 0,2 \
--max_jobs_per_gpu 1 \
--batch_size 4096 \
--epochs 10 \
--launch_mode queued \
--output_dir logs/market_dlinear120_wave \
> logs/market_dlinear120_wave/launcher.log 2>&1 < /dev/null &
```

```bash
mkdir -p logs/market_patchtst20_wave
setsid /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
scripts/market_daily/launch_round1.py \
--python /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
--models PatchTST \
--fold_years 2019,2021 \
--seq_lens 20 \
--gpus 1,3 \
--max_jobs_per_gpu 1 \
--batch_size 4096 \
--epochs 10 \
--launch_mode queued \
--output_dir logs/market_patchtst20_wave \
> logs/market_patchtst20_wave/launcher.log 2>&1 < /dev/null &
```

第二波：

- `PatchTST 2019/2021 seq_len=60`
- `iTransformer 2019/2021 seq_len=20`

推荐命令：

```bash
mkdir -p logs/market_patchtst60_wave
setsid /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
scripts/market_daily/launch_round1.py \
--python /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
--models PatchTST \
--fold_years 2019,2021 \
--seq_lens 60 \
--gpus 0,1 \
--max_jobs_per_gpu 1 \
--batch_size 4096 \
--epochs 10 \
--launch_mode queued \
--output_dir logs/market_patchtst60_wave \
> logs/market_patchtst60_wave/launcher.log 2>&1 < /dev/null &
```

```bash
mkdir -p logs/market_itransformer20_wave
setsid /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
scripts/market_daily/launch_round1.py \
--python /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
--models iTransformer \
--fold_years 2019,2021 \
--seq_lens 20 \
--gpus 2,3 \
--max_jobs_per_gpu 1 \
--batch_size 4096 \
--epochs 10 \
--launch_mode queued \
--output_dir logs/market_itransformer20_wave \
> logs/market_itransformer20_wave/launcher.log 2>&1 < /dev/null &
```

第三波：

- `iTransformer 2019/2021 seq_len=60`
- `TimeMixer 2019/2021 seq_len=20`

推荐命令：

```bash
mkdir -p logs/market_itransformer60_wave
setsid /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
scripts/market_daily/launch_round1.py \
--python /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
--models iTransformer \
--fold_years 2019,2021 \
--seq_lens 60 \
--gpus 0,1 \
--max_jobs_per_gpu 1 \
--batch_size 4096 \
--epochs 10 \
--launch_mode queued \
--output_dir logs/market_itransformer60_wave \
> logs/market_itransformer60_wave/launcher.log 2>&1 < /dev/null &
```

```bash
mkdir -p logs/market_timemixer20_wave
setsid /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
scripts/market_daily/launch_round1.py \
--python /huanghb28/aigc/cv_banc/zsw/zhuangcailin/envs/tslib/bin/python \
--models TimeMixer \
--fold_years 2019,2021 \
--seq_lens 20 \
--gpus 2,3 \
--max_jobs_per_gpu 1 \
--batch_size 4096 \
--epochs 10 \
--launch_mode queued \
--output_dir logs/market_timemixer20_wave \
> logs/market_timemixer20_wave/launcher.log 2>&1 < /dev/null &
```

实际执行备注：

- `iTransformer 60` 已完成，结论是不值得继续扩更长序列
- `TimeMixer 20` 已做两轮兼容修复，仍然在市场接口层失败，当前先暂停
- 因此第四波应回到：
  - `PatchTST 120`
  - `DLinear / PatchTST / iTransformer` 的 `FeatureSet_B/C`

### 监控命令

查看 GPU：

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
```

查看队列状态：

```bash
cat logs/market_dlinear120_wave/job_manifest.json
cat logs/market_patchtst20_wave/job_manifest.json
```

查看日志尾部：

```bash
tail -n 40 logs/market_dlinear120_wave/*.log
tail -n 40 logs/market_patchtst20_wave/*.log
```

## 下一步执行优先级

### P0：立即执行

目标：

- 只扩展当前最有价值的两个 regime winner
- 优先回答“特征增强是否有效”

任务：

| 优先级 | 模型 | Fold | `seq_len` | 特征集 | 目的 |
| --- | --- | --- | --- | --- | --- |
| `P0` | `PatchTST` | `2019`, `2021` | `60` | `B` | 验证 `2019` 强模型能否被 rank 特征进一步强化 |
| `P0` | `iTransformer` | `2019`, `2021` | `20` | `B` | 验证 `2021` 强模型能否保收益并改善 `2019` |

P0 已完成，结果如下：

| 模型 | Fold | 配置 | `FeatureSet_A` | `FeatureSet_B` | 结论 |
| --- | --- | --- | --- | --- | --- |
| `PatchTST` | `2019` | `sl60` | `mean=0.010555`, `sharpe=2.1724` | `mean=0.005087`, `sharpe=0.9520` | 明显变差 |
| `PatchTST` | `2021` | `sl60` | `mean=0.002336`, `sharpe=0.4329` | `mean=0.005914`, `sharpe=1.1211` | 有改善，但仍远弱于主力候选 |
| `iTransformer` | `2019` | `sl20` | `mean=0.003217`, `sharpe=0.4542` | `mean=0.000012`, `sharpe=0.0019` | 明显变差 |
| `iTransformer` | `2021` | `sl20` | `mean=0.018434`, `sharpe=2.1917` | `mean=0.012087`, `sharpe=1.3981` | 明显变差 |

P0 结论：

- `FeatureSet_B` 不是当前 regime winner 的正确方向
- 不再继续 `FeatureSet_C/D` 扩展
- 研究主线切换为：直接做全 rolling folds 验证

### P1：紧随其后

原计划仅当 `P0` 至少一组有效时再继续。

由于 `P0` 已验证 `FeatureSet_B` 对主力候选总体无效，`P1` 当前暂停，不继续消耗 GPU。

| 优先级 | 模型 | Fold | `seq_len` | 特征集 | 目的 |
| --- | --- | --- | --- | --- | --- |
| `P1` | `PatchTST` | `2019`, `2021` | `60` | `C` | 测试波动/成交额归一化增强 |
| `P1` | `iTransformer` | `2019`, `2021` | `20` | `C` | 测试轻量增强特征能否提高跨 regime 泛化 |

### P2：selector 阶段

当 `P0/P1` 跑完后再进入：

- 用所有有效单模型预测文件构建 `year selector`
- 再扩展到 `regime selector`
- 不优先继续普通 `pair ensemble`

## 当前新一轮执行状态

### Full-fold 主验证波次

已于 `2026-05-16` 启动以下 3 条主线，覆盖 `2015-2024` 全部 rolling test years。

截至 `2026-05-17`，3 条主线已全部跑完：

| 波次 | 模型 | 配置 | 日志目录 | 状态 |
| --- | --- | --- | --- | --- |
| `Wave-F1` | `DLinear` | `FeatureSet_A, seq_len=120` | `logs/market_full_dlinear120A/` | `done` |
| `Wave-F2` | `PatchTST` | `FeatureSet_A, seq_len=60` | `logs/market_full_patchtst60A/` | `done` |
| `Wave-F3` | `iTransformer` | `FeatureSet_A, seq_len=20` | `logs/market_full_itransformer20A/` | `done` |

调度方式：

- `GPU0`：`DLinear sl120`
- `GPU1`：`PatchTST sl60`
- `GPU2`：`iTransformer sl20`
- `GPU3`：预留给后续补跑 / selector / 异常复核

当前目标：

- 已完成三条主线在 `2015-2024` 全部 folds 上的真实收益排名
- 当前进入：
  - 最稳健单模型确认
  - year selector / regime selector 复核

## Full-fold 结果汇总

### 单模型汇总

| 模型 | 配置 | folds | 平均 `mean_return` | 中位 `mean_return` | 平均 `sharpe` | 中位 `sharpe` | 正收益年份数 | `sharpe > 1` 年份数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DLinear` | `seq_len=120` | `10` | `0.006357` | `0.002908` | `0.5644` | `0.7682` | `7/10` | `4/10` |
| `iTransformer` | `seq_len=20` | `10` | `0.004025` | `0.001852` | `0.4236` | `0.2898` | `6/10` | `3/10` |
| `PatchTST` | `seq_len=60` | `10` | `0.000237` | `0.000081` | `-0.0949` | `0.0593` | `5/10` | `1/10` |

当前结论：

- `DLinear seq_len=120` 是当前最稳健的全周期单模型
- `iTransformer seq_len=20` 仍然是高弹性模型：
  - 在 `2021`、`2024` 非常强
  - 但在若干年份显著失效
- `PatchTST seq_len=60` 只适合作为局部年份特化候选，不适合作为全周期主策略

### 按年份 winner

| Test Year | Winner |
| --- | --- |
| `2015` | `iTransformer sl20` |
| `2016` | `PatchTST sl60` |
| `2017` | `iTransformer sl20` |
| `2018` | `DLinear sl120` |
| `2019` | `DLinear sl120` |
| `2020` | `DLinear sl120` |
| `2021` | `iTransformer sl20` |
| `2022` | `iTransformer sl20` |
| `2023` | `DLinear sl120` |
| `2024` | `DLinear sl120` |

### Year selector 复核

使用上述逐年 winner 做离线 selector 后：

- `selector_mean_return = 0.008550`
- `selector_sharpe = 1.2541`

说明：

- `year selector` 的平均日收益高于单一 `DLinear sl120`
- 但 `sharpe` 仍未显著高到可以直接替代稳健单模型
- 下一步更合理的是从“按年份 selector”继续推进到“按市场状态 / regime selector”

### 当前最优决策

如果现在必须从当前结果中定一个主策略，建议分两层：

1. `主提交基线`：`DLinear sl120`
2. `研究增强线`：`DLinear sl120 + iTransformer sl20` 的 regime selector

1. 先完成 `DLinear seq_len=120`，确认是否继续提升。
2. 立刻拿到 `PatchTST seq_len=20` 的 `2019/2021` 双折结果，判断是否值得继续。
3. 如果 `PatchTST` 显著强于 `DLinear`，则优先补 `PatchTST 60/120`。
4. 如果 `PatchTST` 不占优，则把资源优先切到 `iTransformer 20/60`。
5. 只有当前三类结果都拿到之后，才决定 `TimeMixer` 是否继续全量。

## 2026-05-17 按“次日开盘涨停不可买”口径重跑结论

这一节覆盖上面的旧全折汇总。旧结果没有剔除 `t+1` 开盘即涨停、实际无法买入的样本；本节结果才是当前应使用的正式口径。

### 评估口径修正

新增交易约束：

- 若股票在 `t+1` 开盘价已经达到对应涨停价，则该样本记为 `tradable=False`
- 日度 `top1` 选择时，先在当日候选中剔除 `tradable=False` 的股票，再从剩余股票中选预测分数最高者

当前已实现的涨跌停规则：

- 主板默认 `10%`
- `SH688*` 科创板 `20%`
- `SZ300*` 创业板自 `2020-08-24` 起 `20%`
- `BJ*` 北交所 `30%`

注意：

- 由于当前数据集中没有 ST 标记，本轮没有实现 `5%` ST 涨停规则

### 重跑实验矩阵

| 波次 | 模型 | 配置 | folds | 日志目录 | 状态 |
| --- | --- | --- | --- | --- | --- |
| `Rerun-LU-1` | `DLinear` | `FeatureSet_A, seq_len=120` | `2015-2024` | `logs/market_full_dlinear120A_limitup_rerun/` | `done` |
| `Rerun-LU-2` | `PatchTST` | `FeatureSet_A, seq_len=60` | `2015-2024` | `logs/market_full_patchtst60A_limitup_rerun/` | `done` |
| `Rerun-LU-3` | `iTransformer` | `FeatureSet_A, seq_len=20` | `2015-2024` | `logs/market_full_itransformer20A_limitup_rerun/` | `done` |

### GPU 调度表

本轮为了最短墙钟时间，使用了非对称调度：

| GPU | 任务 | 说明 |
| --- | --- | --- |
| `GPU0` | `DLinear sl120` | 单卡串行跑 10 个 fold |
| `GPU1` | `PatchTST sl60` | 与 `GPU3` 分摊最慢模型 |
| `GPU2` | `iTransformer sl20` | 单卡串行跑 10 个 fold |
| `GPU3` | `PatchTST sl60` | 与 `GPU1` 并行，总共双卡 2 实例 |

并发过程中发现并修复了主特征缓存并发写损坏问题：

- `cache/market_daily_features.parquet` 改为文件锁 + 原子写
- 已损坏缓存会自动删除并重建
- 修复后 30/30 folds 全部成功完成，零失败

### 新口径全折结果

以下指标基于 10 个 test years 全部串接后的真实日收益序列统计：

| 模型 | 配置 | folds | days | 日均收益 | 累计收益 | Sharpe | 胜率 | 最大回撤 | 正收益年份 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DLinear` | `seq_len=120` | `10` | `2340` | `0.004779` | `4.6605` | `0.7905` | `0.4919` | `-0.9804` | `6/10` |
| `iTransformer` | `seq_len=20` | `10` | `2340` | `0.002515` | `-0.9271` | `0.4760` | `0.4876` | `-0.9984` | `4/10` |
| `PatchTST` | `seq_len=60` | `10` | `2340` | `0.000548` | `-0.9958` | `0.1135` | `0.4739` | `-0.9979` | `4/10` |

结论变化非常明显：

- `DLinear sl120` 现在是唯一一个全周期串接后仍然保持显著正累计收益的主模型
- `iTransformer sl20` 仍然有高弹性年份，但剔除涨停不可买样本后，全周期优势大幅回落
- `PatchTST sl60` 的旧优势主要集中在少数年份，放到新交易口径下不再适合作为主策略

### 分年最佳模型

按每个 test year 的 `mean_return` 选 winner：

| Test Year | Winner | mean_return | cumulative_return |
| --- | --- | --- | --- |
| `2015` | `iTransformer sl20` | `0.006992` | `1.7032` |
| `2016` | `PatchTST sl60` | `0.000502` | `-0.1674` |
| `2017` | `iTransformer sl20` | `0.000390` | `-0.1359` |
| `2018` | `DLinear sl120` | `0.002857` | `0.1358` |
| `2019` | `DLinear sl120` | `0.012708` | `2.4300` |
| `2020` | `DLinear sl120` | `0.014037` | `4.7446` |
| `2021` | `iTransformer sl20` | `0.014232` | `5.1295` |
| `2022` | `iTransformer sl20` | `0.003126` | `0.0892` |
| `2023` | `DLinear sl120` | `0.004613` | `1.3964` |
| `2024` | `DLinear sl120` | `0.018629` | `0.0305` |

对应的 hindsight year selector 上界：

- 日均收益：`0.007428`
- Sharpe：`1.1512`
- 累计收益：`716.7218`

但这个 selector 只是一条离线收益上界，不能直接当成可部署结论，因为它使用了事后最优年份标签。

### 当前正式结论

如果现在要定主策略，建议直接改成：

1. `正式主策略`：`DLinear + FeatureSet_A + seq_len=120`
2. `研究增强线`：只把 `iTransformer sl20` 作为候选副模型，用于后续 regime selector，而不是与 `DLinear` 平权混合

原因：

- `DLinear sl120` 在新交易口径下全周期最稳，且 `2019/2020/2023/2024` 这些关键年份收益最强
- `iTransformer sl20` 的优势只在 `2015/2021/2022` 等局部阶段成立，跨周期回撤过深
- `PatchTST sl60` 不再值得作为主线继续追加大规模算力

### 下一步最优实验顺序

基于这轮重跑后的新口径，后续实验优先级应调整为：

1. 以 `DLinear sl120` 作为不可动摇的主 baseline，先做特征集 `B/C/D` 和轻量 loss 复核
2. 只对 `iTransformer sl20` 做少量增强实验，目标不是单独替代主策略，而是验证能否补足 `DLinear` 的局部 regime
3. 暂停 `PatchTST sl60` 的全量扩展，除非有明确的新特征/新损失假设

## 2026-05-17 Loss Upgrade 第一轮结果

这一轮只做训练目标改造，不碰组合策略。

### 设计

- Backbone 不变：
  - `DLinear seq_len=120`
  - `iTransformer seq_len=20`
- 在预测层外侧增加通用 `cls_head`
- 总损失：

```text
loss = Huber + alpha * BCE
```

- 正式评估口径保持不变：
  - 先过滤 `t+1` 开盘涨停不可买样本
  - 再做按日 `top1`

### 实验矩阵

| Wave | Model | Folds | Alpha | Log Dir | Status |
| --- | --- | --- | --- | --- | --- |
| `L1` | `DLinear sl120` | `2019,2020,2021,2023` | `0.2` | `logs/market_loss_dlinear120_hbce_a02/` | `done` |
| `L2` | `DLinear sl120` | `2019,2020,2021,2023` | `0.5` | `logs/market_loss_dlinear120_hbce_a05/` | `done` |
| `L3` | `DLinear sl120` | `2019,2020,2021,2023` | `1.0` | `logs/market_loss_dlinear120_hbce_a10/` | `done` |
| `L4` | `iTransformer sl20` | `2015,2021,2022,2024` | `0.2` | `logs/market_loss_itransformer20_hbce_a02/` | `done` |
| `L5` | `iTransformer sl20` | `2015,2021,2022,2024` | `0.5` | `logs/market_loss_itransformer20_hbce_a05/` | `done` |

总计：

## 2026-05-18 P1 稳定性审计与代理实盘验证

这一节覆盖上面关于 “P1 是否可以进入实盘前验证” 的最新结论。当前正式比较对象已经固定为：

- 左模型：`DLinear sl120 + Huber + BCE(alpha=0.1)`
- 右模型：`TimesNet sl20 + Huber + BCE(alpha=0.1)`
- 全部结果都只按 `可交易收益率` 统计：
  - 先剔除 `t+1` 开盘即涨停、无法买入的股票
  - 再按日选 `top1`
  - 收益定义始终是 `open_{t+2} / open_{t+1} - 1`

重要校正：

- 2026-05-18 之前产出的所有 `market_daily` 结果都混入了 `target_shifted` 泄露特征
- 这意味着旧的 `P0/P1` 收益和 selector 结论不能再当作正式上线依据
- 从这一节开始，才是修正后的正式口径

### P1 稳定性审计结论

先对原始 `top1_gap selector` 做稳定性审计，关键结果如下：

| 指标 | 数值 |
| --- | --- |
| `num_days` | `1372` |
| `mean_return` | `0.014907` |
| `sharpe` | `0.9750` |
| `switch_rate` | `35.30%` |
| `left_usage_rate` | `23.91%` |
| `right_usage_rate` | `76.09%` |
| `disagreement_rate` | `95.48%` |
| `alpha_vs_left_mean` | `+0.006704` |
| `alpha_vs_right_mean` | `+0.000603` |

解释：

- 原始 selector 不是伪 alpha，确实比两边单模型都多赚
- 但它换手太高，而且大部分时间都在重写 `TimesNet` 的 `top1`
- 所以它更像 “`TimesNet` 主导 + 少量 `DLinear` 修正”，还不够像可以直接部署的稳定策略

### 门限 selector 复核

随后加入门限 gating：

- 只在 `abs(confidence_edge)` 足够大时才启用 selector
- 否则回退到主基线 `DLinear`

关键候选如下：

| 策略 | 规则 | 日均收益 | 累计收益 | Sharpe | switch_rate |
| --- | --- | --- | --- | --- | --- |
| `gated_q80_left` | `q=0.80, fallback=DLinear` | `0.017567` | `355.3074` | `1.1233` | `15.32%` |
| `gated_q90_left` | `q=0.90, fallback=DLinear` | `0.013553` | `322.0044` | `1.2723` | `7.73%` |
| `selector_top1_gap` | 原始 selector | `0.014907` | `9.9207` | `0.9750` | `35.30%` |
| `dlinear_a01` | 单模型主线 | `0.008203` | `27.4181` | `1.0244` | - |

结论：

- 如果目标是 `最高收益`，当前第一名已经变成 `gated_q80_left`
- 如果目标是 `更稳的 Sharpe / 更低切换`，`gated_q90_left` 是最强 shadow candidate
- 原始 selector 不再应该作为正式候选，因为它被门限版完全支配

对应文件：

- `logs/market_p1_selector_audit/selector_summary.json`
- `logs/market_p1_selector_audit/selector_year_audit.csv`
- `logs/market_p1_selector_audit/threshold_grid.csv`

### 代理实盘验证

为了回答 “离实盘还差多远”，新增了代理实盘成本验证：

- 先把策略物化成逐日 `top1` 持仓序列
- 如果连续两天选中同一只股票，则视为连续持有，不重复扣双边换手成本
- 费用只作为代理情景，不改主研究口径

默认三组情景：

- `low`：买入 `5bps`，卖出 `10bps`
- `base`：买入 `8bps`，卖出 `13bps`
- `high`：买入 `12bps`，卖出 `17bps`

`base` 情景结果如下：

| 策略 | 日均净收益 | 净累计收益 | 净 Sharpe | switch_count | same_code_streak_days |
| --- | --- | --- | --- | --- | --- |
| `gated_q80_left` | `0.015719` | `28.0525` | `1.0061` | `1192` | `179` |
| `selector_top1_gap` | `0.013189` | `0.0534` | `0.8634` | `1112` | `259` |
| `timesnet_hbce_a01` | `0.012202` | `-0.4584` | `0.8048` | `1085` | `286` |
| `gated_q90_left` | `0.011668` | `24.0418` | `1.0971` | `1216` | `155` |
| `dlinear_a01` | `0.006261` | `1.0213` | `0.7832` | `1257` | `114` |

解读：

- 在引入交易成本代理之后，`gated_q80_left` 仍然是 `净收益第一`
- `gated_q90_left` 的净 Sharpe 最高，说明它更适合作为低切换、稳健版备选
- `TimesNet` 单独部署已经不成立；净值在 `base/high` 情景下明显劣化
- `DLinear` 单独部署仍然是合格 baseline，但已经被门限 selector 显著超过

对应结果文件：

- `logs/market_live_proxy/gross_summary.csv`
- `logs/market_live_proxy/proxy_summary.csv`
- `logs/market_live_proxy/proxy_year_summary.csv`
- `logs/market_live_proxy/gated_q80_left_daily.csv`
- `logs/market_live_proxy/gated_q90_left_daily.csv`

### 当前正式候选

如果现在要进入 `pre-live / paper trading`，建议正式候选改成两条：

1. `主候选`：`gated_q80_left`
2. `稳健影子组合`：`gated_q90_left`

不再建议把下面两条当成上线主方案：

- 原始 `selector_top1_gap`
- `TimesNet` 单模型

### 距离实盘还差什么

到这一步，高价值的模型层探索已经基本收敛，但还不能直接说“可以实盘上线”。还差至少四类检查：

1. `交易规则补全`
   - 当前没有 ST 标记，因此还没有纳入 `5%` 涨跌停约束
   - 需要确认数据里是否还有停牌、一字板、异常复权等可交易性问题
2. `代理成本再细化`
   - 当前是 bps 情景法
   - 上线前最好按成交额、换手率、价位分层做更细的冲击成本代理
3. `稳健性闸门`
   - 要为 `2017/2022` 这类弱年设计停用规则或风险闸门
   - 至少需要一个简单的市场状态过滤器，否则回撤年份还是偏硬
4. `仿真上线`
   - 先跑一段严格 paper trading / replay
   - 验证每日候选生成、不可交易过滤、真实委托可成交性和监控告警链路

### 当前判断

到 2026-05-18 这个节点，可以认为：

- `模型层`：高价值探索已经接近收敛
- `策略层`：`门限 selector` 已经替代纯单模型成为当前最优候选
- `上线条件`：还差交易规则补全、成本代理细化、风险闸门和一段仿真盘验证

也就是说，下一步最值钱的工作，不再是继续大规模扩 backbone，而是：

1. 把 `gated_q80_left / gated_q90_left` 接入 paper trading
2. 做交易规则与成本建模补全
3. 再决定最终上线是选 `q80` 还是 `q90`

- `20` 个实验
- `0` 失败

### 汇总结果

| Model | Alpha | Folds | 平均 `mean_return` | 相对原基线平均增量 | 平均 `sharpe` | 相对原基线平均增量 |
| --- | --- | --- | --- | --- | --- | --- |
| `DLinear` | `0.2` | `4` | `0.011290` | `+0.002383` | `1.4916` | `+0.1145` |
| `DLinear` | `0.5` | `4` | `0.012380` | `+0.003472` | `1.4538` | `+0.0767` |
| `DLinear` | `1.0` | `4` | `0.009349` | `+0.000441` | `1.0883` | `-0.2888` |
| `iTransformer` | `0.2` | `4` | `0.002272` | `-0.006442` | `0.2849` | `-1.0300` |
| `iTransformer` | `0.5` | `4` | `0.003729` | `-0.004984` | `0.4211` | `-0.8939` |

### 关键观察

- `DLinear + alpha=0.2` 是这一轮唯一明确值得保留的方向：
  - `2019` 从 `0.012708` 提升到 `0.013542`
  - `2021` 从 `0.004273` 提升到 `0.016154`
- `DLinear + alpha=0.5` 看起来均值不差，但提升几乎集中在 `2021`
- `DLinear + alpha=1.0` 明显过强，已经开始伤害主任务排序
- `iTransformer + BCE` 在代表年份上几乎全面退化，不值得继续烧 GPU

### 当前决策更新

因此，loss 改造后的优先级变成：

1. `DLinear sl120 + Huber + BCE(alpha=0.2)`：继续扩大验证
2. `DLinear sl120 + Huber + BCE(alpha=0.1 / 0.3)`：做小范围精细化搜索
3. 暂停 `iTransformer` 的 BCE 多任务线

### 下一轮建议

下一轮不要再横向扩 `iTransformer` 或 `PatchTST` 的 loss。

最优路径是：

1. 把 `DLinear sl120 + alpha=0.2` 扩大到 `2015-2024` 全部 folds
2. 同时补 `alpha=0.1 / 0.3`
3. 如果全折仍成立，再考虑把它作为新主 baseline 接入后续 regime selector

## 2026-05-17 Loss Upgrade 全折扩展结果

上面那轮代表 fold 之后，已经继续把 `DLinear sl120 + Huber + BCE` 扩大到 `2015-2024` 全部 folds，并且仍然只按可交易收益率统计：

- 先剔除 `t+1` 开盘涨停不可买样本
- 再按日做 `top1`

### 全折实验矩阵

| Wave | Model | Folds | Alpha | Log Dir | Status |
| --- | --- | --- | --- | --- | --- |
| `L6` | `DLinear sl120` | `2015-2024` | `0.1` | `logs/market_loss_dlinear120_hbce_a01_full/` | `done` |
| `L7` | `DLinear sl120` | `2015-2024` | `0.2` | `logs/market_loss_dlinear120_hbce_a02_full/` | `done` |
| `L8` | `DLinear sl120` | `2015-2024` | `0.3` | `logs/market_loss_dlinear120_hbce_a03_full/` | `done` |

总计：

- `30` 个 fold
- `0` 失败

### 全折汇总

对照 baseline：

- `DLinear sl120 baseline`
  - `mean_return = 0.004779`
  - `cumulative_return = 4.6605`
  - `sharpe = 0.7905`

新结果：

| Alpha | Folds | Days | Mean Return | Cumulative Return | Sharpe | Win Rate | Max Drawdown | Positive Years | Avg mean delta vs baseline | Avg sharpe delta vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0.1` | `10` | `2340` | `0.006059` | `45.2618` | `0.8831` | `0.4868` | `-0.9838` | `6/10` | `+0.001608` | `+0.0541` |
| `0.2` | `10` | `2340` | `0.004440` | `0.4513` | `0.6863` | `0.4859` | `-0.9936` | `4/10` | `-0.000470` | `-0.2226` |
| `0.3` | `10` | `2340` | `0.005225` | `5.5790` | `0.7932` | `0.4838` | `-0.9891` | `4/10` | `+0.000105` | `-0.1073` |

### 结论修正

这轮全折结果说明：

- 代表 fold 上看起来最好的 `alpha=0.2`，扩到全折后并不成立
- 真正全折最优的是 `alpha=0.1`
- `alpha=0.3` 有一定收益弹性，但稳定性和 Sharpe 仍不如 `alpha=0.1`

因此，loss 改造线的正式结论应更新为：

1. `DLinear sl120 + Huber + BCE(alpha=0.1)` 是当前最优单模型升级版
2. `DLinear sl120 baseline` 仍然保留为最重要对照
3. `alpha=0.2` 不再继续
4. `iTransformer + BCE` 不再继续

### 当前主策略更新

如果现在要在“纯单模型”层面做正式主策略更新，优先级应该改成：

1. `正式主策略候选A`：`DLinear sl120 + Huber + BCE(alpha=0.1)`
2. `正式主策略候选B`：原始 `DLinear sl120 baseline`

后续只需要围绕这两条线继续比较：

- 是否在更多年份 / 更新数据上仍然保持 `alpha=0.1` 优势
- 是否要把 `alpha=0.1` 版本作为 regime selector 的主输入之一

## 2026-05-17 代表性模型类别扩展方案

当前已经确认：

- 正式主口径固定为“可交易 `Top1` 收益率”
- `DLinear sl120 + Huber + BCE(alpha=0.1)` 是当前最优单模型
- 但不能只靠 `DLinear` 做后续研究，需要补齐其他结构类别的代表性证据

这一轮的目标不是立刻替代 `DLinear`，而是回答两个问题：

1. 除线性族之外，是否还有值得保留的模型类别
2. 哪些类别应进入下一轮更深的 loss / 超参 / 组合研究

### 类别分层

| 类别 | 代表模型 | 当前状态 | 研究目的 |
| --- | --- | --- | --- |
| 线性稳健型 | `DLinear` | 已完成 baseline 和 loss 升级 | 作为主 baseline 与正式对照 |
| 截面 transformer | `iTransformer` | baseline 已完成，`BCE` 线失败 | 判断是否只适合局部 regime |
| patch/local encoder | `PatchTST` | baseline 已完成，整体偏弱 | 判断是否仍有局部年份价值 |
| MLP mixing | `TSMixer` | 待补类别证据 | 判断是否具备比 `DLinear` 更强的低复杂度表达 |
| 频域 / 多周期 | `TimesNet` | 待补类别证据 | 判断多周期结构能否适应 A 股高噪声日频 |

### 本轮假设

- `TSMixer` 可能是“比 `DLinear` 更有表达力，但仍保持轻量稳健”的候选
- `TimesNet` 可能在部分趋势年份受益，但也更容易被高噪声扰动
- 若这两个类别在代表年份都明显弱于 `DLinear`，则下一轮不再为它们投入 loss 改造算力

### 实验矩阵

只跑代表性 probe，不直接上全折：

| Wave | Model | Seq Len | Folds | Loss | Des | 目的 |
| --- | --- | --- | --- | --- | --- | --- |
| `C1` | `TSMixer` | `20` | `2015,2019,2021,2024` | `Huber` | `market_class_tsmixer20_probe` | 短窗 MLP mixing probe |
| `C2` | `TSMixer` | `60` | `2015,2019,2021,2024` | `Huber` | `market_class_tsmixer60_probe` | 中窗 MLP mixing probe |
| `C3` | `TimesNet` | `20` | `2015,2019,2021,2024` | `Huber` | `market_class_timesnet20_probe` | 短窗多周期 probe |
| `C4` | `TimesNet` | `60` | `2015,2019,2021,2024` | `Huber` | `market_class_timesnet60_probe` | 中窗多周期 probe |

总计：

- `2` 个新类别
- `4` 条代表性配置
- `16` 个实验

### GPU 调度表

本轮优先“多实验并行”，不做单任务多卡。

| GPU | 任务 | 并发 | 说明 |
| --- | --- | --- | --- |
| `GPU0` | `TSMixer sl20` | `1` | 4 个 fold 串行 |
| `GPU1` | `TSMixer sl60` | `1` | 4 个 fold 串行 |
| `GPU2` | `TimesNet sl20` | `1` | 4 个 fold 串行 |
| `GPU3` | `TimesNet sl60` | `1` | 4 个 fold 串行 |

如某类显存占用明显偏低，再把该 GPU 提升到 `max_jobs_per_gpu=2` 做第二轮并发。

### 成功标准

某个模型类别只有在满足以下条件时，才进入下一轮深挖：

- 相比自身两个 `seq_len`，至少有一个窗口表现稳定
- 相比 `DLinear` 正式 baseline，不要求全面超越，但不能在代表年份上全面退化
- 至少 `4` 个代表年份中的 `2` 个年份，`mean_return` 不差于当前次优类别

### 本轮后的决策门槛

跑完后按以下优先级做决策：

1. 若 `TSMixer` 明显优于 `TimesNet`，则下一轮只保留 `TSMixer`
2. 若二者都明显弱于 `DLinear` 和 `iTransformer`，则停止新增类别扩展
3. 若其中一个类别在局部年份很强但宽度不足，则降级为“后续 ensemble / selector 候选”，不晋升为主单模型线

## 2026-05-17 类别扩展执行进展

### 当前运行目录

- `logs/market_class_tsmixer20_probe/`
- `logs/market_class_tsmixer60_probe/`
- `logs/market_class_timesnet20_probe_v3/`
- `logs/market_class_timesnet60_probe_v3/`

旧的 `TimesNet` 失败目录保留为工程证据，不纳入模型优劣比较：

- `logs/market_class_timesnet20_probe/`
- `logs/market_class_timesnet60_probe/`
- `logs/market_class_timesnet20_probe_v2/`
- `logs/market_class_timesnet60_probe_v2/`

### TSMixer 已完成结果

截至当前，`TSMixer` 已经完成以下代表 fold：

| Model | Seq Len | Fold | Mean Return | Cumulative Return | Sharpe | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| `TSMixer` | `20` | `2015` | `0.000082` | `-0.4499` | `0.0184` | 基本无效 |
| `TSMixer` | `20` | `2019` | `0.014736` | `2.4888` | `1.9613` | 很强 |
| `TSMixer` | `20` | `2021` | `-0.005082` | `-0.9625` | `-0.6925` | 失效 |
| `TSMixer` | `20` | `2024` | `0.037933` | `9.9616` | `2.3768` | 极强 |
| `TSMixer` | `60` | `2015` | `0.006221` | `1.3696` | `1.3461` | 较好 |
| `TSMixer` | `60` | `2019` | `0.000390` | `-0.9790` | `0.0368` | 很弱 |
| `TSMixer` | `60` | `2021` | `0.011506` | `1.8633` | `1.5145` | 较强 |
| `TSMixer` | `60` | `2024` | `0.016234` | `-0.5088` | `1.0974` | 收益均值尚可，但累计路径差 |

当前判断：

- `TSMixer` 不是稳健型替代品
- `sl20` 和 `sl60` 对不同年份敏感，明显存在 regime 依赖
- 这类模型更适合保留为：
  - `ensemble` 候选
  - `selector` 候选
  - 特定年份增强器

但暂时不应直接替代当前主线 `DLinear + BCE(alpha=0.1)`。

### TimesNet 工程兼容问题

`TimesNet` 在当前 `market_daily` 链路上先后暴露出两类实现兼容问题：

#### 问题1：时间嵌入维度不匹配

原始 `TimesNet + embed=timeF + freq=d` 直接报错：

- `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

根因：

- `market_daily` 当前时间特征是 `4` 维：`month/day/weekday/hour`
- `TimeFeatureEmbedding(freq='d')` 只接受 `3` 维输入

修复：

- 对 `TimesNet` 实验固定改用 `--embed fixed`

#### 问题2：AMP 下 FFT 长度限制

修复时间嵌入后，又出现：

- `RuntimeError: cuFFT only supports dimensions whose sizes are powers of two when computing in half precision`

根因：

- `TimesNet` 内部直接走 `torch.fft.rfft`
- 当前 `seq_len + pred_len` 为 `21` 或 `61`
- 在 AMP 半精度下触发 cuFFT 长度限制

修复：

- 将 `TimesNet` 从 `AMP` 白名单中移除
- 保留其他模型继续使用 `AMP`

因此，当前正式有效的 `TimesNet` 运行版本是：

- `embed = fixed`
- `use_amp = False`

### 当前 promotion 判断

如果 `TimesNet v3` 能顺利跑通，下一步只需要回答一个问题：

- 它是否能在代表年份上比 `TSMixer` 更稳

如果不能，则类别扩展线的正式结论很可能会收敛到：

1. `DLinear + BCE(alpha=0.1)` 继续作为主线
2. `TSMixer` 保留为高弹性次级类别
3. `TimesNet` 不再继续投入 full-fold 算力

## 2026-05-17 类别扩展最终汇总与下一步决策

到这一轮结束，代表性类别实验已经形成足够完整的证据：

- `DLinear + Huber + BCE(alpha=0.1)`：主线已稳定
- `TSMixer`：高弹性，但不稳
- `TimesNet sl20/sl60`：类别可用性已验证
- `TimesNet sl20 + BCE(alpha=0.05/0.1/0.2)`：loss 放大效应已验证

### 一、主线模型仍然是 DLinear

当前最强、最稳的正式单模型仍是：

- `DLinear sl120 + Huber + BCE(alpha=0.1)`

近年代表 fold 结果：

| Fold | Mean Return | Cumulative Return | Sharpe |
| --- | --- | --- | --- |
| `2019` | `0.012025` | `2.0041` | `1.3477` |
| `2021` | `0.013487` | `3.0722` | `1.3455` |
| `2023` | `0.004331` | `1.2772` | `1.5796` |
| `2024` | `0.029599` | `3.8452` | `1.9862` |

平均：

- `avg_mean_return = 0.014861`
- `avg_sharpe = 1.564715`

结论：

- 这条线不需要再做重复复核
- 后续若继续提升主线，重点应该从 `BCE alpha` 微调转向更接近 `Top1` 目标的排序型 loss

### 二、TSMixer 结论：保留为高弹性候选，但停止继续做 BCE

`TSMixer` 基线结果显示出明显 regime 依赖：

- `sl20` 在 `2019/2024` 很强
- `sl60` 在 `2015/2021` 更稳
- 但两者都不能形成稳定主线

进一步加入 `BCE(alpha=0.1)` 后：

- `sl20`：
  - `2019` 退化
  - `2021` 仅小幅改善
  - `2024` 退化
- `sl60`：
  - `2019` 改善
  - `2021` 明显退化
  - `2024` 基本持平

因此正式结论是：

1. `TSMixer` 不适合作为当前主线替代
2. `TSMixer + BCE(alpha=0.1)` 不值得继续扩展
3. `TSMixer` 只保留为：
   - 后续 `ensemble` 候选
   - `selector` 候选
   - 局部年份增强器

### 三、TimesNet 结论：sl20 值得保留，sl60 只作为次级对照

#### 1. TimesNet 基线

`TimesNet sl20`：

| Fold | Mean Return | Cumulative Return | Sharpe |
| --- | --- | --- | --- |
| `2015` | `0.006938` | `1.6854` | `1.4560` |
| `2019` | `0.001497` | `-0.7971` | `0.2019` |
| `2021` | `0.016013` | `3.1815` | `1.7223` |
| `2024` | `0.020722` | `1.5538` | `1.8489` |

`TimesNet sl60`：

| Fold | Mean Return | Cumulative Return | Sharpe |
| --- | --- | --- | --- |
| `2015` | `0.000706` | `-0.3800` | `0.1538` |
| `2019` | `0.009445` | `0.3214` | `1.0044` |
| `2021` | `0.012320` | `1.4679` | `1.4503` |
| `2024` | `0.020959` | `0.0777` | `1.6105` |

解读：

- `sl20` 总体上比 `TSMixer` 更稳
- `sl60` 也能在部分年份工作，但训练太慢，且并不明显优于 `sl20`

因此：

- 保留 `TimesNet sl20`
- `TimesNet sl60` 不进入主优先级，只保留为次级结构对照

#### 2. TimesNet sl20 的 BCE loss 对照

为检验 `TimesNet sl20` 是否适合加入方向监督，已完成：

- `alpha=0.05`
- `alpha=0.1`
- `alpha=0.2`

结果如下。

`alpha=0.05`：

| Fold | Mean Return | Sharpe |
| --- | --- | --- |
| `2019` | `-0.000770` | `-0.0949` |
| `2021` | `0.011809` | `1.1947` |
| `2024` | `0.096587` | `2.4052` |

`alpha=0.1`：

| Fold | Mean Return | Sharpe |
| --- | --- | --- |
| `2015` | `0.004305` | `0.8905` |
| `2019` | `0.004525` | `0.5195` |
| `2021` | `0.012473` | `1.2693` |
| `2024` | `0.094689` | `2.3616` |

`alpha=0.2`：

| Fold | Mean Return | Sharpe |
| --- | --- | --- |
| `2019` | `0.000317` | `0.0393` |
| `2021` | `0.017585` | `1.8550` |
| `2024` | `0.083199` | `2.0886` |

对比基线可得：

- 在 `2019` 这种较弱年份，`alpha=0.1` 最有帮助
- 在 `2021`，`alpha=0.2` 最强
- 在 `2024`，`alpha=0.05/0.1/0.2` 都会显著放大收益，`0.05` 均值最高
- 但三者都没有形成一个“所有年份都稳定占优”的统一 alpha

正式判断：

1. `TimesNet sl20 + BCE` 这条线是有效的
2. 它不是稳定型改进，而是收益放大型改进
3. 当前最值得保留的是：
   - `alpha=0.1`
   - `alpha=0.2`
4. `alpha=0.05` 在 `2024` 非常强，但泛化宽度偏弱，作为次级对照保留

### 四、当前类别排序

如果只看当前单模型/类别层面的综合排序，可写成：

1. `DLinear sl120 + BCE(alpha=0.1)`：正式主线
2. `TimesNet sl20 baseline / BCE(alpha=0.1~0.2)`：最值得保留的 challenger 类别
3. `TSMixer sl20/sl60 baseline`：只保留为高弹性辅助候选
4. `TimesNet sl60`：完成证据即可，不再继续深挖

### 五、下一步决策

后续不再继续做这些方向：

- `TSMixer + BCE`
- `DLinear` 的重复复核
- `TimesNet sl60` 的更多扩展
- `BCE alpha` 的无限细化搜索

下一步只保留两个高价值方向：

#### 方向A：DLinear 的排序型 loss

目标：

- 把当前最稳主线从“回归 + 方向监督”推进到更接近 `Top1` 决策目标

建议实验：

- `DLinear sl120`
- `Huber + pairwise rank loss`
- 代表年份：
  - `2019`
  - `2021`
  - `2024`

这是下一轮最重要的方向。

#### 方向B：TimesNet sl20 的正式 challenger 验证

目标：

- 判断 `TimesNet sl20 + BCE(alpha=0.1 / 0.2)` 是否值得升级为正式 challenger

建议实验：

- 只保留 `sl20`
- 只保留 `alpha=0.1` 和 `alpha=0.2`
- 从代表年份扩大到更宽窗口

例如：

- `2015,2017,2019,2021,2022,2024`

如果这轮宽样本仍成立，再考虑 full-fold。

## P0 收口与 P1 自动接续

### 当前 P0 状态

- `DLinear sl120 + Huber + BCE(alpha=0.1)` 仍是当前正式主线
- `DLinear + pairwise rank loss` 已完成代表 fold 验证，但没有超过主线，不再扩展
- `TimesNet sl20` 的最后一轮宽样本 challenger 对照已经补齐到：
  - baseline
  - `BCE(alpha=0.1)`
  - `BCE(alpha=0.2)`
  - 对齐年份：`2015,2017,2019,2021,2022,2024`

P0 的最后动作不再是继续训练新 backbone，而是：

1. 完成 `TimesNet` challenger 的 6 年对齐比较
2. 正式选择唯一 challenger
3. 关闭不再继续的方向：
   - `TSMixer + BCE`
   - `TimesNet sl60`
   - `DLinear rank loss`

### P1 的正式定义

P1 不再是“继续横向扫更多 backbone 或特征集”，而是进入策略层研究：

- 输入：P0 已完成的单模型预测文件 `top1_predictions.csv`
- 协议：仍然严格使用可交易收益率
- 目标：在不改变标签与交易规则的前提下，验证是否可以通过双模型集成/选择器提升最终 `top1` 可执行收益

P1 只允许回答一个核心问题：

> `DLinear` 主线和 `TimesNet` challenger 的差异性，能否转化为比任一单模型更高的真实可交易 `top1` 收益。

### P1 候选结构

P1 当前只保留两个模型族：

1. `DLinear sl120 + Huber + BCE(alpha=0.1)`：主模型
2. `TimesNet sl20` 的最终 challenger 版本：副模型

不再引入第三个 backbone，原因是：

- 当前其他结构缺乏足够宽样本证据
- 现在的高价值问题不是“还有没有别的弱模型”，而是“主模型与 challenger 是否存在可利用的互补性”

### P1 实验矩阵

所有 P1 评估都只基于 `2015,2017,2019,2021,2022,2024` 这 6 个已对齐年份。

#### P1-A：平权集成

- `rank_mean`
- `zscore_mean`
- `raw mean`

目的：

- 判断简单双模型融合是否能在不增加策略复杂度的情况下改善宽样本收益

#### P1-B：置信度选择器

- `top1_gap`
- `top1_gap_rank`

机制：

- 每个交易日分别计算两个模型的 `top1-top2` 分数间距
- 间距更大的模型被视为当日更有把握
- 当日全部股票分数采用该模型输出

目的：

- 判断“只在更有把握时切换模型”是否比平权融合更有效

### 自动化实施

已补充两个脚本：

- `scripts/market_daily/evaluate_ensembles.py`
  - 输入多个 `top1_predictions.csv`
  - 输出单模型 / 平权集成 / selector 的逐年与 aggregate 指标
- `scripts/market_daily/run_p1_research.py`
  - 先等待 P0 最后一个训练进程结束
  - 自动汇总 `TimesNet` challenger 三个版本
  - 选择正式 challenger
  - 立即执行 P1 集成与 selector 评估
  - 产出：
    - `logs/market_p1/timesnet_challenger_summary.csv`
    - `logs/market_p1/p1_ensemble_summary.csv`
    - `logs/market_p1/p1_decision.json`

### P1 的晋升标准

只有满足以下条件，P1 结果才算对实盘前推进有价值：

1. aggregate `mean_return` 高于当前 `DLinear` 主线
2. aggregate `sharpe` 不明显恶化
3. 6 个年份中至少 `4/6` 年不差于主线
4. 改进不是只由 `2024` 单一年份驱动
5. 策略逻辑足够简单，可被实盘系统稳定复现

如果 P1 不满足以上条件，则结论应当是：

- 单模型 `DLinear` 继续作为正式候选
- `TimesNet` 只保留为研究辅助，不进入实盘第一版

## P0 完成后的正式收口结果

### 1. TimesNet challenger 最终选择

`P0` 完成后，`TimesNet sl20` 的 6 年对齐 challenger 结果如下：

| Variant | Years | Mean Return | Cumulative Return | Sharpe |
| --- | --- | --- | --- | --- |
| `TimesNet baseline` | `2015,2017,2019,2021,2022,2024` | `0.006123` | `0.010599` | `0.8732` |
| `TimesNet + BCE(alpha=0.1)` | `2015,2017,2019,2021,2022,2024` | `0.013878` | `4.304503` | `0.9145` |
| `TimesNet + BCE(alpha=0.2)` | `2015,2017,2019,2021,2022,2024` | `0.012437` | `0.835562` | `0.8304` |

正式 challenger 选择为：

- `TimesNet sl20 + Huber + BCE(alpha=0.1)`

原因：

- 6 年 aggregate `mean_return` 最高
- aggregate `sharpe` 也优于 `alpha=0.2`
- 在 `2024` 上维持最强爆发，同时没有牺牲掉 `2015/2019/2021`

### 2. P1 正式结果

修正 aggregate 口径后，`P1` 的 6 年总汇总结果如下：

| Kind | Method | Members | Mean Return | Cumulative Return | Sharpe |
| --- | --- | --- | --- | --- | --- |
| `selector_aggregate` | `top1_gap` | `DLinear + TimesNet(a0.1)` | `0.014907` | `9.920656` | `0.9750` |
| `single_aggregate` | `single` | `TimesNet(a0.1)` | `0.013878` | `4.304503` | `0.9145` |
| `pair_aggregate` | `zscore_mean` | `DLinear + TimesNet(a0.1)` | `0.013514` | `73.924211` | `0.9156` |
| `single_aggregate` | `single` | `DLinear(a0.1)` | `0.008203` | `27.418124` | `1.0244` |

解读：

- 如果只看 `mean_return`，`top1_gap selector` 是当前 6 年 aggregate 最优
- 但它的 `sharpe=0.9750` 仍然低于单模型 `DLinear` 的 `1.0244`
- `TimesNet` 本身和集成线都在收益均值上明显强于 `DLinear`
- 但波动控制和跨年稳定性还没有强到可以直接替代单模型主线

### 3. P1 是否已经足够晋升

当前结论是：

- `P1` 已经证明 `TimesNet challenger` 与 `DLinear` 存在真实互补性
- `top1_gap selector` 是目前最值得保留的策略层增强线
- 但现在还**不建议**直接把它作为第一版实盘正式策略

原因：

1. `selector` 的 aggregate `sharpe` 没有超过单模型 `DLinear`
2. `2017` 仍是明显弱点年份
3. 当前收益提升有一部分仍明显依赖 `2024` 的高弹性行情

### 4. 当前正式结论

如果此刻必须决定“谁先进入第一版实盘候选”，排序应为：

1. `DLinear sl120 + Huber + BCE(alpha=0.1)`：正式主候选
2. `DLinear + TimesNet(a0.1) + top1_gap selector`：最强研究增强线
3. `TimesNet sl20 + BCE(alpha=0.1)`：正式 challenger，但暂不单独上线

### 5. 下一步最高价值方向

在当前节点，不再优先做新的 backbone 扩展，而是只做两件高价值工作：

1. 对 `top1_gap selector` 做更严格稳定性验证
   - 检查逐年换手
   - 检查切换频率
   - 检查是否只在少数极端日贡献收益
2. 做实盘前代理验证
   - 加入手续费/滑点
   - 加入实际下单约束
   - 做部署级 rerun 与可复现实验

如果这两项通过，才接近“可以实盘第一版”的标准。

## Selector 稳定化审计与门控结果

在正式 `P1` 结果之上，已经继续做了 `top1_gap selector` 的稳定性审计与阈值门控实验。

### 1. 原始 selector 的稳定性审计

对象：

- `DLinear sl120 + BCE(alpha=0.1)`
- `TimesNet sl20 + BCE(alpha=0.1)`
- `selector = top1_gap`

6 年 aggregate 审计结果：

| Metric | Value |
| --- | --- |
| `mean_return` | `0.014907` |
| `sharpe` | `0.9750` |
| `switch_rate` | `35.30%` |
| `left_usage_rate` (`DLinear`) | `23.91%` |
| `right_usage_rate` (`TimesNet`) | `76.09%` |
| `disagreement_rate` | `95.48%` |
| `alpha_vs_left_mean` | `+0.006704` |
| `alpha_vs_right_mean` | `+0.000603` |

解读：

- 原始 selector 的收益提升是真实的，不是靠极少数异常日硬拉出来的
- 但切换率偏高，且几乎每天都在改写两边单模型的 top1
- 它本质上更像“以 `TimesNet` 为主、偶尔用 `DLinear` 修补”的高收益增强线

因此，原始 selector 还不适合直接作为第一版实盘主策略。

### 2. 门控思路

为降低 churn，测试以下规则：

- 先计算 `abs(confidence_edge) = abs(left_gap - right_gap)`
- 只有当 `abs(confidence_edge)` 超过某个分位阈值时，才启用 selector
- 其余日期直接回退到固定主模型

比较了两类 fallback：

1. `fallback = DLinear`
2. `fallback = TimesNet`

### 3. 最优门控结果

最佳综合配置为：

- `quantile = 0.80`
- `fallback = DLinear`

即：

- 只有 `abs(confidence_edge)` 进入全样本前 `20%` 的高置信度日，才启用 selector
- 其余 `80%` 左右的普通日，直接使用 `DLinear`

aggregate 结果：

| Strategy | Mean Return | Cumulative Return | Sharpe | Switch Rate | Active Selector Rate |
| --- | --- | --- | --- | --- | --- |
| 原始 `top1_gap selector` | `0.014907` | `9.920656` | `0.9750` | `35.30%` | `100%` |
| `q80 + fallback=DLinear` | `0.017567` | `355.307362` | `1.1233` | `15.32%` | `20.04%` |

这是当前最重要的发现：

- 收益均值进一步提高
- `Sharpe` 从 `0.9750` 提升到 `1.1233`
- 切换率从 `35.30%` 压到 `15.32%`

因此，这个版本比原始 selector 明显更接近可部署状态。

### 4. 逐年表现

`q80 + fallback=DLinear` 的逐年结果：

| Fold | Mean Return | Sharpe | Switch Rate |
| --- | --- | --- | --- |
| `2015` | `0.001013` | `0.2199` | `12.40%` |
| `2017` | `-0.000196` | `-0.0932` | `4.96%` |
| `2019` | `0.011353` | `1.0309` | `16.53%` |
| `2021` | `0.027339` | `2.1862` | `20.75%` |
| `2022` | `0.000448` | `0.1477` | `15.00%` |
| `2024` | `0.090128` | `2.2555` | `26.42%` |

解读：

- `2017` 仍然是最后的弱点年份，但已经比原始 selector 更接近中性
- `2019/2021/2024` 这几个关键收益年份表现明显更强
- `2022` 仍然偏弱，但保持正值

### 5. 当前最新结论

此时策略层的正式排序应更新为：

1. `门控 selector: q80 + fallback=DLinear`
2. `DLinear sl120 + BCE(alpha=0.1)`
3. `原始 top1_gap selector`
4. `TimesNet sl20 + BCE(alpha=0.1)`

其中第 1 条已经不是单纯研究想法，而是当前最值得进入“实盘前代理验证”的正式候选。
