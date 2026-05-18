# 实盘上线前检查清单

日期：`2026-05-18`

## 1. 研究与回测口径检查

### 已检查并通过

- `[PASS]` corrected rerun 已全部完成
  - `DLinear 10/10`
  - `iTransformer 10/10`
  - `PatchTST 10/10`
  - `TimesNet 10/10`
  - `TimeMixer 10/10`

- `[PASS]` 研究主文档已更新为 corrected 结论
  - 文件：[2010_top1_quant_research_plan.md](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/docs/2010_top1_quant_research_plan.md)

- `[PASS]` 当前主方案与备选方案结果文件存在
  - 单模总表：[core5_model_ranking.csv](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/logs/market_corrected_p1/core5_model_ranking.csv)
  - 最优 pair：[pair_itransformer_patchtst.csv](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/logs/market_corrected_p1/pair_itransformer_patchtst.csv)
  - live proxy：[live_proxy_with_gated_q50_right.csv](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/logs/market_corrected_p1/itransformer_patchtst/live_proxy_with_gated_q50_right.csv)

- `[PASS]` 交易收益口径为可交易收益率
  - 已剔除 `t+1` 开盘即不可买入样本
  - 数据集中通过 `can_buy_on_next_open` 标记并在评估时过滤

- `[PASS]` 滚动验证口径为 `5年训练 -> 1年测试`
  - 研究过程未使用随机切分作为正式结论依据

### 结论

- 当前 corrected 口径下的主实盘候选：
  - `iTransformer + PatchTST / rank_mean`
- 当前 corrected 口径下的研究备选：
  - `iTransformer + PatchTST / top1_gap + q50 + fallback=PatchTST`

## 2. 未来信息与数据泄漏检查

### 已检查并通过

- `[PASS]` 历史泄漏点已移除
  - 原泄漏字段 `target_shifted` 已不再出现在数据加载主流程中
  - 测试中保留断言，确认它已被删除

- `[PASS]` 当前特征输入来自同日或历史可观测字段
  - `avg_price = amount / volume` 已替代原泄漏特征
  - 相关实现位于 [market_research.py](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/utils/market_research.py)

- `[PASS]` 数据加载器使用 `feature_columns` 取特征，不包含 `label_prev_shift`
  - 数据加载逻辑位于 [data_loader.py](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/data_provider/data_loader.py)
  - `label_prev_shift` 仅用于样本完整性过滤，不作为模型输入

- `[PASS]` 相关单测通过
  - 命令：`/huanghb28/.../envs/tslib/bin/python -m unittest tests.test_market_research`
  - 结果：`19 tests OK`

## 3. 策略选择检查

### 已检查并通过

- `[PASS]` 单模 corrected 排名已生成
  - `iTransformer`、`TimeMixer`、`PatchTST` 为第一梯队

- `[PASS]` 高价值 pair 中，当前最优已确认
  - `iTransformer + PatchTST / rank_mean`
  - 聚合指标：
  - `mean_return = 0.014044`
  - `cumulative_return = 12.154646`
  - `sharpe = 0.756673`

- `[PASS]` selector 与 gating 已完成首轮收口
  - `top1_gap` selector 毛收益更高，但 live-proxy 后不如 `rank_mean` 稳健
  - `gated_q50_right` 毛收益最高，但 live-proxy 后仍未成为当前第一实盘候选

- `[PASS]` live-proxy 已覆盖 `low / base / high` 三档成本
  - 当前主方案 `rank_mean` 在 `base` 场景下：
  - `mean_return = 0.013352`
  - `cumulative_return = 1.675380`
  - `sharpe = 0.720090`

## 4. 工程与可复现性检查

### 已检查并通过

- `[PASS]` corrected 关键结果文件已落盘，可复现回看
- `[PASS]` 关键研究脚本存在
  - [evaluate_ensembles.py](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/scripts/market_daily/evaluate_ensembles.py)
  - [audit_selector_stability.py](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/scripts/market_daily/audit_selector_stability.py)
  - [evaluate_selector_thresholds.py](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/scripts/market_daily/evaluate_selector_thresholds.py)
  - [validate_live_proxy.py](/huanghb28/aigc/cv_banc/zsw/zhuangcailin/project/Time-Series-Library2/scripts/market_daily/validate_live_proxy.py)

### 仍未自动验证

- `[TODO]` 一键从“最新日数据 -> 主策略打分 -> top1 输出” 的正式推理入口尚未固定
- `[TODO]` 生产配置文件尚未固化
  - 模型路径
  - 特征缓存路径
  - 当日推理输入路径
  - 输出选股路径

## 5. 实盘运营检查

### 仍需人工或交易系统联调

- `[TODO]` 券商/柜台/OMS 接口检查
  - 是否能在 `t+1` 开盘前完成下单
  - 是否支持集合竞价/开盘成交策略

- `[TODO]` 成交约束与滑点建模检查
  - 当前 live-proxy 是简化成本模型
  - 尚未纳入真实盘口冲击、开盘可成交量、撮合失败率

- `[TODO]` 股票池准入规则固化
  - ST/退市风险
  - 停牌
  - 风险警示板
  - 北交所/创业板/科创板的差异化交易限制

- `[TODO]` 风控开关
  - 单日不交易条件
  - 主策略失败时回退逻辑
  - 模型文件缺失或当日数据不完整时的熔断逻辑

- `[TODO]` 监控与告警
  - 每日是否成功产出 top1
  - 产出时间是否晚于阈值
  - 今日候选是否可交易
  - 与前一日是否发生异常频繁切换

## 6. 上线判断

### 当前判断

- 结论：`研究层面接近可上线，交易执行层面还不能直接视为已可实盘`

### 原因

- 研究与 corrected 回测口径已经基本收口
- 主策略候选已经明确
- 但执行链路、风控链路、监控链路、成交约束还没有完成最后联调

### 建议上线顺序

1. 先固化主策略推理入口
2. 做 `2-4` 周 paper trading / shadow trading
3. 比较 paper trading 与研究结果偏差
4. 小资金灰度上线
5. 观察换手、成交、滑点和异常日表现
6. 再决定是否放量
