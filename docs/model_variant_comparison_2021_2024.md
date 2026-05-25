# Model Variant Comparison 2021-2024
范围：这张表统一整理三组结果。`baseline_before` 是改动前 5 个 backbone 基线；`Transformer` 是旧版主线 3 个变体；`modified_stage2topheavy_topk` 是当前改动后的截面打分主线。
说明：`NA` 表示该 `year x model x variant` 组合当前没有完整结果，不表示收益为 0。年度收益率字段使用 `annual_cum_return`。
| year | model | variant | top1_mean | top1_sharpe | annual_cum_return | top3_mean | top5_mean | ic | rank_ic | unique | max_rep | debias015_top1_mean | num_days |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021 | Transformer | transformer_headreg | 0.002026 | 0.503227 | 0.012383 | 0.005273 | 0.004205 | 0.014876 | 0.021843 | 95 | 30 | 0.000986 | 242 |
| 2021 | Transformer | transformer_stage2topheavy | -0.001387 | -0.455809 | -0.469325 | 0.002528 | 0.003400 | 0.018303 | 0.031929 | 162 | 5 | -0.001717 | 242 |
| 2021 | Transformer | transformer_stage2topheavy_topk | 0.012555 | 1.381440 | 1.508062 | 0.011496 | 0.008340 | 0.018800 | 0.015643 | 113 | 28 | 0.009357 | 242 |
| 2021 | TimesNet | baseline_before | 0.041299 | 1.969189 | 4.965690 | 0.021852 | 0.015461 | 0.020870 | -0.017859 | 20 | 123 | 0.043803 | 242 |
| 2021 | TimesNet | modified_stage2topheavy_topk | 0.006779 | 2.458573 | 3.087543 | 0.005209 | 0.003206 | 0.009094 | 0.017945 | 211 | 4 | 0.006861 | 242 |
| 2021 | TimeMixer | baseline_before | 0.038966 | 1.849638 | 1.990550 | 0.018228 | 0.012921 | 0.006808 | -0.038733 | 11 | 146 | 0.037224 | 242 |
| 2021 | TimeMixer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2021 | WPMixer | baseline_before | 0.038663 | 1.837578 | 1.873409 | 0.017404 | 0.013447 | 0.006657 | -0.038403 | 13 | 146 | 0.038324 | 242 |
| 2021 | WPMixer | modified_stage2topheavy_topk | 0.000270 | 0.147142 | -0.037419 | 0.000460 | 0.000337 | 0.007900 | 0.022561 | 230 | 2 | -0.000804 | 242 |
| 2021 | PatchTST | baseline_before | 0.037442 | 1.779154 | 1.146404 | 0.017772 | 0.012938 | 0.007148 | -0.038047 | 11 | 148 | 0.038759 | 242 |
| 2021 | PatchTST | modified_stage2topheavy_topk | 0.002735 | 1.480819 | 0.746709 | 0.002949 | 0.002801 | 0.010045 | 0.012403 | 225 | 3 | NA | 242 |
| 2021 | iTransformer | baseline_before | 0.034579 | 1.650212 | 0.187569 | 0.017361 | 0.011386 | 0.009173 | -0.033099 | 17 | 137 | 0.038904 | 242 |
| 2021 | iTransformer | modified_stage2topheavy_topk | 0.003805 | 2.235856 | 1.299316 | 0.001581 | 0.000472 | 0.003782 | 0.006916 | 228 | 3 | 0.003783 | 242 |
| 2022 | Transformer | transformer_headreg | 0.004013 | 1.130508 | 0.807593 | 0.001965 | 0.000455 | 0.036992 | 0.042106 | 134 | 31 | -0.000037 | 241 |
| 2022 | Transformer | transformer_stage2topheavy | -0.002753 | -0.946039 | -0.603547 | 0.000373 | 0.000250 | 0.043029 | 0.051360 | 172 | 10 | -0.003677 | 241 |
| 2022 | Transformer | transformer_stage2topheavy_topk | 0.000568 | 0.126995 | -0.383431 | 0.002647 | 0.001025 | 0.034738 | 0.034937 | 94 | 35 | -0.001331 | 241 |
| 2022 | TimesNet | baseline_before | 0.002906 | 0.636121 | 0.067673 | 0.002596 | 0.001958 | 0.008762 | -0.020788 | 26 | 82 | 0.004712 | 241 |
| 2022 | TimesNet | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2022 | TimeMixer | baseline_before | -0.003036 | -0.618950 | -0.763951 | -0.003095 | -0.002197 | -0.010771 | -0.052273 | 16 | 61 | -0.001342 | 241 |
| 2022 | TimeMixer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2022 | WPMixer | baseline_before | -0.008185 | -1.620830 | -0.935966 | -0.003252 | -0.002573 | -0.011385 | -0.053295 | 15 | 68 | -0.002604 | 241 |
| 2022 | WPMixer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2022 | PatchTST | baseline_before | -0.008318 | -1.662964 | -0.937830 | -0.004158 | -0.003085 | -0.011066 | -0.052921 | 18 | 57 | -0.006252 | 241 |
| 2022 | PatchTST | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2022 | iTransformer | baseline_before | -0.000574 | -0.120085 | -0.560941 | 0.000692 | -0.001101 | -0.006083 | -0.045480 | 21 | 59 | -0.002605 | 241 |
| 2022 | iTransformer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2023 | Transformer | transformer_headreg | 0.004245 | 0.890323 | 0.539537 | 0.003785 | 0.002325 | 0.025475 | 0.016499 | 110 | 22 | 0.003632 | 241 |
| 2023 | Transformer | transformer_stage2topheavy | 0.005026 | 1.100649 | 0.959606 | 0.000701 | -0.000490 | 0.028769 | 0.019625 | 159 | 13 | 0.007893 | 241 |
| 2023 | Transformer | transformer_stage2topheavy_topk | 0.004318 | 0.766986 | 0.179391 | 0.002521 | 0.001116 | 0.019711 | 0.004244 | 93 | 51 | 0.005069 | 241 |
| 2023 | TimesNet | baseline_before | 0.007356 | 1.110991 | 0.758245 | 0.003197 | 0.003228 | 0.002208 | -0.031837 | 17 | 151 | 0.007657 | 241 |
| 2023 | TimesNet | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2023 | TimeMixer | baseline_before | 0.008779 | 1.518716 | 2.350265 | 0.002597 | 0.001855 | -0.012639 | -0.060946 | 15 | 86 | 0.011480 | 241 |
| 2023 | TimeMixer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2023 | WPMixer | baseline_before | 0.010352 | 1.804487 | 3.955177 | 0.003345 | 0.002627 | -0.012591 | -0.061046 | 15 | 95 | 0.010357 | 241 |
| 2023 | WPMixer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2023 | PatchTST | baseline_before | 0.009135 | 1.584930 | 2.670969 | 0.002659 | 0.002588 | -0.012393 | -0.060681 | 16 | 88 | 0.009733 | 241 |
| 2023 | PatchTST | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2023 | iTransformer | baseline_before | 0.006592 | 0.858814 | 0.057363 | 0.005548 | 0.002167 | -0.009038 | -0.055092 | 18 | 90 | 0.005625 | 241 |
| 2023 | iTransformer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2024 | Transformer | transformer_headreg | 0.001283 | 0.167959 | -0.807959 | 0.004925 | 0.006493 | 0.045285 | 0.042521 | 118 | 18 | -0.004211 | 241 |
| 2024 | Transformer | transformer_stage2topheavy | 0.001334 | 0.187061 | -0.765098 | 0.000765 | 0.003444 | 0.050560 | 0.056789 | 143 | 15 | 0.007538 | 241 |
| 2024 | Transformer | transformer_stage2topheavy_topk | 0.006591 | 0.662014 | -0.825711 | 0.008693 | 0.012274 | 0.046340 | 0.042637 | 84 | 47 | 0.010190 | 241 |
| 2024 | TimesNet | baseline_before | 0.059849 | 1.735461 | -0.778370 | 0.035773 | 0.024647 | 0.029236 | -0.017806 | 12 | 65 | 0.064532 | 241 |
| 2024 | TimesNet | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2024 | TimeMixer | baseline_before | 0.073536 | 2.026193 | 2.949457 | 0.031569 | 0.022654 | 0.010234 | -0.048706 | 9 | 66 | 0.070839 | 241 |
| 2024 | TimeMixer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2024 | WPMixer | baseline_before | 0.078408 | 2.163328 | 12.983341 | 0.032330 | 0.021734 | 0.009912 | -0.048892 | 8 | 77 | 0.078851 | 241 |
| 2024 | WPMixer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2024 | PatchTST | baseline_before | 0.076930 | 2.120126 | 7.679263 | 0.033610 | 0.022367 | 0.009468 | -0.047891 | 9 | 70 | 0.077917 | 241 |
| 2024 | PatchTST | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| 2024 | iTransformer | baseline_before | 0.077925 | 2.137960 | 7.811691 | 0.033451 | 0.021153 | 0.019504 | -0.036259 | 12 | 78 | 0.077690 | 241 |
| 2024 | iTransformer | modified_stage2topheavy_topk | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## Brief Notes
- 2021 年改动后主线里，`TimesNet` 的 `modified_stage2topheavy_topk` 最完整也最有代表性：`top1_mean=0.006779`，`sharpe=2.458573`，`annual_cum_return=3.087543`，同时 `unique=211`、`max_rep=4`，集中度明显低于旧 baseline。
- 改动前 baseline 在 2024 年的 top1 均值和年度收益率整体更高，尤其 `WPMixer / iTransformer / PatchTST / TimeMixer`。这说明“去集中后的截面打分线”目前更像是在换稳定性，不是普遍换来更高收益。
- 当前改动后主线并没有完成 2022-2024 全年矩阵，所以这份表可以作为正式对比底稿，但不能把 2021 的改动后结果直接当成跨年结论。
