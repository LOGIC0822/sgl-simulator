# A100 PCIe：SGLang 数据采集式 ML 预测器复测

## 范围

本轮只覆盖 3 个 dense checkpoint：Qwen2.5-3B-Instruct、Qwen2.5-7B-Instruct 和 Qwen3-32B。每个 checkpoint 单独采集、单独训练、单独回放；没有跨模型复用 ML bundle。

每个模型的真实服务运行 5 次 saturation_1024x256_inf 负载。每次真实运行产生独立 raw request trace；CPU 模拟器回放同一次 trace 的 token IDs、完成输出长度和相对到达时间。

## 本轮结果

| 模型 | 校准 batch 数 | 实际 batch 评分 MAPE | 实际 batch MAE (ms) | E2E 实测均值 (ms) | E2E 预测均值 (ms) | E2E MAPE | TTFT MAPE | TPOT MAPE | 总吞吐 MAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 1600 | 2.55% | 2.79 | 4461.87 | 4564.17 | 2.31% | 34.61% | 16.48% | 8.95% |
| Qwen2.5-7B-Instruct | 1814 | 3.48% | 2.82 | 4107.47 | 4224.65 | 2.85% | 31.21% | 8.87% | 6.70% |
| Qwen3-32B | 1093 | 14.62% | 10.49 | 11543.92 | 15344.45 | 32.97% | 8.23% | 34.31% | 19.99% |

实际 batch 评分使用真实服务记录下来的 Scheduler batch 组成，预测不会改变后续 batch。端到端回放会让预测时延参与调度，因而两行结果不应互相替代。

![实际 batch 与端到端 E2E 的 MAPE 对比](../test-results/pr33824_data_tools_dense3_20260917/aggregate/figures/teacher_forced_vs_e2e_mape.png)

![端到端各指标的 MAPE 对比](../test-results/pr33824_data_tools_dense3_20260917/aggregate/figures/e2e_metric_mape.png)

## 端到端真实值与预测值

表中均值和标准差来自 5 个独立真机运行及其一一对应的回放。带符号偏差为 `(预测−实测)/实测×100%`。

| 模型 | 指标 | 实测均值 | 实测标准差 | 预测均值 | 预测标准差 | 带符号偏差 | MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | TTFT (ms) | 691.67 | 12.14 | 452.17 | 48.44 | -34.61% | 34.61% |
| Qwen2.5-3B-Instruct | TPOT (ms) | 39.19 | 0.45 | 45.64 | 0.32 | 16.48% | 16.48% |
| Qwen2.5-3B-Instruct | E2E (ms) | 4461.87 | 54.16 | 4564.17 | 44.93 | 2.31% | 2.31% |
| Qwen2.5-3B-Instruct | 总吞吐 (token/s) | 2099.66 | 21.28 | 2287.38 | 16.62 | 8.95% | 8.95% |
| Qwen2.5-7B-Instruct | TTFT (ms) | 785.01 | 9.08 | 539.93 | 4.11 | -31.21% | 31.21% |
| Qwen2.5-7B-Instruct | TPOT (ms) | 39.40 | 0.13 | 42.89 | 0.38 | 8.87% | 8.87% |
| Qwen2.5-7B-Instruct | E2E (ms) | 4107.47 | 21.59 | 4224.65 | 38.83 | 2.85% | 2.85% |
| Qwen2.5-7B-Instruct | 总吞吐 (token/s) | 2406.83 | 15.00 | 2568.07 | 24.67 | 6.70% | 6.70% |
| Qwen3-32B | TTFT (ms) | 2650.98 | 7.60 | 2432.88 | 13.26 | -8.23% | 8.23% |
| Qwen3-32B | TPOT (ms) | 114.84 | 2.72 | 154.18 | 1.68 | 34.31% | 34.31% |
| Qwen3-32B | E2E (ms) | 11543.92 | 283.46 | 15344.45 | 171.63 | 32.97% | 32.97% |
| Qwen3-32B | 总吞吐 (token/s) | 955.47 | 28.23 | 764.02 | 9.68 | -19.99% | 19.99% |

## 校准与运行配置

| 项目 | 设置 |
| --- | --- |
| 真实硬件 | NVIDIA A100 80GB PCIe, GPU-588c855d-6c74-a953-1bd7-b3144460900a, 00000000:D8:00.0, 81920 MiB, 0 MiB, 535.230.02 |
| 源码 revision | 59799a368793b9f795baf59b067233a65ad8e38e |
| 采集器来源 | https://github.com/littlefatfat/sglang/tree/simulator-data-tools-0914 @ 9cfddcedc9d1e00885c28db75a45513eddbb6271 |
| Python / PyTorch | 3.10.21 / 2.9.1+cu126（CUDA 12.6） |
| SGLang 包版本记录 | 0.0.0；实际代码以 revision 为准 |
| sgl-kernel 包版本记录 | 0.4.6.post1+cu129；运行时覆盖路径见各模型 package_versions.txt |
| Attention / kernel | triton / isolated_SM80_AOT; SGLANG_FORCE_FUSED_OP_BACKEND=unset |
| 调度设置 | overlap disabled；CUDA graph disabled；context=2048；max_running_requests=32；chunked_prefill_size=512 |
| 长度采样 | random_range_ratio=0：输入和输出长度分别在 1 到命名上限之间采样 |
| 校准负载 | latency_256x128_r1 与 service_1024x256_r4，warmup_requests=4，forward modes=1,2 |
| 测试负载 | saturation_1024x256_inf；同一 seed 的 5 次独立服务启动和真实执行；每条 trace 单独回放 |

| 模型 | 静态显存比例 | 校准集内 p50 分层切分 MAPE | 校准集内 p50 MAE (ms) | 持出实际 batch 数 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 0.30 | 1.48% | 1.04 | 1195 |
| Qwen2.5-7B-Instruct | 0.30 | 1.13% | 0.78 | 1195 |
| Qwen3-32B | 0.93 | 2.07% | 2.69 | 1195 |

校准集内分层切分只用于检查训练数据拟合情况；最终表中的实际 batch 评分和端到端结果来自未用于训练的 saturation 负载。

## 结果边界

- 结果只适用于表中 checkpoint、A100 PCIe、当前源码 revision、Triton attention、关闭 overlap 和 CUDA graph 的组合。
- ML predictor 只给 forward 逻辑时长；真实请求处理、CPU 调度和网络处理不会被单独建模。端到端误差包含这些因素与调度反馈的共同影响。
- 同一 seed 的五次重复用于观察同一负载下的运行波动，不代表不同请求分布下的泛化误差。
- 32B 的结果不应外推到 MoE、量化 checkpoint 或更长上下文；本轮未测这些条件。

## 已有三种预测方式图表（历史实验）

下图保留此前 AIC Silicon（退化后端、SM80 兼容 kernel）和旧 ExtraTrees ML 实验的记录。两批 AIC 实验使用的真实基线并非只改变一个变量；旧 ML 也不是本轮的 collector + HGB bundle。因此图仅用于追溯历史结果，不能与本轮表格作单一变量因果比较。

![三种预测方式的固定单步误差对比](../test-results/pr33824_predictor_method_comparison_dense3_20260915_231314/single_step_mape.png)

![三种预测方式的端到端 E2E 误差对比](../test-results/pr33824_predictor_method_comparison_dense3_20260915_231314/end_to_end_e2e_mape.png)

历史图表的逐项数据：`../test-results/pr33824_predictor_method_comparison_dense3_20260915_231314/comparison_data.csv`。

## 结果文件

- `aggregate/model_summary.csv`：模型级汇总。
- `aggregate/e2e_detail.csv`：各指标的实测、预测、标准差和误差。
- 每个模型的 `holdout/teacher_forced/`：实际 Scheduler batch 的逐 batch 评分。
- 每个模型的 `holdout/e2e_comparison/`：5 次真实 trace 与回放的逐次配对。
