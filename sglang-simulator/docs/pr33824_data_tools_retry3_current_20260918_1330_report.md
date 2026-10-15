# A100 PCIe：SGLang 数据采集式 ML 预测器复测

## 范围

本轮覆盖 3 个 checkpoint：Qwen2.5-14B-Instruct-1M、Qwen3-30B-A3B、Qwen3-30B-A3B-Instruct-2507。每个 checkpoint 单独采集、单独训练、单独回放；没有跨模型复用 ML bundle。

每个模型的真实服务运行 5 次 saturation_1024x256_inf 负载。每次真实运行产生独立 raw request trace；CPU 模拟器回放同一次 trace 的 token IDs、完成输出长度和相对到达时间。

## 本轮结果

| 模型 | 校准 batch 数 | 实际 batch 评分 MAPE | 实际 batch MAE (ms) | E2E 实测均值 (ms) | E2E 预测均值 (ms) | E2E MAPE | TTFT MAPE | TPOT MAPE | 总吞吐 MAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-14B-Instruct-1M | 382 | 8.26% | 60.66 | 63875.93 | 53614.40 | 15.36% | 16.52% | 16.22% | 25.03% |
| Qwen3-30B-A3B | 471 | 4.72% | 21.18 | 12955.32 | 11133.27 | 13.94% | 47.25% | 10.98% | 31.15% |
| Qwen3-30B-A3B-Instruct-2507 | 467 | 3.84% | 19.56 | 12943.85 | 11125.93 | 14.04% | 46.24% | 11.58% | 29.93% |

实际 batch 评分使用真实服务记录下来的 Scheduler batch 组成，预测不会改变后续 batch。端到端回放会让预测时延参与调度，因而两行结果不应互相替代。

![实际 batch 与端到端 E2E 的 MAPE 对比](../test-results/pr33824_data_tools_retry3_current_20260918_1330/aggregate/figures/teacher_forced_vs_e2e_mape.png)

![端到端各指标的 MAPE 对比](../test-results/pr33824_data_tools_retry3_current_20260918_1330/aggregate/figures/e2e_metric_mape.png)

## 端到端真实值与预测值

表中均值和标准差来自 5 个独立真机运行及其一一对应的回放。带符号偏差为 `(预测−实测)/实测×100%`。

| 模型 | 指标 | 实测均值 | 实测标准差 | 预测均值 | 预测标准差 | 带符号偏差 | MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-14B-Instruct-1M | TTFT (ms) | 24484.55 | 2376.68 | 20271.48 | 8.78 | -16.52% | 16.52% |
| Qwen2.5-14B-Instruct-1M | TPOT (ms) | 661.79 | 63.76 | 549.92 | 0.43 | -16.22% | 16.22% |
| Qwen2.5-14B-Instruct-1M | E2E (ms) | 63875.93 | 6205.47 | 53614.40 | 41.58 | -15.36% | 15.36% |
| Qwen2.5-14B-Instruct-1M | 总吞吐 (token/s) | 229.69 | 19.61 | 284.76 | 0.27 | 25.03% | 25.03% |
| Qwen3-30B-A3B | TTFT (ms) | 2018.90 | 79.33 | 1062.78 | 43.09 | -47.25% | 47.25% |
| Qwen3-30B-A3B | TPOT (ms) | 124.56 | 4.71 | 110.74 | 0.39 | -10.98% | 10.98% |
| Qwen3-30B-A3B | E2E (ms) | 12955.32 | 494.83 | 11133.27 | 50.02 | -13.94% | 13.94% |
| Qwen3-30B-A3B | 总吞吐 (token/s) | 709.90 | 24.90 | 929.89 | 3.64 | 31.15% | 31.15% |
| Qwen3-30B-A3B-Instruct-2507 | TTFT (ms) | 1987.66 | 17.65 | 1068.59 | 6.10 | -46.24% | 46.24% |
| Qwen3-30B-A3B-Instruct-2507 | TPOT (ms) | 125.33 | 1.69 | 110.80 | 0.49 | -11.58% | 11.58% |
| Qwen3-30B-A3B-Instruct-2507 | E2E (ms) | 12943.85 | 105.58 | 11125.93 | 47.97 | -14.04% | 14.04% |
| Qwen3-30B-A3B-Instruct-2507 | 总吞吐 (token/s) | 716.78 | 6.02 | 931.19 | 3.83 | 29.93% | 29.93% |

## 校准与运行配置

| 项目 | 设置 |
| --- | --- |
| 真实硬件 | NVIDIA A100 80GB PCIe, GPU-588c855d-6c74-a953-1bd7-b3144460900a, 00000000:D8:00.0, 81920 MiB, 0 MiB, 535.230.02 |
| 源码 revision | 59799a368793b9f795baf59b067233a65ad8e38e |
| 采集器来源 | https://github.com/littlefatfat/sglang/tree/simulator-data-tools-0914 @ 9cfddcedc9d1e00885c28db75a45513eddbb6271 |
| Python / PyTorch | 3.10.21 / 2.9.1+cu126（CUDA 12.6） |
| SGLang 包版本记录 | 0.0.0；实际代码以 revision 为准 |
| sgl-kernel 包版本记录 | 0.4.6.post1+cu129；运行时覆盖路径见各模型 package_versions.txt |
| Attention / kernel | dual_chunk_flash_attn / isolated_SM80_AOT; SGLANG_FORCE_FUSED_OP_BACKEND=unset |
| 调度设置 | overlap disabled；CUDA graph disabled；context=2048；max_running_requests=32；chunked_prefill_size=512 |
| 长度采样 | random_range_ratio=0：输入和输出长度分别在 1 到命名上限之间采样 |
| 校准负载 | latency_256x128_r1 与 service_1024x256_r4，warmup_requests=4，forward modes=1,2 |
| 测试负载 | saturation_1024x256_inf；同一 seed 的 5 次独立服务启动和真实执行；每条 trace 单独回放 |

| 模型 | 静态显存比例 | 校准集内 p50 分层切分 MAPE | 校准集内 p50 MAE (ms) | 持出实际 batch 数 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-14B-Instruct-1M | 0.60 | 26.19% | 91.31 | 1235 |
| Qwen3-30B-A3B | 0.90 | 3.42% | 13.38 | 1195 |
| Qwen3-30B-A3B-Instruct-2507 | 0.90 | 4.37% | 14.94 | 1195 |

校准集内分层切分只用于检查训练数据拟合情况；最终表中的实际 batch 评分和端到端结果来自未用于训练的 saturation 负载。

## 结果边界

- 结果只适用于表中 checkpoint、A100 PCIe、当前源码 revision、Triton attention、关闭 overlap 和 CUDA graph 的组合。
- ML predictor 只给 forward 逻辑时长；真实请求处理、CPU 调度和网络处理不会被单独建模。端到端误差包含这些因素与调度反馈的共同影响。
- 同一 seed 的五次重复用于观察同一负载下的运行波动，不代表不同请求分布下的泛化误差。
- 每个 checkpoint 的结果只适用于其自身的模型格式、量化方式和上下文设置；不同架构之间不共享 ML bundle。

## 结果文件

- `aggregate/model_summary.csv`：模型级汇总。
- `aggregate/e2e_detail.csv`：各指标的实测、预测、标准差和误差。
- 每个模型的 `holdout/teacher_forced/`：实际 Scheduler batch 的逐 batch 评分。
- 每个模型的 `holdout/e2e_comparison/`：5 次真实 trace 与回放的逐次配对。
