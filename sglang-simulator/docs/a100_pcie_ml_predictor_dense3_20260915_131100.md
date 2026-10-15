# A100 PCIe：原生 SGLang ML 预测器实验

## 2026-09-17 采集式复测更新

下面三行是当前应引用的复测结果。复测在真实 SGLang Scheduler 中采集设备同步的 batch 时延，每个模型以 5 次真实 saturation 运行及其一一对应的 CPU 回放汇总。完整的真实值、预测值、标准差、配置和原始结果路径见 [采集式复测报告](a100_pcie_pr33824_data_collector_ml_dense3_20260917.md)。

| 模型 | 实际 batch MAPE | E2E 实测均值 (ms) | E2E 预测均值 (ms) | E2E MAPE | TTFT MAPE | TPOT MAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 2.55% | 4461.87 | 4564.17 | 2.31% | 34.61% | 16.48% |
| Qwen2.5-7B-Instruct | 3.48% | 4107.47 | 4224.65 | 2.85% | 31.21% | 8.87% |
| Qwen3-32B | 14.62% | 11543.92 | 15344.45 | 32.97% | 8.23% | 34.31% |

本页其余内容保留 2026-09-15 的 `one_batch` + ExtraTrees 实验记录。旧实验与本轮的采集方式、ML 模型和部分运行设置不同，不能把两组数字作为只改动一个变量的前后对照。

## 结果摘要

- 已完成 3 个 dense 模型、3 组工作负载和 5 次真机基线，共 45 组真实值/ML 预测值配对。
- 本轮预测器是每个 checkpoint 单独训练的 sklearn ExtraTrees 模型；训练目标来自本机 A100 上原生 SGLang `one_batch` 的 CUDA 同步时延。训练、预测和汇总均未调用 AIConfigurator。
- 三个模型的 E2E MAPE 平均为 1028.86%；具体真实值、预测值和误差在下表逐项列出。
- 本轮端到端 E2E MAPE 分别为 Qwen2.5-3B-Instruct 1913.46%；Qwen2.5-7B-Instruct 954.19%；Qwen3-32B 218.93%。
- 静态单批 holdout 和端到端矩阵是两类测量：前者使用留出的 batch/sequence 形状，后者比较固定请求 trace 的服务结果。

## 实验边界

- 真机基线：`/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_a100_dense3_sm80compat_matrix_20260910_164000`，A100 PCIe、完整 checkpoint、原生 SGLang、Triton attention、piecewise CUDA graph 和 overlap scheduler；每个固定 trace 跑 5 次。
- ML 校准：`/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_ml_native_calibration_dense3_20260915_130105`。3B/7B 使用静态 Cartesian 网格；32B 使用按真实 trace 的 prefill 上限和 decode 并发组织的可行分组网格。
- 模拟：PR #33824 CPU 离线 scheduler，`MLTimePredictor` 仅预测每次 forward 的逻辑时长，不执行物理 CUDA forward。本次每个工作负载的 ML 模拟均完成 5 次，汇总时取其均值；真机侧也保留全部 5 次。

| 模型 | 校准形状 | 留出形状 | 校准 mem_fraction | 真机基线 mem_fraction | 采样方式 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-3B-Instruct | 90 / 90 | 16 / 16 | 0.80 | 0.80 | static Cartesian grid |
| Qwen2.5-7B-Instruct | 90 / 90 | 16 / 16 | 0.80 | 0.80 | static Cartesian grid |
| Qwen3-32B | 70 / 70 | 10 / 10 | 0.96 | 0.96 | trace-feasible grouped grid |

- 32B 的最初 32×1024 静态 prefill 在 80GB 上触发 OOM，且该形状不在固定服务 trace 中：trace 的单轮 prefill 总扩展 token 不超过 2048。失败日志已保留，最终 32B 结果改为同样 `mem_fraction_static=0.96` 下的 trace-feasible 网格。

## 静态批次 holdout

| 模型 | 阶段 | 样本数 | MAPE | 中位 APE | P90 APE | MAE (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | prefill | 16 | 27.17% | 23.06% | 51.58% | 87.90 |
| Qwen2.5-3B-Instruct | decode | 2032 | 30.13% | 30.05% | 53.25% | 3.26 |
| Qwen2.5-7B-Instruct | prefill | 16 | 49.75% | 45.41% | 66.84% | 233.00 |
| Qwen2.5-7B-Instruct | decode | 2032 | 56.28% | 54.98% | 77.09% | 6.48 |
| Qwen3-32B | prefill | 10 | 10.37% | 10.77% | 13.40% | 39.64 |
| Qwen3-32B | decode | 6198 | 0.64% | 0.62% | 1.12% | 0.29 |

## 与 AIC 预测的并列对照

该节使用已生成的三方法比较表。单步均为 batch=1、1024-token prefill 和首个 decode token；端到端均为 3 个负载、每负载 5 次真实/预测配对。

### 单步 Prefill MAPE

| 模型 | AIC：退化后端 | AIC：SM80 兼容 kernel | ML：SM80 兼容 kernel |
| --- | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 26.33% | 12.44% | 0.52% |
| Qwen2.5-7B-Instruct | 15.91% | 23.26% | 97.07% |
| Qwen3-32B | 15.27% | 27.41% | 46.09% |

### 单步 Decode MAPE

| 模型 | AIC：退化后端 | AIC：SM80 兼容 kernel | ML：SM80 兼容 kernel |
| --- | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 39.37% | 10.09% | 4.96% |
| Qwen2.5-7B-Instruct | 27.43% | 8.63% | 96.25% |
| Qwen3-32B | 12.19% | 33.00% | 17.06% |

### 端到端 E2E MAPE

| 模型 | AIC：退化后端 | AIC：SM80 兼容 kernel | ML：SM80 兼容 kernel |
| --- | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 6.74% | 55.50% | 1913.46% |
| Qwen2.5-7B-Instruct | 2.72% | 26.44% | 954.19% |
| Qwen3-32B | 8.86% | 15.17% | 218.93% |

- AIC 退化后端和 AIC SM80 兼容 kernel 分别与各自批次的真实基线配对；两批运行不是只改变一个 kernel 的受控对照。
- “SM80 兼容 kernel”描述真机负载使用的兼容包；AIC 查询仍使用 Silicon 静态算子表，不直接执行 CUDA kernel。
- ML 的固定单步点位使用兼容 kernel 的真机均值。input=1024 位于 ML 校准网格内，因此这两个单点不是独立 holdout。
- ML 的端到端结果复用兼容 kernel 矩阵的真机基线。
- 图表、逐项数据和来源：`/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_predictor_method_comparison_dense3_20260915_231314`。

![三种预测方式的固定单步误差对比](../test-results/pr33824_predictor_method_comparison_dense3_20260915_231314/single_step_mape.png)

![三种预测方式的端到端误差对比](../test-results/pr33824_predictor_method_comparison_dense3_20260915_231314/end_to_end_e2e_mape.png)

## 采集期间的 GPU 状态

- 动态遥测从 2026-09-15T13:13:13+08:00 到 2026-09-15T14:29:18+08:00，共 1471 个两秒采样点；平均 GPU 利用率为 59.70%，最小空闲显存为 39.00 MiB。
- 其中 142 个采样点同时发现至少两个 GPU compute 进程（最多 2 个）。遥测在 3B 采集完成后才开始，因此它只能说明后续时间窗口的资源竞争，不能反证 3B 采集是否干净。

## 端到端模型汇总

| 模型 | TTFT MAPE | TPOT MAPE | E2E MAPE | 总吞吐 MAPE |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 504.74% | 1982.87% | 1913.46% | 74.52% |
| Qwen2.5-7B-Instruct | 318.33% | 1066.93% | 954.19% | 75.09% |
| Qwen3-32B | 88.34% | 225.47% | 218.93% | 64.87% |

## 端到端真实值与预测值

表中值为同一工作负载 5 次真机结果和 5 次 ML 模拟结果的均值；signed error 为 `(预测−真实)/真实×100%`。

| 模型 | 工作负载 | 指标 | 真实均值 | ML 预测 | signed error | MAPE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | latency_256x128_r1 | TTFT (ms) | 37.56 | 371.21 | 942.38% | 942.38% |
| Qwen2.5-3B-Instruct | latency_256x128_r1 | TPOT (ms) | 6.15 | 154.73 | 2445.61% | 2445.61% |
| Qwen2.5-3B-Instruct | latency_256x128_r1 | E2E (ms) | 408.44 | 9453.30 | 2244.15% | 2244.15% |
| Qwen2.5-3B-Instruct | latency_256x128_r1 | Total throughput (token/s) | 187.79 | 84.18 | -55.17% | 55.17% |
| Qwen2.5-3B-Instruct | saturation_1024x256_inf | TTFT (ms) | 356.57 | 967.33 | 174.04% | 174.04% |
| Qwen2.5-3B-Instruct | saturation_1024x256_inf | TPOT (ms) | 14.65 | 152.95 | 956.03% | 956.03% |
| Qwen2.5-3B-Instruct | saturation_1024x256_inf | E2E (ms) | 1581.05 | 22904.48 | 1366.18% | 1366.18% |
| Qwen2.5-3B-Instruct | saturation_1024x256_inf | Total throughput (token/s) | 8404.14 | 520.03 | -93.75% | 93.75% |
| Qwen2.5-3B-Instruct | service_1024x256_r4 | TTFT (ms) | 128.39 | 268.73 | 381.37% | 397.80% |
| Qwen2.5-3B-Instruct | service_1024x256_r4 | TPOT (ms) | 7.70 | 195.85 | 2546.97% | 2546.97% |
| Qwen2.5-3B-Instruct | service_1024x256_r4 | E2E (ms) | 912.25 | 18464.04 | 2130.06% | 2130.06% |
| Qwen2.5-3B-Instruct | service_1024x256_r4 | Total throughput (token/s) | 1502.23 | 380.79 | -74.64% | 74.64% |
| Qwen2.5-7B-Instruct | latency_256x128_r1 | TTFT (ms) | 41.56 | 243.39 | 493.13% | 493.13% |
| Qwen2.5-7B-Instruct | latency_256x128_r1 | TPOT (ms) | 11.60 | 170.34 | 1397.32% | 1397.32% |
| Qwen2.5-7B-Instruct | latency_256x128_r1 | E2E (ms) | 629.77 | 8444.09 | 1264.32% | 1264.32% |
| Qwen2.5-7B-Instruct | latency_256x128_r1 | Total throughput (token/s) | 178.77 | 69.04 | -61.37% | 61.37% |
| Qwen2.5-7B-Instruct | saturation_1024x256_inf | TTFT (ms) | 1095.91 | 2177.55 | 100.59% | 100.59% |
| Qwen2.5-7B-Instruct | saturation_1024x256_inf | TPOT (ms) | 24.98 | 174.62 | 599.55% | 599.55% |
| Qwen2.5-7B-Instruct | saturation_1024x256_inf | E2E (ms) | 4116.33 | 25654.21 | 524.38% | 524.38% |
| Qwen2.5-7B-Instruct | saturation_1024x256_inf | Total throughput (token/s) | 3971.50 | 574.38 | -85.52% | 85.52% |
| Qwen2.5-7B-Instruct | service_1024x256_r4 | TTFT (ms) | 96.61 | 429.25 | 361.28% | 361.28% |
| Qwen2.5-7B-Instruct | service_1024x256_r4 | TPOT (ms) | 14.93 | 188.96 | 1203.93% | 1203.93% |
| Qwen2.5-7B-Instruct | service_1024x256_r4 | E2E (ms) | 2034.41 | 23159.41 | 1073.88% | 1073.88% |
| Qwen2.5-7B-Instruct | service_1024x256_r4 | Total throughput (token/s) | 2045.22 | 441.94 | -78.37% | 78.37% |
| Qwen3-32B | latency_256x128_r1 | TTFT (ms) | 141.17 | 482.45 | 241.95% | 241.95% |
| Qwen3-32B | latency_256x128_r1 | TPOT (ms) | 46.94 | 207.26 | 341.51% | 341.51% |
| Qwen3-32B | latency_256x128_r1 | E2E (ms) | 3506.54 | 14861.72 | 323.83% | 323.83% |
| Qwen3-32B | latency_256x128_r1 | Total throughput (token/s) | 174.01 | 62.14 | -64.29% | 64.29% |
| Qwen3-32B | saturation_1024x256_inf | TTFT (ms) | 3356.84 | 3544.40 | 5.59% | 5.59% |
| Qwen3-32B | saturation_1024x256_inf | TPOT (ms) | 69.72 | 217.22 | 211.58% | 211.58% |
| Qwen3-32B | saturation_1024x256_inf | E2E (ms) | 11800.94 | 30993.06 | 162.63% | 162.63% |
| Qwen3-32B | saturation_1024x256_inf | Total throughput (token/s) | 1229.55 | 421.28 | -65.74% | 65.74% |
| Qwen3-32B | service_1024x256_r4 | TTFT (ms) | 1483.36 | 1742.57 | 17.48% | 17.48% |
| Qwen3-32B | service_1024x256_r4 | TPOT (ms) | 120.39 | 268.82 | 123.31% | 123.31% |
| Qwen3-32B | service_1024x256_r4 | E2E (ms) | 9602.57 | 25958.54 | 170.33% | 170.33% |
| Qwen3-32B | service_1024x256_r4 | Total throughput (token/s) | 1258.40 | 445.85 | -64.57% | 64.57% |

## 说明

- 每个模型的 ML bundle 只使用本模型的静态真机采样；没有用这 45 条端到端真实基线拟合或调参。
- 模型输入是 `MLTimePredictor` 的 18 个调度批次长度特征。它不包含模型结构字段，因此不能把一个模型的 bundle 直接套到另一个 checkpoint。
- 结果文件中的 `per-model/<model>/analysis/paired_repeat_metrics.csv` 保留逐次真实值与预测值；`analysis/operator_holdout_summary.csv` 保留单步 holdout 误差。
