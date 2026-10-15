# A100 PCIe：SGLang 数据采集式 ML 预测器同分布复测

## 范围

本轮覆盖 3 个 checkpoint：Qwen2.5-3B-Instruct、Qwen2.5-7B-Instruct、Qwen3-32B。每个 checkpoint 单独采集、单独训练、单独回放；没有跨模型复用 ML bundle。

每个 checkpoint 用 5 条新采集的 saturation_1024x256_inf trace 训练，并只在来自独立参考目录的 5 条同工作负载 trace 上评分和回放。训练与测试的 ScheduleBatch 行没有复用。

## 本轮结果

| 模型 | 校准 batch 数 | 实际 batch 评分 MAPE | 实际 batch MAE (ms) | E2E 实测均值 (ms) | E2E 预测均值 (ms) | E2E MAPE | TTFT MAPE | TPOT MAPE | 总吞吐 MAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 1448 | 3.48% | 3.10 | 4506.70 | 4454.77 | 2.18% | 39.21% | 13.81% | 12.39% |
| Qwen2.5-7B-Instruct | 1448 | 4.06% | 2.88 | 4029.25 | 4311.57 | 7.09% | 23.56% | 13.65% | 4.65% |
| Qwen3-32B | 1448 | 3.77% | 4.26 | 11215.76 | 11674.00 | 4.36% | 10.14% | 8.20% | 4.10% |

实际 batch 评分使用真实服务记录下来的 Scheduler batch 组成，预测不会改变后续 batch。端到端回放会让预测时延参与调度，因而两行结果不应互相替代。

![实际 batch 与端到端 E2E 的 MAPE 对比](../test-results/pr33824_data_tools_same_distribution_dense3_20260918_1430/aggregate/figures/teacher_forced_vs_e2e_mape.png)

![端到端各指标的 MAPE 对比](../test-results/pr33824_data_tools_same_distribution_dense3_20260918_1430/aggregate/figures/e2e_metric_mape.png)

## 端到端真实值与预测值

表中均值和标准差来自 5 个独立真机运行及其一一对应的回放。带符号偏差为 `(预测−实测)/实测×100%`。

| 模型 | 指标 | 实测均值 | 实测标准差 | 预测均值 | 预测标准差 | 带符号偏差 | MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | TTFT (ms) | 718.63 | 26.19 | 436.08 | 7.92 | -39.21% | 39.21% |
| Qwen2.5-3B-Instruct | TPOT (ms) | 39.36 | 0.46 | 44.79 | 0.59 | 13.81% | 13.81% |
| Qwen2.5-3B-Instruct | E2E (ms) | 4506.70 | 65.05 | 4454.77 | 50.70 | -1.12% | 2.18% |
| Qwen2.5-3B-Instruct | 总吞吐 (token/s) | 2084.00 | 27.93 | 2341.65 | 28.30 | 12.39% | 12.39% |
| Qwen2.5-7B-Instruct | TTFT (ms) | 777.25 | 5.26 | 594.11 | 6.63 | -23.56% | 23.56% |
| Qwen2.5-7B-Instruct | TPOT (ms) | 38.62 | 1.17 | 43.86 | 0.62 | 13.65% | 13.65% |
| Qwen2.5-7B-Instruct | E2E (ms) | 4029.25 | 121.16 | 4311.57 | 67.67 | 7.09% | 7.09% |
| Qwen2.5-7B-Instruct | 总吞吐 (token/s) | 2456.58 | 95.42 | 2552.11 | 48.49 | 4.03% | 4.65% |
| Qwen3-32B | TTFT (ms) | 2677.21 | 47.53 | 2405.11 | 11.35 | -10.14% | 10.14% |
| Qwen3-32B | TPOT (ms) | 111.21 | 2.53 | 120.26 | 0.82 | 8.20% | 8.20% |
| Qwen3-32B | E2E (ms) | 11215.76 | 287.16 | 11674.00 | 75.82 | 4.16% | 4.36% |
| Qwen3-32B | 总吞吐 (token/s) | 990.68 | 30.64 | 1030.19 | 8.34 | 4.10% | 4.10% |

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
| 训练负载 | saturation_1024x256_inf；真实训练 trace 数=5；训练 seed=20260918,20260919,20260920,20260921,20260922；warmup_requests=4；forward modes=1,2 |
| 测试负载 | saturation_1024x256_inf；真实 trace 数=5；每条 trace 单独回放 |

| 模型 | 静态显存比例 | 校准集内 p50 分层切分 MAPE | 校准集内 p50 MAE (ms) | 持出实际 batch 数 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 0.30 | 2.71% | 4.37 | 1195 |
| Qwen2.5-7B-Instruct | 0.35 | 2.21% | 4.09 | 1195 |
| Qwen3-32B | 0.93 | 1.84% | 3.92 | 1195 |

校准集内分层切分只用于检查训练数据拟合情况；最终表中的实际 batch 评分和端到端结果来自未用于训练的 held-out trace。

## 训练/测试 Scheduler batch 覆盖

下表来自真实设备同步的 Scheduler batch。`测试超出训练范围`表示测试行至少有一个 18 维输入特征落在训练逐维范围外；它描述支持集覆盖，不等同于预测误差。

| 模型 | phase | 训练 batch | 测试 batch | 完全同形状 | 测试超出训练范围 | 训练/测试 batch p50 | 训练/测试 extend p50 | 训练/测试 past p50 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Qwen2.5-3B-Instruct | all | 1448 | 1195 | 0.42% | 1.26% | 14 / 12 | 17 / 18 | 7481 / 5994 |
| Qwen2.5-3B-Instruct | prefill | 183 | 155 | 3.23% | 3.23% | 2 / 2 | 512 / 512 | 340 / 322 |
| Qwen2.5-3B-Instruct | decode | 1265 | 1040 | 0.00% | 6.73% | 15 / 17 | 15 / 17 | 8149 / 9668 |
| Qwen2.5-7B-Instruct | all | 1448 | 1195 | 0.42% | 1.26% | 14 / 12 | 17 / 18 | 7481 / 5994 |
| Qwen2.5-7B-Instruct | prefill | 183 | 155 | 3.23% | 3.23% | 2 / 2 | 512 / 512 | 340 / 322 |
| Qwen2.5-7B-Instruct | decode | 1265 | 1040 | 0.00% | 6.73% | 15 / 17 | 15 / 17 | 8149 / 9668 |
| Qwen3-32B | all | 1448 | 1195 | 0.42% | 1.26% | 14 / 12 | 17 / 18 | 7481 / 5994 |
| Qwen3-32B | prefill | 183 | 155 | 3.23% | 3.23% | 2 / 2 | 512 / 512 | 340 / 322 |
| Qwen3-32B | decode | 1265 | 1040 | 0.00% | 6.73% | 15 / 17 | 15 / 17 | 8149 / 9668 |

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
