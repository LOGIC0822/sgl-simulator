# 全模型预测方法对比实验

## 实验口径

本轮对 13 个本地 checkpoint 分别进行真实 SGLang 采集、ML 训练与 AIConfigurator Silicon 预测。每个模型训练和留出测试均为 5 个独立 seed 的饱和请求 trace；每个 trace 为 64 个请求，因此训练和测试各 320 个请求。工作负载 `saturation_1024x256_inf` 表示输入目标为 1024 token、输出目标为 256 token，持续保持请求队列饱和。真实服务的每一轮都加载完整 checkpoint 并记录设备同步的 scheduler iteration 时延。

ML 仅使用训练 trace 的 ScheduleBatch 时延拟合 HGB 模型。AIC 的宽覆盖表和 trace-targeted 算子表也只使用训练 trace 的形状；留出 trace 的真实时延不参与 ML 拟合或 AIC 选点。批级比较采用 teacher-forced 方式，端到端比较则使用同一条留出原始请求 trace 分别运行 CPU 逻辑模拟器。

本轮比较的预测方法是 ML（HGB）和 AIC Silicon。此前的“退化后端”记录对应另一套真实 kernel 路径，不能与本轮同一兼容 kernel 的基线构成受控比较，因此不作为第三种方法并入本表。

预测器对于固定 trace 是确定性的，因此每条不同 seed 的留出 trace 只运行一次预测；真实 GPU 侧保留 5 条独立 trace。报告中的 n=5 是五个真实留出工作负载配对结果，而不是五次复制同一个预测结果。

## 版本与运行设置

- SGLang 源码 revision：`59799a368793b9f795baf59b067233a65ad8e38e`
- Python：`/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python`
- 训练请求：320；留出请求：320。
- 训练 seed：`20260918,20260919,20260920,20260921,20260922`；留出 seed：`20261018,20261019,20261020,20261021,20261022`。
- 调度设置：attention=per-model (triton except 1M dual_chunk_flash_attn); 1M_page_size=256_for_FA2_paged_KV; overlap=disabled; cuda_graph=disabled; context_length=2048; max_running_requests=32; chunked_prefill_size=512; random_range_ratio=0。
- kernel 设置：isolated_SM80_AOT; SGLANG_FORCE_FUSED_OP_BACKEND=unset；阶段目录：`/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4`。
- AIC：AIConfigurator 0.10，backend schema `0.5.10`，源码 revision `c5f139662be28b177ca9f62f83543bbf10f133a0`。
- AIC 算子采集：AIC collector default: 3 warmups plus 20 timed iterations per point。
- AIC 规则插值支撑：regular attention grid from training plans plus context_length=2048 and max_running_requests=32; holdout paths were not read。
- AIC 评分表处理：for relevant context-attention slices, retain only direct regular-grid measurements so sparse training keys cannot shadow Silicon batch interpolation; all other operator tables are copied unchanged。
- AIC 的 backend schema 是其性能表查询命名空间；真实 SGLang 服务的身份以上述源码 revision、每条 trace 的 package_versions.txt 和兼容 kernel 路径为准，二者不混写为同一个版本号。

## 数据量与 AIC 采样覆盖

| 项目 | 数值 |
| --- | --- |
| 规则 batch 网格 | 1–32（32 个值） |
| prefill 长度网格 | 21 个值，最大 2048 |
| decode 长度网格 | 22 个值，最大 2304 |
| prefill 支撑点 | 6720（新测 6260，复用 460） |
| decode 支撑点 | 7040（新测 6964，复用 76） |
| 规则网格总点 | 13760 |

| 评分库 context 表处理 | 行数 |
| --- | --- |
| 原始 context 表 | 9814 |
| 评分库保留的完整规则 context 行 | 6720 |
| 保留的非本轮模型 context 行 | 240 |
| 从评分 context 切片排除的散点/不兼容行 | 2854 |

规则网格只由训练 trace 计划和固定的 `context_length=2048`、`max_running_requests=32` 生成。它用于保证 AIC Silicon 对独立留出 trace 的批大小和长度组合有完整的三维插值支撑；留出 trace 不参与该选点过程。
原始 AIC 表仍保留所有 trace-targeted 行。正式评分使用独立评分库：对本轮 10 个 attention 形状，只保留完整规则 context 网格，避免 AIC 先选到一个单 batch 的相邻长度行；generation、GEMM 和 MoE 表没有改写。

| 模型 | 训练请求 | 留出请求 | 训练 batch | 留出 batch | 留出超出训练特征范围 | AIC attention 目标 | AIC attention 新测 / 复用 | AIC GEMM 新测 / 复用 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 320 | 320 | 2359 | 2269 | 0.22% | 614 | 608 / 6 | 695 / 45 |
| Qwen2.5-3B-Instruct | 320 | 320 | 2359 | 2269 | 0.22% | 614 | 608 / 6 | 695 / 45 |
| llama-2-7b | 320 | 320 | 2364 | 2269 | 0.48% | 619 | 617 / 2 | 695 / 50 |
| Mistral-7B-Instruct-v0.2 | 320 | 320 | 2361 | 2268 | 0.35% | 621 | 618 / 3 | 611 / 159 |
| Qwen2.5-7B-Instruct | 320 | 320 | 2359 | 2269 | 0.22% | 614 | 608 / 6 | 695 / 45 |
| Qwen3-8B | 320 | 320 | 2359 | 2269 | 0.22% | 614 | 474 / 140 | 551 / 189 |
| Qwen2.5-14B-Instruct | 320 | 320 | 2359 | 2269 | 0.22% | 614 | 608 / 6 | 695 / 45 |
| Qwen2.5-14B-Instruct-1M | 320 | 320 | 2408 | 2318 | 0.09% | 602 | 130 / 472 | 600 / 695 |
| Qwen3-14B | 320 | 320 | 2359 | 2269 | 0.22% | 614 | 0 / 614 | 417 / 323 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 320 | 320 | 2358 | 2268 | 10.49% | 613 | 610 / 3 | 705 / 40 |
| Qwen3-30B-A3B | 320 | 320 | 2359 | 2269 | 0.22% | 614 | 608 / 6 | 592 / 148 |
| Qwen3-30B-A3B-Instruct-2507 | 320 | 320 | 2359 | 2269 | 0.22% | 614 | 0 / 614 | 0 / 740 |
| Qwen3-32B | 320 | 320 | 2359 | 2269 | 0.22% | 614 | 476 / 138 | 556 / 184 |

同样的请求预算并不保证产生同样数量的 scheduler batch：较快或较慢的模型在饱和队列下会形成不同的动态 batch 组合。因此训练和留出 batch 数均单独列出，作为实际时延样本量。
AIC 的“复用”表示该算子形状已在本轮相同 A100、相同采集后端下直接测得，因而不重复执行同一测点；它不是使用留出 trace，也不是把 ML 训练样本在模型间转移。

![scheduler batch samples](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/scheduler_batch_samples.png)

![aic points](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/aic_trace_targeted_points.png)

![aic regular support](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/aic_regular_interpolation_support.png)

![ml feature support](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/ml_feature_support.png)

训练侧还保留了一个按 trace 分层的内部检查，用于确认模型拟合过程；它与后面的独立留出 trace 结果分开报告，不能替代正式测试。

## ML 训练内核查（非正式留出结果）

| 模型 | 最终拟合全部行 | 内部训练 / 检查行 | p50 HGB MAPE | MAE ms |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 2359 | 1887 / 472 | 1.45% | 0.41 |
| Qwen2.5-3B-Instruct | 2359 | 1887 / 472 | 1.63% | 0.50 |
| llama-2-7b | 2364 | 1891 / 473 | 1.36% | 1.03 |
| Mistral-7B-Instruct-v0.2 | 2361 | 1889 / 472 | 1.20% | 0.81 |
| Qwen2.5-7B-Instruct | 2359 | 1887 / 472 | 1.54% | 0.43 |
| Qwen3-8B | 2359 | 1887 / 472 | 1.31% | 0.45 |
| Qwen2.5-14B-Instruct | 2359 | 1887 / 472 | 1.19% | 0.52 |
| Qwen2.5-14B-Instruct-1M | 2408 | 1928 / 480 | 3.07% | 27.30 |
| Qwen3-14B | 2359 | 1887 / 472 | 1.00% | 0.40 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 2358 | 1887 / 471 | 1.13% | 1.72 |
| Qwen3-30B-A3B | 2359 | 1887 / 472 | 1.78% | 6.93 |
| Qwen3-30B-A3B-Instruct-2507 | 2359 | 1887 / 472 | 1.73% | 6.47 |
| Qwen3-32B | 2359 | 1887 / 472 | 0.88% | 0.64 |

每条 real trace 的完整启动命令、SGLang/sgl-kernel 包版本和原始 server log 均保存在对应的 `calibration/real/repeat-*/` 或 `holdout/real/repeat-*/` 目录。下表给出模型间不同的运行参数；其余 scheduler 设置统一采用本报告前述口径。

## 模型级运行参数

| 模型 | 显存比例 | attention 后端 | page size | FA2 兼容层 | 运行时 shim |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 0.20 | triton | 默认 | 否 | 否 |
| Qwen2.5-3B-Instruct | 0.30 | triton | 默认 | 否 | 否 |
| llama-2-7b | 0.72 | triton | 默认 | 否 | 否 |
| Mistral-7B-Instruct-v0.2 | 0.35 | triton | 默认 | 否 | 否 |
| Qwen2.5-7B-Instruct | 0.35 | triton | 默认 | 否 | 否 |
| Qwen3-8B | 0.42 | triton | 默认 | 否 | 否 |
| Qwen2.5-14B-Instruct | 0.60 | triton | 默认 | 否 | 否 |
| Qwen2.5-14B-Instruct-1M | 0.60 | dual_chunk_flash_attn | 256 | 是 | 是 |
| Qwen3-14B | 0.60 | triton | 默认 | 否 | 否 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 0.80 | triton | 默认 | 否 | 否 |
| Qwen3-30B-A3B | 0.90 | triton | 默认 | 否 | 是 |
| Qwen3-30B-A3B-Instruct-2507 | 0.90 | triton | 默认 | 否 | 是 |
| Qwen3-32B | 0.93 | triton | 默认 | 否 | 否 |

## 批级时延结果（teacher-forced）

| 模型 | ML 全部 MAPE | AIC 全部 MAPE | ML prefill | AIC prefill | ML decode | AIC decode |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 2.16% | 84.98% | 2.37% | 73.89% | 2.11% | 87.24% |
| Qwen2.5-3B-Instruct | 5.52% | 73.01% | 4.02% | 42.61% | 5.83% | 79.21% |
| llama-2-7b | 3.66% | 35.62% | 1.96% | 15.55% | 4.01% | 39.71% |
| Mistral-7B-Instruct-v0.2 | 1.64% | 43.29% | 1.55% | 10.59% | 1.66% | 49.93% |
| Qwen2.5-7B-Instruct | 6.22% | 44.81% | 3.31% | 10.45% | 6.82% | 51.81% |
| Qwen3-8B | 4.94% | 49.58% | 1.98% | 11.72% | 5.54% | 57.29% |
| Qwen2.5-14B-Instruct | 3.89% | 34.34% | 2.06% | 9.04% | 4.26% | 39.49% |
| Qwen2.5-14B-Instruct-1M | 8.00% | 79.85% | 2.86% | 94.20% | 9.18% | 76.55% |
| Qwen3-14B | 5.20% | 29.05% | 1.31% | 5.50% | 6.00% | 33.84% |
| Meta-Llama-3.1-70B-Instruct-AWQ | 5.71% | 44.08% | 1.34% | 17.82% | 6.60% | 49.41% |
| Qwen3-30B-A3B | 2.43% | 51.55% | 2.83% | 15.12% | 2.34% | 58.98% |
| Qwen3-30B-A3B-Instruct-2507 | 2.16% | 51.29% | 2.16% | 14.72% | 2.17% | 58.74% |
| Qwen3-32B | 1.64% | 6.86% | 1.26% | 7.03% | 1.72% | 6.83% |

批级平均实测值与预测值（全部 ScheduleBatch）：

| 模型 | 实测 ms | ML 预测 ms | ML 偏差 | AIC 预测 ms | AIC 偏差 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 25.04 | 23.83 | -0.29% | 3.58 | -84.98% |
| Qwen2.5-3B-Instruct | 31.05 | 29.67 | -0.88% | 8.03 | -73.01% |
| llama-2-7b | 28.47 | 26.79 | -2.04% | 18.05 | -35.62% |
| Mistral-7B-Instruct-v0.2 | 28.34 | 27.20 | -0.41% | 16.25 | -43.28% |
| Qwen2.5-7B-Instruct | 27.19 | 25.83 | -0.73% | 15.14 | -44.76% |
| Qwen3-8B | 33.00 | 32.68 | 2.65% | 16.70 | -49.49% |
| Qwen2.5-14B-Instruct | 45.12 | 42.52 | -3.67% | 30.25 | -34.33% |
| Qwen2.5-14B-Instruct-1M | 280.96 | 292.65 | 7.37% | 29.99 | -79.85% |
| Qwen3-14B | 41.83 | 38.94 | -4.86% | 30.32 | -29.03% |
| Meta-Llama-3.1-70B-Instruct-AWQ | 121.06 | 120.77 | 1.33% | 137.83 | 38.34% |
| Qwen3-30B-A3B | 85.00 | 76.62 | -2.24% | 37.42 | -51.55% |
| Qwen3-30B-A3B-Instruct-2507 | 84.65 | 76.63 | -1.71% | 37.42 | -51.29% |
| Qwen3-32B | 70.38 | 68.71 | -1.18% | 64.70 | -6.84% |

![teacher forced mape](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/teacher_forced_all_mape.png)

![teacher forced prefill mape](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/teacher_forced_prefill_mape.png)

![teacher forced decode mape](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/teacher_forced_decode_mape.png)

![teacher forced actual predicted](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/teacher_forced_actual_vs_predicted.png)

![teacher forced signed error](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/teacher_forced_signed_error.png)

批级实测值为实际 SGLang 一次 forward iteration 的设备同步时延；AIC 为静态算子组合估计，ML 为基于同配置训练 trace 的回归估计。两者都没有将 CPU 调度、网络和后续 batch 重排单独拆入这个批级数值。

## 端到端留出 trace 结果

| 模型 | 实测 E2E ms | ML 预测 ms / MAPE | AIC 预测 ms / MAPE | ML TTFT MAPE | AIC TTFT MAPE |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 6638.71 | 6822.71 / 3.09% | 2231.93 / 66.32% | 3.89% | 68.89% |
| Qwen2.5-3B-Instruct | 8060.43 | 8181.26 / 3.33% | 3222.19 / 59.95% | 6.10% | 58.60% |
| llama-2-7b | 7558.93 | 8363.99 / 10.58% | 6455.22 / 14.68% | 3.94% | 19.17% |
| Mistral-7B-Instruct-v0.2 | 7585.63 | 8222.72 / 8.37% | 5764.10 / 23.96% | 2.91% | 23.98% |
| Qwen2.5-7B-Instruct | 7440.23 | 7863.61 / 5.79% | 5090.42 / 31.67% | 4.72% | 30.71% |
| Qwen3-8B | 8795.25 | 10115.94 / 14.85% | 5811.17 / 33.77% | 7.81% | 33.00% |
| Qwen2.5-14B-Instruct | 12213.11 | 12589.33 / 3.25% | 9744.32 / 20.22% | 4.75% | 16.84% |
| Qwen2.5-14B-Instruct-1M | 99032.34 | 102279.17 / 3.29% | 9972.89 / 89.94% | 1.85% | 92.32% |
| Qwen3-14B | 11592.64 | 11790.90 / 2.10% | 9885.56 / 14.53% | 2.67% | 15.24% |
| Meta-Llama-3.1-70B-Instruct-AWQ | 37095.63 | 39783.62 / 7.16% | 40235.26 / 8.39% | 2.71% | 9.09% |
| Qwen3-30B-A3B | 20928.74 | 19473.42 / 7.01% | 11636.80 / 44.40% | 13.73% | 43.35% |
| Qwen3-30B-A3B-Instruct-2507 | 20823.25 | 19755.17 / 5.22% | 11653.22 / 44.04% | 12.25% | 42.92% |
| Qwen3-32B | 20048.26 | 20879.38 / 4.14% | 19371.57 / 3.37% | 1.53% | 7.06% |

![end to end mape](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/end_to_end_mape.png)

![end to end actual predicted](../test-results/pr33824_comprehensive_all13_predictors_20260918_1605/analysis/figures/end_to_end_actual_vs_predicted.png)

端到端误差会包含单步预测偏差在后续调度中的累积影响。表格同时保留实测均值和预测均值，避免只看误差百分比而忽略绝对时延规模。

## 汇总比较

- 批级全部行 MAPE 按模型等权平均：ML 4.09%，AIC Silicon 48.33%；中位数分别为 3.89% 和 44.81%。
- 批级 MAPE 较小的模型数：ML 13 个，AIC Silicon 0 个；其余为并列。
- 端到端平均 E2E MAPE 按模型等权平均：ML 6.01%，AIC Silicon 35.02%。
- 端到端 E2E MAPE 较小的模型数：ML 12 个，AIC Silicon 1 个；其余为并列。
- 上述统计只覆盖本报告的统一 1024/256 饱和工作负载、A100 PCIe 和当前兼容 kernel；不将其外推为其他 batch、上下文或硬件上的结论。

## 兼容性说明

| 模型 | AWQ BF16 代理 | dual-chunk Triton 代理 | MoE expert trace-targeted 表缺失 |
| --- | --- | --- | --- |
| Qwen3-0.6B | 否 | 否 | 否 |
| Qwen2.5-3B-Instruct | 否 | 否 | 否 |
| llama-2-7b | 否 | 否 | 否 |
| Mistral-7B-Instruct-v0.2 | 否 | 否 | 否 |
| Qwen2.5-7B-Instruct | 否 | 否 | 否 |
| Qwen3-8B | 否 | 否 | 否 |
| Qwen2.5-14B-Instruct | 否 | 否 | 否 |
| Qwen2.5-14B-Instruct-1M | 否 | 是 | 否 |
| Qwen3-14B | 否 | 否 | 否 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 是 | 否 | 否 |
| Qwen3-30B-A3B | 否 | 否 | 是 |
| Qwen3-30B-A3B-Instruct-2507 | 否 | 否 | 是 |
| Qwen3-32B | 否 | 否 | 否 |

AWQ checkpoint 保留原始权重参与真实基线，但 AIC 0.10 在本机 A100/SM80 没有原生 dense-AWQ GEMM 表，因此该行使用 BF16 算子代理。1M checkpoint 的真实短上下文路径使用 dual-chunk FlashAttention；AIC trace-targeted attention 表为 Triton 代理。Qwen3 MoE 两行保留 AIC 的通用 MoE 表，当前没有按 trace 动态路由分布补采 expert 表。上述行仍纳入结果表，但不应与原生 BF16 dense 行作“仅差模型大小”的解释。

## 结果位置

- 原始实验根目录：`/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_comprehensive_all13_predictors_20260918_1605`
- 模型级 real、ML、AIC 命令、日志、原始请求 trace 和 metrics 均在 `models/<模型名>/` 下。
- 本报告的结构化汇总在同级 `analysis/` CSV 文件中；图位于 `analysis/figures/`。
