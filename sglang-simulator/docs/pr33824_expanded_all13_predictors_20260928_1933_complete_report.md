# 全模型预测方法对比实验

## 实验口径

本轮对 13 个本地 checkpoint 分别进行真实 SGLang 采集、ML 训练与 AIConfigurator Silicon 预测。每个模型训练请求 800 个，独立测试请求 1280 个；训练与测试 seed 分开。训练负载：`latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf`；测试负载：`latency_256x128_r1,service_1024x256_r4,saturation_1024x256_inf,latency_128x64_r1,saturation_1536x384_inf`。真实服务每轮加载完整 checkpoint，记录同步墙钟 scheduler iteration 时延。

ML 仅使用训练 trace 的 ScheduleBatch 时延拟合 HGB 模型。AIC 的宽覆盖表和 trace-targeted 算子表也只使用训练 trace 的形状；留出 trace 的真实时延不参与 ML 拟合或 AIC 选点。批级比较采用 teacher-forced 方式，端到端比较则使用同一条留出原始请求 trace 分别运行 CPU 逻辑模拟器。

本轮比较的预测方法是 ML（HGB）和 AIC Silicon。此前的“退化后端”记录对应另一套真实 kernel 路径，不能与本轮同一兼容 kernel 的基线构成受控比较，因此不作为第三种方法并入本表。

每条不同 seed 的留出 trace 运行一次预测；真实 GPU 对每类负载保留五轮。每负载 n=5；多负载汇总 n 为实际配对数，不能混称为同一负载的五次结果。多负载汇总表为描述性总览，同负载与跨负载差异须查看分负载结果。

## 版本与运行设置

- SGLang 源码 revision：`59799a368793b9f795baf59b067233a65ad8e38e`
- Python：`/home/ms-admin/workspace1/.miniforge3/envs/hbf-cxl/bin/python`
- 训练请求：800；留出请求：1280。
- 训练 seed：`202609280,202609281,202609282,202609283,202609284`；留出 seed：`202609380,202609381,202609382,202609383,202609384`。
- 调度设置：attention=per-model (triton except 1M dual_chunk_flash_attn); 1M_page_size=256_for_FA2_paged_KV; overlap=disabled; cuda_graph=disabled; context_length=2048; max_running_requests=32; chunked_prefill_size=512; random_range_ratio=0。
- kernel 设置：isolated_SM80_AOT; SGLANG_FORCE_FUSED_OP_BACKEND=unset；阶段目录：`/mnt/data/sglang-pr33824-merged/test-results/pr33824_sglang_kernel_sm80_compat_20260908_1915/minimal-stage-v4`。
- AIC：AIConfigurator 0.10，backend schema `0.5.10`，源码 revision `c5f139662be28b177ca9f62f83543bbf10f133a0`。
- AIC 算子采集：attention: 3 warmups,min 20 CUDA-event iterations; GEMM: 3 warmups,min 6 CUDA-event iterations (each iteration contains a cache-aware op list); adaptive run count and CUDA graph replay as defined by collector/helper.py。
- AIC 规则插值支撑：regular attention grid from training plans plus context_length=2048 and max_running_requests=32; holdout paths were not read。
- AIC 评分表处理：for relevant context-attention slices, retain only direct regular-grid measurements so sparse training keys cannot shadow Silicon batch interpolation; all other operator tables are copied unchanged。
- AIC 的 backend schema 是其性能表查询命名空间；真实 SGLang 服务的身份以上述源码 revision、每条 trace 的 package_versions.txt 和兼容 kernel 路径为准，二者不混写为同一个版本号。

## 数据量与 AIC 采样覆盖

| 项目 | 数值 |
| --- | --- |
| 规则 batch 网格 | 1–32（32 个值） |
| prefill 长度网格 | 38 个值，最大 2048 |
| decode 长度网格 | 42 个值，最大 2304 |
| prefill 支撑点 | 12160（新测 11647，复用 513） |
| decode 支撑点 | 13440（新测 13220，复用 220） |
| 规则网格总点 | 25600 |

| 评分库 context 表处理 | 行数 |
| --- | --- |
| 原始 context 表 | 17367 |
| 评分库保留的完整规则 context 行 | 12160 |
| 保留的非本轮模型 context 行 | 240 |
| 从评分 context 切片排除的散点/不兼容行 | 4967 |

规则网格只由训练 trace 计划和固定的 `context_length=2048`、`max_running_requests=32` 生成。它用于保证 AIC Silicon 对独立留出 trace 的批大小和长度组合有完整的三维插值支撑；留出 trace 不参与该选点过程。
原始 AIC 表仍保留所有 trace-targeted 行。正式评分使用独立评分库：对本轮 10 个 attention 形状，只保留完整规则 context 网格，避免 AIC 先选到一个单 batch 的相邻长度行；generation、GEMM 和 MoE 表没有改写。

| 模型 | 训练请求 | 留出请求 | 训练 batch | 留出 batch | 留出超出训练特征范围 | AIC attention 目标 | AIC attention 新测 / 复用 | AIC GEMM 新测 / 复用 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 800 | 1280 | 11533 | 19080 | 17.96% | 1597 | 1586 / 11 | 1490 / 55 |
| Qwen2.5-3B-Instruct | 800 | 1280 | 10557 | 17628 | 19.29% | 1570 | 1560 / 10 | 1480 / 55 |
| llama-2-7b | 800 | 1280 | 11532 | 18500 | 18.80% | 1560 | 1549 / 11 | 1520 / 55 |
| Mistral-7B-Instruct-v0.2 | 800 | 1280 | 11319 | 18306 | 19.20% | 1541 | 1533 / 8 | 1159 / 466 |
| Qwen2.5-7B-Instruct | 800 | 1280 | 11598 | 19018 | 17.85% | 1571 | 1561 / 10 | 1475 / 55 |
| Qwen3-8B | 800 | 1280 | 10328 | 17291 | 19.61% | 1549 | 1187 / 362 | 1052 / 503 |
| Qwen2.5-14B-Instruct | 800 | 1280 | 9073 | 15701 | 21.53% | 1452 | 1443 / 9 | 1470 / 55 |
| Qwen2.5-14B-Instruct-1M | 800 | 1280 | 5862 | 9974 | 33.86% | 884 | 291 / 593 | 410 / 1235 |
| Qwen3-14B | 800 | 1280 | 9256 | 16069 | 21.10% | 1450 | 231 / 1219 | 869 / 601 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 800 | 1280 | 7069 | 12674 | 24.28% | 1038 | 1030 / 8 | 1140 / 50 |
| Qwen3-30B-A3B | 800 | 1280 | 6951 | 12223 | 27.40% | 1266 | 1255 / 11 | 1179 / 256 |
| Qwen3-30B-A3B-Instruct-2507 | 800 | 1280 | 6931 | 12188 | 27.49% | 1279 | 97 / 1182 | 97 / 1358 |
| Qwen3-32B | 800 | 1280 | 7930 | 13889 | 24.24% | 1256 | 930 / 326 | 1025 / 285 |

同样的请求预算并不保证产生同样数量的 scheduler batch：较快或较慢的模型在饱和队列下会形成不同的动态 batch 组合。因此训练和留出 batch 数均单独列出，作为实际时延样本量。
AIC 的“复用”表示该算子形状已在本轮相同 A100、相同采集后端下直接测得，因而不重复执行同一测点；它不是使用留出 trace，也不是把 ML 训练样本在模型间转移。

![scheduler batch samples](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/scheduler_batch_samples.png)

![aic points](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/aic_trace_targeted_points.png)

![aic regular support](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/aic_regular_interpolation_support.png)

![ml feature support](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/ml_feature_support.png)

训练侧还保留了一个按 trace 分层的内部检查，用于确认模型拟合过程；它与后面的独立留出 trace 结果分开报告，不能替代正式测试。

## ML 训练内核查（非正式留出结果）

| 模型 | 最终 fit 输入行 | 内部训练 / 检查行 | p50 HGB MAPE | MAE ms |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 11533 | 9227 / 2306 | 2.24% | 1.63 |
| Qwen2.5-3B-Instruct | 10557 | 8445 / 2112 | 1.89% | 1.52 |
| llama-2-7b | 11532 | 9226 / 2306 | 1.41% | 1.37 |
| Mistral-7B-Instruct-v0.2 | 11319 | 9055 / 2264 | 1.84% | 1.33 |
| Qwen2.5-7B-Instruct | 11598 | 9277 / 2321 | 1.50% | 1.07 |
| Qwen3-8B | 10328 | 8262 / 2066 | 1.83% | 1.67 |
| Qwen2.5-14B-Instruct | 9073 | 7257 / 1816 | 1.30% | 0.68 |
| Qwen2.5-14B-Instruct-1M | 5862 | 4688 / 1174 | 7.77% | 27.62 |
| Qwen3-14B | 9256 | 7402 / 1854 | 3.09% | 2.40 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 7069 | 5656 / 1413 | 2.63% | 3.00 |
| Qwen3-30B-A3B | 6951 | 5561 / 1390 | 2.15% | 12.22 |
| Qwen3-30B-A3B-Instruct-2507 | 6931 | 5545 / 1386 | 2.54% | 14.86 |
| Qwen3-32B | 7930 | 6344 / 1586 | 1.61% | 1.31 |

每条 real trace 的完整启动命令、SGLang/sgl-kernel 包版本和原始 server log 均保存在对应的 `calibration/real/repeat-*/` 或 `holdout/real/repeat-*/` 目录。下表给出模型间不同的运行参数；其余 scheduler 设置统一采用本报告前述口径。

## 模型级运行参数

| 模型 | 显存比例 | attention 后端 | page size | FA2 兼容层 | 运行时 shim |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 0.65 | triton | 默认 | 否 | 否 |
| Qwen2.5-3B-Instruct | 0.8 | triton | 默认 | 否 | 否 |
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
| Qwen3-0.6B | 6.21% | 88.75% | 4.59% | 76.34% | 6.38% | 90.03% |
| Qwen2.5-3B-Instruct | 4.55% | 77.54% | 3.33% | 48.39% | 4.69% | 80.80% |
| llama-2-7b | 3.84% | 49.61% | 2.38% | 22.39% | 4.00% | 52.51% |
| Mistral-7B-Instruct-v0.2 | 5.42% | 53.49% | 2.86% | 16.78% | 5.70% | 57.43% |
| Qwen2.5-7B-Instruct | 6.04% | 52.11% | 3.57% | 16.26% | 6.29% | 55.81% |
| Qwen3-8B | 4.55% | 58.01% | 2.79% | 19.25% | 4.75% | 62.45% |
| Qwen2.5-14B-Instruct | 3.52% | 42.30% | 2.32% | 12.78% | 3.68% | 46.01% |
| Qwen2.5-14B-Instruct-1M | 12.18% | 82.54% | 5.95% | 94.84% | 13.51% | 79.92% |
| Qwen3-14B | 6.16% | 38.66% | 3.90% | 9.64% | 6.44% | 42.22% |
| Meta-Llama-3.1-70B-Instruct-AWQ | 4.73% | 30.35% | 2.20% | 19.06% | 5.13% | 32.11% |
| Qwen3-30B-A3B | 4.62% | 63.92% | 6.88% | 28.17% | 4.24% | 69.80% |
| Qwen3-30B-A3B-Instruct-2507 | 4.76% | 63.51% | 5.85% | 27.31% | 4.58% | 69.48% |
| Qwen3-32B | 2.85% | 15.56% | 1.58% | 4.74% | 3.03% | 17.09% |

批级平均实测值与预测值（全部 ScheduleBatch）：

| 模型 | 实测 ms | ML 预测 ms | ML 平均相对偏差 | AIC 预测 ms | AIC 平均相对偏差 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 25.97 | 25.88 | 3.08% | 2.83 | -88.75% |
| Qwen2.5-3B-Instruct | 31.96 | 31.25 | 0.63% | 6.96 | -77.54% |
| llama-2-7b | 28.37 | 27.49 | -0.09% | 14.23 | -49.61% |
| Mistral-7B-Instruct-v0.2 | 29.03 | 28.60 | 1.57% | 13.53 | -53.47% |
| Qwen2.5-7B-Instruct | 26.51 | 25.93 | 1.16% | 12.70 | -52.08% |
| Qwen3-8B | 33.99 | 32.60 | -1.33% | 14.23 | -57.94% |
| Qwen2.5-14B-Instruct | 44.41 | 43.39 | -0.06% | 26.03 | -42.29% |
| Qwen2.5-14B-Instruct-1M | 310.80 | 300.60 | 2.74% | 29.58 | -82.54% |
| Qwen3-14B | 41.34 | 40.24 | 0.28% | 25.76 | -38.57% |
| Meta-Llama-3.1-70B-Instruct-AWQ | 111.92 | 110.90 | 0.68% | 124.70 | 26.30% |
| Qwen3-30B-A3B | 95.98 | 84.32 | -3.44% | 31.07 | -63.92% |
| Qwen3-30B-A3B-Instruct-2507 | 95.22 | 85.30 | -1.47% | 31.13 | -63.51% |
| Qwen3-32B | 68.90 | 67.22 | -0.90% | 58.51 | -15.54% |

平均相对偏差为逐 batch 的 (预测−实测)/实测 的平均值，正值表示平均相对高估；它不是 (预测均值−实测均值)/实测均值，两者可能符号不同。MAPE 对逐 batch 相对误差取绝对值后平均。


![teacher forced mape](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/teacher_forced_all_mape.png)

![teacher forced prefill mape](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/teacher_forced_prefill_mape.png)

![teacher forced decode mape](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/teacher_forced_decode_mape.png)

![teacher forced actual predicted](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/teacher_forced_actual_vs_predicted.png)

![teacher forced signed error](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/teacher_forced_signed_error.png)

批级实测为同步墙钟 run_batch 时延，包含函数内 CPU 发射、采样/拷贝和同步等待，但不包含函数外全部调度及网络时延。AIC 为 GPU 算子组合估计，ML 拟合上述同步墙钟标签；两者计时口径并非天然相同。

## 端到端留出 trace 结果

| 模型 | 实测 E2E ms | ML 预测 ms / MAPE | AIC 预测 ms / MAPE | ML TTFT MAPE | AIC TTFT MAPE |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 5008.01 | 5390.30 / 10.13% | 2009.41 / 67.00% | 22.41% | 72.46% |
| Qwen2.5-3B-Instruct | 6037.78 | 6294.81 / 6.33% | 2398.02 / 65.72% | 19.69% | 65.89% |
| llama-2-7b | 5652.75 | 6193.06 / 11.21% | 4817.25 / 28.99% | 12.11% | 35.15% |
| Mistral-7B-Instruct-v0.2 | 5703.27 | 6288.98 / 12.02% | 4433.44 / 32.87% | 13.54% | 36.53% |
| Qwen2.5-7B-Instruct | 5365.69 | 5823.26 / 11.46% | 3607.24 / 40.16% | 18.80% | 44.23% |
| Qwen3-8B | 6623.54 | 7022.07 / 8.41% | 4531.04 / 40.19% | 17.25% | 45.06% |
| Qwen2.5-14B-Instruct | 9080.83 | 9340.98 / 5.84% | 7329.57 / 26.75% | 18.98% | 35.69% |
| Qwen2.5-14B-Instruct-1M | 86307.17 | 82786.88 / 5.82% | 7432.00 / 92.39% | 7.58% | 96.22% |
| Qwen3-14B | 8629.82 | 8906.92 / 7.07% | 7359.89 / 22.28% | 18.91% | 31.62% |
| Meta-Llama-3.1-70B-Instruct-AWQ | 28531.64 | 28523.31 / 3.18% | 30929.26 / 15.63% | 13.03% | 11.39% |
| Qwen3-30B-A3B | 17637.20 | 16215.96 / 8.45% | 7968.21 / 66.48% | 32.68% | 69.73% |
| Qwen3-30B-A3B-Instruct-2507 | 17482.99 | 16798.12 / 6.42% | 7963.89 / 66.31% | 32.77% | 70.31% |
| Qwen3-32B | 15322.04 | 15671.32 / 3.19% | 14838.52 / 7.35% | 14.45% | 19.48% |

![end to end mape](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/end_to_end_mape.png)

![TTFT mape](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/ttft_mape.png)

![TPOT mape](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/tpot_mape.png)

![end to end actual predicted](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/figures/end_to_end_actual_vs_predicted.png)

端到端误差会包含单步预测偏差在后续调度中的累积影响。表格同时保留实测均值和预测均值，避免只看误差百分比而忽略绝对时延规模。

## 汇总比较

- 批级全部行 MAPE 按模型等权平均：ML 5.34%，AIC Silicon 55.10%；中位数分别为 4.73% 和 53.49%。
- 批级 MAPE 较小的模型数：ML 13 个，AIC Silicon 0 个；其余为并列。
- 端到端平均 E2E MAPE 按模型等权平均：ML 7.66%，AIC Silicon 44.01%。
- 端到端 E2E MAPE 较小的模型数：ML 13 个，AIC Silicon 0 个；其余为并列。
- 上述统计仅覆盖本报告实际采集的工作负载、A100 PCIe 和当前兼容 kernel，不外推至未测试配置。

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

- 原始实验根目录：`/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933`
- 模型级 real、ML、AIC 命令、日志、原始请求 trace 和 metrics 均在 `models/<模型名>/` 下。
- 本报告的结构化汇总在同级 `analysis/` CSV 文件中；图位于 `analysis/figures/`。

## 实际运行版本核验

下表来自每组真实采集的 package_versions.txt，而不是 conda 环境名称。核对了训练与独立测试组版本一致、AIC 采集的 Torch/CUDA/kernel 运行版本及实际 native 导入路径一致。采集前另以原生二进制 SHA256 对照正式基线快照。SGLang 元数据为 0.0.0，源码身份以上文 revision 和 dirty 源码快照为准。

| 模型 | 核验真实组数 | SGLang 元数据 | Torch / CUDA | kernel 运行 / 包元数据 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Qwen2.5-3B-Instruct | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| llama-2-7b | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Mistral-7B-Instruct-v0.2 | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Qwen2.5-7B-Instruct | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Qwen3-8B | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Qwen2.5-14B-Instruct | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Qwen2.5-14B-Instruct-1M | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Qwen3-14B | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Qwen3-30B-A3B | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Qwen3-30B-A3B-Instruct-2507 | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |
| Qwen3-32B | 10 | 0.0.0 | 2.9.1+cu126 / 12.6 | 0.4.6.post1 / 0.4.6.post1+cu129 |

AIC attention 采集 page size 为 1，KV split 上限为 8（非固定 split 数）。普通 BF16 Triton 基线按 page=1 对齐；1M dual-chunk/page=256、AWQ 精度和 MoE 路由的代理差异不因版本核验而消失。

## AIC 采集源码记录

采集前归档了 385 个 collector/SDK/配置源文件，并核验归档内容与 SHA256 清单一致；实际导入的 SDK 与声明 checkout 的 133 个 Python 文件逐一相同。checkout dirty=True，因此源码身份由 revision、dirty patch 和文件归档共同确定。原生 kernel 的一致性另按二进制 SHA256 检查。

归档位于 aic_calibration/provenance_before_wide_sweep/；算子性能表与本次重新测量记录位于同一校准目录，不混入源码归档。

## ML 训练记录

正式 predictor 为每模型独立 HGB p50，18 维 batch/长度特征；不按独立测试误差选择 p50 或 L2。训练集合内先按每条 trace 用例随机划分 80%/20%作内部检查，再将全部训练行传给最终 fit；HGB 另取训练侧 15%作 early-stopping 验证，不使用正式测试请求。

上游 data-tools 的 trainer 读取已采集的 batch 文件，不负责生成独立测试请求。其默认内部检查按每个用例随机分行，比例 50%/50%、seed=42、forward_modes=1；检查后保存的模型重新使用全部输入行拟合。这个划分没有按完整请求隔离，同一请求的多个 batch 可能落在两侧，不能作为跨请求或跨负载泛化结果。本轮正式测试另行采集完整请求，内部检查只用于核查拟合过程。[上游训练脚本](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/tools/train_latency_model.py)，[上游验证说明](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/README.md#train-an-ml-latency-predictor)。

上游 serving 单测与上述训练验证是两件事：它使用 examples 中的 ML 配置，在 CPU 模拟器上发送 3 条 ShareGPT 请求，每条输出 4 token，断言完成数和指标有效性，没有真机误差对照。配套资产生成脚本仅以一行全零特征、0.001 秒标签构建固定 1 ms 的 DummyRegressor；它是功能示例，不是 GPU 校准结果。本轮不使用该示例预测器，而是每模型分别采集、训练并在独立请求上评价。[上游 serving 单测](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/test/test_simulation_sglang_serving.py)，[ML 示例配置](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/examples/sim_configs/ml.json)，[示例资产生成脚本](https://github.com/littlefatfat/sglang/blob/9cfddcedc9d1e00885c28db75a45513eddbb6271/tools/sglang-simulator/examples/build_example_assets.py)。

| 参数 | 设置 |
| --- | --- |
| loss / quantile | quantile / 0.5 |
| max_iter / max_leaf_nodes | 400 / 63 |
| learning_rate / l2_regularization | 0.04 / 0.1 |
| early_stopping / validation_fraction / n_iter_no_change | True / 0.15 / 20 |
| 标签与过滤 | iter_latency，秒；训练只保留有效 mode=1/2、非空 batch、0 < latency ≤ 30 秒；正式评分保留所有有限正时延 |

| 模型 | 最终 fit 输入行 | 训练 trace 用例 | 实际迭代数 |
| --- | --- | --- | --- |
| Qwen3-0.6B | 11533 | 15 | 285 |
| Qwen2.5-3B-Instruct | 10557 | 15 | 286 |
| llama-2-7b | 11532 | 15 | 400 |
| Mistral-7B-Instruct-v0.2 | 11319 | 15 | 400 |
| Qwen2.5-7B-Instruct | 11598 | 15 | 393 |
| Qwen3-8B | 10328 | 15 | 400 |
| Qwen2.5-14B-Instruct | 9073 | 15 | 313 |
| Qwen2.5-14B-Instruct-1M | 5862 | 15 | 229 |
| Qwen3-14B | 9256 | 15 | 274 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 7069 | 15 | 242 |
| Qwen3-30B-A3B | 6951 | 15 | 246 |
| Qwen3-30B-A3B-Instruct-2507 | 6931 | 15 | 293 |
| Qwen3-32B | 7930 | 15 | 313 |

全部特征名称、单调约束、random_state 和实际参数另存 training_parameters.csv；原始 bundle 和训练命令同时保留。

### 上游分行方法的请求重叠核对

将上游提交中的原始划分函数应用到本轮下列模型的训练记录，seed=42、50%/50%，只检查 request ID 和行索引，不拟合。这不是上游作者的数据或准确率，也没有改变本轮训练与独立测试的划分。默认仅保留 prefill；第二列对照显式加入 decode，并非上游默认设置。比例以两侧请求 ID 的并集为分母。

| 模型 | 默认 prefill：两侧共享请求 / 全部请求 | 包含 decode：两侧共享请求 / 全部请求 |
| --- | --- | --- |
| Mistral-7B-Instruct-v0.2 | 216 / 800（27.00%） | 792 / 800（99.00%） |
| Qwen2.5-3B-Instruct | 217 / 800（27.12%） | 793 / 800（99.12%） |
| Qwen3-0.6B | 222 / 800（27.75%） | 794 / 800（99.25%） |
| llama-2-7b | 224 / 800（28.00%） | 795 / 800（99.38%） |

两侧没有相同的原始行，但同一请求的不同步骤仍可能分到两侧；因此这种内部检查不等于独立请求测试。此表不测量它使误差降低多少，不能拿来替代前面的冻结预测器对照。原始函数、执行命令和输入校验值保存在 verification/upstream_row_split_identity_*/；明细复制到 upstream_split_request_overlap.csv。

## 历史 ML 请求与本轮变化

另行核对了 2026-09-18 全模型矩阵的原始请求与回放 trace，共 130 个闭合用例、8320 个请求。那轮已经采用独立训练/测试 seed，并非上游脚本按 batch 行划分的内部验证；两者不能统称为同请求测试。此处核对请求身份，不将其当作拟合过程或准确率的证明。

历史训练 seed：20260918, 20260919, 20260920, 20260921, 20260922；测试 seed：20261018, 20261019, 20261020, 20261021, 20261022。两侧 request ID 无复用，输入 token 与输出长度均与各自回放 trace 对应。下表 prompt 重叠以测试请求数为分母。

| 历史模型 | 训练/测试请求 | 测试 prompt 与训练重叠 | 输入 ID+输出长度重叠 | 回放 ID 核验请求 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 320 / 320 | 0.00% | 0.00% | 640 |
| Qwen2.5-3B-Instruct | 320 / 320 | 0.00% | 0.00% | 640 |
| llama-2-7b | 320 / 320 | 0.00% | 0.00% | 640 |
| Mistral-7B-Instruct-v0.2 | 320 / 320 | 0.00% | 0.00% | 640 |
| Qwen2.5-7B-Instruct | 320 / 320 | 0.00% | 0.00% | 640 |
| Qwen3-8B | 320 / 320 | 0.00% | 0.00% | 640 |
| Qwen2.5-14B-Instruct | 320 / 320 | 0.00% | 0.00% | 640 |
| Qwen2.5-14B-Instruct-1M | 320 / 320 | 0.00% | 0.00% | 640 |
| Qwen3-14B | 320 / 320 | 0.00% | 0.00% | 640 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 320 / 320 | 0.00% | 0.00% | 640 |
| Qwen3-30B-A3B | 320 / 320 | 0.00% | 0.00% | 640 |
| Qwen3-30B-A3B-Instruct-2507 | 320 / 320 | 0.00% | 0.00% | 640 |
| Qwen3-32B | 320 / 320 | 0.00% | 0.00% | 640 |

本轮相对该历史矩阵同时增加请求量和训练负载覆盖，不能把两轮的误差变化仅解释成测试请求不同。训练侧回评分、上游内部划分验证、历史独立请求和本轮独立请求分别列示；旧预测器在本轮相同真实步骤上的评分另作配对比较。

明细见 historical_train_test_request_identity.csv；原始审计命令、执行时源码及输入校验值保存在 verification/historical_request_identity_all13_*/。未重新训练、预测或修改历史数据。

## 训练与测试请求核对

直接检查真机保存的 input_ids、实际输出长度和 request ID，不只依靠 seed 声明；逐条核验回放 trace 的输入 ID/输出长度未改变。下表的重叠比例以测试请求数为分母。相同 prompt、相同 18 维特征与同一条时延标签是不同概念；少量 prompt 重复不自动构成时延标签泄漏。正式拟合仍只读取训练目录。

| 模型 | 训练/测试请求 | 测试 prompt 与训练重叠 | 输入 ID+输出长度同时重叠 | 回放 ID 核验请求 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Qwen2.5-3B-Instruct | 800 / 1280 | 0.00% | 0.00% | 2080 |
| llama-2-7b | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Mistral-7B-Instruct-v0.2 | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Qwen2.5-7B-Instruct | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Qwen3-8B | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Qwen2.5-14B-Instruct | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Qwen2.5-14B-Instruct-1M | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Qwen3-14B | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Qwen3-30B-A3B | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Qwen3-30B-A3B-Instruct-2507 | 800 / 1280 | 0.00% | 0.00% | 2080 |
| Qwen3-32B | 800 / 1280 | 0.00% | 0.00% | 2080 |

## 离线回放的计算开销

回放日志同时记录 predicted forward、cpu_overhead 和 predictor_time_cost。当前 OFFLINE 路径以相邻处理点的墙钟间隔推进 CPU 时间，预测函数在该间隔内执行；predictor_time_cost 另行汇总，但没有从 cpu_overhead 中扣除。所以下表是模拟器自身的执行开销，不是独立测得的原生服务 CPU 时延，也不是 GPU 算子时延。依据为本轮归档的 src/sglang_simulator/simulation/sglang/scheduler.py。

均值按回放 iteration 数加权；不同预测方法形成的动态 batch 数可能不同。每个用例、每次重复的明细保存在 replay_wall_time_components.csv。保留原始端到端结果，没有事后扣除平均开销；扣除会改变后续调度，效果需要另行回放验证。

| 模型 | 方法 | 核对用例数 | 回放 iteration 数 | 预测函数墙钟 ms/iter | 计入时钟的 CPU ms/iter | predicted forward ms/iter |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | ML | 25 | 18446 | 3.034 | 4.062 | 25.872 |
| Qwen3-0.6B | AIC | 25 | 27680 | 5.419 | 6.347 | 2.572 |
| Qwen2.5-3B-Instruct | ML | 25 | 17268 | 3.081 | 4.147 | 30.925 |
| Qwen2.5-3B-Instruct | AIC | 25 | 26677 | 3.372 | 4.287 | 6.641 |
| llama-2-7b | ML | 25 | 17745 | 4.216 | 5.297 | 27.428 |
| llama-2-7b | AIC | 25 | 22089 | 4.611 | 5.609 | 13.878 |
| Mistral-7B-Instruct-v0.2 | ML | 25 | 17434 | 4.225 | 5.308 | 28.524 |
| Mistral-7B-Instruct-v0.2 | AIC | 25 | 21976 | 5.212 | 6.209 | 13.103 |
| Qwen2.5-7B-Instruct | ML | 25 | 18199 | 4.308 | 5.376 | 25.713 |
| Qwen2.5-7B-Instruct | AIC | 25 | 23565 | 2.609 | 3.573 | 12.318 |
| Qwen3-8B | ML | 25 | 16821 | 4.286 | 5.392 | 32.572 |
| Qwen3-8B | AIC | 25 | 21644 | 5.438 | 6.429 | 13.749 |
| Qwen2.5-14B-Instruct | ML | 25 | 15522 | 3.446 | 4.567 | 43.077 |
| Qwen2.5-14B-Instruct | AIC | 25 | 18208 | 6.403 | 7.475 | 25.359 |
| Qwen2.5-14B-Instruct-1M | ML | 25 | 9987 | 2.571 | 4.117 | 300.300 |
| Qwen2.5-14B-Instruct-1M | AIC | 25 | 18268 | 6.311 | 7.513 | 25.374 |
| Qwen3-14B | ML | 25 | 15923 | 3.043 | 4.169 | 40.270 |
| Qwen3-14B | AIC | 25 | 18264 | 6.318 | 7.381 | 25.158 |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 25 | 12626 | 2.824 | 4.123 | 109.888 |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 25 | 11992 | 7.343 | 8.616 | 126.859 |
| Qwen3-30B-A3B | ML | 25 | 12712 | 2.706 | 3.897 | 84.227 |
| Qwen3-30B-A3B | AIC | 25 | 21437 | 2.474 | 3.449 | 20.119 |
| Qwen3-30B-A3B-Instruct-2507 | ML | 25 | 12668 | 3.193 | 4.445 | 86.155 |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 25 | 21418 | 2.477 | 3.461 | 20.138 |
| Qwen3-32B | ML | 25 | 13932 | 3.470 | 4.665 | 67.636 |
| Qwen3-32B | AIC | 25 | 14449 | 6.958 | 8.120 | 58.151 |

## 单步与端到端预测的 KV 输入口径

本轮离线训练、ML 单步评分及 AIC 单步评分读取 prefix_indices_len；端到端调用预测器时使用 prefix_indices_len + output_ids_len。因此原单步表和端到端回放并非完全相同的输入口径。此处保留原始结果，使用同一冻结 ML bundle、同一真实步骤和时延，只将评分输入改为当前端到端表达式；没有重新训练、调整参数或重跑真机。

这个对照检验的是调用口径，不是独立测量 GPU seq_lens；也不是按新口径重新训练后的准确率。AIC 的上下文查询也受该输入影响，但本表只测 ML，不能把 ML 的变化量当成 AIC 的变化量。原始记录保留已生成 token 数，后续口径统一可复用这些采集数据。

| 模型 | 独立步骤数 | 实测 ms | prefix-only / 端到端长度口径预测 ms | MAPE（旧 / 新） | 改善（百分点） |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 19080 | 25.97 | 25.88 / 25.84 | 6.21% / 6.29% | -0.073 |
| Qwen2.5-3B-Instruct | 17628 | 31.96 | 31.25 / 30.91 | 4.55% / 3.87% | +0.686 |
| llama-2-7b | 18500 | 28.37 | 27.49 / 27.38 | 3.84% / 3.72% | +0.119 |
| Mistral-7B-Instruct-v0.2 | 18306 | 29.03 | 28.60 / 28.52 | 5.42% / 5.45% | -0.026 |
| Qwen2.5-7B-Instruct | 19018 | 26.51 | 25.93 / 25.69 | 6.04% / 5.44% | +0.595 |
| Qwen3-8B | 17291 | 33.99 | 32.60 / 32.55 | 4.55% / 4.47% | +0.081 |
| Qwen2.5-14B-Instruct | 15701 | 44.41 | 43.39 / 43.06 | 3.52% / 3.39% | +0.132 |
| Qwen2.5-14B-Instruct-1M | 9974 | 310.80 | 300.60 / 300.56 | 12.18% / 9.38% | +2.798 |
| Qwen3-14B | 16069 | 41.34 | 40.24 / 40.24 | 6.16% / 5.88% | +0.283 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 12674 | 111.92 | 110.90 / 109.77 | 4.73% / 3.73% | +1.000 |
| Qwen3-30B-A3B | 12223 | 95.98 | 84.32 / 84.21 | 4.62% / 4.55% | +0.062 |
| Qwen3-30B-A3B-Instruct-2507 | 12188 | 95.22 | 85.30 / 86.23 | 4.76% / 5.40% | -0.639 |
| Qwen3-32B | 13889 | 68.90 | 67.22 / 67.68 | 2.85% / 3.44% | -0.589 |

![冻结模型 KV 输入口径对照](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/kv_input_contract/ml_frozen_kv_input_mape.png)

kv_input_contract/input_contract_audit.csv 分列训练/测试、负载、阶段和输入变化量；frozen_input_comparison.csv 与 frozen_input_per_batch/ 保留全部分组和逐步预测。source_manifest.json 保存原始文件、源码、模型及评分文件校验值。原口径预测已逐条复现，原始文件和 bundle 在对照前后保持不变。

### 复用原训练数据的 KV 口径修正拟合

冻结模型的输入对照不等于修正训练后的结果。这里另做一次 CPU 拟合：复用每个模型原有训练时延、步骤顺序、HGB p50 参数、seed 和 early-stopping 规则，只将训练特征的 past 改为 prefix+output。独立测试请求不进入拟合；测试记录在拟合、保存 bundle 后才读取，也不用于选参数或选择是否采用修正。原始采集文件和预测器保持不变。

| 模型 | 训练 / 测试时延条数 | 实测 ms | 预测 ms（原口径 / 冻结模型新口径 / 新口径重拟合） | MAPE（原 / 冻结 / 重拟合） |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 11533 / 19080 | 25.97 | 25.88 / 25.84 / 25.63 | 6.21% / 6.29% / 5.31% |
| Qwen2.5-3B-Instruct | 10557 / 17628 | 31.96 | 31.25 / 30.91 / 30.71 | 4.55% / 3.87% / 3.49% |
| llama-2-7b | 11532 / 18500 | 28.37 | 27.49 / 27.38 / 27.99 | 3.84% / 3.72% / 5.21% |
| Mistral-7B-Instruct-v0.2 | 11319 / 18306 | 29.03 | 28.60 / 28.52 / 28.73 | 5.42% / 5.45% / 5.84% |
| Qwen2.5-7B-Instruct | 11598 / 19018 | 26.51 | 25.93 / 25.69 / 25.68 | 6.04% / 5.44% / 5.87% |
| Qwen3-8B | 10328 / 17291 | 33.99 | 32.60 / 32.55 / 32.55 | 4.55% / 4.47% / 4.47% |
| Qwen2.5-14B-Instruct | 9073 / 15701 | 44.41 | 43.39 / 43.06 / 43.24 | 3.52% / 3.39% / 3.54% |
| Qwen2.5-14B-Instruct-1M | 5862 / 9974 | 310.80 | 300.60 / 300.56 / 296.77 | 12.18% / 9.38% / 8.34% |
| Qwen3-14B | 9256 / 16069 | 41.34 | 40.24 / 40.24 / 40.11 | 6.16% / 5.88% / 6.27% |
| Meta-Llama-3.1-70B-Instruct-AWQ | 7069 / 12674 | 111.92 | 110.90 / 109.77 / 111.00 | 4.73% / 3.73% / 4.40% |
| Qwen3-30B-A3B | 6951 / 12223 | 95.98 | 84.32 / 84.21 / 84.40 | 4.62% / 4.55% / 4.59% |
| Qwen3-30B-A3B-Instruct-2507 | 6931 / 12188 | 95.22 | 85.30 / 86.23 / 85.06 | 4.76% / 5.40% / 4.81% |
| Qwen3-32B | 7930 / 13889 | 68.90 | 67.22 / 67.68 / 67.46 | 2.85% / 3.44% / 2.99% |

![ML KV 口径修正拟合对照](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/ml_runtime_refit/ml_runtime_kv_refit_mape.png)

ml_runtime_refit/ 保存完整分负载/阶段统计、逐步骤三个预测值、新 bundle、构造参数、实际训练轮数、CPU 拟合用时、拟合特征/目标及原始文件校验值。此表仅检验修正口径后重新拟合的单步误差，没有用新 bundle 重做端到端回放，不能把单步改善直接写成 E2E 改善；也不是独立验证 GPU seq_lens。

### AIC 使用相同实测步骤的输入对照

下面固定本轮 AIC Silicon 数据库、模型配置和实测步骤，先只把 past 长度从 prefix-only 改为 prefix+output，再单独计入阶段判断的差异。原口径预测先逐条复现；没有用测试时延调整算子数据、比例系数或选择模型。该结果与上面的 ML 输入对照分别计算，不互相代替。

离线 AIC 按真机 forward_mode 判断 prefill/decode；当前回放的 ScheduleBatch.is_decode() 则在所有 extend 长度都不大于 1 时判为 decode，单 token 的 prefill 因而可能走另一条查询路径。表中依次列出原评分、只改长度、长度与阶段都按回放输入的结果。分组仍按真机阶段，未改写真实标签。

| 模型 | 独立步骤数 / 阶段不一致数 | 实测 ms | 预测 ms（原 / 只改长度 / 长度+阶段） | MAPE（原 / 只改长度 / 长度+阶段） | 只改长度改善（百分点） |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 19080 / 4 | 25.97 | 2.83 / 2.90 / 2.90 | 88.75% / 88.47% / 88.47% | +0.284 |
| Qwen2.5-3B-Instruct | 17628 / 4 | 31.96 | 6.96 / 6.99 / 6.99 | 77.54% / 77.44% / 77.44% | +0.100 |
| llama-2-7b | 18500 / 4 | 28.37 | 14.23 / 14.58 / 14.58 | 49.61% / 48.31% / 48.31% | +1.292 |
| Mistral-7B-Instruct-v0.2 | 18306 / 5 | 29.03 | 13.53 / 13.61 / 13.61 | 53.49% / 53.18% / 53.18% | +0.307 |
| Qwen2.5-7B-Instruct | 19018 / 4 | 26.51 | 12.70 / 12.74 / 12.74 | 52.11% / 51.96% / 51.97% | +0.148 |
| Qwen3-8B | 17291 / 4 | 33.99 | 14.23 / 14.33 / 14.33 | 58.01% / 57.71% / 57.71% | +0.300 |
| Qwen2.5-14B-Instruct | 15701 / 4 | 44.41 | 26.03 / 26.17 / 26.17 | 42.30% / 41.95% / 41.95% | +0.350 |
| Qwen2.5-14B-Instruct-1M | 9974 / 0 | 310.80 | 29.58 / 29.80 / 29.80 | 82.54% / 82.36% / 82.36% | +0.183 |
| Qwen3-14B | 16069 / 4 | 41.34 | 25.76 / 25.88 / 25.88 | 38.66% / 38.34% / 38.34% | +0.321 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 12674 / 4 | 111.92 | 124.70 / 124.99 / 124.99 | 30.35% / 30.73% / 30.73% | -0.382 |
| Qwen3-30B-A3B | 12223 / 3 | 95.98 | 31.07 / 31.18 / 31.18 | 63.92% / 63.80% / 63.80% | +0.121 |
| Qwen3-30B-A3B-Instruct-2507 | 12188 / 4 | 95.22 | 31.13 / 31.24 / 31.24 | 63.51% / 63.39% / 63.39% | +0.122 |
| Qwen3-32B | 13889 / 4 | 68.90 | 58.51 / 58.72 / 58.72 | 15.56% / 15.19% / 15.19% | +0.369 |

![AIC KV 输入口径对照](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/aic_kv_input_contract/aic_frozen_kv_input_mape.png)

aic_kv_input_contract/aic_input_comparison.csv 保留全部模型、负载和阶段；per_batch/ 保留逐步骤真实值和三种输入解释的预测。source_manifest.json 记录原评分、实测记录、checkpoint config、数据库、kernel 二进制和调用源码校验值。这里只检验输入变化对静态预测的影响，没有重新做端到端回放，也没有独立测量 GPU seq_lens；剩余误差不能仅凭此表归因于 CPU 或某个算子。

## AIC 评分缓存与实际查询参数核对

评分缓存键与预测调用对平均长度的处理存在差异。这里执行当前源码中的实际方法，仅捕获查询参数，按原评分文件顺序检查后续步骤是否错误复用首个查询；没有修改评分器、重新采样或计算新的预测时延。不同请求被汇总为相同的实际查询，不算缓存冲突。

| 模型 | 训练 / 测试步骤数 | 训练 / 测试参数不同的复用步骤 |
| --- | --- | --- |
| Qwen3-0.6B | 11533 / 19080 | 0 / 0 |
| Qwen2.5-3B-Instruct | 10557 / 17628 | 0 / 0 |
| llama-2-7b | 11532 / 18500 | 0 / 0 |
| Mistral-7B-Instruct-v0.2 | 11319 / 18306 | 0 / 0 |
| Qwen2.5-7B-Instruct | 11598 / 19018 | 0 / 0 |
| Qwen3-8B | 10328 / 17291 | 0 / 0 |
| Qwen2.5-14B-Instruct | 9073 / 15701 | 0 / 0 |
| Qwen2.5-14B-Instruct-1M | 5862 / 9974 | 0 / 0 |
| Qwen3-14B | 9256 / 16069 | 0 / 0 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 7069 / 12674 | 0 / 0 |
| Qwen3-30B-A3B | 6951 / 12223 | 0 / 0 |
| Qwen3-30B-A3B-Instruct-2507 | 6931 / 12188 | 0 / 0 |
| Qwen3-32B | 7930 / 13889 | 0 / 0 |

本次已采集步骤未出现这种错误复用，因此不能用该缓存键差异解释本次预测误差。这不表示所有未测试输入都没有冲突。

aic_cache_contract/ 保留逐模型计数、冲突样例（若有）、原数据及调用源码校验值。浮点差异同时记录严格相等与 1e-12 容差口径；表格采用后者。


## 原生单步 CUDA 诊断

沿用 3B、7B、32B 正式真实基线的模型、kernel、attention 后端、显存比例、上下文、并行设置及客户端请求参数。每类负载跳过前两个 prefill/decode 步骤，各记录随后三个，共 90 个样本。此处每负载只有一次带 profiler 的运行，不替代前面五次正式实测；采样和导出可能改变动态批次与墙钟时间。

按 CPU launch correlation 归属 GPU 事件，GPU busy 时间使用时间区间并集；算子族使用持续时间之和，两个量不混用。AIC 在同一条诊断样本上分别查询原 prefix、runtime prefix+output、实际 prepared seq_lens−extend 三种输入，保持原生阶段标签。generation 查询内部使用 isl+1，因此第三种输入扣除本次新 token。三种结果都保留，不按测试误差选择口径。

| 模型 | 阶段 / 样本 | 带 profiling 墙钟 / GPU busy ms | AIC 原 prefix / runtime / 实际 seq 输入 ms | 实际 seq 输入对墙钟 / GPU busy MAPE | GPU 事件时长归属比例 |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B-Instruct | prefill / 15 | 45.546 / 14.646 | 15.449 / 15.449 / 15.449 | 64.91% / 5.64% | 100.00% |
| Qwen2.5-3B-Instruct | decode / 15 | 46.592 / 6.250 | 6.089 / 6.091 / 6.090 | 86.62% / 3.05% | 100.00% |
| Qwen2.5-7B-Instruct | prefill / 15 | 38.264 / 27.473 | 28.851 / 28.851 / 28.851 | 26.89% / 4.95% | 100.00% |
| Qwen2.5-7B-Instruct | decode / 15 | 34.936 / 10.683 | 11.240 / 11.248 / 11.246 | 67.36% / 5.29% | 100.00% |
| Qwen3-32B | prefill / 15 | 139.525 / 124.995 | 127.759 / 127.759 / 127.759 | 12.54% / 3.77% | 100.00% |
| Qwen3-32B | decode / 15 | 82.480 / 44.963 | 48.695 / 48.695 / 48.692 | 40.68% / 8.34% | 100.00% |

墙钟−GPU busy 包含 CPU 调度、launch 间隙、同步、未归属事件及 profiler 开销，不能直接称为纯 CPU 时间。未归属事件及实际 KV 长度不一致的请求数保留在 per_sample.csv 和逐样本 JSON 中。若归属不完整，相应 busy 时间只代表已归属部分，不作完整 GPU 执行时间结论。

### 同类算子时长对照

| 模型 | 阶段 | Native GEMM/GEMV / AIC GEMM ms | Native Triton attention / AIC attention ms | Native 其他 kernel ms | AIC 其他项 ms |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B-Instruct | prefill | 11.170 / 12.195 | 1.964 / 2.132 | 1.483 | 1.122 |
| Qwen2.5-3B-Instruct | decode | 4.796 / 5.202 | 0.648 / 0.529 | 0.804 | 0.359 |
| Qwen2.5-7B-Instruct | prefill | 23.670 / 25.387 | 2.163 / 2.139 | 1.585 | 1.325 |
| Qwen2.5-7B-Instruct | decode | 9.374 / 10.294 | 0.648 / 0.655 | 0.644 | 0.296 |
| Qwen3-32B | prefill | 110.497 / 114.282 | 8.794 / 9.495 | 5.596 | 3.982 |
| Qwen3-32B | decode | 40.370 / 45.281 | 2.661 / 2.662 | 1.880 | 0.749 |

Native 按 CUDA kernel 名称分族；AIC GEMM 为 *_gemm，attention 为 context_attention/generation_attention。融合、准备及采样边界不完全相同，这张表是同类时长定位，不是逐算子一一对应的准确率。完整 kernel 名、correlation、流、区间与 AIC 每项预测都保留，不能把 other 项差值直接归因为某个算子。

![原生 GPU 单步与 AIC 诊断](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/matched_native_profiles/analysis/native_gpu_vs_aic_diagnostic.png)

matched_native_profiles/ 保存原命令校验值、诊断命令/环境、版本源码归档、GPU UUID/容量检查、原生 collector 数据、trace、退出码及完成标记。所有正式单步/E2E 表仍使用未启用本 profiler 的原始实验结果。


## 同负载与跨负载测试

前三类负载出现在训练数据中，但测试请求和 seed 独立；后两类负载不用于训练。跨负载不代表所有特征都在训练范围之外。以下按模型与负载列出实测值和预测值，避免把五类负载混成一个精度结论。

总 token 吞吐采用 collector/trace 的实际输入+输出 token 数，除以原始 benchmark duration；原始输入计数和原报吞吐保留在 comparison.csv。随机 ID 解码成文本后再编码可能改变输入长度，此处统一 token 分子，不消除 client/模拟器时间窗差异。std 为五次重复的总体标准差（分母 n），不是置信区间。

| 用例名称 | 生成输入/输出长度上限 | 到达设置 | 每组请求 | 用于训练 |
| --- | --- | --- | --- | --- |
| latency_256x128_r1 | 256 / 128 token | 1 请求/秒 | 32 | 是 |
| service_1024x256_r4 | 1024 / 256 token | 4 请求/秒 | 64 | 是 |
| saturation_1024x256_inf | 1024 / 256 token | 一次提交，不限速 | 64 | 是 |
| latency_128x64_r1 | 128 / 64 token | 1 请求/秒 | 32 | 否 |
| saturation_1536x384_inf | 1536 / 384 token | 一次提交，不限速 | 64 | 否 |

长度是随机生成上限，不是每个请求的固定长度；range_ratio=0 从 1 到上限采样。1536/384 长用例使用 --tokenize-prompt 直接发送随机 ID，避免文本再编码使输入+输出超出 2048 上下文；其他用例沿用随机 ID 解码成文本再发送，实际输入 ID 保存在 trace。长用例相对训练还改变了输入路径，不能将其差异只归因于长度。r1/r4 为平均到达率，有限速请求间隔为指数分布；inf 是突发提交，不是无限并发，服务最多同时运行 32 个请求。

### latency_256x128_r1（同负载独立测试）

| 模型 | 方法 | 单步实测/预测 ms | 单步 MAPE / MAE ms | E2E 实测 ± std / 预测 ms | E2E MAPE | TTFT 实测 ± std / 预测 ms（MAPE） | TPOT 实测 ± std / 预测 ms（MAPE） | 总 token 吞吐实测/预测 tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 24.91 / 25.94 | 7.23% / 2.22 | 1694.90 ± 121.11 / 1958.77 | 15.39% | 81.31 ± 41.17 / 44.17（35.20%） | 26.75 ± 0.59 / 30.92（15.62%） | 200.71 / 198.78 |
| Qwen3-0.6B | AIC | 24.91 / 2.14 | 91.28% / 22.78 | 1694.90 ± 121.11 / 371.70 | 78.19% | 81.31 ± 41.17 / 11.24（83.56%） | 26.75 ± 0.59 / 6.05（77.35%） | 200.71 / 213.94 |
| Qwen2.5-3B-Instruct | ML | 31.66 / 30.84 | 3.27% / 1.55 | 2152.77 ± 155.42 / 2292.63 | 6.46% | 75.84 ± 9.02 / 52.30（30.03%） | 33.75 ± 1.12 / 36.22（7.39%） | 196.67 / 195.64 |
| Qwen2.5-3B-Instruct | AIC | 31.66 / 5.83 | 81.25% / 25.83 | 2152.77 ± 155.42 / 555.04 | 74.27% | 75.84 ± 9.02 / 16.91（77.33%） | 33.75 ± 1.12 / 8.84（73.76%） | 196.67 / 211.84 |
| llama-2-7b | ML | 27.08 / 26.52 | 4.20% / 1.59 | 1826.29 ± 143.86 / 2052.58 | 12.51% | 58.97 ± 5.09 / 47.78（18.45%） | 28.77 ± 1.59 / 32.31（12.62%） | 204.51 / 203.42 |
| llama-2-7b | AIC | 27.08 / 10.27 | 61.40% / 16.81 | 1826.29 ± 143.86 / 928.62 | 49.10% | 58.97 ± 5.09 / 26.24（55.04%） | 28.77 ± 1.59 / 14.63（48.95%） | 204.51 / 216.26 |
| Mistral-7B-Instruct-v0.2 | ML | 27.55 / 27.58 | 4.78% / 1.75 | 1857.05 ± 148.68 / 2160.04 | 16.41% | 59.70 ± 6.60 / 49.64（16.30%） | 29.21 ± 1.72 / 33.89（16.44%） | 202.50 / 200.53 |
| Mistral-7B-Instruct-v0.2 | AIC | 27.55 / 10.89 | 59.77% / 16.66 | 1857.05 ± 148.68 / 975.01 | 47.42% | 59.70 ± 6.60 / 29.21（50.37%） | 29.21 ± 1.72 / 15.23（47.68%） | 202.50 / 214.03 |
| Qwen2.5-7B-Instruct | ML | 24.66 / 25.49 | 7.64% / 2.24 | 1676.05 ± 156.27 / 1966.41 | 17.41% | 61.87 ± 6.47 / 43.35（29.02%） | 26.28 ± 1.87 / 31.06（18.67%） | 200.57 / 198.41 |
| Qwen2.5-7B-Instruct | AIC | 24.66 / 10.64 | 56.06% / 14.02 | 1676.05 ± 156.27 / 853.56 | 49.02% | 61.87 ± 6.47 / 25.58（58.21%） | 26.28 ± 1.87 / 13.49（48.46%） | 200.57 / 208.87 |
| Qwen3-8B | ML | 32.73 / 31.90 | 3.56% / 1.69 | 2217.29 ± 203.59 / 2405.40 | 8.75% | 80.44 ± 11.38 / 54.00（31.50%） | 34.55 ± 2.53 / 37.92（11.37%） | 195.90 / 194.25 |
| Qwen3-8B | AIC | 32.73 / 11.61 | 63.94% / 21.12 | 2217.29 ± 203.59 / 1033.25 | 53.38% | 80.44 ± 11.38 / 29.09（63.08%） | 34.55 ± 2.53 / 16.35（52.43%） | 195.90 / 208.09 |
| Qwen2.5-14B-Instruct | ML | 41.29 / 40.51 | 4.05% / 2.28 | 2799.79 ± 267.44 / 2961.89 | 7.45% | 96.75 ± 12.59 / 66.62（30.06%） | 43.68 ± 3.34 / 46.62（9.55%） | 190.63 / 189.05 |
| Qwen2.5-14B-Instruct | AIC | 41.29 / 20.77 | 48.92% / 20.52 | 2799.79 ± 267.44 / 1754.44 | 37.31% | 96.75 ± 12.59 / 48.96（48.43%） | 43.68 ± 3.34 / 27.59（36.54%） | 190.63 / 202.32 |
| Qwen2.5-14B-Instruct-1M | ML | 288.52 / 295.81 | 24.33% / 38.22 | 25057.76 ± 4388.06 / 23443.00 | 6.76% | 3474.98 ± 1598.77 / 2728.39（14.33%） | 565.16 ± 176.33 / 518.59（7.15%） | 132.22 / 134.97 |
| Qwen2.5-14B-Instruct-1M | AIC | 288.52 / 24.07 | 82.99% / 264.45 | 25057.76 ± 4388.06 / 1766.19 | 92.77% | 3474.98 ± 1598.77 / 50.69（98.30%） | 565.16 ± 176.33 / 27.76（94.46%） | 132.22 / 202.20 |
| Qwen3-14B | ML | 38.02 / 37.28 | 6.84% / 3.14 | 2567.28 ± 151.78 / 2712.14 | 8.35% | 90.22 ± 17.74 / 62.92（27.48%） | 40.12 ± 2.06 / 42.78（7.08%） | 193.06 / 191.61 |
| Qwen3-14B | AIC | 38.02 / 20.32 | 45.72% / 17.74 | 2567.28 ± 151.78 / 1724.95 | 32.83% | 90.22 ± 17.74 / 52.40（39.51%） | 40.12 ± 2.06 / 27.11（32.18%） | 193.06 / 202.50 |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 83.31 / 84.65 | 6.18% / 5.78 | 5883.50 ± 870.62 / 6181.98 | 6.78% | 256.64 ± 12.92 / 218.30（15.16%） | 90.70 ± 7.02 / 95.92（7.73%） | 170.15 / 168.87 |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 83.31 / 98.48 | 22.50% / 18.53 | 5883.50 ± 870.62 / 7667.56 | 31.39% | 256.64 ± 12.92 / 228.92（10.81%） | 90.70 ± 7.02 / 120.35（33.37%） | 170.15 / 158.82 |
| Qwen3-30B-A3B | ML | 99.24 / 84.99 | 5.50% / 15.23 | 6532.61 ± 689.74 / 6085.21 | 6.55% | 273.08 ± 55.79 / 127.20（51.41%） | 102.33 ± 4.72 / 96.18（6.23%） | 157.49 / 165.83 |
| Qwen3-30B-A3B | AIC | 99.24 / 18.98 | 78.63% / 80.26 | 6532.61 ± 689.74 / 759.27 | 88.40% | 273.08 ± 55.79 / 56.01（78.58%） | 102.33 ± 4.72 / 11.83（88.43%） | 157.49 / 210.91 |
| Qwen3-30B-A3B-Instruct-2507 | ML | 97.23 / 85.36 | 3.83% / 13.62 | 6446.22 ± 479.95 / 6176.25 | 4.31% | 319.56 ± 47.52 / 126.65（59.08%） | 101.33 ± 4.39 / 97.74（4.77%） | 162.06 / 165.30 |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 97.23 / 19.02 | 78.06% / 78.21 | 6446.22 ± 479.95 / 767.20 | 88.14% | 319.56 ± 47.52 / 56.13（82.08%） | 101.33 ± 4.39 / 11.94（88.17%） | 162.06 / 210.87 |
| Qwen3-32B | ML | 59.89 / 57.37 | 3.20% / 2.74 | 4101.41 ± 322.16 / 4158.71 | 4.56% | 213.44 ± 97.37 / 117.45（36.78%） | 63.28 ± 3.05 / 65.09（5.45%） | 180.60 / 180.30 |
| Qwen3-32B | AIC | 59.89 / 47.17 | 20.38% / 12.74 | 4101.41 ± 322.16 / 3641.01 | 11.22% | 213.44 ± 97.37 / 106.18（42.55%） | 63.28 ± 3.05 / 56.97（9.76%） | 180.60 / 184.58 |

按阶段展开单步结果。n 为五次独立真机实验合并后的 batch 条数，P95 为逐 batch 绝对百分比误差的 95 分位。

| 模型 | 方法 | n（prefill/decode） | prefill 实测/预测 ms | prefill MAPE / MAE ms | decode 实测/预测 ms | decode MAPE / MAE ms | P95 APE（prefill/decode） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 157 / 5294 | 25.85 / 25.18 | 4.61% / 1.52 | 24.88 / 25.96 | 7.31% / 2.24 | 17.00% / 16.20% |
| Qwen3-0.6B | AIC | 157 / 5294 | 25.85 / 4.05 | 84.06% / 21.80 | 24.88 / 2.08 | 91.50% / 22.81 | 88.09% / 92.75% |
| Qwen2.5-3B-Instruct | ML | 157 / 4568 | 31.57 / 30.44 | 3.76% / 1.28 | 31.66 / 30.85 | 3.25% / 1.56 | 15.54% / 12.08% |
| Qwen2.5-3B-Instruct | AIC | 157 / 4568 | 31.57 / 9.48 | 69.88% / 22.09 | 31.66 / 5.70 | 81.64% / 25.96 | 79.28% / 83.60% |
| llama-2-7b | ML | 160 / 5061 | 28.03 / 27.38 | 3.51% / 1.04 | 27.05 / 26.49 | 4.22% / 1.61 | 11.91% / 12.07% |
| llama-2-7b | AIC | 160 / 5061 | 28.03 / 15.92 | 43.15% / 12.10 | 27.05 / 10.09 | 61.97% / 16.96 | 61.04% / 67.08% |
| Mistral-7B-Instruct-v0.2 | ML | 160 / 5000 | 27.95 / 27.91 | 4.31% / 1.24 | 27.54 / 27.56 | 4.79% / 1.77 | 11.18% / 13.09% |
| Mistral-7B-Instruct-v0.2 | AIC | 160 / 5000 | 27.95 / 18.64 | 33.21% / 9.31 | 27.54 / 10.64 | 60.62% / 16.90 | 56.17% / 65.97% |
| Qwen2.5-7B-Instruct | ML | 158 / 5331 | 25.36 / 25.49 | 5.16% / 1.33 | 24.63 / 25.49 | 7.71% / 2.26 | 11.49% / 14.46% |
| Qwen2.5-7B-Instruct | AIC | 158 / 5331 | 25.36 / 17.17 | 32.37% / 8.20 | 24.63 / 10.44 | 56.76% / 14.19 | 52.23% / 61.30% |
| Qwen3-8B | ML | 158 / 4470 | 33.58 / 31.89 | 4.91% / 1.89 | 32.70 / 31.90 | 3.51% / 1.68 | 15.95% / 11.77% |
| Qwen3-8B | AIC | 158 / 4470 | 33.58 / 19.24 | 42.17% / 14.34 | 32.70 / 11.34 | 64.71% / 21.36 | 59.95% / 67.75% |
| Qwen2.5-14B-Instruct | ML | 155 / 3782 | 42.49 / 40.98 | 5.14% / 2.40 | 41.24 / 40.49 | 4.01% / 2.28 | 20.81% / 11.98% |
| Qwen2.5-14B-Instruct | AIC | 155 / 3782 | 42.49 / 32.21 | 24.09% / 10.30 | 41.24 / 20.31 | 49.94% / 20.94 | 48.59% / 55.34% |
| Qwen2.5-14B-Instruct-1M | ML | 111 / 739 | 1445.45 / 1368.44 | 4.54% / 82.97 | 114.75 / 134.70 | 27.30% / 31.50 | 21.68% / 334.55% |
| Qwen2.5-14B-Instruct-1M | AIC | 111 / 739 | 1445.45 / 39.45 | 97.12% / 1406.00 | 114.75 / 21.76 | 80.86% / 92.98 | 98.49% / 84.75% |
| Qwen3-14B | ML | 156 / 4013 | 39.53 / 38.65 | 5.28% / 2.13 | 37.96 / 37.22 | 6.90% / 3.18 | 12.67% / 14.53% |
| Qwen3-14B | AIC | 156 / 4013 | 39.53 / 34.86 | 14.24% / 5.54 | 37.96 / 19.75 | 46.94% / 18.21 | 39.01% / 52.74% |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 155 / 2129 | 151.59 / 152.35 | 2.26% / 3.15 | 78.33 / 79.72 | 6.46% / 5.97 | 7.02% / 16.37% |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 155 / 2129 | 151.59 / 157.15 | 20.33% / 23.95 | 78.33 / 94.21 | 22.65% / 18.13 | 57.78% / 29.36% |
| Qwen3-30B-A3B | ML | 151 / 1918 | 146.24 / 78.33 | 14.61% / 68.92 | 95.54 / 85.51 | 4.78% / 11.01 | 87.50% / 9.97% |
| Qwen3-30B-A3B | AIC | 151 / 1918 | 146.24 / 48.13 | 46.20% / 98.11 | 95.54 / 16.69 | 81.19% / 78.85 | 93.08% / 92.30% |
| Qwen3-30B-A3B-Instruct-2507 | ML | 150 / 1905 | 157.25 / 77.90 | 13.22% / 79.76 | 92.51 / 85.95 | 3.09% / 8.42 | 87.78% / 10.06% |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 150 / 1905 | 157.25 / 47.86 | 45.44% / 109.39 | 92.51 / 16.75 | 80.63% / 75.76 | 93.22% / 92.35% |
| Qwen3-32B | ML | 152 / 2805 | 77.66 / 77.23 | 2.31% / 1.71 | 58.92 / 56.29 | 3.25% / 2.80 | 11.54% / 12.34% |
| Qwen3-32B | AIC | 152 / 2805 | 77.66 / 73.89 | 5.76% / 4.20 | 58.92 / 45.72 | 21.18% / 13.20 | 16.59% / 30.15% |

### service_1024x256_r4（同负载独立测试）

| 模型 | 方法 | 单步实测/预测 ms | 单步 MAPE / MAE ms | E2E 实测 ± std / 预测 ms | E2E MAPE | TTFT 实测 ± std / 预测 ms（MAPE） | TPOT 实测 ± std / 预测 ms（MAPE） | 总 token 吞吐实测/预测 tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 26.03 / 25.97 | 6.20% / 2.41 | 4200.17 ± 310.26 / 4694.00 | 11.80% | 94.64 ± 10.00 / 64.25（31.53%） | 31.83 ± 1.18 / 35.59（11.93%） | 1929.29 / 1876.61 |
| Qwen3-0.6B | AIC | 26.03 / 3.08 | 87.71% / 22.95 | 4200.17 ± 310.26 / 1764.49 | 57.96% | 94.64 ± 10.00 / 22.52（75.95%） | 31.83 ± 1.18 / 13.18（58.51%） | 1929.29 / 2216.73 |
| Qwen2.5-3B-Instruct | ML | 31.69 / 31.63 | 5.10% / 2.53 | 5137.12 ± 377.07 / 5623.72 | 9.44% | 116.03 ± 17.50 / 81.63（28.64%） | 39.00 ± 1.56 / 42.67（9.57%） | 1828.15 / 1781.32 |
| Qwen2.5-3B-Instruct | AIC | 31.69 / 7.44 | 75.71% / 24.25 | 5137.12 ± 377.07 / 1839.48 | 64.18% | 116.03 ± 17.50 / 38.67（66.07%） | 39.00 ± 1.56 / 13.90（64.30%） | 1828.15 / 2230.75 |
| llama-2-7b | ML | 28.91 / 27.92 | 4.02% / 2.03 | 4697.12 ± 292.04 / 5240.29 | 11.69% | 108.15 ± 12.87 / 89.29（16.80%） | 35.88 ± 2.75 / 40.16（12.25%） | 1951.42 / 1905.28 |
| llama-2-7b | AIC | 28.91 / 15.51 | 45.88% / 13.40 | 4697.12 ± 292.04 / 3578.85 | 23.79% | 108.15 ± 12.87 / 68.91（35.83%） | 35.88 ± 2.75 / 26.78（25.15%） | 1951.42 / 2103.63 |
| Mistral-7B-Instruct-v0.2 | ML | 29.12 / 29.12 | 6.10% / 2.46 | 4755.72 ± 510.83 / 5442.59 | 14.84% | 115.69 ± 17.01 / 91.42（19.84%） | 35.74 ± 2.99 / 41.72（16.96%） | 1938.07 / 1868.98 |
| Mistral-7B-Instruct-v0.2 | AIC | 29.12 / 14.49 | 50.66% / 14.64 | 4755.72 ± 510.83 / 3490.08 | 26.30% | 115.69 ± 17.01 / 74.19（35.00%） | 35.74 ± 2.99 / 26.25（26.21%） | 1938.07 / 2086.51 |
| Qwen2.5-7B-Instruct | ML | 26.85 / 25.51 | 3.72% / 1.86 | 4355.34 ± 474.35 / 4938.26 | 13.59% | 115.61 ± 12.77 / 85.01（25.84%） | 33.21 ± 2.40 / 37.52（13.32%） | 1929.70 / 1871.05 |
| Qwen2.5-7B-Instruct | AIC | 26.85 / 13.41 | 50.24% / 13.47 | 4355.34 ± 474.35 / 2608.50 | 39.82% | 115.61 ± 12.77 / 66.41（42.05%） | 33.21 ± 2.40 / 19.66（40.57%） | 1929.70 / 2157.76 |
| Qwen3-8B | ML | 34.05 / 32.86 | 4.10% / 2.36 | 5615.14 ± 652.06 / 6188.40 | 11.66% | 134.82 ± 36.86 / 104.20（20.41%） | 42.54 ± 3.10 / 47.06（13.22%） | 1788.82 / 1737.30 |
| Qwen3-8B | AIC | 34.05 / 15.28 | 55.03% / 18.82 | 5615.14 ± 652.06 / 3695.60 | 33.83% | 134.82 ± 36.86 / 75.58（40.45%） | 42.54 ± 3.10 / 28.02（33.85%） | 1788.82 / 1997.66 |
| Qwen2.5-14B-Instruct | ML | 46.35 / 45.03 | 4.56% / 3.17 | 8065.93 ± 756.73 / 8454.11 | 7.19% | 375.69 ± 173.42 / 409.57（26.89%） | 61.78 ± 2.34 / 63.73（5.17%） | 1608.53 / 1565.19 |
| Qwen2.5-14B-Instruct | AIC | 46.35 / 29.10 | 38.35% / 17.28 | 8065.93 ± 756.73 / 6344.44 | 21.07% | 375.69 ± 173.42 / 165.40（48.56%） | 61.78 ± 2.34 / 48.87（20.80%） | 1608.53 / 1784.37 |
| Qwen2.5-14B-Instruct-1M | ML | 298.99 / 303.52 | 5.25% / 7.89 | 102509.47 ± 2546.31 / 104188.15 | 1.71% | 50860.55 ± 2957.54 / 51142.91（1.68%） | 617.22 ± 197.86 / 625.69（1.82%） | 290.03 / 284.14 |
| Qwen2.5-14B-Instruct-1M | AIC | 298.99 / 30.25 | 81.74% / 268.74 | 102509.47 ± 2546.31 / 6407.47 | 93.74% | 50860.55 ± 2957.54 / 172.14（99.66%） | 617.22 ± 197.86 / 49.56（91.14%） | 290.03 / 1781.33 |
| Qwen3-14B | ML | 43.40 / 41.53 | 5.75% / 3.70 | 7499.61 ± 779.59 / 7881.60 | 10.26% | 300.44 ± 109.90 / 330.85（23.41%） | 56.88 ± 5.20 / 60.17（10.93%） | 1683.09 / 1632.15 |
| Qwen3-14B | AIC | 43.40 / 29.00 | 34.42% / 14.51 | 7499.61 ± 779.59 / 6361.08 | 14.57% | 300.44 ± 109.90 / 171.51（38.22%） | 56.88 ± 5.20 / 49.37（12.88%） | 1683.09 / 1793.39 |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 127.41 / 126.68 | 5.12% / 5.75 | 32559.47 ± 1859.68 / 32456.92 | 1.30% | 12649.47 ± 1760.51 / 11022.17（12.86%） | 181.51 ± 33.09 / 222.36（22.53%） | 678.93 / 683.32 |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 127.41 / 137.71 | 32.39% / 34.56 | 32559.47 ± 1859.68 / 34777.16 | 6.90% | 12649.47 ± 1760.51 / 10362.82（18.18%） | 181.51 ± 33.09 / 234.98（30.36%） | 678.93 / 600.16 |
| Qwen3-30B-A3B | ML | 94.91 / 84.19 | 4.49% / 12.24 | 18504.79 ± 2132.05 / 17044.92 | 7.59% | 3562.78 ± 1229.67 / 2645.63（23.47%） | 125.85 ± 12.86 / 113.36（9.95%） | 898.94 / 954.03 |
| Qwen3-30B-A3B | AIC | 94.91 / 37.32 | 56.48% / 57.59 | 18504.79 ± 2132.05 / 7548.18 | 58.94% | 3562.78 ± 1229.67 / 265.33（93.13%） | 125.85 ± 12.86 / 57.57（53.73%） | 898.94 / 1759.67 |
| Qwen3-30B-A3B-Instruct-2507 | ML | 95.83 / 83.79 | 4.99% / 12.75 | 18521.25 ± 1189.78 / 17426.59 | 6.22% | 3479.92 ± 832.08 / 2715.33（21.98%） | 126.05 ± 7.42 / 116.30（7.46%） | 889.97 / 938.21 |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 95.83 / 37.30 | 56.68% / 58.53 | 18521.25 ± 1189.78 / 7548.50 | 59.23% | 3479.92 ± 832.08 / 267.17（92.81%） | 126.05 ± 7.42 / 57.63（54.09%） | 889.97 / 1757.72 |
| Qwen3-32B | ML | 74.02 / 73.19 | 2.26% / 2.68 | 15089.80 ± 875.43 / 15696.68 | 4.01% | 2694.80 ± 695.13 / 2728.55（4.27%） | 101.47 ± 5.17 / 105.40（3.87%） | 1130.95 / 1086.90 |
| Qwen3-32B | AIC | 74.02 / 65.12 | 12.54% / 8.91 | 15089.80 ± 875.43 / 14526.52 | 3.70% | 2694.80 ± 695.13 / 2339.89（13.42%） | 101.47 ± 5.17 / 98.92（2.50%） | 1130.95 / 1158.24 |

按阶段展开单步结果。n 为五次独立真机实验合并后的 batch 条数，P95 为逐 batch 绝对百分比误差的 95 分位。

| 模型 | 方法 | n（prefill/decode） | prefill 实测/预测 ms | prefill MAPE / MAE ms | decode 实测/预测 ms | decode MAPE / MAE ms | P95 APE（prefill/decode） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 479 / 3588 | 24.98 / 24.62 | 3.85% / 1.08 | 26.17 / 26.15 | 6.51% / 2.59 | 12.41% / 14.77% |
| Qwen3-0.6B | AIC | 479 / 3588 | 24.98 / 5.87 | 76.24% / 19.10 | 26.17 / 2.71 | 89.24% / 23.46 | 87.42% / 90.85% |
| Qwen2.5-3B-Instruct | ML | 473 / 3083 | 30.48 / 30.15 | 2.62% / 0.85 | 31.88 / 31.86 | 5.48% / 2.79 | 10.99% / 15.28% |
| Qwen2.5-3B-Instruct | AIC | 473 / 3083 | 30.48 / 15.97 | 47.43% / 14.51 | 31.88 / 6.13 | 80.05% / 25.75 | 78.33% / 82.14% |
| llama-2-7b | ML | 472 / 3181 | 35.08 / 34.94 | 2.49% / 0.82 | 28.00 / 26.88 | 4.25% / 2.20 | 9.52% / 10.35% |
| llama-2-7b | AIC | 472 / 3181 | 35.08 / 28.97 | 19.61% / 6.11 | 28.00 / 13.51 | 49.78% / 14.48 | 59.16% / 62.34% |
| Mistral-7B-Instruct-v0.2 | ML | 467 / 3172 | 36.02 / 35.94 | 2.46% / 0.80 | 28.11 / 28.12 | 6.63% / 2.70 | 9.85% / 12.74% |
| Mistral-7B-Instruct-v0.2 | AIC | 467 / 3172 | 36.02 / 31.83 | 13.77% / 4.22 | 28.11 / 11.93 | 56.09% / 16.17 | 51.56% / 61.74% |
| Qwen2.5-7B-Instruct | ML | 470 / 3491 | 33.86 / 33.59 | 2.92% / 0.91 | 25.91 / 24.43 | 3.82% / 1.99 | 11.06% / 12.41% |
| Qwen2.5-7B-Instruct | AIC | 470 / 3491 | 33.86 / 30.28 | 13.35% / 3.77 | 25.91 / 11.13 | 55.21% / 14.77 | 50.67% / 59.90% |
| Qwen3-8B | ML | 471 / 2922 | 37.82 / 37.27 | 2.54% / 0.97 | 33.44 / 32.15 | 4.36% / 2.58 | 11.30% / 11.37% |
| Qwen3-8B | AIC | 471 / 2922 | 37.82 / 32.61 | 16.34% / 5.63 | 33.44 / 12.49 | 61.27% / 20.95 | 57.48% / 66.15% |
| Qwen2.5-14B-Instruct | ML | 451 / 2338 | 64.54 / 64.24 | 2.34% / 1.33 | 42.84 / 41.32 | 4.98% / 3.52 | 11.75% / 12.55% |
| Qwen2.5-14B-Instruct | AIC | 451 / 2338 | 64.54 / 59.52 | 10.12% / 5.22 | 42.84 / 23.23 | 43.79% / 19.61 | 41.11% / 52.74% |
| Qwen2.5-14B-Instruct-1M | ML | 459 / 2004 | 1118.67 / 1115.04 | 1.10% / 12.32 | 111.25 / 117.64 | 6.20% / 6.88 | 2.73% / 18.16% |
| Qwen2.5-14B-Instruct-1M | AIC | 459 / 2004 | 1118.67 / 59.50 | 94.36% / 1059.17 | 111.25 / 23.55 | 78.85% / 87.70 | 97.61% / 81.89% |
| Qwen3-14B | ML | 451 / 2393 | 67.82 / 65.57 | 3.49% / 3.18 | 38.80 / 36.99 | 6.18% / 3.79 | 12.10% / 15.52% |
| Qwen3-14B | AIC | 451 / 2393 | 67.82 / 63.33 | 7.59% / 5.20 | 38.80 / 22.53 | 39.48% / 16.26 | 36.24% / 51.67% |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 414 / 2016 | 365.37 / 364.84 | 1.33% / 3.90 | 78.55 / 77.78 | 5.89% / 6.13 | 5.78% / 15.32% |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 414 / 2016 | 365.37 / 305.02 | 17.91% / 65.34 | 78.55 / 103.35 | 35.37% / 28.24 | 24.99% / 47.49% |
| Qwen3-30B-A3B | ML | 440 / 2037 | 95.30 / 77.68 | 6.84% / 18.72 | 94.82 / 85.59 | 3.98% / 10.84 | 11.29% / 9.12% |
| Qwen3-30B-A3B | AIC | 440 / 2037 | 95.30 / 59.94 | 26.64% / 35.36 | 94.82 / 32.43 | 62.92% / 62.39 | 52.71% / 91.89% |
| Qwen3-30B-A3B-Instruct-2507 | ML | 444 / 2033 | 92.04 / 77.09 | 5.32% / 15.18 | 96.66 / 85.26 | 4.92% / 12.22 | 11.17% / 10.06% |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 444 / 2033 | 92.04 / 59.74 | 26.04% / 32.31 | 96.66 / 32.41 | 63.37% / 64.26 | 52.69% / 91.92% |
| Qwen3-32B | ML | 434 / 2070 | 147.20 / 146.70 | 1.06% / 1.45 | 58.67 / 57.77 | 2.51% / 2.94 | 2.92% / 7.22% |
| Qwen3-32B | AIC | 434 / 2070 | 147.20 / 142.22 | 3.94% / 5.05 | 58.67 / 48.96 | 14.34% / 9.72 | 10.79% / 20.78% |

### saturation_1024x256_inf（同负载独立测试）

| 模型 | 方法 | 单步实测/预测 ms | 单步 MAPE / MAE ms | E2E 实测 ± std / 预测 ms | E2E MAPE | TTFT 实测 ± std / 预测 ms（MAPE） | TPOT 实测 ± std / 预测 ms（MAPE） | 总 token 吞吐实测/预测 tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 27.17 / 25.78 | 5.25% / 2.82 | 7460.88 ± 403.39 / 7714.14 | 6.43% | 2691.61 ± 193.61 / 2673.23（3.20%） | 41.69 ± 4.02 / 42.34（7.01%） | 3036.02 / 2938.66 |
| Qwen3-0.6B | AIC | 27.17 / 3.56 | 86.06% / 23.61 | 7460.88 ± 403.39 / 2777.94 | 62.67% | 2691.61 ± 193.61 / 885.51（67.02%） | 41.69 ± 4.02 / 15.80（61.85%） | 3036.02 / 7187.66 |
| Qwen2.5-3B-Instruct | ML | 33.26 / 31.82 | 6.51% / 3.51 | 8969.10 ± 844.32 / 9109.91 | 4.10% | 3223.91 ± 353.68 / 3207.87（4.93%） | 48.75 ± 3.84 / 49.88（4.79%） | 2518.26 / 2495.48 |
| Qwen2.5-3B-Instruct | AIC | 33.26 / 8.11 | 74.25% / 25.15 | 8969.10 ± 844.32 / 3553.26 | 60.17% | 3223.91 ± 353.68 / 1356.34（57.59%） | 48.75 ± 3.84 / 19.94（59.22%） | 2518.26 / 6443.34 |
| llama-2-7b | ML | 30.39 / 28.90 | 4.89% / 2.77 | 8430.28 ± 334.59 / 9180.03 | 8.94% | 3291.98 ± 127.90 / 3438.14（5.46%） | 45.41 ± 4.68 / 50.17（10.70%） | 2746.41 / 2549.09 |
| llama-2-7b | AIC | 30.39 / 18.05 | 39.32% / 12.34 | 8430.28 ± 334.59 / 7052.86 | 16.33% | 3291.98 ± 127.90 / 2675.98（18.73%） | 45.41 ± 4.68 / 38.76（14.67%） | 2746.41 / 3394.20 |
| Mistral-7B-Instruct-v0.2 | ML | 30.62 / 29.87 | 5.68% / 2.94 | 8444.04 ± 516.94 / 9160.53 | 8.63% | 3321.56 ± 179.18 / 3451.49（4.19%） | 45.08 ± 4.51 / 50.09（11.07%） | 2722.40 / 2533.90 |
| Mistral-7B-Instruct-v0.2 | AIC | 30.62 / 16.46 | 46.18% / 14.17 | 8444.04 ± 516.94 / 6459.42 | 23.40% | 3321.56 ± 179.18 / 2523.20（23.97%） | 45.08 ± 4.51 / 35.65（21.21%） | 2722.40 / 3584.55 |
| Qwen2.5-7B-Instruct | ML | 28.70 / 26.96 | 7.00% / 3.21 | 8121.52 ± 380.37 / 8503.72 | 5.97% | 3222.70 ± 137.11 / 3219.26（3.43%） | 43.96 ± 4.69 / 46.73（7.04%） | 2889.30 / 2786.56 |
| Qwen2.5-7B-Instruct | AIC | 28.70 / 15.21 | 47.02% / 13.51 | 8121.52 ± 380.37 / 5556.49 | 31.52% | 3222.70 ± 137.11 / 2257.10（29.92%） | 43.96 ± 4.69 / 30.74（30.34%） | 2889.30 / 4443.80 |
| Qwen3-8B | ML | 34.27 / 33.30 | 2.67% / 2.22 | 9470.77 ± 415.08 / 10079.72 | 6.43% | 3663.27 ± 171.83 / 3698.62（1.33%） | 51.17 ± 5.85 / 55.28（7.92%） | 2450.61 / 2288.54 |
| Qwen3-8B | AIC | 34.27 / 16.83 | 50.84% / 17.51 | 9470.77 ± 415.08 / 6533.12 | 30.97% | 3663.27 ± 171.83 / 2539.04（30.63%） | 51.17 ± 5.85 / 36.08（29.65%） | 2450.61 / 3552.85 |
| Qwen2.5-14B-Instruct | ML | 49.08 / 46.38 | 4.82% / 3.51 | 13752.23 ± 286.47 / 13563.13 | 3.80% | 5312.74 ± 119.15 / 5303.09（2.93%） | 76.65 ± 10.85 / 74.09（5.62%） | 1743.57 / 1775.12 |
| Qwen2.5-14B-Instruct | AIC | 49.08 / 30.36 | 39.42% / 18.74 | 13752.23 ± 286.47 / 10543.94 | 23.30% | 5312.74 ± 119.15 / 4265.09（19.73%） | 76.65 ± 10.85 / 57.84（24.45%） | 1743.57 / 2306.78 |
| Qwen2.5-14B-Instruct-1M | ML | 332.28 / 303.52 | 10.76% / 37.45 | 123490.23 ± 12247.12 / 112447.76 | 9.71% | 66441.87 ± 7554.84 / 59321.75（11.30%） | 680.24 ± 220.92 / 626.34（8.19%） | 263.17 / 283.69 |
| Qwen2.5-14B-Instruct-1M | AIC | 332.28 / 30.25 | 83.21% / 302.03 | 123490.23 ± 12247.12 / 10801.56 | 91.17% | 66441.87 ± 7554.84 / 4440.18（93.24%） | 680.24 ± 220.92 / 60.19（90.67%） | 263.17 / 2274.23 |
| Qwen3-14B | ML | 44.84 / 42.80 | 5.00% / 3.98 | 13084.77 ± 1183.94 / 12851.11 | 5.09% | 5465.67 ± 507.81 / 5108.36（5.86%） | 72.37 ± 18.52 / 70.29（6.98%） | 1906.00 / 1903.54 |
| Qwen3-14B | AIC | 44.84 / 30.51 | 33.01% / 14.43 | 13084.77 ± 1183.94 / 10702.16 | 17.61% | 5465.67 ± 507.81 / 4399.40（18.77%） | 72.37 ± 18.52 / 58.74（17.19%） | 1906.00 / 2299.47 |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 125.67 / 127.07 | 5.34% / 5.65 | 40381.89 ± 946.81 / 40539.66 | 0.84% | 20861.91 ± 664.37 / 19200.47（7.93%） | 178.32 ± 32.67 / 220.73（23.93%） | 691.25 / 688.34 |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 125.67 / 138.03 | 36.01% / 36.78 | 40381.89 ± 946.81 / 42833.25 | 6.08% | 20861.91 ± 664.37 / 18446.55（11.55%） | 178.32 ± 32.67 / 233.28（31.83%） | 691.25 / 603.42 |
| Qwen3-30B-A3B | ML | 94.54 / 84.25 | 3.72% / 11.70 | 24031.41 ± 1305.78 / 22078.80 | 8.06% | 8881.12 ± 614.12 / 7747.26（12.66%） | 136.16 ± 12.61 / 120.17（11.62%） | 921.68 / 1018.90 |
| Qwen3-30B-A3B | AIC | 94.54 / 37.42 | 55.77% / 57.12 | 24031.41 ± 1305.78 / 12320.02 | 48.66% | 8881.12 ± 614.12 / 4745.88（46.42%） | 136.16 ± 12.61 / 67.51（50.47%） | 921.68 / 2122.89 |
| Qwen3-30B-A3B-Instruct-2507 | ML | 94.59 / 83.85 | 3.65% / 11.67 | 24037.44 ± 1677.94 / 22454.58 | 6.43% | 8887.82 ± 620.98 / 7827.29（11.77%） | 135.91 ± 18.21 / 122.22（9.49%） | 925.27 / 997.16 |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 94.59 / 37.42 | 55.84% / 57.18 | 24037.44 ± 1677.94 / 12311.74 | 48.63% | 8887.82 ± 620.98 / 4741.02（46.48%） | 135.91 ± 18.21 / 67.47（50.23%） | 925.27 / 2124.72 |
| Qwen3-32B | ML | 74.90 / 73.81 | 2.23% / 2.72 | 22247.90 ± 638.76 / 22537.41 | 1.29% | 9762.00 ± 295.49 / 9557.72（2.08%） | 112.14 ± 14.06 / 122.69（9.21%） | 1154.90 / 1125.98 |
| Qwen3-32B | AIC | 74.90 / 65.87 | 12.64% / 9.07 | 22247.90 ± 638.76 / 21211.51 | 4.66% | 9762.00 ± 295.49 / 9044.12（7.34%） | 112.14 ± 14.06 / 115.66（7.23%） | 1154.90 / 1201.88 |

按阶段展开单步结果。n 为五次独立真机实验合并后的 batch 条数，P95 为逐 batch 绝对百分比误差的 95 分位。

| 模型 | 方法 | n（prefill/decode） | prefill 实测/预测 ms | prefill MAPE / MAE ms | decode 实测/预测 ms | decode MAPE / MAE ms | P95 APE（prefill/decode） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 416 / 2004 | 25.24 / 24.43 | 4.52% / 1.26 | 27.57 / 26.06 | 5.40% / 3.14 | 14.82% / 12.68% |
| Qwen3-0.6B | AIC | 416 / 2004 | 25.24 / 6.35 | 74.62% / 18.89 | 27.57 / 2.98 | 88.43% / 24.58 | 87.13% / 91.77% |
| Qwen2.5-3B-Instruct | ML | 414 / 2004 | 31.02 / 30.07 | 4.40% / 1.46 | 33.72 / 32.18 | 6.95% / 3.94 | 13.36% / 13.80% |
| Qwen2.5-3B-Instruct | AIC | 414 / 2004 | 31.02 / 17.43 | 43.36% / 13.59 | 33.72 / 6.18 | 80.63% / 27.54 | 78.13% / 82.78% |
| llama-2-7b | ML | 415 / 2004 | 37.03 / 36.79 | 2.16% / 0.75 | 29.01 / 27.27 | 5.46% / 3.19 | 9.60% / 14.19% |
| llama-2-7b | AIC | 415 / 2004 | 37.03 / 31.58 | 16.30% / 5.45 | 29.01 / 15.25 | 44.08% / 13.76 | 58.25% / 62.85% |
| Mistral-7B-Instruct-v0.2 | ML | 412 / 2004 | 38.12 / 38.01 | 1.90% / 0.67 | 29.08 / 28.20 | 6.46% / 3.41 | 6.44% / 13.78% |
| Mistral-7B-Instruct-v0.2 | AIC | 412 / 2004 | 38.12 / 34.48 | 11.10% / 3.68 | 29.08 / 12.75 | 53.39% / 16.32 | 45.77% / 61.94% |
| Qwen2.5-7B-Instruct | ML | 414 / 2004 | 36.06 / 35.85 | 2.42% / 0.81 | 27.18 / 25.12 | 7.95% / 3.71 | 10.36% / 14.70% |
| Qwen2.5-7B-Instruct | AIC | 414 / 2004 | 36.06 / 33.03 | 10.31% / 3.17 | 27.18 / 11.53 | 54.61% / 15.64 | 46.20% / 61.20% |
| Qwen3-8B | ML | 414 / 2004 | 39.15 / 39.07 | 1.44% / 0.56 | 33.27 / 32.11 | 2.93% / 2.57 | 3.68% / 8.14% |
| Qwen3-8B | AIC | 414 / 2004 | 39.15 / 35.49 | 11.57% / 4.09 | 33.27 / 12.98 | 58.96% / 20.29 | 52.75% / 64.85% |
| Qwen2.5-14B-Instruct | ML | 414 / 2004 | 68.26 / 67.61 | 2.25% / 1.26 | 45.11 / 42.00 | 5.35% / 3.97 | 11.12% / 11.50% |
| Qwen2.5-14B-Instruct | AIC | 414 / 2004 | 68.26 / 63.33 | 9.01% / 5.06 | 45.11 / 23.55 | 45.70% / 21.56 | 41.04% / 54.87% |
| Qwen2.5-14B-Instruct-1M | ML | 459 / 2004 | 1247.10 / 1115.04 | 9.93% / 138.61 | 122.75 / 117.64 | 10.95% / 14.28 | 22.07% / 21.57% |
| Qwen2.5-14B-Instruct-1M | AIC | 459 / 2004 | 1247.10 / 59.50 | 94.87% / 1187.61 | 122.75 / 23.55 | 80.54% / 99.20 | 97.87% / 84.61% |
| Qwen3-14B | ML | 414 / 2004 | 75.08 / 69.45 | 6.28% / 6.31 | 38.59 / 37.30 | 4.74% / 3.50 | 36.85% / 12.35% |
| Qwen3-14B | AIC | 414 / 2004 | 75.08 / 67.33 | 9.98% / 8.35 | 38.59 / 22.90 | 37.77% / 15.69 | 38.66% / 46.10% |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 411 / 2004 | 367.80 / 367.33 | 1.18% / 3.34 | 76.02 / 77.80 | 6.20% / 6.13 | 4.78% / 17.86% |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 411 / 2004 | 367.80 / 306.86 | 18.23% / 65.98 | 76.02 / 103.40 | 39.65% / 30.79 | 24.51% / 49.06% |
| Qwen3-30B-A3B | ML | 414 / 2004 | 87.80 / 77.59 | 3.60% / 11.50 | 95.93 / 85.62 | 3.75% / 11.74 | 10.27% / 9.48% |
| Qwen3-30B-A3B | AIC | 414 / 2004 | 87.80 / 61.21 | 22.26% / 26.59 | 95.93 / 32.50 | 62.69% / 63.42 | 51.17% / 91.97% |
| Qwen3-30B-A3B-Instruct-2507 | ML | 414 / 2004 | 87.81 / 76.98 | 3.63% / 11.36 | 96.00 / 85.27 | 3.66% / 11.73 | 10.82% / 9.81% |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 414 / 2004 | 87.81 / 61.21 | 22.43% / 26.60 | 96.00 / 32.50 | 62.74% / 63.49 | 51.05% / 91.93% |
| Qwen3-32B | ML | 414 / 2004 | 152.30 / 151.52 | 0.95% / 1.35 | 58.90 / 57.76 | 2.50% / 3.00 | 2.43% / 7.64% |
| Qwen3-32B | AIC | 414 / 2004 | 152.30 / 147.37 | 3.92% / 5.19 | 58.90 / 49.03 | 14.45% / 9.87 | 11.68% / 21.27% |

### latency_128x64_r1（跨负载测试）

| 模型 | 方法 | 单步实测/预测 ms | 单步 MAPE / MAE ms | E2E 实测 ± std / 预测 ms | E2E MAPE | TTFT 实测 ± std / 预测 ms（MAPE） | TPOT 实测 ± std / 预测 ms（MAPE） | 总 token 吞吐实测/预测 tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 25.99 / 25.98 | 6.66% / 2.37 | 936.12 ± 120.59 / 1007.90 | 9.30% | 59.67 ± 15.09 / 39.85（29.70%） | 27.89 ± 1.20 / 30.52（9.61%） | 101.72 / 101.45 |
| Qwen3-0.6B | AIC | 25.99 / 2.03 | 91.98% / 23.96 | 936.12 ± 120.59 / 162.53 | 82.52% | 59.67 ± 15.09 / 9.55（83.17%） | 27.89 ± 1.20 / 4.82（82.64%） | 101.72 / 104.85 |
| Qwen2.5-3B-Instruct | ML | 31.26 / 30.60 | 2.95% / 1.61 | 1115.19 ± 95.01 / 1188.06 | 6.42% | 67.87 ± 7.58 / 48.11（28.44%） | 33.16 ± 0.53 / 35.94（8.39%） | 101.01 / 100.76 |
| Qwen2.5-3B-Instruct | AIC | 31.26 / 5.83 | 80.90% / 25.43 | 1115.19 ± 95.01 / 279.76 | 74.89% | 67.87 ± 7.58 / 13.61（79.70%） | 33.16 ± 0.53 / 8.47（74.47%） | 101.01 / 104.39 |
| llama-2-7b | ML | 26.61 / 26.28 | 2.06% / 1.15 | 932.39 ± 78.17 / 1066.40 | 14.38% | 51.80 ± 4.89 / 44.08（14.08%） | 27.91 ± 0.56 / 32.26（15.67%） | 103.57 / 105.36 |
| llama-2-7b | AIC | 26.61 / 10.21 | 60.73% / 16.40 | 932.39 ± 78.17 / 443.57 | 52.34% | 51.80 ± 4.89 / 20.06（60.87%） | 27.91 ± 0.56 / 13.44（51.83%） | 103.57 / 108.37 |
| Mistral-7B-Instruct-v0.2 | ML | 28.76 / 27.87 | 6.55% / 2.50 | 1014.41 ± 113.15 / 1123.31 | 11.14% | 57.73 ± 4.43 / 45.66（20.58%） | 30.21 ± 1.38 / 34.03（12.85%） | 102.78 / 104.43 |
| Mistral-7B-Instruct-v0.2 | AIC | 28.76 / 10.84 | 61.28% / 17.91 | 1014.41 ± 113.15 / 464.12 | 53.95% | 57.73 ± 4.43 / 22.81（60.24%） | 30.21 ± 1.38 / 14.06（53.37%） | 102.78 / 107.48 |
| Qwen2.5-7B-Instruct | ML | 26.03 / 25.39 | 6.82% / 2.33 | 932.91 ± 118.70 / 1041.85 | 12.22% | 67.60 ± 26.26 / 42.61（30.65%） | 27.43 ± 1.95 / 31.56（15.62%） | 101.63 / 101.21 |
| Qwen2.5-7B-Instruct | AIC | 26.03 / 10.61 | 58.03% / 15.42 | 932.91 ± 118.70 / 427.47 | 53.86% | 67.60 ± 26.26 / 19.88（67.48%） | 27.43 ± 1.95 / 12.85（52.91%） | 101.63 / 103.78 |
| Qwen3-8B | ML | 32.91 / 32.12 | 5.02% / 2.43 | 1162.93 ± 95.14 / 1270.80 | 10.93% | 75.11 ± 11.94 / 51.16（30.08%） | 34.45 ± 2.14 / 38.46（12.48%） | 100.81 / 100.52 |
| Qwen3-8B | AIC | 32.91 / 11.54 | 63.98% / 21.36 | 1162.93 ± 95.14 / 495.78 | 57.32% | 75.11 ± 11.94 / 22.99（68.43%） | 34.45 ± 2.14 / 14.90（56.60%） | 100.81 / 103.54 |
| Qwen2.5-14B-Instruct | ML | 40.52 / 40.24 | 2.12% / 1.60 | 1434.27 ± 125.68 / 1525.84 | 6.41% | 86.62 ± 13.45 / 62.81（25.66%） | 42.70 ± 1.20 / 46.23（8.37%） | 99.80 / 99.55 |
| Qwen2.5-14B-Instruct | AIC | 40.52 / 20.46 | 48.53% / 20.05 | 1434.27 ± 125.68 / 850.09 | 40.72% | 86.62 ± 13.45 / 37.23（56.13%） | 42.70 ± 1.20 / 25.65（39.87%） | 99.80 / 102.29 |
| Qwen2.5-14B-Instruct-1M | ML | 357.05 / 389.96 | 30.39% / 35.60 | 17294.39 ± 2387.55 / 17184.71 | 2.46% | 2821.48 ± 738.14 / 2687.42（6.18%） | 630.42 ± 94.62 / 631.78（3.23%） | 78.94 / 79.15 |
| Qwen2.5-14B-Instruct-1M | AIC | 357.05 / 22.66 | 83.72% / 334.39 | 17294.39 ± 2387.55 / 854.81 | 94.99% | 2821.48 ± 738.14 / 37.92（98.57%） | 630.42 ± 94.62 / 25.76（95.78%） | 78.94 / 102.26 |
| Qwen3-14B | ML | 38.32 / 36.90 | 6.25% / 3.21 | 1370.14 ± 167.90 / 1429.69 | 5.86% | 87.41 ± 14.11 / 56.19（34.10%） | 40.38 ± 3.31 / 43.42（9.17%） | 100.11 / 99.96 |
| Qwen3-14B | AIC | 38.32 / 20.13 | 46.14% / 18.19 | 1370.14 ± 167.90 / 836.61 | 38.62% | 87.41 ± 14.11 / 39.26（53.99%） | 40.38 ± 3.31 / 25.25（37.15%） | 100.11 / 102.35 |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 81.55 / 77.37 | 6.49% / 6.48 | 2985.57 ± 390.71 / 2933.13 | 5.85% | 177.64 ± 16.02 / 138.41（21.50%） | 87.57 ± 7.28 / 87.99（6.70%） | 95.97 / 96.15 |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 81.55 / 95.06 | 19.33% / 16.00 | 2985.57 ± 390.71 / 3738.11 | 25.93% | 177.64 ± 16.02 / 170.93（7.51%） | 87.57 ± 7.28 / 112.92（29.61%） | 95.97 / 93.48 |
| Qwen3-30B-A3B | ML | 100.35 / 84.90 | 4.05% / 16.20 | 3606.13 ± 278.20 / 3162.95 | 12.37% | 366.47 ± 68.95 / 124.40（64.79%） | 103.30 ± 5.49 / 95.80（7.06%） | 93.53 / 94.17 |
| Qwen3-30B-A3B | AIC | 100.35 / 15.02 | 82.96% / 85.34 | 3606.13 ± 278.20 / 375.34 | 89.59% | 366.47 ± 68.95 / 46.28（86.98%） | 103.30 ± 5.49 / 10.61（89.71%） | 93.53 / 104.15 |
| Qwen3-30B-A3B-Instruct-2507 | ML | 100.93 / 85.98 | 5.45% / 17.54 | 3647.67 ± 380.08 / 3205.68 | 12.00% | 393.18 ± 109.90 / 125.92（65.99%） | 104.02 ± 6.21 / 97.11（6.38%） | 94.11 / 94.14 |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 100.93 / 15.13 | 82.99% / 85.80 | 3647.67 ± 380.08 / 372.89 | 89.75% | 393.18 ± 109.90 / 46.00（87.59%） | 104.02 ± 6.21 / 10.54（89.83%） | 94.11 / 104.15 |
| Qwen3-32B | ML | 59.61 / 56.31 | 4.08% / 3.51 | 2152.64 ± 249.60 / 2110.99 | 3.55% | 115.89 ± 9.89 / 87.42（24.08%） | 64.43 ± 4.16 / 63.86（4.49%） | 97.45 / 97.61 |
| Qwen3-32B | AIC | 59.61 / 45.81 | 21.71% / 13.81 | 2152.64 ± 249.60 / 1793.25 | 16.47% | 115.89 ± 9.89 / 80.75（29.90%） | 64.43 ± 4.16 / 54.03（15.85%） | 97.45 / 98.76 |

按阶段展开单步结果。n 为五次独立真机实验合并后的 batch 条数，P95 为逐 batch 绝对百分比误差的 95 分位。

| 模型 | 方法 | n（prefill/decode） | prefill 实测/预测 ms | prefill MAPE / MAE ms | decode 实测/预测 ms | decode MAPE / MAE ms | P95 APE（prefill/decode） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 159 / 3383 | 26.99 / 25.34 | 6.27% / 2.06 | 25.94 / 26.01 | 6.68% / 2.39 | 19.45% / 15.58% |
| Qwen3-0.6B | AIC | 159 / 3383 | 26.99 / 3.51 | 86.81% / 23.48 | 25.94 / 1.96 | 92.22% / 23.98 | 88.93% / 93.57% |
| Qwen2.5-3B-Instruct | ML | 157 / 3172 | 31.20 / 30.47 | 3.05% / 1.02 | 31.26 / 30.61 | 2.95% / 1.64 | 15.37% / 11.37% |
| Qwen2.5-3B-Instruct | AIC | 157 / 3172 | 31.20 / 7.40 | 76.23% / 23.80 | 31.26 / 5.75 | 81.14% / 25.51 | 79.63% / 83.04% |
| llama-2-7b | ML | 160 / 3447 | 27.17 / 27.32 | 2.13% / 0.58 | 26.58 / 26.23 | 2.05% / 1.18 | 4.94% / 8.24% |
| llama-2-7b | AIC | 160 / 3447 | 27.17 / 12.04 | 55.68% / 15.13 | 26.58 / 10.12 | 60.96% / 16.46 | 60.53% / 63.18% |
| Mistral-7B-Instruct-v0.2 | ML | 160 / 3331 | 29.22 / 27.99 | 6.95% / 2.09 | 28.73 / 27.87 | 6.53% / 2.52 | 12.02% / 14.99% |
| Mistral-7B-Instruct-v0.2 | AIC | 160 / 3331 | 29.22 / 14.61 | 49.82% / 14.62 | 28.73 / 10.66 | 61.83% / 18.07 | 61.46% / 65.72% |
| Qwen2.5-7B-Instruct | ML | 159 / 3391 | 26.51 / 25.26 | 7.73% / 2.11 | 26.01 / 25.39 | 6.78% / 2.34 | 13.45% / 14.64% |
| Qwen2.5-7B-Instruct | AIC | 159 / 3391 | 26.51 / 13.37 | 49.28% / 13.14 | 26.01 / 10.48 | 58.45% / 15.53 | 59.22% / 62.62% |
| Qwen3-8B | ML | 157 / 3095 | 33.27 / 31.94 | 5.08% / 1.89 | 32.89 / 32.13 | 5.02% / 2.46 | 15.34% / 13.34% |
| Qwen3-8B | AIC | 157 / 3095 | 33.27 / 14.92 | 54.88% / 18.35 | 32.89 / 11.37 | 64.45% / 21.52 | 63.81% / 68.15% |
| Qwen2.5-14B-Instruct | ML | 158 / 2799 | 40.72 / 40.45 | 2.89% / 1.29 | 40.50 / 40.23 | 2.08% / 1.62 | 11.46% / 6.80% |
| Qwen2.5-14B-Instruct | AIC | 158 / 2799 | 40.72 / 25.20 | 37.98% / 15.52 | 40.50 / 20.20 | 49.12% / 20.31 | 46.28% / 52.85% |
| Qwen2.5-14B-Instruct-1M | ML | 109 / 448 | 1379.44 / 1381.50 | 1.03% / 13.13 | 108.30 / 148.71 | 37.54% / 41.07 | 2.41% / 342.92% |
| Qwen2.5-14B-Instruct-1M | AIC | 109 / 448 | 1379.44 / 28.02 | 97.82% / 1351.42 | 108.30 / 21.36 | 80.29% / 86.94 | 98.58% / 82.40% |
| Qwen3-14B | ML | 156 / 2882 | 38.38 / 36.97 | 6.28% / 2.52 | 38.31 / 36.90 | 6.25% / 3.25 | 13.04% / 14.54% |
| Qwen3-14B | AIC | 156 / 2882 | 38.38 / 27.62 | 27.86% / 10.77 | 38.31 / 19.73 | 47.13% / 18.59 | 44.25% / 52.54% |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 156 / 1789 | 93.98 / 92.11 | 4.55% / 4.21 | 80.46 / 76.08 | 6.66% / 6.68 | 14.20% / 14.53% |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 156 / 1789 | 93.98 / 116.07 | 26.99% / 22.65 | 80.46 / 93.22 | 18.66% / 15.42 | 57.60% / 28.41% |
| Qwen3-30B-A3B | ML | 145 / 1514 | 148.57 / 79.54 | 13.53% / 70.28 | 95.74 / 85.41 | 3.14% / 11.02 | 87.55% / 9.80% |
| Qwen3-30B-A3B | AIC | 145 / 1514 | 148.57 / 39.92 | 55.97% / 108.65 | 95.74 / 12.63 | 85.54% / 83.11 | 93.27% / 92.34% |
| Qwen3-30B-A3B-Instruct-2507 | ML | 144 / 1494 | 149.08 / 78.82 | 14.26% / 70.68 | 96.29 / 86.67 | 4.60% / 12.42 | 87.38% / 10.58% |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 144 / 1494 | 149.08 / 40.02 | 56.49% / 109.06 | 96.29 / 12.73 | 85.54% / 83.56 | 93.08% / 92.38% |
| Qwen3-32B | ML | 154 / 2256 | 60.20 / 57.58 | 4.83% / 3.10 | 59.56 / 56.22 | 4.02% / 3.54 | 15.16% / 13.67% |
| Qwen3-32B | AIC | 154 / 2256 | 60.20 / 54.64 | 9.18% / 5.70 | 59.56 / 45.20 | 22.56% / 14.36 | 20.69% / 31.03% |

### saturation_1536x384_inf（跨负载测试）

| 模型 | 方法 | 单步实测/预测 ms | 单步 MAPE / MAE ms | E2E 实测 ± std / 预测 ms | E2E MAPE | TTFT 实测 ± std / 预测 ms（MAPE） | TPOT 实测 ± std / 预测 ms（MAPE） | 总 token 吞吐实测/预测 tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 26.69 / 25.66 | 4.89% / 2.25 | 10747.97 ± 733.51 / 11576.70 | 7.72% | 3726.92 ± 340.71 / 4179.71（12.41%） | 41.41 ± 5.69 / 43.18（4.53%） | 3088.01 / 2910.87 |
| Qwen3-0.6B | AIC | 26.69 / 3.90 | 84.73% / 22.79 | 10747.97 ± 733.51 / 4970.39 | 53.67% | 3726.92 ± 340.71 / 1758.17（52.61%） | 41.41 ± 5.69 / 18.74（54.99%） | 3088.01 / 6352.65 |
| Qwen2.5-3B-Instruct | ML | 32.37 / 31.64 | 5.86% / 2.81 | 12814.73 ± 532.11 / 13259.75 | 5.23% | 4532.65 ± 297.41 / 4833.34（6.41%） | 49.73 ± 6.81 / 49.60（5.24%） | 2566.04 / 2517.02 |
| Qwen2.5-3B-Instruct | AIC | 32.37 / 8.23 | 73.56% / 24.14 | 12814.73 ± 532.11 / 5762.59 | 55.07% | 4532.65 ± 297.41 / 2323.17（48.77%） | 49.73 ± 6.81 / 22.02（56.20%） | 2566.04 / 6059.55 |
| llama-2-7b | ML | 30.09 / 28.74 | 4.22% / 2.21 | 12377.66 ± 803.18 / 13426.00 | 8.54% | 4906.33 ± 397.26 / 5188.73（5.76%） | 46.49 ± 9.00 / 50.66（9.16%） | 2778.52 / 2587.49 |
| llama-2-7b | AIC | 30.09 / 20.15 | 32.06% / 9.94 | 12377.66 ± 803.18 / 12082.35 | 3.40% | 4906.33 ± 397.26 / 4657.90（5.28%） | 46.49 ± 9.00 / 45.64（3.10%） | 2778.52 / 3009.54 |
| Mistral-7B-Instruct-v0.2 | ML | 30.26 / 29.42 | 4.41% / 2.22 | 12445.16 ± 734.90 / 13558.44 | 9.08% | 4988.30 ± 411.52 / 5259.19（6.80%） | 46.71 ± 9.07 / 51.34（9.93%） | 2775.34 / 2547.08 |
| Mistral-7B-Instruct-v0.2 | AIC | 30.26 / 16.97 | 44.71% / 13.32 | 12445.16 ± 734.90 / 10778.60 | 13.27% | 4988.30 ± 411.52 / 4328.11（13.10%） | 46.71 ± 9.07 / 41.08（12.57%） | 2775.34 / 3289.82 |
| Qwen2.5-7B-Instruct | ML | 27.95 / 26.93 | 4.73% / 2.21 | 11742.65 ± 1060.12 / 12666.06 | 8.09% | 4739.78 ± 493.16 / 4959.24（5.04%） | 43.60 ± 8.12 / 47.81（9.68%） | 2960.46 / 2764.41 |
| Qwen2.5-7B-Instruct | AIC | 27.95 / 15.46 | 45.74% / 12.52 | 11742.65 ± 1060.12 / 8590.17 | 26.59% | 4739.78 ± 493.16 / 3611.18（23.52%） | 43.60 ± 8.12 / 33.02（25.28%） | 2960.46 / 4306.42 |
| Qwen3-8B | ML | 36.34 / 33.22 | 7.07% / 3.44 | 14651.55 ± 1016.22 / 15166.05 | 4.31% | 5679.18 ± 501.54 / 5782.74（2.91%） | 54.35 ± 9.23 / 57.03（5.60%） | 2322.61 / 2255.20 |
| Qwen3-8B | AIC | 36.34 / 17.30 | 52.63% / 19.08 | 14651.55 ± 1016.22 / 10897.44 | 25.43% | 5679.18 ± 501.54 / 4381.56（22.69%） | 54.35 ± 9.23 / 41.49（24.24%） | 2322.61 / 3241.50 |
| Qwen2.5-14B-Instruct | ML | 46.37 / 45.84 | 2.43% / 1.94 | 19351.91 ± 968.31 / 20199.91 | 4.37% | 7413.46 ± 511.75 / 8103.29（9.35%） | 77.32 ± 14.40 / 76.70（4.29%） | 1829.51 / 1758.79 |
| Qwen2.5-14B-Instruct | AIC | 46.37 / 31.06 | 34.95% / 15.32 | 19351.91 ± 968.31 / 17154.97 | 11.38% | 7413.46 ± 511.75 / 7086.00（5.58%） | 77.32 ± 14.40 / 65.53（16.06%） | 1829.51 / 2136.58 |
| Qwen2.5-14B-Instruct-1M | ML | 302.39 / 284.09 | 12.21% / 31.46 | 163183.99 ± 19617.05 / 156670.76 | 8.46% | 83352.26 ± 8182.02 / 81744.26（4.39%） | 661.82 ± 244.25 / 647.19（12.06%） | 289.09 / 296.08 |
| Qwen2.5-14B-Instruct-1M | AIC | 302.39 / 31.02 | 82.35% / 271.37 | 163183.99 ± 19617.05 / 17329.99 | 89.29% | 83352.26 ± 8182.02 / 7220.92（91.31%） | 661.82 ± 244.25 / 66.92（89.45%） | 289.09 / 2124.22 |
| Qwen3-14B | ML | 43.74 / 43.75 | 6.40% / 3.35 | 18627.33 ± 993.36 / 19660.07 | 5.77% | 7798.04 ± 512.40 / 8043.98（3.68%） | 70.02 ± 17.88 / 74.84（7.45%） | 1933.66 / 1851.85 |
| Qwen3-14B | AIC | 43.74 / 31.05 | 31.34% / 12.75 | 18627.33 ± 993.36 / 17174.66 | 7.76% | 7798.04 ± 512.40 / 7205.65（7.59%） | 70.02 ± 17.88 / 65.83（6.12%） | 1933.66 / 2154.41 |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 126.80 / 124.17 | 2.20% / 3.82 | 60847.78 ± 2384.58 / 60504.86 | 1.10% | 31508.83 ± 1118.32 / 29074.03（7.72%） | 159.01 ± 22.67 / 237.56（45.90%） | 686.33 / 689.09 |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 126.80 / 139.62 | 36.12% / 37.07 | 60847.78 ± 2384.58 / 65630.23 | 7.87% | 31508.83 ± 1118.32 / 28701.77（8.93%） | 159.01 ± 22.67 / 252.80（56.68%） | 686.33 / 589.75 |
| Qwen3-30B-A3B | ML | 93.79 / 83.82 | 5.06% / 10.40 | 35511.07 ± 2874.62 / 32707.91 | 7.70% | 13277.27 ± 1443.37 / 11786.03（11.08%） | 134.67 ± 22.88 / 121.85（8.92%） | 925.35 / 1008.36 |
| Qwen3-30B-A3B | AIC | 93.79 / 36.87 | 57.30% / 56.93 | 35511.07 ± 2874.62 / 18838.23 | 46.80% | 13277.27 ± 1443.37 / 7462.14（43.56%） | 134.67 ± 22.88 / 71.32（47.22%） | 925.35 / 2105.66 |
| Qwen3-30B-A3B-Instruct-2507 | ML | 91.48 / 86.97 | 5.57% / 10.62 | 34762.36 ± 2436.87 / 34727.50 | 3.15% | 13022.11 ± 1301.52 / 12359.95（5.03%） | 132.19 ± 23.69 / 129.20（3.83%） | 947.33 / 954.34 |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 91.48 / 36.87 | 56.19% / 54.61 | 34762.36 ± 2436.87 / 18819.15 | 45.78% | 13022.11 ± 1301.52 / 7453.04（42.60%） | 132.19 ± 23.69 / 71.27（46.26%） | 947.33 / 2107.88 |
| Qwen3-32B | ML | 74.93 / 74.03 | 2.56% / 2.57 | 33018.45 ± 1513.97 / 33852.82 | 2.53% | 13959.38 ± 827.05 / 14638.74（5.06%） | 135.65 ± 28.54 / 130.27（7.92%） | 1146.54 / 1108.62 |
| Qwen3-32B | AIC | 74.93 / 66.78 | 11.53% / 8.17 | 33018.45 ± 1513.97 / 33020.31 | 0.69% | 13959.38 ± 827.05 / 14291.31（4.20%） | 135.65 ± 28.54 / 127.02（8.09%） | 1146.54 / 1148.25 |

按阶段展开单步结果。n 为五次独立真机实验合并后的 batch 条数，P95 为逐 batch 绝对百分比误差的 95 分位。

| 模型 | 方法 | n（prefill/decode） | prefill 实测/预测 ms | prefill MAPE / MAE ms | decode 实测/预测 ms | decode MAPE / MAE ms | P95 APE（prefill/decode） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 575 / 3025 | 25.12 / 24.29 | 4.80% / 1.35 | 26.99 / 25.93 | 4.91% / 2.42 | 14.08% / 12.55% |
| Qwen3-0.6B | AIC | 575 / 3025 | 25.12 / 6.80 | 72.66% / 18.32 | 26.99 / 3.35 | 87.03% / 23.64 | 85.10% / 91.52% |
| Qwen2.5-3B-Instruct | ML | 575 / 3025 | 30.62 / 30.01 | 3.11% / 1.04 | 32.71 / 31.94 | 6.38% / 3.14 | 12.41% / 11.70% |
| Qwen2.5-3B-Instruct | AIC | 575 / 3025 | 30.62 / 18.47 | 39.35% / 12.15 | 32.71 / 6.29 | 80.06% / 26.42 | 73.38% / 82.38% |
| llama-2-7b | ML | 575 / 3025 | 38.80 / 38.70 | 2.22% / 0.84 | 28.44 / 26.84 | 4.60% / 2.48 | 8.85% / 11.35% |
| llama-2-7b | AIC | 575 / 3025 | 38.80 / 33.74 | 14.01% / 5.05 | 28.44 / 17.56 | 35.49% / 10.87 | 51.29% / 61.29% |
| Mistral-7B-Instruct-v0.2 | ML | 575 / 3025 | 39.98 / 39.69 | 2.33% / 0.90 | 28.42 / 27.47 | 4.81% / 2.47 | 7.77% / 8.84% |
| Mistral-7B-Instruct-v0.2 | AIC | 575 / 3025 | 39.98 / 36.71 | 9.52% / 3.46 | 28.42 / 13.22 | 51.40% / 15.20 | 36.65% / 61.97% |
| Qwen2.5-7B-Instruct | ML | 575 / 3025 | 37.81 / 37.59 | 3.34% / 1.27 | 26.08 / 24.91 | 5.00% / 2.39 | 14.87% / 13.30% |
| Qwen2.5-7B-Instruct | AIC | 575 / 3025 | 37.81 / 34.78 | 9.37% / 3.16 | 26.08 / 11.78 | 52.65% / 14.30 | 39.58% / 58.63% |
| Qwen3-8B | ML | 575 / 3025 | 41.50 / 40.85 | 2.76% / 1.13 | 35.36 / 31.78 | 7.89% / 3.88 | 12.09% / 12.84% |
| Qwen3-8B | AIC | 575 / 3025 | 41.50 / 37.38 | 11.14% / 4.37 | 35.36 / 13.49 | 60.51% / 21.88 | 49.43% / 67.38% |
| Qwen2.5-14B-Instruct | ML | 575 / 3025 | 71.68 / 71.44 | 1.42% / 0.97 | 41.56 / 40.97 | 2.62% / 2.12 | 4.16% / 10.39% |
| Qwen2.5-14B-Instruct | AIC | 575 / 3025 | 71.68 / 66.96 | 7.61% / 4.76 | 41.56 / 24.23 | 40.15% / 17.33 | 30.39% / 51.49% |
| Qwen2.5-14B-Instruct-1M | ML | 617 / 3024 | 1185.41 / 1088.34 | 7.71% / 103.25 | 122.23 / 120.00 | 13.13% / 16.81 | 21.84% / 28.71% |
| Qwen2.5-14B-Instruct-1M | AIC | 617 / 3024 | 1185.41 / 64.26 | 94.23% / 1121.15 | 122.23 / 24.24 | 79.93% / 97.99 | 97.72% / 84.51% |
| Qwen3-14B | ML | 575 / 3025 | 73.62 / 73.50 | 1.49% / 1.00 | 38.05 / 38.09 | 7.33% / 3.79 | 5.14% / 13.00% |
| Qwen3-14B | AIC | 575 / 3025 | 73.62 / 71.02 | 4.82% / 3.02 | 38.05 / 23.46 | 36.38% / 14.60 | 13.93% / 49.32% |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 575 / 3025 | 392.96 / 385.03 | 2.89% / 10.21 | 76.21 / 74.59 | 2.06% / 2.60 | 11.45% / 11.44% |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 575 / 3025 | 392.96 / 324.13 | 17.98% / 71.56 | 76.21 / 104.55 | 39.57% / 30.51 | 22.45% / 49.53% |
| Qwen3-30B-A3B | ML | 575 / 3025 | 90.75 / 77.28 | 5.58% / 14.14 | 94.37 / 85.06 | 4.96% / 9.69 | 10.69% / 10.61% |
| Qwen3-30B-A3B | AIC | 575 / 3025 | 90.75 / 63.18 | 21.85% / 27.57 | 94.37 / 31.87 | 64.04% / 62.51 | 44.65% / 92.58% |
| Qwen3-30B-A3B-Instruct-2507 | ML | 575 / 3025 | 88.73 / 76.73 | 3.82% / 12.72 | 92.00 / 88.91 | 5.90% / 10.23 | 10.34% / 9.18% |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 575 / 3025 | 88.73 / 63.18 | 19.77% / 25.55 | 92.00 / 31.87 | 63.12% / 60.13 | 44.02% / 92.07% |
| Qwen3-32B | ML | 575 / 3025 | 162.23 / 160.40 | 1.36% / 2.19 | 58.34 / 57.62 | 2.78% / 2.65 | 3.69% / 10.80% |
| Qwen3-32B | AIC | 575 / 3025 | 162.23 / 155.63 | 4.49% / 6.66 | 58.34 / 49.89 | 12.87% / 8.45 | 12.49% / 21.01% |

![ML batch 分负载误差](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/ml_batch_workload_heatmap.png)

![ML e2e 分负载误差](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/ml_e2e_workload_heatmap.png)

![AIC batch 分负载误差](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/aic_batch_workload_heatmap.png)

![AIC e2e 分负载误差](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/aic_e2e_workload_heatmap.png)

## 训练量递增对比

测试集保持不变，以前 1、3、5 组训练 trace 分别拟合，训练请求数为 160、480、800；下面曲线按负载展示。增加样本是否有效以实测误差变化为准，不预设单调改善。

![模型 1–4 训练量曲线](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/learning_curves_models_1_4.png)

![模型 5–8 训练量曲线](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/learning_curves_models_5_8.png)

![模型 9–12 训练量曲线](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/learning_curves_models_9_12.png)

![模型 13–13 训练量曲线](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/learning_curves_models_13_13.png)

完整 prefill/decode、P95 APE、偏差与样本数见同目录 workload_batch_metrics.csv；全部 E2E、TTFT、TPOT、吞吐及实机标准差见 workload_end_to_end_metrics.csv。

## 结果解释的边界

同步墙钟单步时延包含 run_batch 内 CPU 发射、采样/拷贝和同步等待，AIC 算子 GPU 时延求和不包含全部这些开销。因此增加 AIC 算子点与提高端到端准确性不是同一件事。AWQ BF16 代理、dual-chunk attention 代理与 MoE 路由覆盖限制沿用前文单列说明。当前 GPU 其他进程及显存状态已记录；本轮不是无干扰的硬件性能标定。

AIC 的 Silicon 模式也不是所有算子都查实测表：此版本将 RMSNorm、SwiGLU 等归为 ElementWise，并调用 query_mem_op 的带宽/固定延迟公式。当前配置为带宽 1.935 TB/s、经验缩放 0.8、固定延迟 3 μs；重新采集 attention/GEMM 不会更新这些项。这里保留标准 SDK 实现，解析项与本机实测表分别说明。源码：[ElementWise](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/src/aiconfigurator/sdk/operations/elementwise.py)、[query_mem_op](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/src/aiconfigurator/sdk/perf_database.py)。

attention harness 在计时前调用 init_forward_metadata，测量时仅重复执行 RadixAttention，并先 warmup 三次；真机 run_batch 的同步墙钟标签包含相关元数据准备。两条路径都构造 TritonAttnBackend，KV-split 上限 8 不代表每个请求固定使用 8 个 split。前文的 3B、7B、32B 抽样 profile 用于检查实际时序和算子族；它不是元数据准备、编译等开销的逐项受控对照，不能直接分摊总误差。

14B-1M 与普通 14B 都是 48 层、40 个 query heads，但原生 attention 路径不同。当前 1M 的非稀疏 prefill 按请求、按 head 循环，并包含 KV gather、长度回传及标量读取；单个非空序列按源码展开为 1,920 次子函数调用，这不是实测 GPU kernel 数或时延。其有效 chunk 长度为 253,952，本轮 context=2,048，decode 不进入 successor/inter 分支，但仍有缩放及 LSE 归并。AIC 普通 Triton/page1 表未对齐 dual-chunk/FA2/page256 的调用图，该模型只作后端代理比较，不据此解释其余普通 dense 模型的误差。实际配置、加载路径和源码校验和见结果目录 verification/qwen25_14b_1m_attention_route_20260929_0818.md。

算子采样默认使用 CUDA Graph：benchmark_with_power 在 capture 前后分别预热，再用 CUDA event 计时重放；本轮真机则关闭 graph，测量完整 run_batch 的同步墙钟。GEMM harness 按显存预算建立 1–6 组独立矩阵，依次执行后按组数取平均；attention harness 重用一组层输入与 KV 池。计时循环内没有显式缓存驱逐步骤，但这不足以证明实际缓存命中率或它造成的误差，不能把所有 GEMM 采样概括为反复读取同一份权重，也不能据此给预测值补一个固定延迟。前文的抽样 GPU profile 不单独测量缓存命中率，不能据此量化缓存或权重复用对误差的独立影响。依据：[采样计时](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/collector/helper.py)、[GEMM harness](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/collector/sglang/collect_gemm.py)；实际 collector 修改与源码校验和已存入本轮 provenance。

Qwen3 的 QK norm 还存在阶段差异：ContextAttention 用解析项附加其时延；GenerationAttention 虽保留 use_qk_norm 标志，query 中没有对应附加项。本机 attention harness 计时的是 RadixAttention 调用，不包含模型层前面的 QK norm/旋转位置编码调用，因此加密这张表不等于补采这些操作。CPU 调用核验在 B=1、S=128 和 B=8、S=1024 下，开关 QK norm 的 prefill 差值均为约 0.0264 ms/层，decode 差值为零；这是 SDK 公式/调用行为，不是 GPU 实测时延，也不用于修改预测值。算子族统计不等于隔离 QK norm 的单项计时，不能由这项公式差值量化其对总误差的贡献。依据：[attention 查询](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/src/aiconfigurator/sdk/operations/attention.py)、[attention harness](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/collector/sglang/collect_attn.py)；调用记录见 preflight/aic_attention_query_scope_20260928.json。

AIC checkout 在本轮开始前已有三处 collector 修改：两份 attention YAML 增补 head=28，以及 collect_attn.py 的 runner 兼容字段和可选 Torch SDPA 路径。正式采集显式选择 Triton，不进入可选 SDPA 分支；这些原有修改未回滚，也未在本轮改写。provenance 同时保存 AIC dirty patch、collector/SDK 源文件及 SHA256，不能仅用 HEAD 表示实际采集代码。

## 按负载等权的同负载/跨负载结果

以下先分别计算每类负载的误差，再对负载等权平均，避免低到达率用例较多的单请求 decode 行主导总览。前三种是同负载独立请求，后两种是跨负载；这个划分不等同于严格的特征分布内/分布外。原始逐行加权结果仍在前文保留。

| 模型 | 方法 | 同负载单步 MAPE | 跨负载单步 MAPE | 同负载 E2E MAPE | 跨负载 E2E MAPE |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 6.23% | 5.77% | 11.21% | 8.51% |
| Qwen3-0.6B | AIC | 88.35% | 88.36% | 66.28% | 68.09% |
| Qwen2.5-3B-Instruct | ML | 4.96% | 4.41% | 6.67% | 5.83% |
| Qwen2.5-3B-Instruct | AIC | 77.07% | 77.23% | 66.21% | 64.98% |
| llama-2-7b | ML | 4.37% | 3.14% | 11.05% | 11.46% |
| llama-2-7b | AIC | 48.86% | 46.39% | 29.74% | 27.87% |
| Mistral-7B-Instruct-v0.2 | ML | 5.52% | 5.48% | 13.29% | 10.11% |
| Mistral-7B-Instruct-v0.2 | AIC | 52.20% | 52.99% | 32.38% | 33.61% |
| Qwen2.5-7B-Instruct | ML | 6.12% | 5.78% | 12.32% | 10.15% |
| Qwen2.5-7B-Instruct | AIC | 51.11% | 51.89% | 40.12% | 40.22% |
| Qwen3-8B | ML | 3.45% | 6.05% | 8.94% | 7.62% |
| Qwen3-8B | AIC | 56.60% | 58.30% | 39.39% | 41.38% |
| Qwen2.5-14B-Instruct | ML | 4.48% | 2.27% | 6.15% | 5.39% |
| Qwen2.5-14B-Instruct | AIC | 42.23% | 41.74% | 27.22% | 26.05% |
| Qwen2.5-14B-Instruct-1M | ML | 13.45% | 21.30% | 6.06% | 5.46% |
| Qwen2.5-14B-Instruct-1M | AIC | 82.65% | 83.03% | 92.56% | 92.14% |
| Qwen3-14B | ML | 5.86% | 6.33% | 7.90% | 5.82% |
| Qwen3-14B | AIC | 37.71% | 38.74% | 21.67% | 23.19% |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 5.55% | 4.34% | 2.98% | 3.47% |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 30.30% | 27.72% | 14.79% | 16.90% |
| Qwen3-30B-A3B | ML | 4.57% | 4.55% | 7.40% | 10.03% |
| Qwen3-30B-A3B | AIC | 63.63% | 70.13% | 65.33% | 68.19% |
| Qwen3-30B-A3B-Instruct-2507 | ML | 4.16% | 5.51% | 5.65% | 7.58% |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 63.52% | 69.59% | 65.33% | 67.77% |
| Qwen3-32B | ML | 2.57% | 3.32% | 3.29% | 3.04% |
| Qwen3-32B | AIC | 15.19% | 16.62% | 6.53% | 8.58% |

## 同一 ML 预测器的训练样本回评分与独立测试

固定每模型最终 p50 bundle，回评分传入最终 fit 的训练样本，并与同一 bundle 的独立测试比较；不重新训练、不选择新的参数。训练侧沿用 bundle 的时延过滤，核对行数/用例数及数据来源；测试侧保留全部有效正时延，逐行复算并核验与正式评分完全一致。

训练样本回评分不是独立验证；最终 HGB fit 内仍有 early-stopping 保留部分，因此也不表示每行都用于构建树。它与前文 80%/20% 内部分行验证的模型不同。下表各格为负载等权的实测/预测均值 ms（MAPE）；差距同时包含请求组成、batch 分布与采集时段变化，不能归因于请求不同这一单一因素。此处仅比较单步，不补跑训练请求的端到端服务。

| 模型 | 训练样本回评分 ms（MAPE） | 同负载独立测试 ms（MAPE） | 跨负载独立测试 ms（MAPE） |
| --- | --- | --- | --- |
| Qwen3-0.6B | 26.808/25.756（2.32%） | 26.036/25.894（6.23%） | 26.340/25.823（5.77%） |
| Qwen2.5-3B-Instruct | 32.107/30.920（1.88%） | 32.204/31.430（4.96%） | 31.815/31.118（4.41%） |
| llama-2-7b | 28.875/27.924（1.35%） | 28.795/27.778（4.37%） | 28.348/27.509（3.14%） |
| Mistral-7B-Instruct-v0.2 | 29.665/28.813（1.91%） | 29.096/28.856（5.52%） | 29.510/28.644（5.48%） |
| Qwen2.5-7B-Instruct | 26.779/25.840（1.49%） | 26.734/25.986（6.12%） | 26.992/26.158（5.78%） |
| Qwen3-8B | 33.760/32.703（1.61%） | 33.684/32.689（3.45%） | 34.625/32.672（6.05%） |
| Qwen2.5-14B-Instruct | 44.881/43.744（1.46%） | 45.574/43.972（4.48%） | 43.444/43.042（2.27%） |
| Qwen2.5-14B-Instruct-1M | 314.460/302.783（5.46%） | 306.598/300.949（13.45%） | 329.721/337.025（21.30%） |
| Qwen3-14B | 42.617/40.598（3.07%） | 42.087/40.536（5.86%） | 41.027/40.324（6.33%） |
| Meta-Llama-3.1-70B-Instruct-AWQ | 115.214/113.649（2.30%） | 112.131/112.803（5.55%） | 104.175/100.769（4.34%） |
| Qwen3-30B-A3B | 93.792/84.435（1.84%） | 96.227/84.475（4.57%） | 97.074/84.359（4.55%） |
| Qwen3-30B-A3B-Instruct-2507 | 93.822/84.276（2.03%） | 95.888/84.334（4.16%） | 96.204/86.473（5.51%） |
| Qwen3-32B | 69.891/68.566（1.70%） | 69.600/68.123（2.57%） | 67.270/65.170（3.32%） |

![同一冻结 ML 预测器的三类样本对照](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/ml_fit_input_vs_independent_mape.png)

逐负载、prefill/decode、样本数和 bundle SHA256 见 ml_fit_input_vs_independent_by_workload.csv；等权汇总及相对训练回评分的百分点差见 ml_fit_input_vs_independent_macro.csv。训练逐行预测保存在 ml_fit_input_readback/，正式独立测试文件不修改。

### 相同特征 batch 的补充核对

为了减少 batch 形状变化的影响，又从独立测试中筛出 18 维 ML 特征与训练样本完全相同的 batch。对每条入选测试 batch，先取相同特征训练样本的平均绝对百分比误差，再按测试集中的特征出现次数加权，得到表中的“训练回评分”；测试误差用同一批入选测试行计算。仍使用原来冻结的模型和已保存的真机时延，没有重新拟合或运行服务。以下是逐 batch MAPE，不是 E2E 误差。

| 模型 | 测试 batch 特征完全匹配比例 | 匹配特征训练回评分 MAPE | 匹配特征独立测试 MAPE | 差值（百分点） |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 8.97% | 3.85% | 6.21% | +2.36 |
| Qwen2.5-3B-Instruct | 8.46% | 1.99% | 3.55% | +1.55 |
| llama-2-7b | 8.33% | 1.84% | 2.61% | +0.78 |
| Mistral-7B-Instruct-v0.2 | 9.40% | 2.94% | 3.95% | +1.00 |
| Qwen2.5-7B-Instruct | 8.51% | 1.80% | 5.15% | +3.35 |
| Qwen3-8B | 8.35% | 1.48% | 3.57% | +2.09 |
| Qwen2.5-14B-Instruct | 7.09% | 1.60% | 3.43% | +1.83 |
| Qwen2.5-14B-Instruct-1M | 7.83% | 6.90% | 6.13% | -0.76 |
| Qwen3-14B | 7.46% | 5.49% | 4.96% | -0.53 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 4.17% | 1.38% | 1.87% | +0.49 |
| Qwen3-30B-A3B | 5.31% | 3.10% | 6.86% | +3.77 |
| Qwen3-30B-A3B-Instruct-2507 | 5.26% | 3.79% | 5.90% | +2.11 |
| Qwen3-32B | 5.81% | 0.90% | 2.58% | +1.68 |

13 模型等权平均：完全匹配比例 7.30%，匹配特征训练回评分 2.85%，独立测试 4.37%，增加 1.52 个百分点；11/13 个模型增加。表中差值由未取整数据计算，可能与显示的两列相减相差 0.01 个百分点。同负载独立请求中的匹配行是 2.85%→4.12%，跨负载中的匹配行是 2.85%→4.69%。这说明即使描述 batch 形状的特征相同，独立采集的误差也不一定维持在训练回评分水平；但匹配行只占少数，且训练值仍是样本内误差。请求内容、未编码的运行状态、采集时段及动态调度没有单独控制，不能将差值全部归因于“换了 prompt”。逐模型、阶段及负载组的原始汇总和只读核对脚本见 `verification/exact_feature_conditioned_ml_20260930_0401/summary_all13.csv`、`audit.py`。

## 单步误差的尾部集中情况

按平方残差排序检查误差集中程度，所有样本仍计入正式评分。表中的占比是总平方误差占比，不是 MAPE 占比；少量长时延可能影响 RMSE，不能据此判断平均相对误差或确定其成因。ML/AIC 已逐条核对来源文件、行号、batch 特征和真实时延，不能仅以样本数与均值相同代替这项检查。

同一组步骤对整体 MAPE 的贡献为其 APE 之和除以全部步骤数，单位为百分点；它们按平方残差选出，不是按 APE 选出。这不是删除样本后 MAPE 会下降的幅度，因为删除会改变分母。

| 模型 | 方法 | 全部步骤 | 实测中位数 / 最大值 ms | RMSE ms | 最大 100 项的实际数量 / 样本占比 | 平方误差占比 | 对整体 MAPE 贡献（百分点） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 19080 | 24.82 / 568.71 | 19.48 | 100 / 0.524% | 99.01% | 0.3068 |
| Qwen3-0.6B | AIC | 19080 | 24.82 / 568.71 | 30.25 | 100 / 0.524% | 45.89% | 0.4940 |
| Qwen2.5-3B-Instruct | ML | 17628 | 30.68 / 563.02 | 20.23 | 100 / 0.567% | 99.14% | 0.2785 |
| Qwen2.5-3B-Instruct | AIC | 17628 | 30.68 / 563.02 | 32.34 | 100 / 0.567% | 43.62% | 0.5108 |
| llama-2-7b | ML | 18500 | 26.22 / 752.19 | 19.28 | 100 / 0.541% | 99.43% | 0.2634 |
| llama-2-7b | AIC | 18500 | 26.22 / 752.19 | 24.28 | 100 / 0.541% | 67.29% | 0.4280 |
| Mistral-7B-Instruct-v0.2 | ML | 18306 | 26.63 / 538.68 | 18.89 | 100 / 0.546% | 99.04% | 0.2570 |
| Mistral-7B-Instruct-v0.2 | AIC | 18306 | 26.63 / 538.68 | 24.75 | 100 / 0.546% | 62.57% | 0.4309 |
| Qwen2.5-7B-Instruct | ML | 19018 | 24.24 / 558.55 | 18.84 | 100 / 0.526% | 98.96% | 0.2512 |
| Qwen2.5-7B-Instruct | AIC | 19018 | 24.24 / 558.55 | 23.55 | 100 / 0.526% | 67.80% | 0.4040 |
| Qwen3-8B | ML | 17291 | 31.63 / 721.24 | 20.19 | 100 / 0.578% | 98.92% | 0.2904 |
| Qwen3-8B | AIC | 17291 | 31.63 / 721.24 | 28.63 | 100 / 0.578% | 53.92% | 0.4621 |
| Qwen2.5-14B-Instruct | ML | 15701 | 40.17 / 547.86 | 20.66 | 100 / 0.637% | 98.95% | 0.3124 |
| Qwen2.5-14B-Instruct | AIC | 15701 | 40.17 / 547.86 | 28.10 | 100 / 0.637% | 58.89% | 0.4619 |
| Qwen2.5-14B-Instruct-1M | ML | 9974 | 112.89 / 2394.40 | 76.03 | 100 / 1.003% | 37.59% | 1.3718 |
| Qwen2.5-14B-Instruct-1M | AIC | 9974 | 112.89 / 2394.40 | 517.20 | 100 / 1.003% | 17.20% | 0.9804 |
| Qwen3-14B | ML | 16069 | 36.18 / 696.09 | 21.60 | 100 / 0.622% | 98.04% | 0.3691 |
| Qwen3-14B | AIC | 16069 | 36.18 / 696.09 | 26.91 | 100 / 0.622% | 67.79% | 0.4039 |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 12674 | 74.71 / 653.13 | 24.13 | 100 / 0.789% | 93.60% | 0.3232 |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 12674 | 74.71 / 653.13 | 40.96 | 100 / 0.789% | 32.59% | 0.3519 |
| Qwen3-30B-A3B | ML | 12223 | 85.27 / 2419.16 | 112.99 | 100 / 0.818% | 96.99% | 0.7352 |
| Qwen3-30B-A3B | AIC | 12223 | 85.27 / 2419.16 | 131.57 | 100 / 0.818% | 77.12% | 0.7744 |
| Qwen3-30B-A3B-Instruct-2507 | ML | 12188 | 85.04 / 2390.11 | 112.06 | 100 / 0.820% | 96.70% | 0.7371 |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 12188 | 85.04 / 2390.11 | 130.44 | 100 / 0.820% | 77.06% | 0.7774 |
| Qwen3-32B | ML | 13889 | 57.01 / 587.95 | 22.47 | 100 / 0.720% | 98.54% | 0.3251 |
| Qwen3-32B | AIC | 13889 | 57.01 / 587.95 | 24.96 | 100 / 0.720% | 84.08% | 0.4010 |

完整排序占比见 residual_concentration.csv；每种方法最大的十个残差及其原始文件/行号见 largest_batch_residuals.csv。未删除异常值、重新拟合或修正任何预测。零总平方误差时，占比记为 0。

## 分负载的训练特征覆盖

下面按 all/prefill/decode 分别计算训练集 18 维特征的逐维最小/最大值，统计每种测试负载至少一维超出范围的 batch 比例。训练范围采用最终 fit 的输入过滤口径；测试保留所有有效正时延，且核对每格样本数与 ML/AIC 正式评分一致。范围内不代表同分布，范围外也不是误差的唯一原因。batch_size、sum_extend、sum_past 的训练/测试中位数和训练范围保存在 workload_feature_coverage.csv。

![all 分负载训练范围覆盖](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/workload_feature_coverage_all.png)

![prefill 分负载训练范围覆盖](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/workload_feature_coverage_prefill.png)

![decode 分负载训练范围覆盖](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/workload_feature_coverage_decode.png)

![训练覆盖率与 ML 误差](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/feature_coverage_vs_ml_error.png)

## 固定新测试集的历史校准对照（仅单步）

下面将 9 月 18 日的 ML bundle / AIC Silicon 表和本轮扩充结果评分于完全相同的本轮真实 batch；逐条核对原始文件、行号、phase、batch 大小、token 统计和实测时延，再核对分组样本数及实测均值。所有来源必须指向本轮测试目录；样本数和均值相同不能替代逐条核对。旧方法原有 E2E 不移入此表。ML 的训练负载覆盖和请求量同时改变，不能把改善全部归因于样本数量；前面的 160/480/800 请求曲线用于固定覆盖下的训练量检查。AIC 重新采集、加密规则长度网格，并将 Triton page size 从旧采集的 64 对齐为普通真机基线的 1；比较的是整体校准效果，不能单独归因于多采样。1M/量化/MoE 的代理限制保持单列。

| 模型 | 方法 | 实测 ms | 旧预测 / 新预测 ms | 旧 MAPE / 新 MAPE | 改善（百分点） |
| --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | ML | 25.97 | 23.59 / 25.88 | 6.07% / 6.21% | -0.14 |
| Qwen3-0.6B | AIC | 25.97 | 2.90 / 2.83 | 88.48% / 88.75% | -0.27 |
| Qwen2.5-3B-Instruct | ML | 31.96 | 29.11 / 31.25 | 6.72% / 4.55% | +2.17 |
| Qwen2.5-3B-Instruct | AIC | 31.96 | 6.97 / 6.96 | 77.49% / 77.54% | -0.05 |
| llama-2-7b | ML | 28.37 | 25.58 / 27.49 | 7.28% / 3.84% | +3.44 |
| llama-2-7b | AIC | 28.37 | 14.41 / 14.23 | 48.92% / 49.61% | -0.69 |
| Mistral-7B-Instruct-v0.2 | ML | 29.03 | 25.88 / 28.60 | 8.45% / 5.42% | +3.03 |
| Mistral-7B-Instruct-v0.2 | AIC | 29.03 | 13.43 / 13.53 | 53.81% / 53.49% | +0.32 |
| Qwen2.5-7B-Instruct | ML | 26.51 | 24.37 / 25.93 | 7.41% / 6.04% | +1.37 |
| Qwen2.5-7B-Instruct | AIC | 26.51 | 12.74 / 12.70 | 51.91% / 52.11% | -0.20 |
| Qwen3-8B | ML | 33.99 | 32.38 / 32.60 | 6.18% / 4.55% | +1.63 |
| Qwen3-8B | AIC | 33.99 | 14.18 / 14.23 | 58.09% / 58.01% | +0.08 |
| Qwen2.5-14B-Instruct | ML | 44.41 | 40.29 / 43.39 | 7.61% / 3.52% | +4.09 |
| Qwen2.5-14B-Instruct | AIC | 44.41 | 26.04 / 26.03 | 42.17% / 42.30% | -0.13 |
| Qwen2.5-14B-Instruct-1M | ML | 310.80 | 281.91 / 300.60 | 14.75% / 12.18% | +2.57 |
| Qwen2.5-14B-Instruct-1M | AIC | 310.80 | 29.47 / 29.58 | 82.44% / 82.54% | -0.11 |
| Qwen3-14B | ML | 41.34 | 36.21 / 40.24 | 10.61% / 6.16% | +4.45 |
| Qwen3-14B | AIC | 41.34 | 25.73 / 25.76 | 38.50% / 38.66% | -0.17 |
| Meta-Llama-3.1-70B-Instruct-AWQ | ML | 111.92 | 105.68 / 110.90 | 8.40% / 4.73% | +3.67 |
| Meta-Llama-3.1-70B-Instruct-AWQ | AIC | 111.92 | 124.48 / 124.70 | 30.24% / 30.35% | -0.11 |
| Qwen3-30B-A3B | ML | 95.98 | 76.73 / 84.32 | 12.14% / 4.62% | +7.52 |
| Qwen3-30B-A3B | AIC | 95.98 | 31.09 / 31.07 | 63.91% / 63.92% | -0.02 |
| Qwen3-30B-A3B-Instruct-2507 | ML | 95.22 | 77.09 / 85.30 | 10.94% / 4.76% | +6.18 |
| Qwen3-30B-A3B-Instruct-2507 | AIC | 95.22 | 31.15 / 31.13 | 63.49% / 63.51% | -0.02 |
| Qwen3-32B | ML | 68.90 | 62.33 / 67.22 | 9.50% / 2.85% | +6.65 |
| Qwen3-32B | AIC | 68.90 | 57.76 / 58.51 | 16.25% / 15.56% | +0.70 |

![固定新测试集的校准对照](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/old_new_calibration_mape.png)

![ML 分负载校准差值](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/ml_old_new_calibration_workloads.png)

![AIC 分负载校准差值](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/aic_old_new_calibration_workloads.png)

## 历史低误差结果的基线变化

9 月 8 日的低 E2E 误差来自 Torch 回退算子路径；9 月 10 日采用 SM80 兼容 native kernel 并更换算子表。两次历史实验都启用 CUDA Graph 和 overlap，context=8192，同一用例重复固定 seed；显存比例从 0.80 变为 0.96。本轮则禁用 graph/overlap、context=2048，并重新采集五个独立 seed。源码 revision 相同不代表 dirty 补丁、kernel 和执行路径相同。这些数据是配置变化的历史对照，不是固定真机基线上的单因素校准实验。

先列饱和用例的实际时延与预测时延。两次历史实验均每轮 32 请求；本轮同名用例为 64 请求，不能仅按用例名称直接相减。

| 历史实验 | 模型 | 真机 E2E ms | AIC E2E ms | E2E MAPE |
| --- | --- | --- | --- | --- |
| 2026-09-08 fallback | Qwen2.5-3B-Instruct | 2258.37 | 2093.47 | 7.30% |
| 2026-09-08 fallback | Qwen2.5-7B-Instruct | 3849.78 | 3653.71 | 5.09% |
| 2026-09-08 fallback | Qwen3-32B | 19448.89 | 17680.21 | 9.09% |
| 2026-09-10 compatible | Qwen2.5-3B-Instruct | 1581.05 | 2229.65 | 42.73% |
| 2026-09-10 compatible | Qwen2.5-7B-Instruct | 4116.33 | 4358.86 | 6.15% |
| 2026-09-10 compatible | Qwen3-32B | 11800.94 | 13137.21 | 11.32% |

三个历史负载等权汇总如下。E2E 小于 20%并不表示首 token 或单步也小于 20%；表中不把饱和用例误差与三负载总误差混写。

| 历史实验 | 模型 | E2E MAPE（三负载） | TTFT MAPE（三负载） | 配对数 |
| --- | --- | --- | --- | --- |
| 2026-09-08 fallback | Qwen2.5-3B-Instruct | 6.74% | 46.52% | 15 |
| 2026-09-08 fallback | Qwen2.5-7B-Instruct | 2.72% | 36.82% | 15 |
| 2026-09-08 fallback | Qwen3-32B | 8.86% | 23.80% | 15 |
| 2026-09-10 compatible | Qwen2.5-3B-Instruct | 55.50% | 31.92% | 15 |
| 2026-09-10 compatible | Qwen2.5-7B-Instruct | 26.44% | 24.69% | 15 |
| 2026-09-10 compatible | Qwen3-32B | 15.17% | 19.57% | 15 |

![历史基线与预测值变化](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/historical_aic_real_and_predicted.png)

历史各负载数值、源码、graph 配置、context、显存比例和固定 seed 策略保存于 historical_aic_baselines.csv。版本、实际算子路径、请求量及单步计时边界都需要对齐后才可归因；仅有同名模型和同一源码提交不够。

## 历史固定形状单步结果与本轮统计口径

以下重新读取两次历史实验的原始五次 one_batch 测量和 AIC 预测 JSON，复算真实均值、样本标准差及误差，并核验与原 CSV 一致。B=1、输入=1024、输出=2，每轮只含一个 prefill 和一个首 decode；启用 prefill piecewise/decode full graph，context=8192。它不是本轮 graph 关闭、动态 batch、随机长度服务请求的同输入对照。

原 single_step_comparison.csv 的 absolute_error_pct 是 |预测−五次真实均值|/真实均值，不是逐次 APE 的平均，也不是本轮逐 batch MAPE。早期 ML/AIC 比较表曾将该字段标成 MAPE；这里按原始公式更正名称，不改写历史文件。逐次 APE 平均另列在新 CSV 的 individual_repeat_mape_pct，仍只代表这五次固定形状测量。历史 std 为样本标准差（n−1），与本轮 E2E 表采用的总体 std 不同。

| 历史实验 | 模型 | 阶段 | 真机均值 ± 样本 std ms | AIC 预测 ms | 均值相对误差 |
| --- | --- | --- | --- | --- | --- |
| Sep 08 fallback | Qwen2.5-3B-Instruct | prefill | 47.424 ± 0.336 | 34.936 | 26.33% |
| Sep 08 fallback | Qwen2.5-3B-Instruct | decode | 10.174 ± 0.086 | 6.169 | 39.37% |
| Sep 08 fallback | Qwen2.5-7B-Instruct | prefill | 87.211 ± 0.498 | 73.340 | 15.91% |
| Sep 08 fallback | Qwen2.5-7B-Instruct | decode | 13.967 ± 0.044 | 10.135 | 27.43% |
| Sep 08 fallback | Qwen3-32B | prefill | 371.144 ± 1.221 | 314.467 | 15.27% |
| Sep 08 fallback | Qwen3-32B | decode | 49.679 ± 0.149 | 43.624 | 12.19% |
| Sep 10 compatible | Qwen2.5-3B-Instruct | prefill | 36.924 ± 0.389 | 41.517 | 12.44% |
| Sep 10 compatible | Qwen2.5-3B-Instruct | decode | 6.893 ± 0.050 | 6.198 | 10.09% |
| Sep 10 compatible | Qwen2.5-7B-Instruct | prefill | 74.558 ± 0.683 | 91.897 | 23.26% |
| Sep 10 compatible | Qwen2.5-7B-Instruct | decode | 11.136 ± 0.033 | 12.097 | 8.63% |
| Sep 10 compatible | Qwen3-32B | prefill | 322.922 ± 1.025 | 411.448 | 27.41% |
| Sep 10 compatible | Qwen3-32B | decode | 43.152 ± 0.127 | 57.393 | 33.00% |

在本表这些固定样本中，两种误差公式的最大数值差为 0.0100 个百分点；这是同一批历史重复的复算结果，不是新旧服务实验的精度差。


历史单步也存在超过 20%的误差；不能把某次饱和负载的低 E2E 误差推广到全部阶段与形状。当前 one_batch 的计时源码与所记录 revision 一致：它同样使用同步墙钟，包含 batch 准备、forward 和 sampling，并非仅 CUDA event 算子时间。当前 collector 则包围 Scheduler.run_batch，二者的调用范围和 graph 设置均不同；没有同配置 profile 时，不能给 CPU、编译或 kernel 差异分配具体误差比例。

![历史固定形状的真实值与预测值](../../../../../../../../../mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/analysis/workloads/historical_aic_one_batch_actual_vs_predicted.png)

原始数据位置、五次样本误差、graph、context、kernel stage、实际命令和预测表来源保存于 historical_aic_fixed_shape_steps.csv；它与本轮独立服务请求表分开列示。

## GPU 状态与采集质量

每次服务启动前记录显存和驻留进程，另每 30 秒采样一次 GPU 功率、频率、温度和进程。下表只统计真实 profile 时间窗；未采到的窗口记为未覆盖，不当作没有干扰。其他进程驻留不等同于同时执行计算，30 秒采样也不能排除短时干扰。

| 模型 | 真实用例总数 | 遥测覆盖用例 | 采到其他驻留进程的用例 | profile 内日志记录的慢编译 |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 40 | 25 | 2 | 0 |
| Qwen2.5-3B-Instruct | 40 | 32 | 0 | 0 |
| llama-2-7b | 40 | 34 | 0 | 0 |
| Mistral-7B-Instruct-v0.2 | 40 | 30 | 0 | 0 |
| Qwen2.5-7B-Instruct | 40 | 28 | 0 | 0 |
| Qwen3-8B | 40 | 37 | 0 | 0 |
| Qwen2.5-14B-Instruct | 40 | 34 | 0 | 0 |
| Qwen2.5-14B-Instruct-1M | 40 | 40 | 0 | 0 |
| Qwen3-14B | 40 | 33 | 6 | 0 |
| Meta-Llama-3.1-70B-Instruct-AWQ | 40 | 40 | 0 | 0 |
| Qwen3-30B-A3B | 40 | 40 | 0 | 0 |
| Qwen3-30B-A3B-Instruct-2507 | 40 | 40 | 0 | 0 |
| Qwen3-32B | 40 | 40 | 0 | 0 |

原始遥测和每用例统计均保留；日志慢编译统计不是所有编译事件的完整计数，不据此自动删去较慢样本。

## 工程回归验证（非 GPU 精度实验）

2026-09-28 在 hbf-cxl 执行 `python -m pytest -q test`：47 项通过、1 项失败、5 条警告，耗时 395.20 秒，退出码 1。

该次全套检查使用 CUDA_VISIBLE_DEVICES='' 的 CPU fixture 服务，运行于 20:19–20:26，与 3B 真机训练采集重叠，不是隔离的性能测试。它不覆盖随后新增的测试；后续局部检查通过不能替代这次全套结果，也不能证明模型预测精度。

失败用例：`test/test_simulation_offline_blocking.py::test_request_rate_offline_matches_blocking`。以下从保存的 fixture 输出重新读取，未修改测量值或容差。

| 指标 | offline | blocking | 相对差异 | 测试容差 |
| --- | ---: | ---: | ---: | ---: |
| mean_ttft_ms | 12.762786 | 11.341969 | 12.5271% | 10% |
| mean_e2e_latency_ms | 39.090425 | 39.187272 | 0.2471% | 10% |
| duration | 1.565132 | 1.565379 | 0.0158% | 1% |

TTFT 相对差异超出原有 10% 容差；本次运行尚未隔离具体原因。未放宽容差，也未修改模拟器来隐藏失败。表格只核对列出的三个保存指标，不表示失败之后的全部断言均已执行。具体环境和保存位置见 verification/verification_20260928.md。
