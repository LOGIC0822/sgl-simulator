# A100 PCIe 三个 Dense 模型预测误差汇总

实验日期：2026-09-08

测试对象：Qwen2.5-3B-Instruct、Qwen2.5-7B-Instruct、Qwen3-32B

## 结论先说

- 三个模型的端到端矩阵全部完成，共有 45 组有效实测/预测对照。按请求端到端时延统计，Qwen2.5-7B-Instruct 的误差最小，E2E MAPE 为 2.72%；3B 和 32B 分别为 6.74% 与 8.86%。
- 这个结果不能直接理解为单个 forward 已经预测准确。固定 shape 的单步实验中，AIC 对六个 prefill/decode 点都偏低，误差在 12.19% 到 39.37% 之间；32B 最接近，3B 的 decode 偏差最大。
- 7B 的端到端 E2E 误差小，主要说明该组请求的总时长与预测较接近；它不等于每类算子都对齐。7B 的单步 prefill 和 decode 仍分别低估 15.91% 和 27.43%。
- trace 核对显示，BF16 配置本身是一致的，已采集的 GEMM 和 Triton FMHA 主核也比较接近。主要缺口来自实际运行中的 RMSNorm、SwiGLU、RoPE/QK norm/KV 写入等辅助 kernel，以及 CUDA Graph 和调度边界没有被 AIC 逐项回放。

本报告只汇总 9 月 8 日已经完成的正式结果；不把中途退出的
`pr33824_piecewise_a100_dense3_matrix_20260908_134210` 目录计入统计，也不把后续 SM80 兼容包的 3B smoke 结果混进来。

## 1. 测试条件和数据口径

| 项目 | 本次设置 |
| --- | --- |
| GPU | NVIDIA A100 PCIe |
| SGLang 源码 | PR #33824，本地提交 `59799a368793b9f795baf59b067233a65ad8e38e` |
| Python / Torch / Triton | 3.10.21 / 2.9.1+cu126 / 3.5.1 |
| AIConfigurator | 0.10.0，A100 PCIe Silicon 表，`axis3d_20260908` 校准版本 |
| 模型精度 | checkpoint、SGLang `--dtype auto`、KV cache 和 AIC GEMM/FMHA 均为 BF16 |
| 真机服务 | 完整 checkpoint；Triton attention；prefill `tc_piecewise` CUDA Graph；decode `full` CUDA Graph；overlap scheduler 开启 |
| 预测器 | PR #33824 的 CPU offline logical-time engine；使用 AIC Silicon 算子时延，不执行物理 CUDA Graph replay |

本地 `sglang` 包显示的 metadata 是 `0.0.0`，不能把它当成发布版号；本报告用上面的 PR 提交号定位源码。32B 首次在 `mem_fraction_static=0.90` 做 prefill CUDA Graph capture 时耗尽显存，正式数据采用 `0.80` 重启后得到，attention 和 CUDA Graph 后端没有改变。

端到端误差使用 `(预测值 - 真机值) / 真机值 × 100%` 计算；MAPE 为绝对误差的均值。表中的负数表示预测偏低。

本轮 AIC 表包含 240 条精确 BF16 GEMM、576 条 prefill attention 和 858 条 decode attention 采样行，采样过程没有报错。真机和采样器均使用同一份 PR 源码中的 `TritonAttnBackend`，采样 attention 行的 `kernel_source=triton`。

## 2. 端到端请求矩阵

每个模型的每个工作负载固定随机 token 和 seed，真机、预测器各完成 5 次；因此共有 `3 模型 × 3 工作负载 × 5 次 = 45` 组有效对照。三个工作负载如下。

| 工作负载 | 请求数 | 输入 / 输出 token | 到达速率 | 用途 |
| --- | ---: | --- | --- | --- |
| `latency_256x128_r1` | 8 | 256 / 128 | 1 | 单请求延迟场景 |
| `service_1024x256_r4` | 32 | 1024 / 256 | 4 | 常规服务负载 |
| `saturation_1024x256_inf` | 32 | 1024 / 256 | 无限 | 饱和吞吐场景 |

### 2.1 模型级误差

下表汇总每个模型的 15 组对照。`±` 后为这 15 次绝对误差的样本标准差。

| 模型 | TTFT MAPE | TPOT MAPE | E2E MAPE | 总吞吐 MAPE |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | 46.52% ± 18.95 | 16.70% ± 15.35 | 6.74% ± 8.10 | 28.06% ± 29.37 |
| Qwen2.5-7B-Instruct | 36.82% ± 14.26 | 2.52% ± 1.73 | 2.72% ± 1.78 | 5.67% ± 2.66 |
| Qwen3-32B | 23.80% ± 18.48 | 8.44% ± 5.08 | 8.86% ± 3.20 | 11.03% ± 5.16 |
| 三模型平均 | 35.71% | 9.22% | 6.11% | 14.92% |

E2E 是三个模型中最稳定的一项，但 TTFT 仍有 23.80% 到 46.52% 的误差。也就是说，当前配置下预测器更适合看请求总时长的量级，不能把它当成首 token 时间已经对齐的依据。

### 2.2 每个工作负载的实测值和预测值

下表只展开 E2E 时延，单位为 ms；每项取该工作负载 5 次运行的均值。它给出了报告中最直接的“真实值”和“预测值”。

| 模型 | 工作负载 | 真机 E2E | 预测 E2E | 带符号误差 | E2E MAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | `latency_256x128_r1` | 581.56 | 598.51 | +2.91% | 2.91% |
| Qwen2.5-3B-Instruct | `saturation_1024x256_inf` | 2258.37 | 2093.47 | -7.30% | 7.30% |
| Qwen2.5-3B-Instruct | `service_1024x256_r4` | 1239.92 | 1088.32 | -10.01% | 10.01% |
| Qwen2.5-7B-Instruct | `latency_256x128_r1` | 671.92 | 681.58 | +1.44% | 1.44% |
| Qwen2.5-7B-Instruct | `saturation_1024x256_inf` | 3849.78 | 3653.71 | -5.09% | 5.09% |
| Qwen2.5-7B-Instruct | `service_1024x256_r4` | 2413.60 | 2374.25 | -1.63% | 1.63% |
| Qwen3-32B | `latency_256x128_r1` | 4092.21 | 3888.36 | -4.98% | 4.98% |
| Qwen3-32B | `saturation_1024x256_inf` | 19448.89 | 17680.21 | -9.09% | 9.09% |
| Qwen3-32B | `service_1024x256_r4` | 14847.83 | 12988.88 | -12.52% | 12.52% |

7B 的三个工作负载都落在 1.44% 到 5.09% 之间。32B 则始终偏低，而且负载越重偏差越明显。3B 的 E2E 不算差，但吞吐和 TTFT 波动较大，说明小模型场景中非 GPU 主算子的影响不能忽略。

## 3. 单步 forward 对照

单步实验使用完整 checkpoint 的 SGLang `ModelRunner`，`batch=1`、输入 1024 token、输出 2 token。每个模型保持相同 shape 做 5 次真机测量；prefill 是一次 extend forward，decode 是首次生成后的一个 decode forward。AIC 对同一模型和 shape 做静态算子组合预测，不包含 HTTP、排队和多请求 scheduler 时间。

| 模型 | 阶段 | 真机均值 ms | 真机标准差 ms | AIC 预测 ms | 带符号误差 | MAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | prefill | 47.424 | 0.336 | 34.936 | -26.33% | 26.33% |
| Qwen2.5-3B-Instruct | decode | 10.174 | 0.086 | 6.169 | -39.37% | 39.37% |
| Qwen2.5-7B-Instruct | prefill | 87.211 | 0.498 | 73.340 | -15.91% | 15.91% |
| Qwen2.5-7B-Instruct | decode | 13.967 | 0.044 | 10.135 | -27.43% | 27.43% |
| Qwen3-32B | prefill | 371.144 | 1.221 | 314.467 | -15.27% | 15.27% |
| Qwen3-32B | decode | 49.679 | 0.149 | 43.624 | -12.19% | 12.19% |

单步实验的规律比较一致：AIC 全部低估，且 decode 的相对误差通常比 prefill 更大。随着模型变大，GEMM 占比提高，遗漏的小 kernel 在总时延中的比例下降，所以 32B 的相对误差最小；这和 3B decode 的 -39.37% 形成了对比。

## 4. 端到端和单步为什么会给出不同印象

两张表回答的不是同一个问题。端到端矩阵比较的是完整请求从进入服务到完成的结果；单步表只看固定 shape 的一次 forward。端到端中的多个阶段、到达间隔和调度等待会一起参与结果，不同阶段的偏差可能相互抵消。因此 7B 的 E2E MAPE 为 2.72%，并不能推出它的单个 prefill/decode 算子已经精确。

为了确认不是数据类型配错，又对同一组单步做了 CUDA trace。结论如下。

- 三个模型的权重、KV cache、真机运行和 AIC 查询均为 BF16，没有发现 BF16/FP16 这类外部配置不一致。
- 已采集的 GEMM 与 trace 中 GEMM 主核偏差在 1.97% 到 8.17% 之间；Triton FMHA 主核偏差在 -15.72% 到 +4.62% 之间。它们不是单步总误差的主要来源。
- 真机任务因为已安装的 `sglang-kernel 0.4.6.post1+cu129` 不支持 A100 SM80 且与当前 Torch CUDA ABI 不匹配，实际走了 Torch 的融合算子回退路径。RMSNorm 被拆成 FP32 cast、归一化和回写等多个 kernel；SwiGLU 也被拆成 SiLU 和 BF16 乘法。
- AIC 中的 `add_norm`、`act_gate` 仍是高层 `ElementWise` 估时，和真实 kernel 边界不同。六份 trace 里，RMSNorm 低估 85.4% 到 90.7%，SwiGLU 低估 32.0% 到 63.2%；decode attention 还没有把 RoPE/QK norm/KV 写入按相同边界计入。
- 在这六个 profile 中，已识别的 GEMM、attention 辅助项、RMSNorm 和 SwiGLU 差异能够解释 `trace GPU sum - AIC total` 缺口的 84.8% 到 95.5%。这说明继续只加密 GEMM 或 FMHA 表，不能把当前误差补齐。

另外，真机使用 CUDA Graph replay，而 AIC 是静态算子组合，并不物理回放 CUDA Graph。这个边界对短请求和小模型尤其敏感，也与 TTFT 仍偏差较大的现象一致。

## 5. 结果的适用范围和下一步

这份表反映的是 9 月 8 日的历史基线，不是兼容 kernel 修复后的精度结论。此次正式矩阵中，真实服务、AIC 采样和预测器虽然都记录了同一份 PR 源码、BF16 和 Triton attention，但 `sglang-kernel` 实际没有加载。后续若要判断校准是否真正改善，必须让以下三部分使用同一套可执行 kernel、Torch/CUDA ABI 和 SGLang 源码：

1. AIC 采样器；
2. 真机 SGLang 基线；
3. CPU 预测器读取的 AIC 表。

最小的可比复验是保持本报告的三个模型、三个端到端工作负载和五次真机重复不变，重新完成 45 组端到端对照；同时重做 `3 模型 × prefill/decode × 5 次` 的 30 个真机单步样本，并用同一版本的 AIC 表查询六个预测点。新结果应单独成表，再与本报告逐项比较，不能只比较 MAPE，还要同时给出真实值和预测值。

## 原始结果位置

- 端到端正式矩阵：`test-results/pr33824_piecewise_a100_dense3_matrix_20260908_134523/`，总表在 `analysis/final_report.md` 和 `analysis/paired_repeat_metrics.csv`。
- 单步 forward：`test-results/pr33824_aic_vs_real_single_step_dense3_20260908_155211/`，总表在 `analysis/single_step_comparison.csv`。
- CUDA kernel 定位：`test-results/pr33824_aic_dtype_op_profile_dense3_20260908_1618/`，结论和明细分别在 `analysis/aic_raw_kernel_root_cause.md`、`analysis/aic_raw_kernel_phase_summary.csv`。
