# 全模型扩充采样与预测对比方案

日期：2026-09-28。此文为执行方案，不是新实验结果。

## 范围与已有数据

本轮沿用 PR #33824 合并源码和已有 A100 SM80 兼容 kernel，在 `hbf-cxl` 环境运行。模型已从 `/mnt/data/models` 迁移到 `/home/ms-admin/data/models`，不重新下载，不修改 checkpoint。13 个模型全部纳入，包含两个 Qwen3 MoE、14B-1M 和 Llama-3.1-70B AWQ，启动路径沿用上一轮已通过的模型级设置。

2026-09-18 的完整实验每个模型训练、测试各 320 个请求；32B 的训练单步样本为 2359 条。560 条是更早实验的数据量，不能当作当前训练量。旧实验保留，不覆写。

## 请求预算与划分

每个模型预算完全一致。每类负载进行五轮真机采集；不同模型使用相同 seed 和长度采样规则，训练与测试 seed 分开。

| 负载 | 输入/输出长度上限 | 到达速率（请求/秒） | 每轮请求 | 训练请求 | 独立测试请求 |
| --- | --- | --- | --- | --- | --- |
| latency_256x128_r1 | 256 / 128 | 1 | 32 | 160 | 160 |
| service_1024x256_r4 | 1024 / 256 | 4 | 64 | 320 | 320 |
| saturation_1024x256_inf | 1024 / 256 | 一次送入，无速率限制 | 64 | 320 | 320 |
| latency_128x64_r1 | 128 / 64 | 1 | 32 | 0 | 160 |
| saturation_1536x384_inf | 1536 / 384 | 一次送入，无速率限制 | 64 | 0 | 320 |
| 合计 | — | — | — | 800 | 1280 |

`random_range_ratio=0` 沿用历史规则；实际长度以保存的原始请求为准，不将上限当作每条请求的固定长度。前三类构成同负载独立测试，后两类用于跨负载测试；后两类并非所有特征都超出训练范围，应另报特征覆盖率。五轮合计请求数相同，不强行让 scheduler batch 数相同：batch 数是模型实际调度形成的时延样本数。

## 采集与预测

- ML：采集真实 `Scheduler.run_batch` 同步墙钟时延，包含该函数内的 CPU 发射和同步等待，不等同于纯 GPU kernel 时间。禁用 overlap/CUDA graph，context=2048、max_running_requests=32、chunked_prefill=512；模型差异参数、命令和版本写入每轮目录。
- 训练：仅使用训练 trace，保存 HGB p50、L2 两个 bundle；正式比较沿用预先指定的 p50，不根据测试误差选模型。以 160、480、800 个训练请求绘制学习曲线，固定独立测试集，观察增加样本是否实际降低误差。
- AIC：重新构建本机 Silicon 表，使用相同源码和兼容 kernel，补采训练 trace 对应的 attention/GEMM 形状及规则插值网格。不读取测试时延选点或调参；相同硬件、后端与形状的测点允许复用并记录。算子数与请求数分别统计。
- AIC 测量为算子 GPU 计时，ML 标签为同步墙钟时间；增加算子点不能自动补上 CPU 开销。AWQ 的 BF16 代理、1M 的 attention 代理、MoE 路由覆盖限制必须单列，不能称为原生精确适配。
- 当前 OFFLINE 回放把预测器调用的墙钟耗时计入逻辑 CPU 时间，`predictor_time_cost` 仅单独汇总，没有从时钟扣除。这是模拟器自身的计算开销，不是原生服务 CPU 开销的独立测量。报告另列每 iteration 的预测器耗时、记账 CPU 时间和预测 forward；保持原始端到端指标，不事后减去均值，也不在本轮中途改变 scheduler 计时语义。
- AIC 0.10 的 ElementWise（含 RMSNorm、SwiGLU）不查本机算子表，仍调用 `query_mem_op` 的经验带宽公式。本轮保留 SDK 行为，当前参数为 1.935 TB/s、缩放 0.8、固定延迟 3 μs；attention/GEMM 采样加密不校准这些解析项。依据：[ElementWise 源码](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/src/aiconfigurator/sdk/operations/elementwise.py)、[query_mem_op 源码](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/src/aiconfigurator/sdk/perf_database.py)。
- Qwen3 的 QK norm 在 `ContextAttention` 中作为解析项附加，在 `GenerationAttention.query` 中没有对应附加项；attention harness 只计时 RadixAttention，不包含模型层前面的 QK norm/位置编码调用。本轮先保留 SDK 行为，并把该覆盖边界写入报告，不用测试集拟合补偿。CPU 调用检查只验证公式与调用路径，不是 GPU 时延采样。依据：[attention 查询源码](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/src/aiconfigurator/sdk/operations/attention.py)、[harness 源码](https://github.com/ai-dynamo/aiconfigurator/blob/c5f139662be28b177ca9f62f83543bbf10f133a0/collector/sglang/collect_attn.py)。
- 历史校准对照：将 9 月 18 日的 ML bundle 和 AIC Silicon 表直接评分于本轮同一独立测试 batch，比较旧/新预测值与误差。该对照只做单步，不借用旧真机 E2E；ML 的训练数量和负载覆盖均有变化，不能把差值单独归因于请求量。评分等全部真机采集结束后再执行，不额外占用 GPU。

## 输出与验收

报告同时给出单步 prefill/decode 与端到端 E2E、TTFT、TPOT、吞吐的实测值、预测值、MAPE、MAE、偏差和样本数，按同负载与跨负载分别列出。绘制训练量曲线、预测/实测、模型误差、负载误差热图和采样覆盖图；历史退化后端结果只作有配置说明的参考。

正式 Markdown 和宋体 DOCX 内容一致，日志和数据保存在本轮独立 `test-results/` 目录。只有 13 个模型的采集、训练、AIC 评分、独立请求回放与报告均通过才称为全量完成；等待显存、失败、代理预测不写成完成。其他 GPU 进程不停止，逐项检查显存，不足时每十分钟复查，并记录干扰进程。

启动入口：`bash scripts/run_pr33824_expanded_predictor_matrix.sh`。恢复使用原目录加 `--resume --results-root <原目录>`；旧目录不直接用于本轮。

## 本轮环境与显存记录

启动检查发现用户级 `Torch 2.14.0+cpu` 遮蔽了 conda 的 CUDA Torch，因此本轮显式设置 `PYTHONNOUSERSITE=1`。向 `hbf-cxl` 补齐了原先位于用户目录的采集、训练、绘图及报告依赖，没有替换 Torch、CUDA 或编译 kernel；实际依赖版本以每轮 `package_versions.txt` 为准。一次安装尝试因指定的 certifi 版本不存在而失败，随后采用本机原有的 2025.1.31，后一次安装成功。

首次显存检查时其他进程已占约 65GB，0.6B 尚未采样便进入等待。预检一度尝试 0.6B/3B 显存比例 0.15/0.17，未形成有效训练样本；正式参数见下段。与旧实验相比不能把全部结果变化归因于请求量，训练量曲线才是本轮固定配置、固定测试集下的对比。

服务启动还需要用户目录中的 `gguf` 等可选适配依赖。本轮显式排列 Python 路径为源码/兼容 kernel、conda site-packages、用户 site-packages 后备路径，保证 CUDA Torch 和核心依赖优先从 conda 载入；用户目录只补充 conda 未安装的包。真实采集与 AIC 采用相同的后备路径策略，不称为完全独立于用户目录的环境。原始等待和未产生样本的 gguf 导入失败保存在本轮 `attempts/`，不计入训练或测试。

进一步核对源码 `kv_cache_configurator.py::_profile_available_bytes` 后确认，`mem_fraction_static` 作用于加载前空闲显存，不是卡的物理总量。0.15/0.17 是预检阶段的临时配置，正式矩阵调整为 0.6B=0.65、3B=0.80；其他模型保持历史比例。等待脚本改用模型级空闲显存门槛，小模型分别为 6/12 GiB，30B/32B 为 72 GiB。门槛是保守启动估计，最终容量和 KV token pool 仍以实机启动记录为准。

预热卡住的原因是 PATH 中用户级 Ninja 包装器递归调用 `/usr/local/bin/ninja`；首个 forward 正在编译 QK norm 而不是推理。已将 conda 环境的 Ninja ELF 二进制放到 PATH 前面，仅清理经 PID、命令与编译目录核实属于本轮的编译子进程。正式采样不启用 IO 调试；本轮共享 JIT cache 仅复用相同源码、硬件与类型的编译产物，不复用请求时延标签。

预检 `qwen06_ninja_fixed` 的首个用例导出了 8 个请求、53 条同步单步记录；第二个服务启动时，旧递归编译进程尚未清理完，系统发生主机内存 OOM，服务被终止。随后按旧尝试的精确编译目录清理残留，确认匹配进程数归零后再恢复矩阵。这组预检和未完成的正式尝试均不计入训练/测试；日志保留，成功预检的编译产物仅用于减少重复编译。

正式矩阵第一用例 `Qwen3-0.6B / training-repeat-1 / latency_256x128_r1` 已核对 32 个请求全部成功，导出 915 条单步时延。基线导入的是合并仓库源码，SGLang 安装元数据版本为 `0.0.0`，不能据此当作正式发行版本；源码 revision 才是本轮身份。kernel 运行版本为 `0.4.6.post1`，包元数据为 `0.4.6.post1+cu129`。AIC 导入的是 conda 中的 0.10.0 包，已逐一对比 133 个 Python 源文件，与声明的 AIC checkout 一致。源文件快照、dirty diff、kernel SHA256 和依赖路径保存在本轮 `provenance/`。

由于 7B/8B 沿用的比例为 0.35/0.42，仅空闲 24/32 GiB 并不保证能分配权重和 KV；等待门槛进一步统一到 56 GiB。0.6B/3B 仍为 6/12 GiB，14B、AWQ、Llama-2-7B 为 56 GiB，30B/32B 为 72 GiB。这里只设置保守门槛，不修改 checkpoint 或测量过程中固定的显存比例；最终 replay 使用每用例真实分配的 `max_total_num_tokens`，并校验请求和 token 总数一致。GPU 连续遥测每 30 秒记录，状态汇总仍每十分钟更新。

0.6B 第四组的三个用例均已采集完成，但运行中修改 Bash 文件使旧进程在循环后读到了错误偏移，未写组完成标记。逐项核验 160 请求、2526 行、原始/转换 token、profile 起止和用例状态后恢复组标记，过程记录在该组 `RECOVERY_VALIDATION.md`，不重写测量数据。随后在当前 3B 采集组结束后重启队列，运行中的 Bash 文件不再改动。

请求生成器的 `random-ids` 默认先将随机 ID 解码成文本，服务再编码；benchmark 的输入计数沿用解码前长度，与真实输入 ID 数可有差异。本轮始终回放 collector 保存的真实 ID。总 token 吞吐比较统一为 `(trace 实际输入+输出 token)/各自原始 duration`，保留 benchmark 原始计数和原报吞吐，不改变实测 E2E/TTFT/TPOT。实机 client 与模拟器的时间窗差异另行说明，此修正仅统一 token 分子。

AIC 规则长度网格在原网格上加入步长 64 的测点，prefill 从 21 个长度增至 38 个，decode 覆盖到 2304（补足 AIC 的 0.9–1.1 长度取样范围）。batch 仍为 1–32，不读取测试 trace 选点。旧、新 AIC 表在同一新测试集上比较，差异包含重新测量、训练形状覆盖及网格密度变化，不能只归因于多发请求。算子采集前以原生库 SHA256 核对正式真机采集快照，并验证实际导入路径；若 kernel 已变化则停止采集，不混用数据。算子阶段保留 56 GiB 的保守空闲门槛，十分钟检查一次，不停止其他任务。

进一步核对发现，旧 AIC attention harness 的默认 `page_size=64`，当前 Triton 后端会按 `PAGE_SIZE` 专门化 kernel；普通真实基线实际为 `page_size=1`，两者不能写成完全相同配置。本轮采集用进程内适配显式设为 1（不修改 AIC 仓库），`triton_attention_num_kv_splits=8` 与基线一致。1M 原生 dual-chunk/page=256 仍为 Triton 代理，不按原生对齐解释。旧新 AIC 差异同时含 page size 对齐与采样加密。GPU 计时 helper 的运行次数为自适应，attention/GEMM 分别最低 20/6 次，不能把最低次数写成实际固定重复数。

长用例做了 13 模型 × 5 seed × 64 请求、共 4160 请求的 tokenizer-only 预检。没有输入自身达到 2048，但 10 个 Qwen checkpoint 各有 5 个请求在文本再编码后输入+输出超过 2048；Llama、Mistral、AWQ 的这批候选未超限。正式 `saturation_1536x384_inf` 统一改为 `--tokenize-prompt` 直接发送随机 ID，因此输入+请求输出最多 1920；不启用自动截断，不修改 context=2048，也不修改其他四类用例。长用例相对训练同时改变了输入路径，结果解释须明确这个差异。预检仅检查请求边界，不是新增真机训练数据。

AIC checkout 在本轮开始前已有三处 collector 修改：两份 attention YAML 加入 head=28，`collect_attn.py` 补充 runner 兼容字段及可选 Torch SDPA 分支。本轮不修改该 checkout，正式采集显式选择 Triton，不进入 SDPA 分支。原先只检查了 SDK 的 133 个文件；现补充保存 collector/SDK 源码归档、AIC dirty patch 和 SHA256，使实际采集代码可追溯。该补充不改变已采样的模型时延或 kernel。
