# A800 八卡预测实验迁移方案

日期：2026-10-05。本文是实施方案，尚未在目标 A800 机器上运行或验证。范围是原生 SGLang 实测、AIConfigurator（AIC）Silicon 采样和 ML 预测对比；HBF 联合模拟是另一项验收，不能用本方案的预测精度结果代替。

## 目标与基准

将现有 A100 单卡实验迁移为可配置的单机八卡实验，支持同一模型在 TP=1/2/4/8 下的实测、采样、训练、预测和报告。先用一组模型验证迁移，再测计划中的大模型；不预设所有模型都能以 BF16 装入八卡。正式比较同时列出实测值、预测值、单步误差和端到端误差，不只列 MAPE。

当前代码基于 SGLang 源码 revision `59799a368793b9f795baf59b067233a65ad8e38e` 与 AIC revision `c5f139662be28b177ca9f62f83543bbf10f133a0`，两处工作树均有未提交修改。现有完整 A100 报告的 13 模型等权均值为：ML 单步 MAPE 5.34%、端到端 MAPE 7.66%；AIC Silicon 分别为 55.10%、44.01%。这些数仅作历史参照，不作为 A800 的预计精度，也不能归因于硬件差异一项。

## 上机前要确认的条件

| 项目 | 在 A800 机器上保存的证据 | 用途 |
| --- | --- | --- |
| 八卡身份与拓扑 | `nvidia-smi -L`、`nvidia-smi -q`、`nvidia-smi topo -m`、GPU UUID、MIG 状态 | 确认实际是 PCIe 还是 HGX/NVSwitch、每卡显存、链路和是否有其他进程；不能仅按“A800”填带宽。NVIDIA 文档列有 [A800 PCIe 80GB](https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/reference/ampere-a800.html) 与 [HGX A800](https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/features/nvswitch.html) 两类。 |
| 软件与源码 | 驱动/CUDA、PyTorch、NCCL、Triton、`sgl_kernel`、SGLang/AIC revision、dirty patch、wheel/`.so` SHA256、Python 导入路径 | 实测 SGLang、AIC 算子采集、模拟器必须使用匹配的算子实现、dtype 和 attention 后端。`sglang` 的包元数据版本不能代替源码 revision。 |
| 模型 | 路径、checkpoint revision/hash、参数量、权重格式、量化方式、配置文件、tokenizer | 计算每 rank 权重及 KV 预算，判定支持的 TP/EP 组合。 |
| 主机资源 | CPU/内存、可用本地磁盘、容器/conda 权限、端口 | 大模型加载、JIT、trace、profile 和 AIC 数据库均占用主机资源。 |

显存预检先算权重下界 `参数量 × 每参数字节数 / TP`，再加入 KV cache、激活、CUDA graph workspace、通信缓冲和碎片；以 **每张卡** 的最小剩余显存和实际服务启动结果验收，不能把八卡显存当作无条件共享内存。例如 70B BF16 权重约 140 GB，TP=8 的理想均分下界约 17.5 GB/卡；这不是整机可运行保证。量化模型另按真实 checkpoint 和 kernel 支持检查。先确认八卡空闲或取得独占测量窗口；共享机器上的干扰进程不主动停止，正式时延测量须记录其占用并标记受干扰轮次。

## 现有脚本的迁移改造

当前脚本**不能直接复制到 A800 后改一个 `--tp-size 8` 就运行**：

| 位置 | 当前限制 | 必须改成 |
| --- | --- | --- |
| `scripts/run_pr33824_expanded_predictor_matrix.sh`、`run_pr33824_comprehensive_predictor_matrix.sh` | 本机 conda/Python、模型目录、旧 AIC/兼容 kernel 和结果目录默认值；固定 13 个模型和 A100 历史对照 | 从版本化实验配置读路径、模型清单、TP/EP、负载和方法；启动前检查实际导入路径；阶段可恢复，不把旧结果当运行依赖。 |
| `scripts/run_pr33824_data_tools_baseline.sh` 与矩阵脚本 | 服务固定 `--tp-size 1`，脚本读取 `TP0.*`；显存比例和门槛按 13 个模型名称写死 | 支持 `CUDA_VISIBLE_DEVICES`、TP/EP、逐 rank 产物和配置驱动的显存预检；校验全部预期 rank 与请求完整性。 |
| `src/sglang_simulator/baseline/scheduler_profile.py` | 诊断 profiler 明确要求单 rank TP0 | 为每 rank 导出独立诊断 trace，并按同步边界计算关键路径；正式基线仍用无 profiler 的五轮测量。 |
| `scripts/write_pr33824_aic_config.py`、`a100_pcie_local_aic010_system.yaml` | A100 单卡容量/带宽、`num_device_per_node=1`、`tp_size=1` | 由实机盘点与通信基准生成 A800 系统配置，设实际 TP/EP/DP 与每卡资源；按机器拓扑保留配置版本。 |
| `scripts/run_pr33824_aic010_all13_collector.sh`、`generate_pr33824_aic_gemm_cases.py`、`run_pr33824_trace_targeted_aic_calibration.sh` | A100 数据库与固定模型白名单、固定形状/代理路径 | 从模型 config、运行 trace 和 TP 分片推导算子形状；在 A800 用相同 kernel/后端重新采 Silicon 表，标出原生、代理、缺失算子。 |
| `tools/append_pr33824_expanded_report.py` 与报告入口 | A100 路径、三模型历史 profile、旧目录依赖 | 报告完全读取本轮 manifest；历史 A100 对照可选且单列，不能混进 A800 正式精度。 |

采集器 `src/sglang_simulator/baseline/collector.py` 已有按 TP rank 命名的逻辑，但这**不等于**当前实验具备多卡正确性；需实测检查每 rank 内容、时间边界和请求 ID 是否可对齐。改造时保留原单卡路径，增加 TP=2/4/8 回归测试。将硬编码的环境路径移入配置与启动参数，打包源码快照、未提交补丁及依赖锁定文件；不要只复制 Git commit，因为当前实验依赖 dirty tree 和兼容 kernel 阶段产物。

## 测量与预测口径

1. **真机原生 SGLang**：固定源码/kernel、模型 dtype、KV dtype、attention 后端、page size、context、CUDA graph/overlap 开关、TP/EP、batch 及到达速率。每组服务预热后做五轮独立请求，保存实际 token trace、每轮服务命令、rank 日志、benchmark 输出和 GPU 遥测。原单卡协议禁用 CUDA graph 与 overlap；若另测生产配置，作为独立实验组，重新采样，不与旧组混算。
2. **单步拆账**：分别记录同步 `run_batch` 墙钟、每 rank 的 GPU 活跃区间及关键路径、NCCL/collective 区间、CPU/调度空隙；并行区间按并集或关键路径算，不把八卡 kernel 时间简单相加。profiling 会扰动时延，只用单独诊断组解释误差，不能替换正式无 profile 的实测标签。
3. **AIC Silicon**：在 A800 上用与真机一致的 `sgl_kernel` 二进制、dtype、attention 实现和算子参数采样；记录算子覆盖率、表键和哈希。对 TP=8 另外量测通信：可用 [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) 的 2/4/8 卡 all-reduce、all-gather、reduce-scatter 尺寸扫描，并与实际 SGLang collective profile 对照。AIC 的算子时间应优先对比 GPU/通信关键路径；AIC 总时间与 `run_batch`/端到端差值另外归因，不能靠增加 GEMM 样本自动修正 CPU bubble。AWQ、长上下文、MoE 等若只走代理实现，单列、不宣称原生精度。
4. **ML**：A800 每个明确的 `(模型, TP/EP, dtype, 后端, graph 设置, 硬件拓扑)` 组合单独训练/评估；A100 bundle 可作为迁移基线，但不可当作新机正式已校准模型。保持训练与测试请求 ID/seed 不重叠；给出同负载独立测试和跨负载测试，以及训练量学习曲线。模拟器重复预测若完全相同可去重，真机五轮不能省略。

## 执行顺序与数据规模

| 阶段 | 做什么 | 放行条件 |
| --- | --- | --- |
| 0. 环境冻结 | 盘点八卡、核实 checkpoint、创建独立环境/源码快照，跑 CUDA/NCCL 与 import smoke | 八卡身份、实际路径和版本齐全；二进制与 Python 来源无混用。 |
| 1. 单卡复现 | 选一款小/中型 dense，在 A800 一卡跑原生请求、采样、预测 | 输出请求数和单步 trace 可核对。八张卡跑八个独立 TP1 作业只用于吞吐调度，不等于 TP8。 |
| 2. 多卡正确性 | 同一 dense 依次 TP=2、4、8，比较各 rank、collective、单步和端到端 | 所有 rank 完整、无死锁、请求 ID 可对齐；TP 变更后的算子形状可采样、可预测。 |
| 3. 大模型试点 | 按实际模型清单选一个能以目标精度/量化运行的大型 dense；再覆盖量化、MoE、长上下文各自支持路径 | 容量、TP/EP、kernel、算子覆盖分别通过；不支持项有明确原因。 |
| 4. 正式矩阵 | 固定测试集后执行每模型每配置的五轮真机、AIC/ML 训练和独立预测 | 文件数、请求数、rank 数、版本和退出码全部通过；失败行不计作有效结果。 |
| 5. 汇报 | 生成总表、误差分解、图和环境清单 | 报告可从本轮产物重建，所有数值有原始文件可追。 |

为了和上一轮保持可比，**可容纳这些负载的模型配置**沿用每模型 800 个训练请求、1280 个独立测试请求（同负载与跨负载），正式真机每类负载五轮。大模型若不能承受原并发或长度，先做容量试点，再在实验配置中定义另一套负载；其绝对时延只与同 trace 的预测比较，不与旧 A100 数值直接相减。试点允许缩小请求数，但不得以试点误差冒充正式结论。正式耗时以第一款模型试跑的请求吞吐、加载/JIT 和 AIC 测点速度外推，再锁定完整排期；目前没有目标机器数据，不给虚假的固定工时。

## 产物与验收

建议新增单一入口配置（如 `test/assets/a800_8gpu_experiment.yaml`）和预检、运行、汇总脚本；这是**待实现接口**，当前文件尚不存在。配置至少包含机器/可见 GPU、源码和 kernel 版本、模型清单、TP/EP/精度、真实/采样后端、负载与 seed、训练/测试划分、输出目录。每次运行生成不可覆盖的 `test-results/a800_8gpu_<时间戳>/`，内含 `manifest.json`、硬件/环境快照、各 rank trace、原始 benchmark、AIC 数据库、ML bundle、预测逐行 CSV、错误表、图和状态文件。正式报告放 `docs/`，Markdown 与宋体 DOCX 内容一致。

完成标准：全部计划模型/配置明确标注“通过、失败、跳过及原因”；通过行的请求完成率 100%、五轮真机有效、所有 rank 可对齐；AIC 原生/代理/缺失覆盖率可核对；ML 无训练-测试泄漏；同时列单步和端到端的实测/预测、MAE、MAPE、有符号偏差、p50/p95 与样本数，并按模型、TP、负载、prefill/decode 分组。图至少包含逐模型误差、实测-预测散点、TP 扩展、单步误差组成和训练量曲线。准确率是测量结果，不预设一定优于 A100。

## 开始实施前需要的四项输入

1. 八卡机器的登录/运行方式，以及 `nvidia-smi -L`、`nvidia-smi topo -m`、每卡显存；目前未知 PCIe/HGX 分型。
2. 目标模型的路径/清单、权重格式；是否只做 dense，还是也做量化、MoE、长上下文。
3. 是否允许在新机器创建独立 conda/container 环境，以及可用磁盘位置；不要覆盖原机器或原实验目录。
4. 目标是复现实验口径（关闭 graph/overlap），还是另加生产配置；两者可同时做，但结果必须分组。
