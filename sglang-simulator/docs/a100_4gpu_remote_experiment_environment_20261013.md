# 四卡 A100 远程实验环境清单

日期：2026-10-13。依据 2026-09-28 完整实验目录和当前 simulator `pyproject.toml` 汇总。目标机器尚未 SSH 检查，以下远端条件均待实机核实。

## 现有实验实际使用的环境

| 组件 | 完整实验记录 | 说明 |
| --- | --- | --- |
| GPU | 单张 NVIDIA A100 80GB PCIe，SM80；driver 535.230.02；`nvidia-smi` 显示 CUDA 12.2 | 不是四卡结果。当前租用机须另外确认是 PCIe 还是 SXM、MIG 状态和卡间拓扑。 |
| Python / PyTorch | Python 3.10.21；PyTorch 2.9.1+cu126，PyTorch CUDA 12.6 | 与 `nvidia-smi` 显示的 CUDA 版本分开记录。 |
| SGLang | 源码 checkout revision `59799a368793b9f795baf59b067233a65ad8e38e`（PR #33824 simulator 合并源码）；`sglang` 包元数据为 `0.0.0` | 实验导入的是源码，不是可准确称作 0.5.18 的发行版。工作树含实验改动，迁移时需要源码快照、dirty patch 和未跟踪文件，不能只 clone 此 commit。 |
| SGLang kernel | runtime `0.4.6.post1`；包元数据 `0.4.6.post1+cu129`；通过隔离的 SM80 AOT 兼容目录加载 | `sgl-kernel` 并非从普通安装包加载。部署时要核对真实 `.so` 来源和 SHA256，并做模型 smoke；不能只照抄版本字符串。 |
| 其他运行包 | Triton 3.5.1；FlashInfer 0.6.18；Transformers 5.12.1；NumPy 1.26.4 | 真实实验的依赖快照见 `test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen2.5-7B-Instruct/calibration/real/repeat-1/package_versions.txt`。 |
| AIC | AIConfigurator 0.10.0，SILICON；collector backend 配置字段为 SGLang 0.5.10；AIC checkout revision `c5f139662be28b177ca9f62f83543bbf10f133a0` | `backend_version=0.5.10` 是 AIC 配置值，不代表这次原生服务是 SGLang 0.5.10。AIC collector 工作树也有本地修改，需保存 patch 与源码哈希。 |
| 实测设置 | attention 按模型选后端（Triton；1M 用 dual-chunk FlashAttention）；关闭 CUDA graph 与 overlap；context 2048；max running requests 32；chunked prefill 512；1M page size 256 | 目标机 TP=1 复现时沿用对应模型的设置。生产 graph 配置须作为另一组采集。 |

上述版本与命令来自完整实验根目录 `test-results/pr33824_expanded_all13_predictors_20260928_1933/run_metadata.txt`。该轮共 13 个模型，每模型训练 800 个请求、独立测试 1280 个请求，真机测量五轮。旧误差数据来自单卡 A100 PCIe；TP=4 是新配置，不能直接套用旧的 AIC 数据库或 ML bundle。

## 四卡机器需要先核对

先通过 SSH 执行只读盘点并保存输出：

```bash
nvidia-smi -L
nvidia-smi
nvidia-smi -q
nvidia-smi topo -m
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); print(torch.cuda.nccl.version())'
```

还需记录 OS/内核、驱动、CPU/内存、磁盘空间、网络下载权限、MIG 状态、四卡 UUID 与剩余显存、NCCL 版本，以及模型 checkpoint 的实际路径。四张 80GB 卡的 320GB 是总显存容量，不是单一共享显存池。大模型能否运行需按权重精度、TP 分片、KV cache 和运行时开销逐卡验算。若四卡机器是不同互联拓扑或 PCIe 代际，通信时延也要重新测。

## 推荐的执行顺序

1. 远端先盘点硬件和软件，不安装、不覆盖既有环境；确认工作目录、模型路径、可用空间和 GPU 无其他任务占用。
2. 用一张卡、TP=1，在目标机复跑小模型和原 workload，验证 SGLang 启动、请求采集、算子导入路径和结果文件完整性。这一阶段用于与本地旧 A100 数据核对。
3. 对目标大模型启用 TP=4 后，单独记录命令、四个 rank、NCCL 拓扑、每卡显存、同步单步与端到端时延。TP=4 每个模型/精度/attention 后端需要重新生成 AIC Silicon 数据，并为该配置单独采 ML 训练集和独立测试集。
4. 将原始 trace、AIC 数据库、ML bundle、版本与硬件快照传回工作区，在本机 CPU 上执行训练后分析、AIC/ML 预测评分、模拟器回放、图表和报告。CPU 阶段不需要四卡 GPU；AIC Silicon 算子实测与原生 SGLang 推理必须在远端 GPU 上完成。

模拟器项目基础依赖在 `pyproject.toml` 中为 NumPy、scikit-learn、joblib；AIC 预测/配置工具另需 AIC 0.10.0 及其依赖。CPU 侧应使用独立 Python 环境，先验证能够载入本轮数据库、模型配置和 predictor bundle，再完整回放 trace。模型权重通常只需留在远端跑原生服务；本机 CPU 侧保存配置、请求 trace 和预测器产物即可，若所选工具要求 tokenizer 或额外模型配置，再补充复制对应小文件。

## SSH 接入

可以通过 SSH 在租用机执行硬件检查、安装/启动用户态环境、SGLang 基线、AIC 采集和推理实验，并将结果传回当前工作区。前提是本工作区的执行环境能访问租用机，而且认证方式已配置可用。

请提供主机/IP、SSH 端口、用户名、是否需要跳板机，以及远端模型和结果目录。认证请使用已加入远端 `authorized_keys` 的公钥或本工作区可访问的 SSH agent；不要在聊天里发送密码、私钥或云平台 token。收到连接信息后，先做只读预检和连通性测试，再根据盘点结果确定环境与模型矩阵。远端如不能访问外网，需要预先提供可用的 wheel/源码/模型缓存位置。
