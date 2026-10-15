# 远端 A100 采集、本机 CPU 预测的一键流程

## 用途

脚本把实验分成两个地点和两个阶段：远端 A100 上运行原生 SGLang 请求并采集 AIConfigurator Silicon 数据；原始 trace 和 AIC 数据库同步回工作区后，在本机 CPU 上训练 ML 预测器、计算 AIC/ML 预测并运行 simulator 回放。默认 `all` 会按顺序完成这两段并生成汇总报告。

目前采集入口限定为 **单张 A100 80GB PCIe、TP=1**。四卡机器一次只暴露配置中的一张卡，不会自动使用 TP=4；AIC 系统数据仍是当前 A100 PCIe profile。检测到 SXM、软件版本不符或 kernel 未从指定兼容目录加载时，采集会在启动模型前退出。

## 运行前准备

远端需要可用的 SSH key/agent、`rsync`、`nvidia-smi`、`curl`，以及已经准备好的 GPU Python 环境和模型目录。本机需要 CPU Python 环境（能导入 PyTorch、scikit-learn、joblib、matplotlib、python-docx 和 AIC 0.10.0），AIC checkout、SM80 kernel 兼容 stage，以及 `LOCAL_MODEL_ROOT` 下各模型的配置文件。脚本负责复制本次源码快照、AIC 源码、kernel stage 和实验数据；它不负责安装 CUDA 驱动、编译/安装 kernel 或下载模型。

远端版本检查按已有完整实验环境设置：Python 3.10.21、PyTorch 2.9.1+cu126（CUDA 12.6）、SGLang 源码包元数据 0.0.0、`sglang-kernel` 0.4.6.post1+cu129、Triton 3.5.1、FlashInfer 0.6.18、Transformers 5.12.1、NumPy 1.26.4、scikit-learn 1.6.1、joblib 1.4.2、AIConfigurator 0.10.0。AIC collector 的配置字段仍使用 SGLang backend 0.5.10；它不是本机原生 SGLang 的包版本。脚本会把实际导入路径、包版本和 staged `.so` 的 SHA256 存档；当前代码快照的远端 Git tree hash 也会写入结果目录。版本或硬件不同，需要先明确调整兼容基线，脚本不会静默放行。

## 一键运行

在本机复制并编辑配置文件。`REMOTE_HOST` 可写已配置好的 SSH alias；不要把密码或私钥写进配置。

```bash
cd /home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator
cp scripts/remote_a100_experiment.env.example scripts/remote_a100_experiment.env
```

至少设置 `REMOTE_HOST`、`REMOTE_ROOT`、`REMOTE_PYTHON`、`REMOTE_MODEL_ROOT`、`REMOTE_GPU_ID`、`LOCAL_PYTHON` 和 `LOCAL_MODEL_ROOT`。然后先检查计划，再启动：

```bash
bash scripts/run_pr33824_remote_a100_experiment.sh \
  --config scripts/remote_a100_experiment.env --run-id a100_trial_01 --dry-run

bash scripts/run_pr33824_remote_a100_experiment.sh \
  --config scripts/remote_a100_experiment.env --run-id a100_trial_01
```

默认测试配置中的 13 个模型。每个模型采集 5 轮训练集（每轮 160 个请求，共 800）和 5 轮独立 holdout（每轮 256 个请求，共 1280）；holdout 不参与 ML 拟合或 AIC trace-target 采样。训练负载为 `latency_256x128_r1`、`service_1024x256_r4`、`saturation_1024x256_inf`；holdout 在此基础上增加 `latency_128x64_r1` 和 `saturation_1536x384_inf`。修改模型、轮数或 workload 时，在同一次采集与预测中使用同一配置。

## 分阶段与恢复

如果需要先单独采集、检查结果后再预测，使用相同 run ID：

```bash
bash scripts/run_pr33824_remote_a100_experiment.sh --config scripts/remote_a100_experiment.env \
  --run-id a100_trial_01 --phase collect
bash scripts/run_pr33824_remote_a100_experiment.sh --config scripts/remote_a100_experiment.env \
  --run-id a100_trial_01 --phase predict
```

`collect` 会在远端完成真实请求采集和 AIC Silicon 校准，并将数据同步回本机；`predict` 不连接远端，只用本机 CPU 训练/评分/回放。中断后不要换 run ID；确认远端目录属于本次实验后，使用相同 `--run-id` 加 `--resume`。脚本会拒绝覆盖非空结果目录。

## 产物位置

默认结果根目录为 `tools/sglang-simulator/test-results/a100_remote_<run-id>/`，其中保存请求 trace、server 日志、硬件与环境快照、AIC 数据库、ML bundle、预测与回放 CSV。部署的源码快照在 `deployment/` 下；最终 Markdown 和 DOCX 报告位于部署快照的 `tools/sglang-simulator/docs/`。本机预测与回放命令会将 `CUDA_VISIBLE_DEVICES` 置空，并以 CPU 模式启动 simulator。
