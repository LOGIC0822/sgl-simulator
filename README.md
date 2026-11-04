# SGLang Simulator 推理时延实验

本仓库保存 SGLang Simulator 源码快照，以及 13 个模型在 A100 上的机器学习（ML）和 AIConfigurator（AIC）预测实验数据。实验分别记录真实 SGLang 服务、批级时延预测和模拟器请求回放结果。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`sglang-simulator/`](sglang-simulator/) | 模拟器源码、脚本、测试和实验文档 |
| [`pr33824_expanded_all13_predictors_20260928_1933/`](pr33824_expanded_all13_predictors_20260928_1933/) | 13 模型的采集、训练、预测、分析和运行记录 |

源码快照基于 SGLang 提交 `59799a368793b9f795baf59b067233a65ad8e38e`，包含实验期间的本地修改。模拟器的安装和启动方式见 [`sglang-simulator/README.md`](sglang-simulator/README.md)。

## 13 模型实验

真实服务运行在单张 A100 80GB PCIe 上；每个模型单独采集训练数据并训练一个 ML 预测器。每个模型包含 800 个训练请求、1280 个独立测试请求；测试包括训练中出现过的三类负载和新增的两类负载。真实服务每类负载运行五轮。

| 指标 | ML | AIC Silicon |
| --- | ---: | ---: |
| 批级时延 MAPE，13 模型等权平均 | 5.34% | 55.10% |
| 请求端到端时延 MAPE，13 模型等权平均 | 7.66% | 44.01% |

批级 ML 预测拟合同步墙钟 scheduler iteration，AIC 根据算子数据估算；端到端结果使用测试请求 trace 回放。模型、运行参数、实测值、预测值及方法限制见[完整实验报告](sglang-simulator/docs/pr33824_expanded_all13_predictors_20260928_1933_complete_report.md)。

结构化结果：

- [批级时延汇总](pr33824_expanded_all13_predictors_20260928_1933/analysis/teacher_forced_metrics.csv)
- [端到端时延与吞吐汇总](pr33824_expanded_all13_predictors_20260928_1933/analysis/end_to_end_metrics.csv)
- [训练与测试数据覆盖](pr33824_expanded_all13_predictors_20260928_1933/analysis/distribution_coverage.csv)
- [模型级运行数据](pr33824_expanded_all13_predictors_20260928_1933/models/)

本仓库不含模型 checkpoint。完整实验数据约占 10 GB 磁盘空间；运行时还需要匹配的 SGLang、Python 环境、模型和硬件配置。

## LongBench 自助实验

本工作区的独立 `sglang-hbf` 版本提供 LongBench v2 自助实验入口，默认使用长输入、8192-token prefill chunk、五次真机测试；AIC 真机实验启用 CUDA Graph 和 piecewise CUDA Graph。该入口位于 `sglang-hbf` 工作树，与本仓库中的 13 模型源码快照分别维护。

在配置了该工作区的机器上查看参数：

```bash
cd /home/ms-admin/workspace1/yiheng/ali/sglang-hbf/tools/sglang-simulator
test-results/envs/hbf-cxl-conda/bin/python scripts/run_longbench_experiment.py --help
```

本机的 `docs/longbench_self_service_experiments.md` 包含 ML、AIC、数据准备和结果位置的完整命令。运行前需要准备对应模型的 LongBench 分词文件；AIC Silicon 还需要与目标硬件及运行路径匹配的算子数据。
