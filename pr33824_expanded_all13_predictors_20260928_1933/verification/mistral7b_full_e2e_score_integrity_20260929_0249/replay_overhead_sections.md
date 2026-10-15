## 离线回放的计算开销

回放日志同时记录 predicted forward、cpu_overhead 和 predictor_time_cost。当前 OFFLINE 路径以相邻处理点的墙钟间隔推进 CPU 时间，预测函数在该间隔内执行；predictor_time_cost 另行汇总，但没有从 cpu_overhead 中扣除。所以下表是模拟器自身的执行开销，不是独立测得的原生服务 CPU 时延，也不是 GPU 算子时延。依据为本轮归档的 src/sglang_simulator/simulation/sglang/scheduler.py。

均值按回放 iteration 数加权；不同预测方法形成的动态 batch 数可能不同。每个用例、每次重复的明细保存在 replay_wall_time_components.csv。保留原始端到端结果，没有事后扣除平均开销；扣除会改变后续调度，效果需要另行回放验证。

| 模型 | 方法 | 核对用例数 | 回放 iteration 数 | 预测函数墙钟 ms/iter | 计入时钟的 CPU ms/iter | predicted forward ms/iter |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mistral-7B-Instruct-v0.2 | ML | 25 | 17434 | 4.225 | 5.308 | 28.524 |

