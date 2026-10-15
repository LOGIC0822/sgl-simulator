# 14B-1M attention 路径核对

核对日期：2026-09-29。这里只检查实际配置、源码和加载路径，没有新增 GPU 实验，也没有调整预测值。

## 结论

Qwen2.5-14B-Instruct-1M 与普通 14B 的主要维度相同，但本轮 attention 执行路径不同。1M 模型采用 dual-chunk attention 和本地 FA2 兼容库；AIC 采集脚本采用普通 Triton attention、page-size=1。这一行需要按后端代理比较报告，不能当作完全对齐后的算子预测误差，也不能用它解释其他普通 dense 模型的误差。

## 实际配置

| 项目 | 普通 14B | 14B-1M |
| --- | --- | --- |
| hidden / intermediate size | 5,120 / 13,824 | 5,120 / 13,824 |
| 层数 / query heads / KV heads | 48 / 40 / 8 | 48 / 40 / 8 |
| 本轮服务 context-length | 2,048 | 2,048 |
| 原生 attention | Triton | dual_chunk_flash_attn + FA2 |
| checkpoint dual-chunk 配置 | 无 | chunk-size=262,144；local-size=8,192 |
| 实验是否测百万 token 上下文 | 否 | 否 |

1M checkpoint 的 `max_position_embeddings=1,010,000` 是模型配置上限，不是本轮测试长度。实际 chunk 有效长度为 `262144 - 8192 = 253952`，远大于服务的 2,048 上下文。

## 源码中的实际路径

- prefill 的非稀疏分支先遍历非空请求，再遍历 query head；每个 head 调用 `_dual_chunk_flash_attn_prefill_func`。按 TP=1、48 层和 40 heads 展开，单个非空 prefill 序列对应 **1,920 次该子函数调用**。这是源码调用结构，不是 GPU trace 的 kernel 计数或实际计时。
- prefill 中还包含 KV 页索引与 gather、长度数组 `.cpu().tolist()`、缩放标量 `.item()` 及输出写回。它们位于原生 `run_batch` 的同步墙钟计时范围内。
- 当前配置没有 sparse attention 配置，走非稀疏分支。短上下文也未跨过 253,952 的 chunk 边界；decode 只进入 intra-attention，不调用 successor/inter-attention。不过缩放、mask、stack 和 LSE 归并仍在源码路径中，不能概括为“始终执行三次 attention”。
- FA2 兼容 shim 转接已有的 paged-KV API，并拒绝不支持的 FA3 参数。检查到的 `flash_attn_with_kvcache` 最终调用 `flash_attn_2_cuda.fwd_kvcache`；不是把该接口替换成逐请求的 Torch attention。

## 对实验比较的影响

相同参数量、相同 `sgl_kernel` 版本，不代表相同 attention 调用图。本轮 1M 原生服务仍使用 kernel-v4，另外加载 FlashAttention-v6；后者不是另一个 `sgl_kernel` 版本。AIC 的普通 attention 采样不能仅靠增加相同后端的采样点就覆盖上述不同调用图。

这些事实证明比较路径不等价，但没有给出这些分支各耗时多少，也不能把 1,920 次子函数调用当作 1,920 个已测量 CUDA kernel。不能据此增加固定延迟、按调用数直接放大 AIC 预测，或调整正式结果。原生基线保持不变；其余普通 dense 模型仍需同配置 GPU profile 和算子级比较。

## 可定位证据

实际源码位置：

- `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`：207–333 行元数据；335–519 行 forward；704–860 行逐请求/逐头 prefill；861–1328 行 chunk 分支；1467–1584 行 decode。
- `tools/sglang-simulator/scripts/collector_shims/sitecustomize.py`：290–400 行 FA2 转接与参数拒绝。
- `pr33824_sm80_fa2_compat_v6_20260918_1305/stage/flash_attn/flash_attn_interface.py`：默认 CUDA 扩展导入，以及 1923 行 `fwd_kvcache` 调用。
- AIC `collector/sglang/collect_attn.py`：Triton backend、普通 RadixAttention 和 graph 计时；simulator `time_predictor/aiconfigurator.py` 构造标准 AIC ModelConfig。SDK 源码搜索未找到 dual-chunk 对应配置；这不是对其他 AIC 版本的支持范围判断。
- 两个 checkpoint 的实际 `config.json`、1M 的 `model_settings.txt`、已完成组的原生命令和版本记录。

08:18:59 的原生服务进程 **1772158**，以及其 scheduler 子进程 **1772222**，对应本轮训练 group 3 的 service 用例。已读取服务进程的 memory maps，确认 kernel-v4 `common_ops.abi3.so` 与 FlashAttention-v6 `flash_attn_2_cuda` 同时加载；加载本身不等于已经完成算子级 GPU profiling。

旧 service 的 scheduler 后来正常退出，不能再读取其 `/proc`，也未因此重启实验。08:23:19 转而核对当前 saturation 服务 **1773508** 的实际 scheduler **1773568**（启动时间 08:20:05），其 memory maps 同时包含上述两套库。原始过滤后的映射和父子进程记录保存于 `qwen25_14b_1m_scheduler_libraries_20260929_0823.json`。08:23:59 的环境白名单核对显示 `SGLANG_SIMULATOR_FORCE_FA2_FLASH_ATTN=1`，没有 `FLASH_ATTENTION_TRITON_AMD_ENABLE`；按已检查代码走默认 CUDA 扩展导入，而非 AMD Triton 开关。

SHA256：

| 文件 | SHA256 |
| --- | --- |
| dual-chunk backend | `15a3c811cdec1dd0499d6dfe924971db509ab3a7e1fb231786add79c6b605bbe` |
| collector shim | `61b7d823cac5591d82edd0fd29faf6f8e19944dbe027e4985238de58d740e090` |
| 普通 14B config | `0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e` |
| 14B-1M config | `4d5b7a229eb3144cfe1e42f830d3a46d93f52a36171ed5d541dcf2f77704ff00` |
| FA2 Python interface | `6f9b88a0dc9cba21c1f85805e504fb47de866c67239370336b5771b5016e4d8e` |
| FA2 CUDA binary | `3ba4d5f5ee583c3f4224a607de710f1d2429b10d89cbad0c1b2248799b56f79f` |
| simulator AIC adapter | `8605d0b7a2b4d66266a9cad2caa48648e1e54bf72f192b0e75d4b2743f340aa8` |
