# Collected SGLang ML replay comparison

## Scope

Each row pairs a real SGLang serving result with a CPU simulator replay of that run's converted raw request trace. The replay uses the same token IDs, completed output lengths, relative arrival times, context length, request limit, chunked-prefill size, and real service token-pool limit. The real server uses device-synchronized collection with overlap scheduling and CUDA graph disabled.

Total token throughput uses actual trace input+output counts divided by the original measured benchmark duration; the benchmark's nominal pre-decode input count can differ after text re-tokenization. Original reported throughput and token counts are retained in comparison.csv. Client and simulator timing boundaries remain different; this correction aligns only token counting.

## End-to-end paired metrics

| Metric | Pairs | Real mean | Real std | Predicted mean | MAPE | Mean signed error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_e2e_latency_ms | 25 | 8629.823 | 6535.732 | 7359.893 | 22.280% | -22.280% |
| mean_tpot_ms | 25 | 55.953 | 18.256 | 45.258 | 21.102% | -21.021% |
| mean_ttft_ms | 25 | 2748.356 | 3272.758 | 2373.644 | 31.617% | -31.617% |
| total_throughput | 25 | 1163.184 | 844.639 | 1310.424 | 9.395% | 9.395% |

## Inputs

1. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/real/repeat-1`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/aic_simulator/repeat-1`
2. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/real/repeat-2`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/aic_simulator/repeat-2`
3. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/real/repeat-3`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/aic_simulator/repeat-3`
4. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/real/repeat-4`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/aic_simulator/repeat-4`
5. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/real/repeat-5`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/holdout/aic_simulator/repeat-5`

## Teacher-forced scheduler-batch score

This score uses the real observed batch composition and therefore does not include scheduler feedback.

- batches: 16069
- MAPE: 38.664%
- MAE: 15.636 ms

## Reading the numbers

The table is descriptive. Teacher-forced batch error and end-to-end replay error answer different questions: the latter also changes future batch formation when a predicted latency differs from the real one. These rows do not isolate every runtime component, such as request handling and host scheduling, from GPU forward time.
