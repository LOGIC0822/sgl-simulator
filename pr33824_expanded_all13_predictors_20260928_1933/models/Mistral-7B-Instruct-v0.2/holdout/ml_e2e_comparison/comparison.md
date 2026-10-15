# Collected SGLang ML replay comparison

## Scope

Each row pairs a real SGLang serving result with a CPU simulator replay of that run's converted raw request trace. The replay uses the same token IDs, completed output lengths, relative arrival times, context length, request limit, chunked-prefill size, and real service token-pool limit. The real server uses device-synchronized collection with overlap scheduling and CUDA graph disabled.

Total token throughput uses actual trace input+output counts divided by the original measured benchmark duration; the benchmark's nominal pre-decode input count can differ after text re-tokenization. Original reported throughput and token counts are retained in comparison.csv. Client and simulator timing boundaries remain different; this correction aligns only token counting.

## End-to-end paired metrics

| Metric | Pairs | Real mean | Real std | Predicted mean | MAPE | Mean signed error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_e2e_latency_ms | 25 | 5703.275 | 4283.567 | 6288.982 | 12.019% | 12.019% |
| mean_tpot_ms | 25 | 37.390 | 8.757 | 42.214 | 13.449% | 13.449% |
| mean_ttft_ms | 25 | 1708.595 | 2075.643 | 1779.479 | 13.542% | -9.290% |
| total_throughput | 25 | 1548.217 | 1188.775 | 1450.986 | 4.409% | -3.479% |

## Inputs

1. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/real/repeat-1`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/ml_simulator/repeat-1`
2. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/real/repeat-2`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/ml_simulator/repeat-2`
3. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/real/repeat-3`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/ml_simulator/repeat-3`
4. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/real/repeat-4`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/ml_simulator/repeat-4`
5. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/real/repeat-5`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/holdout/ml_simulator/repeat-5`

## Teacher-forced scheduler-batch score

This score uses the real observed batch composition and therefore does not include scheduler feedback.

- batches: 18306
- MAPE: 5.425%
- MAE: 2.284 ms

## Reading the numbers

The table is descriptive. Teacher-forced batch error and end-to-end replay error answer different questions: the latter also changes future batch formation when a predicted latency differs from the real one. These rows do not isolate every runtime component, such as request handling and host scheduling, from GPU forward time.
