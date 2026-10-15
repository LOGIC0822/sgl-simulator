# Collected SGLang ML replay comparison

## Scope

Each row pairs a real SGLang serving result with a CPU simulator replay of that run's converted raw request trace. The replay uses the same token IDs, completed output lengths, relative arrival times, context length, request limit, chunked-prefill size, and real service token-pool limit. The real server uses device-synchronized collection with overlap scheduling and CUDA graph disabled.

Total token throughput uses actual trace input+output counts divided by the original measured benchmark duration; the benchmark's nominal pre-decode input count can differ after text re-tokenization. Original reported throughput and token counts are retained in comparison.csv. Client and simulator timing boundaries remain different; this correction aligns only token counting.

## End-to-end paired metrics

| Metric | Pairs | Real mean | Real std | Predicted mean | MAPE | Mean signed error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean_e2e_latency_ms | 25 | 28531.642 | 21805.854 | 28523.307 | 3.176% | 0.834% |
| mean_tpot_ms | 25 | 139.422 | 47.976 | 172.910 | 21.356% | 19.936% |
| mean_ttft_ms | 25 | 13090.897 | 12133.145 | 11930.677 | 13.035% | -13.035% |
| total_throughput | 25 | 464.524 | 272.452 | 465.155 | 1.097% | 0.023% |

## Inputs

1. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/real/repeat-1`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/ml_simulator/repeat-1`
2. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/real/repeat-2`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/ml_simulator/repeat-2`
3. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/real/repeat-3`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/ml_simulator/repeat-3`
4. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/real/repeat-4`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/ml_simulator/repeat-4`
5. real: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/real/repeat-5`
   replay: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Meta-Llama-3.1-70B-Instruct-AWQ/holdout/ml_simulator/repeat-5`

## Teacher-forced scheduler-batch score

This score uses the real observed batch composition and therefore does not include scheduler feedback.

- batches: 12674
- MAPE: 4.732%
- MAE: 5.299 ms

## Reading the numbers

The table is descriptive. Teacher-forced batch error and end-to-end replay error answer different questions: the latter also changes future batch formation when a predicted latency differs from the real one. These rows do not isolate every runtime component, such as request handling and host scheduling, from GPU forward time.
