# ML latency predictor training log

- data_roots: ['/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/calibration/real/repeat-1/cases', '/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/calibration/real/repeat-2/cases', '/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/calibration/real/repeat-3/cases', '/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/calibration/real/repeat-4/cases', '/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/calibration/real/repeat-5/cases']
- schedule_glob: TP0*.schedule_batch.jsonl
- rows: 11319
- cases: 15
- forward_modes: [1, 2]
- max_latency_seconds: 30.0
- features: batch_size, sum_extend, max_extend, min_extend, sum_past, max_past, min_past, sum_extend_x_past, sum_extend_squared, sum_past_squared, sum_attn_flops, sum_extend_x_max_past, log1p_sum_past, log1p_sum_attn_flops, batch_size_x_sum_extend, max_past_minus_min_past, is_decode, is_prefill

## Case-stratified holdout

- l2: MAPE=4.907%, MAE=2.182ms, R2=0.04676
- p50: MAPE=1.841%, MAE=1.333ms, R2=0.02288
