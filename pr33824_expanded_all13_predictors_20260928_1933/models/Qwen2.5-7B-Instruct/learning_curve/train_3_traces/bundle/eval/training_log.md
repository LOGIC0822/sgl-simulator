# ML latency predictor training log

- data_roots: ['/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen2.5-7B-Instruct/calibration/real/repeat-1/cases', '/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen2.5-7B-Instruct/calibration/real/repeat-2/cases', '/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen2.5-7B-Instruct/calibration/real/repeat-3/cases']
- schedule_glob: TP0*.schedule_batch.jsonl
- rows: 6808
- cases: 9
- forward_modes: [1, 2]
- max_latency_seconds: 30.0
- features: batch_size, sum_extend, max_extend, min_extend, sum_past, max_past, min_past, sum_extend_x_past, sum_extend_squared, sum_past_squared, sum_attn_flops, sum_extend_x_max_past, log1p_sum_past, log1p_sum_attn_flops, batch_size_x_sum_extend, max_past_minus_min_past, is_decode, is_prefill

## Case-stratified holdout

- p50: MAPE=1.533%, MAE=1.407ms, R2=0.01921
