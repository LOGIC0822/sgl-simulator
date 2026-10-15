# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-32B/learning_curve/train_1_traces/bundle/Qwen3-32B_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 13889
- MAPE: 2.482%
- MAE: 2.714 ms
