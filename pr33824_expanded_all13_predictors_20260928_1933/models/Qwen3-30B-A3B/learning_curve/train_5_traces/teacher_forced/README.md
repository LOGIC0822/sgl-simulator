# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-30B-A3B/learning_curve/train_5_traces/bundle/Qwen3-30B-A3B_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 12223
- MAPE: 4.616%
- MAE: 12.634 ms
