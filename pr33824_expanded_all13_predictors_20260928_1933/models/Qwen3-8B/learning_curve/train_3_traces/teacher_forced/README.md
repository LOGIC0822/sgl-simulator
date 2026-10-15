# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-8B/learning_curve/train_3_traces/bundle/Qwen3-8B_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 17291
- MAPE: 4.487%
- MAE: 2.382 ms
