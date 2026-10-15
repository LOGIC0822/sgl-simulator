# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen2.5-14B-Instruct-1M/learning_curve/train_3_traces/bundle/Qwen2.5-14B-Instruct-1M_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 9974
- MAPE: 16.755%
- MAE: 32.387 ms
