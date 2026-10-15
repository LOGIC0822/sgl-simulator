# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/llama-2-7b/learning_curve/train_5_traces/bundle/llama-2-7b_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 18500
- MAPE: 3.840%
- MAE: 1.867 ms
