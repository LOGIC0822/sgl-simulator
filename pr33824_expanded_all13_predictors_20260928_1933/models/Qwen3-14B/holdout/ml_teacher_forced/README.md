# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-14B/calibration/training/Qwen3-14B_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 16069
- MAPE: 6.160%
- MAE: 3.424 ms
