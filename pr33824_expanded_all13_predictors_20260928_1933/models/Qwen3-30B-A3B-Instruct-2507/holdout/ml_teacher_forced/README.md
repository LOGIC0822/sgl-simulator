# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-30B-A3B-Instruct-2507/calibration/training/Qwen3-30B-A3B-Instruct-2507_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 12188
- MAPE: 4.763%
- MAE: 12.700 ms
