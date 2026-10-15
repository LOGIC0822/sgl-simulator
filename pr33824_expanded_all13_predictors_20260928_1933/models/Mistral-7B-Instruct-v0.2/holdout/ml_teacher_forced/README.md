# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Mistral-7B-Instruct-v0.2/calibration/training/Mistral-7B-Instruct-v0.2_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 18306
- MAPE: 5.425%
- MAE: 2.284 ms
