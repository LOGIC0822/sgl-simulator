# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/models/Qwen3-30B-A3B/calibration/training/Qwen3-30B-A3B_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 12223
- MAPE: 12.139%
- MAE: 19.250 ms
