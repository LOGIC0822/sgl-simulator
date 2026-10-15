# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/models/Qwen3-8B/calibration/training/Qwen3-8B_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 17291
- MAPE: 6.179%
- MAE: 2.928 ms
