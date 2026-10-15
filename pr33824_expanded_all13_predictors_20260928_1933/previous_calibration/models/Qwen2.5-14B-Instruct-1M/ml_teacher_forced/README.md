# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/models/Qwen2.5-14B-Instruct-1M/calibration/training/Qwen2.5-14B-Instruct-1M_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 9974
- MAPE: 14.755%
- MAE: 42.755 ms
