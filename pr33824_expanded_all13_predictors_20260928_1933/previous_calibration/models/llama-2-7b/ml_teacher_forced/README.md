# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/models/llama-2-7b/calibration/training/llama-2-7b_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 18500
- MAPE: 7.284%
- MAE: 2.811 ms
