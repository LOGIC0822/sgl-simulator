# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/models/Meta-Llama-3.1-70B-Instruct-AWQ/calibration/training/Meta-Llama-3.1-70B-Instruct-AWQ_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 12674
- MAPE: 8.401%
- MAE: 8.042 ms
