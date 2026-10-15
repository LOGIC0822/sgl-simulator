# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/models/Qwen2.5-7B-Instruct/calibration/training/Qwen2.5-7B-Instruct_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 19018
- MAPE: 7.411%
- MAE: 2.696 ms
