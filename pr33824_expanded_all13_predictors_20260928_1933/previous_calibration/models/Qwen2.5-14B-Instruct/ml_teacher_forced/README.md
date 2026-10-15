# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/models/Qwen2.5-14B-Instruct/calibration/training/Qwen2.5-14B-Instruct_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 15701
- MAPE: 7.611%
- MAE: 4.151 ms
