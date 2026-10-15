# ML scheduler-batch scoring

This is teacher-forced scoring against real, device-synchronized ScheduleBatch rows. It does not replay a new scheduler sequence.

- bundle: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen2.5-3B-Instruct/calibration/training/Qwen2.5-3B-Instruct_hgbmono_p50.pkl`
- forward modes: `[1, 2]`
- batches: 17628
- MAPE: 4.552%
- MAE: 2.286 ms
