# AIC Silicon scheduler-batch scoring

Teacher-forced comparison against real device-synchronized ScheduleBatch rows.

- sim config: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json`
- model: `/home/ms-admin/data/models/Qwen2.5-7B-Instruct`
- forward modes: `[1, 2]`
- batches: 19018
- unique AIC static queries: 2420
- MAPE: 51.912%
- MAE: 13.774 ms
