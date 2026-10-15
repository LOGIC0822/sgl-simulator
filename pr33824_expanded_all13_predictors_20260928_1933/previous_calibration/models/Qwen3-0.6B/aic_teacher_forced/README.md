# AIC Silicon scheduler-batch scoring

Teacher-forced comparison against real device-synchronized ScheduleBatch rows.

- sim config: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json`
- model: `/home/ms-admin/data/models/Qwen3-0.6B`
- forward modes: `[1, 2]`
- batches: 19080
- unique AIC static queries: 2412
- MAPE: 88.479%
- MAE: 23.070 ms
