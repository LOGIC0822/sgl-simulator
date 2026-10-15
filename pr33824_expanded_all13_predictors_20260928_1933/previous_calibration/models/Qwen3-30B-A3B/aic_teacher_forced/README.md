# AIC Silicon scheduler-batch scoring

Teacher-forced comparison against real device-synchronized ScheduleBatch rows.

- sim config: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json`
- model: `/home/ms-admin/data/models/Qwen3-30B-A3B`
- forward modes: `[1, 2]`
- batches: 12223
- unique AIC static queries: 2174
- MAPE: 63.907%
- MAE: 64.885 ms
