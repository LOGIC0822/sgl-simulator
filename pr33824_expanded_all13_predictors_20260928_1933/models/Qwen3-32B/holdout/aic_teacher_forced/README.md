# AIC Silicon scheduler-batch scoring

Teacher-forced comparison against real device-synchronized ScheduleBatch rows.

- sim config: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json`
- model: `/home/ms-admin/data/models/Qwen3-32B`
- forward modes: `[1, 2]`
- batches: 13889
- unique AIC static queries: 2200
- MAPE: 15.556%
- MAE: 10.411 ms
