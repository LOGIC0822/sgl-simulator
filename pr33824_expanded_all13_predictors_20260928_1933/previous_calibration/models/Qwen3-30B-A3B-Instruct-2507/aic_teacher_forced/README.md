# AIC Silicon scheduler-batch scoring

Teacher-forced comparison against real device-synchronized ScheduleBatch rows.

- sim config: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json`
- model: `/home/ms-admin/data/models/Qwen3-30B-A3B-Instruct-2507`
- forward modes: `[1, 2]`
- batches: 12188
- unique AIC static queries: 2185
- MAPE: 63.493%
- MAE: 64.068 ms
