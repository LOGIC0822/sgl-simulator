# AIC Silicon scheduler-batch scoring

Teacher-forced comparison against real device-synchronized ScheduleBatch rows.

- sim config: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_expanded_all13_predictors_20260928_1933/aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json`
- model: `/home/ms-admin/data/models/Mistral-7B-Instruct-v0.2`
- forward modes: `[1, 2]`
- batches: 18306
- unique AIC static queries: 2485
- MAPE: 53.490%
- MAE: 15.513 ms
