# AIC Silicon scheduler-batch scoring

Teacher-forced comparison against real device-synchronized ScheduleBatch rows.

- sim config: `/mnt/data/sglang-pr33824-merged/test-results/pr33824_comprehensive_all13_predictors_20260918_1605/aic_calibration/simulator-config.a100_pcie_local_aic010_regular_context.json`
- model: `/home/ms-admin/data/models/Qwen2.5-14B-Instruct-1M`
- forward modes: `[1, 2]`
- batches: 9974
- unique AIC static queries: 1652
- MAPE: 82.436%
- MAE: 281.331 ms
