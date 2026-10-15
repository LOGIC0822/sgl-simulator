# SGL Simulator and Predictor Experiments

This repository keeps the simulator source and its ML/AIConfigurator experiment results as ordinary directories.

## Contents

- `sglang-simulator/` — simulator source, tests, examples, scripts, and documentation. This is a working-tree snapshot based on SGLang commit `59799a3687`; local source changes at packaging time are included.
- `pr33824_expanded_all13_predictors_20260928_1933/` — records and data for the expanded 13-model ML and AIConfigurator experiments, including model-level calibration/holdout data, logs, analysis, and reports.

The experiment directory is approximately 10 GB on disk. The upload preserves its regular files and directory structure. Machine-local links were omitted: the source tree's `test-results` link points outside this repository, and 1,173 links under the experiment's `verification/` directory use absolute `/mnt/data/...` paths. The generated source `.pytest_cache/` and private `sglang-simulator/scripts/remote_a100_experiment.env` were also omitted; only the `.env.example` template is included.

This is an experimental snapshot, not a clean upstream SGLang release. Run-specific environment details and results are recorded in the experiment directory's metadata and reports.
