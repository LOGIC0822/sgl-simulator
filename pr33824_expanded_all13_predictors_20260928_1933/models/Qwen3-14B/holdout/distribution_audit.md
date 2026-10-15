# ML 训练/测试 Scheduler batch 分布核对

统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。

| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| all | 9256 | 16069 | 7.46% | 21.10% | 3 / 3 | 5 / 5 | 659 / 603 |
| prefill | 1079 | 1752 | 39.90% | 3.31% | 1 / 1 | 499 / 512 | 0 / 22 |
| decode | 8177 | 14317 | 3.49% | 23.70% | 4 / 4 | 4 / 4 | 920 / 1044 |
