# ML 训练/测试 Scheduler batch 分布核对

统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。

| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| all | 9073 | 15701 | 7.09% | 21.53% | 4 / 3 | 6 / 6 | 691 / 657 |
| prefill | 1079 | 1753 | 39.59% | 3.02% | 1 / 1 | 494 / 512 | 0 / 21 |
| decode | 7994 | 13948 | 3.00% | 24.33% | 4 / 4 | 4 / 4 | 1064 / 1350 |
