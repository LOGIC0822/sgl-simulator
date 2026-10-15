# ML 训练/测试 Scheduler batch 分布核对

统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。

| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| all | 11533 | 19080 | 8.97% | 17.96% | 3 / 2 | 4 / 4 | 570 / 512 |
| prefill | 1103 | 1786 | 42.83% | 4.59% | 1 / 1 | 463 / 500 | 0 / 0 |
| decode | 10430 | 17294 | 5.47% | 19.62% | 3 / 3 | 3 / 3 | 869 / 660 |
