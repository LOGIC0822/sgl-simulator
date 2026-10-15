# ML 训练/测试 Scheduler batch 分布核对

统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。

| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| all | 6951 | 12223 | 5.31% | 27.40% | 7 / 7 | 15 / 16 | 1691 / 2576 |
| prefill | 1052 | 1725 | 37.62% | 3.36% | 1 / 1 | 512 / 512 | 46 / 44 |
| decode | 5899 | 10498 | 0.00% | 32.43% | 9 / 9 | 9 / 9 | 3892 / 6024 |
