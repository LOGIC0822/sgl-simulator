# ML 训练/测试 Scheduler batch 分布核对

统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。

| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| all | 11319 | 18306 | 9.40% | 19.20% | 3 / 3 | 4 / 4 | 513 / 512 |
| prefill | 1079 | 1774 | 34.44% | 6.09% | 1 / 1 | 460 / 507 | 2 / 2 |
| decode | 10240 | 16532 | 6.71% | 21.91% | 3 / 3 | 3 / 3 | 646 / 653 |
