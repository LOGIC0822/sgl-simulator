# ML 训练/测试 Scheduler batch 分布核对

统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。

| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| all | 7069 | 12674 | 4.17% | 24.28% | 6 / 6 | 13 / 13 | 1636 / 1688 |
| prefill | 1009 | 1711 | 30.92% | 6.37% | 1 / 1 | 512 / 512 | 94 / 83 |
| decode | 6060 | 10963 | 0.00% | 27.85% | 8 / 8 | 8 / 8 | 3235 / 4587 |
