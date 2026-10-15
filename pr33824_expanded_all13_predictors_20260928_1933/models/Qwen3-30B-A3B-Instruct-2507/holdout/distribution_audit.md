# ML 训练/测试 Scheduler batch 分布核对

统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。

| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| all | 6931 | 12188 | 5.26% | 27.49% | 7 / 7 | 15 / 16 | 1802 / 2576 |
| prefill | 1051 | 1727 | 37.12% | 3.18% | 1 / 1 | 512 / 512 | 49 / 44 |
| decode | 5880 | 10461 | 0.00% | 32.58% | 9 / 9 | 9 / 9 | 3892 / 6066 |
