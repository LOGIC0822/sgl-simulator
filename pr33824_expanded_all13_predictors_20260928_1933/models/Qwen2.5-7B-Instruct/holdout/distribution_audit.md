# ML 训练/测试 Scheduler batch 分布核对

统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。

| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| all | 11598 | 19018 | 8.51% | 17.85% | 3 / 2 | 4 / 4 | 564 / 494 |
| prefill | 1097 | 1776 | 42.00% | 2.98% | 1 / 1 | 466 / 505 | 0 / 0 |
| decode | 10501 | 17242 | 5.06% | 19.68% | 3 / 3 | 3 / 3 | 869 / 609 |
