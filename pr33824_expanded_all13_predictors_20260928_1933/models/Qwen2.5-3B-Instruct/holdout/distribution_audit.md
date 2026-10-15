# ML 训练/测试 Scheduler batch 分布核对

统计对象为真实设备同步的 ScheduleBatch 行。`exact_feature_match_pct` 是测试行与训练集中完整 18 维特征完全相同的比例；`test_outside_train_feature_range_pct` 是测试行至少一个特征落在训练逐维范围外的比例。前者不应被当成同分布的必要条件，后者只描述特征支持集覆盖情况。

| phase | train rows | test rows | exact match | test outside train range | train / test batch p50 | train / test extend p50 | train / test past p50 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| all | 10557 | 17628 | 8.46% | 19.29% | 3 / 3 | 5 / 5 | 649 / 529 |
| prefill | 1095 | 1776 | 41.72% | 3.27% | 1 / 1 | 467 / 505 | 0 / 0 |
| decode | 9462 | 15852 | 4.73% | 21.40% | 4 / 3 | 4 / 3 | 1056 / 926 |
