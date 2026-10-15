# Collection group recovery

Validated on 2026-09-28 at approximately 20:25 Asia/Shanghai. The launcher failed **after all three measured cases finished**, because its on-disk Bash file was edited while the existing process still had the old read offset. The current file passes `bash -n`; this was not a GPU collection failure. The original failure remains in `runner.log` and `status.tsv`.

All three case `status.txt` files contain `ok`. Independent checks loaded every raw request, batch row, converted trace and final benchmark record. Request counts, unique request IDs, positive finite batch latencies, manifest row counts, raw/trace input tokens, raw/trace/benchmark output tokens and exactly one successful start/stop profiling window per case passed.

| Case | Completed requests | Batch latency rows |
| --- | ---: | ---: |
| latency_256x128_r1 | 32 | 1309 |
| service_1024x256_r4 | 64 | 722 |
| saturation_1024x256_inf | 64 | 495 |

Seed is `202609283`, memory fraction `0.65`; training is disabled in this per-group collector. The expected 160 requests and 2526 latency rows are present. Only the post-loop GPU snapshot and group marker were missing. No historical GPU snapshot is fabricated. The group completion marker was restored after these checks; no measurements were replaced or generated manually.

The benchmark's nominal input-token count differs from actual server input IDs because random IDs are decoded to text and re-encoded. Raw IDs and converted trace counts agree. Paired throughput will use the actual trace token total over the unchanged benchmark duration, retaining the original reported counter separately.
