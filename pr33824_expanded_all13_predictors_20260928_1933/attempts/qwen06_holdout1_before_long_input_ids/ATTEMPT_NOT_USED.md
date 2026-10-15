# Partial held-out attempt — excluded

2026-09-28, approximately 20:41 Asia/Shanghai. The first four cases finished, but the long text-input case was deliberately stopped during startup/warmup after an independent tokenizer-bound check found an input+output overflow candidate. The matrix parents were paused first, and only this verified baseline shell/server/benchmark process tree was terminated. Other users' processes were not stopped.

This entire unfinished group is archived here. It has no group completion marker and does not contribute to the final five-repeat test set, ML fit, learning curves or AIC shape selection. Existing raw measurements, commands and logs are unchanged. A new group with the same test seed will be collected in the canonical model directory. Only the 1536/384 long case adds `--tokenize-prompt`; other four cases and all training configuration remain unchanged.

The tokenizer-only check covered 4160 candidate requests across 13 models and five test seeds. Ten Qwen checkpoint tokenizers each had five candidates with encoded input+requested output beyond context=2048; no candidate's input alone reached 2048. Llama, Mistral and AWQ had no overflow in these candidates. Direct-ID generation bounds the long case by 1536+384=1920 and avoids silently truncating requests.
