import csv, hashlib, json, math, statistics
from collections import Counter
from pathlib import Path
path=Path("/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator/test-results/pr33824_expanded_all13_predictors_20260928_1933/models/Qwen3-30B-A3B-Instruct-2507/holdout/ml_teacher_forced/per_batch.csv")
before=hashlib.sha256(path.read_bytes()).hexdigest()
assert before=="dea1ab5c8aab702a10b6ccfbde206dfe03bdb26df1ddb8e19b638dbf212ff650"
with path.open() as source:
    rows=list(csv.DictReader(source))
assert len(rows)==12188
values=[]
for row in rows:
    actual=float(row["actual_seconds"])*1000
    predicted=float(row["predicted_seconds"])*1000
    absolute=abs(predicted-actual)
    assert actual>0 and math.isfinite(predicted)
    values.append({"case":row["case"],"schedule_path":row["schedule_path"],
                   "line_number":int(row["line_number"]),"forward_mode":int(row["forward_mode"]),
                   "batch_size":int(row["batch_size"]),"sum_extend":int(row["sum_extend"]),
                   "sum_past":int(row["sum_past"]),"actual_ms":actual,"predicted_ms":predicted,
                   "absolute_error_ms":absolute,"ape_pct":absolute/actual*100})
def percentile(numbers,q):
    ordered=sorted(numbers);index=(len(ordered)-1)*q
    low,high=math.floor(index),math.ceil(index)
    return ordered[low]+(ordered[high]-ordered[low])*(index-low)
total_error=sum(row["absolute_error_ms"] for row in values)
total_ape=sum(row["ape_pct"] for row in values)
ranked=sorted(values,key=lambda row:row["absolute_error_ms"],reverse=True)
concentration=[]
for fraction in (.01,.05):
    count=math.ceil(len(values)*fraction);selected=ranked[:count]
    concentration.append({"ranked_by":"absolute_error_ms","top_fraction":fraction,"count":count,
                          "absolute_error_share_pct":sum(row["absolute_error_ms"] for row in selected)/total_error*100,
                          "ape_sum_share_pct":sum(row["ape_pct"] for row in selected)/total_ape*100,
                          "actual_latency_min_ms":min(row["actual_ms"] for row in selected),
                          "actual_latency_max_ms":max(row["actual_ms"] for row in selected),
                          "cases":dict(Counter(row["case"] for row in selected)),
                          "phases":dict(Counter(row["forward_mode"] for row in selected))})
assert hashlib.sha256(path.read_bytes()).hexdigest()==before
print(json.dumps({"source_path":str(path),"source_sha256":before,"source_unchanged":True,
                  "rows":len(values),"no_rows_removed":True,"no_fit_or_prediction":True,
                  "actual_ms_quantiles":{str(q):percentile([row["actual_ms"] for row in values],q) for q in (.5,.95,.99,1)},
                  "prediction_ms_quantiles":{str(q):percentile([row["predicted_ms"] for row in values],q) for q in (.5,.95,.99,1)},
                  "absolute_error_concentration":concentration,"largest_10_absolute_errors":ranked[:10],
                  "limits":"Source-integrity-verified stored wall-clock labels; descriptive heavy-tail statistics only. No claim of GPU stalls, CPU bubbles, JIT compilation or token-content causation; no outlier filtering, refitting or corrected performance score."}))
