"""Build a frozen dense-only report snapshot from completed stored experiments."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
SIM=Path("/home/ms-admin/workspace1/yiheng/ali/sglang-pr33824-merged/tools/sglang-simulator")
NEW=SIM/"test-results/pr33824_expanded_all13_predictors_20260928_1933"
OLD=SIM/"test-results/pr33824_comprehensive_all13_predictors_20260918_1605"
OUT=Path(__file__).resolve().parent
assert not (OUT/"summary.json").exists(), "Refuse to overwrite an earlier snapshot"
os.environ["MPLCONFIGDIR"]=str(OUT/"matplotlib-cache")
MODELS=[
("Qwen3-0.6B","Q3-0.6B"),("Qwen2.5-3B-Instruct","Q2.5-3B"),("llama-2-7b","Llama2-7B"),
("Mistral-7B-Instruct-v0.2","Mistral-7B"),("Qwen2.5-7B-Instruct","Q2.5-7B"),
("Qwen3-8B","Q3-8B"),("Qwen2.5-14B-Instruct","Q2.5-14B"),
("Qwen2.5-14B-Instruct-1M","Q2.5-14B-1M"),("Qwen3-14B","Q3-14B"),
("Meta-Llama-3.1-70B-Instruct-AWQ","Llama3.1-70B-AWQ")]
CASES=["latency_128x64_r1","latency_256x128_r1","service_1024x256_r4",
       "saturation_1024x256_inf","saturation_1536x384_inf"]
CASE_LABELS=["128/64 · 1/s","256/128 · 1/s","1024/256 · 4/s","1024/256 · burst","1536/384 · burst"]
SAT="saturation_1024x256_inf"
METRICS=["mean_e2e_latency_ms","mean_ttft_ms","mean_tpot_ms","total_throughput"]
SHA={}
def blob(path):
    path=path.resolve()
    data=path.read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    if str(path) in SHA: assert SHA[str(path)]==digest, str(path)
    SHA[str(path)]=digest
    return data
def text(path): return blob(path).decode("utf-8")
def rows(path): return list(csv.DictReader(text(path).splitlines()))
def jlines(path): return [json.loads(line) for line in text(path).splitlines() if line.strip()]
def meta(path): return dict(line.split("=",1) for line in text(path).splitlines() if "=" in line)
def one(items,**keys):
    found=[row for row in items if all(row.get(key)==value for key,value in keys.items())]
    assert len(found)==1,keys
    return {key:float(value) if key not in ("group","value") else value for key,value in found[0].items()}
def close(a,b): assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
def check_step(folder):
    summary=rows(folder/"summary.csv")
    samples=rows(folder/"per_batch.csv")
    for group,value in [("all","all"),("forward_mode","1"),("forward_mode","2")]+[("case",case) for case in sorted({s["case"] for s in samples})]:
        subset=[s for s in samples if group=="all" or (s["forward_mode"]==value if group=="forward_mode" else s["case"]==value)]
        prior=one(summary,group=group,value=value)
        actual=[float(s["actual_seconds"]) for s in subset]
        prediction=[float(s["predicted_seconds"]) for s in subset]
        assert len(subset)==int(prior["n"])
        assert all(math.isfinite(v) and v>0 for v in actual+prediction)
        close(statistics.mean(actual)*1000,prior["mean_actual_ms"])
        close(statistics.mean(prediction)*1000,prior["mean_predicted_ms"])
        close(statistics.mean(abs(p-a)/a*100 for a,p in zip(actual,prediction)),prior["mape_pct"])
    return summary
def check_e2e(folder):
    summary=rows(folder/"summary.csv"); pairs=rows(folder/"comparison.csv")
    for metric in METRICS:
        for case in [None]+sorted({row["case"] for row in pairs}):
            subset=[row for row in pairs if row["metric"]==metric and (case is None or row["case"]==case)]
            prior=one(summary,group="metric" if case is None else "case_metric",
                      value=metric if case is None else case+":"+metric)
            assert len(subset)==int(prior["n"])
            actual=[float(row["real_value"]) for row in subset]; pred=[float(row["predicted_value"]) for row in subset]
            close(statistics.mean(actual),prior["real_mean"]); close(statistics.mean(pred),prior["predicted_mean"])
            close(statistics.mean(abs(p-a)/a*100 for a,p in zip(actual,pred)),prior["mape_pct"])
    return summary
def identities(model,part,expected):
    root=NEW/"models"/model/part/"real"
    rids=set(); prompts=set(); count=0; timing_count=0
    for repeat in range(1,6):
        group=root/f"repeat-{repeat}"
        assert text(group/"BASELINE_COLLECTION_COMPLETE").strip()=="complete"
        for path in sorted((group/"cases").glob("*/TP0.raw_request.jsonl")):
            assert text(path.parent/"status.txt").strip()=="ok"
            manifest=json.loads(text(path.parent/"TP0.manifest.json"))
            requests=jlines(path)
            assert len(requests)==manifest["request_count"]
            count+=len(requests); timing_count+=manifest["schedule_batch_count"]
            for req in requests:
                assert req["rid"] not in rids
                rids.add(req["rid"])
                assert len(req["input_ids"])==req["input_length"]
                prompts.add(hashlib.sha256(json.dumps(req["input_ids"],separators=(",",":")).encode()).hexdigest())
    assert count==expected
    return rids,prompts,timing_count

new_protocol=meta(NEW/"run_metadata.txt"); old_protocol=meta(OLD/"run_metadata.txt")
saved_macros=[]
for folder in sorted((NEW/"verification").glob("frozen_fit_input_*")):
    for path in [folder/"ml_fit_input_vs_independent_macro.csv",folder/"results/ml_fit_input_vs_independent_macro.csv"]:
        if path.exists(): saved_macros+=rows(path)
records=[]
for model,label in MODELS:
    nr=NEW/"models"/model; old=OLD/"models"/model
    assert text(nr/"REAL_ML_COMPLETE").strip()=="complete"
    for marker in ["REAL_ML_COMPLETE","AIC_PREDICTOR_COMPLETE"]:
        assert text(old/marker).strip()=="complete"
    current_step=check_step(nr/"holdout/ml_teacher_forced")
    old_step=check_step(old/"holdout/ml_teacher_forced")
    old_aic_step=check_step(old/"holdout/aic_teacher_forced")
    current_e2e=check_e2e(nr/"holdout/ml_e2e_comparison")
    old_e2e=check_e2e(old/"holdout/ml_e2e_comparison")
    old_aic_e2e=check_e2e(old/"holdout/aic_e2e_comparison")
    tr,train_prompts,training_rows=identities(model,"calibration",800)
    te,test_prompts,test_rows=identities(model,"holdout",1280)
    assert not tr & te and not train_prompts & test_prompts
    close(test_rows,one(current_step,group="all",value="all")["n"])
    train_cases=rows(nr/"calibration/training/eval/dataset_cases.csv")
    close(sum(int(row["rows"]) for row in train_cases),training_rows)
    bundle=nr/"calibration/training"/f"{model}_hgbmono_p50.pkl"
    bundle_sha=hashlib.sha256(blob(bundle)).hexdigest()
    macro={}
    for stage in ["fit_input","independent_seen","independent_unseen"]:
        matching=[row for row in saved_macros if row["model"]==model and row["stage"]==stage and row["phase"]=="all"]
        assert len(matching)==1,(model,stage)
        row=matching[0]; assert row["bundle_sha256"]==bundle_sha
        macro[stage]={key:float(row[key]) for key in ("n","mean_actual_ms","mean_predicted_ms","mape_pct","mape_change_from_fit_input_pp")}
    configs={}
    for version,root in [("old",old),("new",nr)]:
        cfg=meta(root/"holdout/real/repeat-1/run_metadata.txt")
        benchmark=jlines(root/"holdout/real/repeat-1/cases"/SAT/"benchmark.jsonl")[-1]
        info=benchmark["server_info"]
        assert info["device"]=="cuda" and info["context_length"]==2048
        assert info["disable_cuda_graph"] and info["disable_overlap_schedule"]
        assert cfg["source_revision"]==new_protocol["source_revision"]
        configs[version]={"mem_fraction_static":cfg["mem_fraction_static"],"attention_backend":cfg["attention_backend"],
                          "page_size":info["page_size"],"dtype_setting":info["dtype"],"kv_cache_dtype_setting":info["kv_cache_dtype"],
                          "versions":meta(root/"holdout/real/repeat-1/package_versions.txt")}
    current_metrics={metric:one(current_e2e,group="metric",value=metric) for metric in METRICS}
    bycase={case:{"step":one(current_step,group="case",value=case),
                 **{metric:one(current_e2e,group="case_metric",value=case+":"+metric) for metric in METRICS}} for case in CASES}
    records.append({"model":model,"label":label,"training_rows":training_rows,"test_rows":test_rows,
                    "training_distinct_inputs":len(train_prompts),"test_distinct_inputs":len(test_prompts),"reused_request_ids":0,"shared_complete_inputs":0,
                    "step":{phase:one(current_step,group=group,value=value) for phase,group,value in
                            [("all","all","all"),("prefill","forward_mode","1"),("decode","forward_mode","2")]},
                    "e2e":current_metrics,"by_case":bycase,"frozen":macro,"settings":configs,
                    "old":{"ml_step":one(old_step,group="all",value="all"),"aic_step":one(old_aic_step,group="all",value="all"),
                           "ml_e2e":{metric:one(old_e2e,group="metric",value=metric) for metric in METRICS},
                           "aic_e2e":{metric:one(old_aic_e2e,group="metric",value=metric) for metric in METRICS}}})
before_finish=datetime.now().astimezone().isoformat(timespec="seconds")
for path,digest in SHA.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
font="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
assert Path(font).exists()
font_manager.fontManager.addfont(font)
plt.rcParams.update({"font.family":font_manager.FontProperties(fname=font).get_name(),"axes.unicode_minus":False,
                     "font.size":10,"axes.spines.top":False,"axes.spines.right":False})
FIG=OUT/"figures"; FIG.mkdir(exist_ok=False)
labels=[r["label"] for r in records]; y=np.arange(len(records)); figures=[]
def save(fig,name):
    fig.savefig(FIG/name,dpi=200,bbox_inches="tight")
    plt.close(fig); figures.append(name)
def barpanel(ax,groups,names,title):
    width=.76/len(groups)
    for i,(values,name) in enumerate(zip(groups,names)):
        centers=y+(i-(len(groups)-1)/2)*width
        bars=ax.barh(centers,values,height=width*.9,label=name)
        ax.bar_label(bars,labels=[f"{v:.2f}" for v in values],padding=3,fontsize=8)
    ax.set_yticks(y,labels); ax.invert_yaxis()
    ax.set_xlabel("MAPE（%）"); ax.set_title(title); ax.grid(axis="x",alpha=.2)
    ax.set_xlim(0,max(max(g) for g in groups)*1.21); ax.legend(loc="lower right",fontsize=9)
fig,axes=plt.subplots(1,2,figsize=(15,7.5))
for ax,key,metric,scale,title in [(axes[0],"step","all",1,"本轮单步均值（ms）"),(axes[1],"e2e","mean_e2e_latency_ms",.001,"本轮端到端延迟均值（s）")]:
    rows_here=[r[key][metric] for r in records]
    real=[row["mean_actual_ms" if key=="step" else "real_mean"]*scale for row in rows_here]
    pred=[row["mean_predicted_ms" if key=="step" else "predicted_mean"]*scale for row in rows_here]
    ax.plot(real,y,"o",label="实测",markersize=7); ax.plot(pred,y,"x",label="ML 预测",markersize=7)
    for pos,a,p in zip(y,real,pred):
        ax.plot([a,p],[pos,pos],color="#777777",alpha=.5)
        ax.annotate(f"{a:.2f} / {p:.2f}",(max(a,p),pos),xytext=(8,0),textcoords="offset points",va="center",fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(min(real+pred)*.75,max(real+pred)*1.7)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_title(title+"（对数轴）"); ax.legend(); ax.grid(axis="x",alpha=.2)
fig.suptitle("10 个 dense 模型：本轮独立测试的实测值 / ML 预测值")
fig.tight_layout(); save(fig,"01_current_real_vs_predicted.png")
fig,axes=plt.subplots(1,2,figsize=(15,7.7))
barpanel(axes[0],[[r["old"]["ml_step"]["mape_pct"] for r in records],[r["by_case"][SAT]["step"]["mape_pct"] for r in records]],["9月18日","本轮"],"共同负载：单步误差")
barpanel(axes[1],[[r["old"]["ml_e2e"]["mean_e2e_latency_ms"]["mape_pct"] for r in records],[r["by_case"][SAT]["mean_e2e_latency_ms"]["mape_pct"] for r in records]],["9月18日","本轮"],"共同负载：端到端延迟误差")
fig.suptitle("1024/256 饱和负载：旧 / 新 ML；各 5 轮 × 64 请求")
fig.tight_layout(); save(fig,"02_matched_workload_old_vs_new_mape.png")
fig,axes=plt.subplots(1,2,figsize=(15,8))
series=[("旧实测","#4e79a7","o"),("旧预测","#4e79a7","x"),("新实测","#f28e2b","o"),("新预测","#f28e2b","x")]
for ax,kind,title,scale in [(axes[0],"step","共同负载单步（ms）",1),(axes[1],"e2e","共同负载端到端延迟（s）",.001)]:
    oldrows=[r["old"]["ml_step"] if kind=="step" else r["old"]["ml_e2e"]["mean_e2e_latency_ms"] for r in records]
    newrows=[r["by_case"][SAT]["step"] if kind=="step" else r["by_case"][SAT]["mean_e2e_latency_ms"] for r in records]
    data=[[row[field]*scale for row in rows_here] for rows_here in [oldrows,newrows] for field in
          (["mean_actual_ms","mean_predicted_ms"] if kind=="step" else ["real_mean","predicted_mean"])]
    for i,(vals,(label,color,marker)) in enumerate(zip(data,series)):
        ax.scatter(vals,y+(i-1.5)*.16,label=label,color=color,marker=marker,s=34)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_title(title+"（对数轴）"); ax.grid(axis="x",alpha=.25); ax.legend(fontsize=8)
fig.suptitle("1024/256 饱和负载：旧 / 新实测与预测，非同一批请求")
fig.tight_layout(); save(fig,"03_matched_workload_real_and_predicted.png")
fig,axes=plt.subplots(1,2,figsize=(15,8))
for ax,kind,title in [(axes[0],"step","本轮单步 MAPE（%）"),(axes[1],"mean_e2e_latency_ms","本轮端到端 MAPE（%）")]:
    matrix=np.array([[r["by_case"][case][kind]["mape_pct"] for case in CASES] for r in records])
    image=ax.imshow(matrix,cmap="YlOrRd",vmin=0,vmax=float(matrix.max()),aspect="auto")
    ax.set_xticks(range(5),CASE_LABELS,rotation=25,ha="right"); ax.set_yticks(y,labels); ax.set_title(title)
    for i in range(10):
        for j in range(5):
            ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",color="white" if matrix[i,j]>matrix.max()*.65 else "black",fontsize=9)
    fig.colorbar(image,ax=ax,shrink=.75)
fig.suptitle("本轮 5 类独立测试负载：输入 / 输出长度为采样上限")
fig.tight_layout(); save(fig,"04_workload_mape_heatmap.png")
fig,ax=plt.subplots(figsize=(12,8))
barpanel(ax,[[r["frozen"][stage]["mape_pct"] for r in records] for stage in ["fit_input","independent_seen","independent_unseen"]],
         ["训练数据回评分","独立请求 · 已见负载","独立请求 · 未见负载"],"同一冻结 ML 预测器：逐负载等权单步 MAPE")
fig.tight_layout(); save(fig,"05_frozen_fit_vs_independent.png")
fig,axes=plt.subplots(1,2,figsize=(15,8))
barpanel(axes[0],[[r["old"]["ml_step"]["mape_pct"] for r in records],[r["old"]["aic_step"]["mape_pct"] for r in records]],["ML","AIC"],"9月18日：单步误差")
barpanel(axes[1],[[r["old"][method]["mean_e2e_latency_ms"]["mape_pct"] for r in records] for method in ["ml_e2e","aic_e2e"]],["ML","AIC"],"9月18日：端到端延迟误差")
fig.suptitle("历史完整矩阵的 ML / AIC 对照（不是本轮新 AIC 结果）")
fig.tight_layout(); save(fig,"06_historical_ml_vs_aic.png")
result={"snapshot_at":before_finish,"models":records,"figures":figures,"new_protocol":new_protocol,"old_protocol":old_protocol,
        "native_requests_total":20800,"training_timing_rows":sum(r["training_rows"] for r in records),
        "heldout_timing_rows":sum(r["test_rows"] for r in records),"source_files_unchanged":len(SHA),
        "limitations":["Historical matched workload is not identical requests or an isolated training-size ablation.",
                       "Current new AIC is pending; historical AIC values are explicitly historical.",
                       "No GPU/native benchmark/model fit/predictor query is invoked to build this report.",
                       "The all13 final experiment is not marked complete by a dense-only stage report."]}
(OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
(OUT/"source_manifest.json").write_text(json.dumps({"source_sha256":SHA,"unchanged_files":len(SHA),"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+"\n")
for filename,fields,table in [
    ("current_single_step.csv",["model","training_rows","test_rows","mean_actual_ms","mean_predicted_ms","mape_pct","prefill_mape_pct","decode_mape_pct"],[
       {"model":r["model"],"training_rows":r["training_rows"],"test_rows":r["test_rows"],
        **{key:r["step"]["all"][key] for key in ("mean_actual_ms","mean_predicted_ms","mape_pct")},
        "prefill_mape_pct":r["step"]["prefill"]["mape_pct"],"decode_mape_pct":r["step"]["decode"]["mape_pct"]} for r in records]),
    ("matched_workload_comparison.csv",["model","old_real_step_ms","old_pred_step_ms","new_real_step_ms","new_pred_step_ms","old_step_mape_pct","new_step_mape_pct","old_real_e2e_ms","old_pred_e2e_ms","new_real_e2e_ms","new_pred_e2e_ms","old_e2e_mape_pct","new_e2e_mape_pct"],[
       {"model":r["model"],"old_real_step_ms":r["old"]["ml_step"]["mean_actual_ms"],"old_pred_step_ms":r["old"]["ml_step"]["mean_predicted_ms"],
        "new_real_step_ms":r["by_case"][SAT]["step"]["mean_actual_ms"],"new_pred_step_ms":r["by_case"][SAT]["step"]["mean_predicted_ms"],
        "old_step_mape_pct":r["old"]["ml_step"]["mape_pct"],"new_step_mape_pct":r["by_case"][SAT]["step"]["mape_pct"],
        "old_real_e2e_ms":r["old"]["ml_e2e"]["mean_e2e_latency_ms"]["real_mean"],"old_pred_e2e_ms":r["old"]["ml_e2e"]["mean_e2e_latency_ms"]["predicted_mean"],
        "new_real_e2e_ms":r["by_case"][SAT]["mean_e2e_latency_ms"]["real_mean"],"new_pred_e2e_ms":r["by_case"][SAT]["mean_e2e_latency_ms"]["predicted_mean"],
        "old_e2e_mape_pct":r["old"]["ml_e2e"]["mean_e2e_latency_ms"]["mape_pct"],"new_e2e_mape_pct":r["by_case"][SAT]["mean_e2e_latency_ms"]["mape_pct"]} for r in records])]:
    with (OUT/filename).open("x",newline="",encoding="utf-8") as output:
        writer=csv.DictWriter(output,fieldnames=fields); writer.writeheader(); writer.writerows(table)
print(json.dumps(result,ensure_ascii=False))

