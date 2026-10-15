#!/usr/bin/env python3
"""Audit completed Qwen7 E2E comparisons without rerunning experiments."""

import csv, hashlib, importlib.util, json, math, re, shlex
from pathlib import Path
sim=Path.cwd(); root=(sim/'test-results/pr33824_expanded_all13_predictors_20260928_1933').resolve()
model='Qwen2.5-7B-Instruct'; mr=root/'models'/model
out=root/'verification/qwen7b_full_e2e_score_integrity_20260929_0420/results'
assert not out.exists()
sha={}
def preserve(path):
    path=path.resolve(); sha[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest(); return path

def tool(name):
    path=preserve(sim/'tools'/f'{name}.py')
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

pair=tool('summarize_pr33824_data_tools_replay')
score=tool('score_pr33824_latency_model')
report=tool('append_pr33824_expanded_report')
assert preserve(mr/'REAL_ML_COMPLETE').read_text().strip()=='complete'
preserve(root/'run_metadata.txt'); preserve(mr/'model_settings.txt')
preserve(Path(__file__))
bundle=preserve(mr/'calibration/training/Qwen2.5-7B-Instruct_hgbmono_p50.pkl')
assert sha[str(bundle)]=='09b75806a25dd7de5337b87883368a6ef64ef811218d5b2877025a8ef4eb19dd'
capacity_checks=[]
def metadata(path):
    return dict(line.split('=',1) for line in preserve(path).read_text().splitlines() if '=' in line)
def argument(command, name):
    matches=[i for i, token in enumerate(command) if token==name]
    assert len(matches)==1, (name, command)
    return command[matches[0]+1]
counts={'latency_256x128_r1':32,'service_1024x256_r4':64,'saturation_1024x256_inf':64,'latency_128x64_r1':32,'saturation_1536x384_inf':64}
versions=[]; recomputed=[]
for repeat in range(1,6):
    real=mr/'holdout/real'/f'repeat-{repeat}'; replay=mr/'holdout/ml_simulator'/f'repeat-{repeat}'
    assert preserve(real/'BASELINE_COLLECTION_COMPLETE').read_text().strip()=='complete'
    assert preserve(replay/'ML_REPLAY_COMPLETE').read_text().strip()=='complete'
    package=preserve(real/'package_versions.txt')
    version=dict(line.split('=',1) for line in package.read_text().splitlines() if '=' in line)
    selected={key:version[key] for key in ('python','torch','torch_cuda','sgl_kernel_file','sgl_kernel_runtime_version','sglang','sglang-kernel','triton')}
    versions.append(selected)
    replay_metadata=metadata(replay/'run_metadata.txt')
    assert Path(replay_metadata['collection_root']).resolve()==real
    assert Path(replay_metadata['ml_bundle']).resolve()==bundle
    assert replay_metadata['source_revision']=='59799a368793b9f795baf59b067233a65ad8e38e'
    assert Path(replay_metadata['compat_stage']).name=='minimal-stage-v4'
    assert replay_metadata['scheduler_overlap']==replay_metadata['cuda_graph']=='disabled'
    config=json.loads(preserve(replay/'simulator-config.json').read_text())
    assert config['predictor']['name']=='ml' and config['predictor']['latency_scale']==1.0
    assert Path(config['predictor']['database_path']).resolve()==bundle
    assert all(config['scheduler'][key]==1 for key in ('tp_size','ep_size','dp_size'))
    assert set(path.name for path in (real/'cases').iterdir() if path.is_dir())==set(counts)
    assert set(path.name for path in (replay/'cases').iterdir() if path.is_dir())==set(counts)
    for case,n in counts.items():
        rc=real/'cases'/case; pc=replay/'cases'/case
        assert preserve(rc/'status.txt').read_text().strip()=='ok'
        assert preserve(pc/'status.txt').read_text().strip()=='ok'
        native=pair.final_jsonl_object(preserve(rc/'benchmark.jsonl'))
        manifest=json.loads(preserve(rc/'autobench_trace.manifest.json').read_text())
        metric=json.loads(preserve(pc/'simulator-output/metrics.json').read_text())
        preserve(pc/'simulator-output/iteration.jsonl')
        preserve(rc/'TP0.schedule_batch.jsonl'); preserve(rc/'TP0.raw_request.jsonl'); preserve(rc/'autobench_trace.jsonl')
        native_command=shlex.split(preserve(real/'cases'/f'{case}.server.command.txt').read_text())
        replay_command=shlex.split(preserve(pc/'server.command.txt').read_text())
        replay_client=shlex.split(preserve(pc/'benchmark.command.txt').read_text())
        native_client=shlex.split(preserve(real/'cases'/f'{case}.benchmark.command.txt').read_text())
        assert argument(replay_client,'--dataset-name')=='autobench'
        assert Path(argument(replay_client,'--dataset-path')).resolve()==(rc/'autobench_trace.jsonl').resolve()
        assert int(argument(replay_client,'--num-prompts'))==int(argument(native_client,'--num-prompts'))==n
        assert '--use-trace-timestamps' in replay_client
        assert argument(native_client,'--dataset-name')=='random-ids'
        assert int(argument(native_client,'--seed'))==202609379+repeat
        assert argument(native_client,'--random-range-ratio')=='0'
        assert ('--tokenize-prompt' in native_client)==(case=='saturation_1536x384_inf')
        assert argument(native_command,'--device')=='cuda' and argument(replay_command,'--device')=='cpu'
        assert Path(argument(native_command,'--model-path')).resolve()==Path(argument(replay_command,'--model-path')).resolve()
        assert argument(native_command,'--tp-size')=='1'
        assert argument(native_command,'--attention-backend')=='triton'
        for name, value in (('--context-length','2048'),('--max-running-requests','32'),('--chunked-prefill-size','512')):
            assert argument(native_command,name)==argument(replay_command,name)==value
        for name in ('--disable-cuda-graph','--disable-overlap-schedule'):
            assert name in native_command and name in replay_command
        native_capacities=re.findall(r'max_total_num_tokens=(\d+)',preserve(real/'cases'/f'{case}.server.log').read_text())
        assert native_capacities
        native_capacity=int(native_capacities[-1])
        assert int(argument(replay_command,'--max-total-tokens'))==native_capacity
        capacity_checks.append({'repeat':repeat,'case':case,'native_max_total_tokens':native_capacity,'replay_max_total_tokens':int(argument(replay_command,'--max-total-tokens'))})
        assert native['completed']==manifest['request_count']==metric['completed']==metric['num_requests']==n
        assert metric['total_input']==manifest['input_tokens']
        assert native['total_output_tokens']==metric['total_output']==manifest['output_tokens']
        assert native['server_info']['device']=='cuda'
        assert Path(native['server_info']['model_path']).name==model
        assert native['server_info']['context_length']==2048
        assert native['server_info']['disable_cuda_graph'] and native['server_info']['disable_overlap_schedule']
    recomputed+=pair.pair_roots(real,replay,f'repeat-{repeat}')
assert all(value==versions[0] for value in versions)
assert versions[0]['torch']=='2.9.1+cu126' and versions[0]['sgl_kernel_runtime_version']=='0.4.6.post1'
recorded=report.read_csv(preserve(mr/'holdout/ml_e2e_comparison/comparison.csv'))
key=lambda row:(row['pair'],row['case'],row['metric'])
expected={(f'repeat-{repeat}',case,metric) for repeat in range(1,6) for case in counts for metric in pair.METRICS}
assert len(recorded)==len(recomputed)==len(expected)==100
saved={key(row):row for row in recorded}; fresh={key(row):row for row in recomputed}
assert set(saved)==set(fresh)==expected
for identity,row in fresh.items():
    previous=saved[identity]
    for field in ('real_value','real_reported_value','predicted_value','signed_error_pct','absolute_percentage_error_pct','trace_request_count','trace_input_tokens','trace_output_tokens','real_benchmark_input_tokens','real_benchmark_output_tokens'):
        assert math.isclose(float(previous[field]),float(row[field]),rel_tol=1e-12,abs_tol=1e-12),(identity,field)
    assert previous['throughput_token_source']==row['throughput_token_source']=='actual_trace'
    for field in ('real_root','replay_root'):
        assert Path(previous[field]).resolve()==Path(row[field]).resolve()
summary=pair.summarize(recomputed)
original=report.read_csv(preserve(mr/'holdout/ml_e2e_comparison/summary.csv'))
summary_key=lambda row:(row['group'],row['value'])
old={summary_key(row):row for row in original}
assert len(old)==len(original)==len(summary)==24
for row in summary:
    prior=old[summary_key(row)]
    assert int(row['n'])==(25 if row['group']=='metric' else 5)
    for field,value in row.items():
        if field not in ('group','value'):
            assert math.isclose(float(prior[field]),float(value),rel_tol=1e-12,abs_tol=1e-12)
observed=score.read_rows([str(mr/'holdout/real')],'TP0*.schedule_batch.jsonl',{1,2})
scored=report.read_csv(preserve(mr/'holdout/ml_teacher_forced/per_batch.csv'))
identity=lambda row:(str(Path(row['schedule_path']).resolve()),int(row['line_number']))
labels={identity(row):row for row in observed}; retained={identity(row):row for row in scored}
assert len(labels)==len(observed)==len(retained)==len(scored)==19018
for step,row in labels.items():
    prior=retained[step]
    assert prior['case']==row['case']
    for field in ('forward_mode','batch_size','sum_extend','sum_past'):
        assert int(prior[field])==row[field]
    assert float(prior['actual_seconds'])==row['actual_seconds']
step_summary=score.summarize(scored)
old_steps=report.read_csv(preserve(mr/'holdout/ml_teacher_forced/summary.csv'))
old_lookup={(row['group'],row['value']):row for row in old_steps}
assert len(step_summary)==len(old_lookup)==len(old_steps)
for row in step_summary:
    prior=old_lookup[row['group'],row['value']]
    for field,value in row.items():
        if field not in ('group','value'):
            assert math.isclose(float(prior[field]),float(value),rel_tol=1e-10,abs_tol=1e-10),(row['group'],row['value'],field)
out.mkdir()
report.write_csv(out/'verified_kv_capacity.csv',capacity_checks)
lines=report.replay_overhead_details(root,[model],out,methods=('ML',))
(out/'replay_overhead_sections.md').write_text('\n'.join(lines)+'\n')
components=report.read_csv(out/'replay_wall_time_components.csv')
assert len(components)==25
iterations=sum(int(row['iterations']) for row in components)
weighted={field:sum(float(row[field]) for row in components)*1000/iterations for field in ('predictor_wall_seconds','accounted_cpu_seconds','predicted_forward_seconds')}
for path,value in sha.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==value,path
report.write_csv(out/'verified_e2e_comparison.csv',recomputed)
report.write_csv(out/'verified_e2e_summary.csv',summary)
report.write_csv(out/'verified_single_step_summary.csv',step_summary)
report.write_csv(out/'source_sha256.csv',[{'path':path,'sha256':value} for path,value in sorted(sha.items())])
result={'model':model,'paired_cases':25,'paired_metric_rows':100,'requests_verified':sum(counts.values())*5,'native_step_rows':len(observed),'original_files_unchanged':len(sha),'native_versions':versions[0],'simulator_iterations':iterations,'weighted_replay_ms_per_iteration':weighted,'fit_gpu_or_latency_adjustment_invoked':False,'bundle_sha256':sha[str(bundle)],'matched_native_replay_capacity_cases':len(capacity_checks),'claim_scope':'All 25 completed native/ML E2E pairs and stored summaries; simulator wall overhead is not native CPU profiling; no fresh AIC or all-13 completion claim.'}
(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
print(json.dumps([row for row in summary if row['group']=='metric']))
print(json.dumps([row for row in summary if row['group']=='case_metric' and row['value'].endswith(':mean_e2e_latency_ms')]))
