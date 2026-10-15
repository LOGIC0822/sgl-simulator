"""Read-only completed-calibration audit for the dual-chunk 14B-1M checkpoint."""
from pathlib import Path
import json
import os
import sys
import time

def run():
    import ast,csv,hashlib,json,math,shlex
    from pathlib import Path
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    def forbidden(*args,**kwargs): raise AssertionError('no diagnostic fit/predict is allowed')
    HistGradientBoostingRegressor.fit=forbidden
    HistGradientBoostingRegressor.predict=forbidden
    sim=Path.cwd(); root=(sim/'test-results/pr33824_expanded_all13_predictors_20260928_1933').resolve()
    model='Qwen2.5-14B-Instruct-1M'; base=root/'models'/model
    out=Path(__file__).resolve().parent/'results'
    assert not out.exists()
    def metadata(path): return dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)
    settings=metadata(base/'model_settings.txt')
    flash_stage=(sim/'test-results/pr33824_sm80_fa2_compat_v6_20260918_1305/stage').resolve()
    assert Path(settings['fa2_compat_stage']).resolve()==flash_stage
    assert settings['attention_backend']=='dual_chunk_flash_attn' and settings['page_size']=='256'
    trainroot=base/'calibration'
    bundle_path=trainroot/'training'/f'{model}_hgbmono_p50.pkl'
    paths={root/'run_metadata.txt',base/'model_settings.txt',bundle_path,trainroot/'train.command.txt',trainroot/'train.log',trainroot/'training/eval/split_summary.csv',trainroot/'training/eval/dataset_cases.csv',sim/'src/sglang_simulator/baseline/collector.py',sim/'src/sglang_simulator/time_predictor/ml_features.py',sim/'tools/train_latency_model.py'}
    paths.add(Path(__file__).resolve())
    paths.add(flash_stage/'flash_attn/__init__.py')
    groups=sorted((trainroot/'real').glob('repeat-*'))
    assert len(groups)==5 and {g.name for g in groups}=={f'repeat-{i}' for i in range(1,6)}
    native=[]
    case_files=[]
    for group in groups:
        marker=group/'BASELINE_COLLECTION_COMPLETE'
        assert marker.read_text().strip()=='complete'
        info=metadata(group/'run_metadata.txt')
        assert info['seed']==settings['training_seeds'].split(',')[int(group.name.split('-')[-1])-1]
        assert info['model']==model and float(info['mem_fraction_static'])==.60
        assert info['source_revision']=='59799a368793b9f795baf59b067233a65ad8e38e'
        assert Path(info['model_path']).resolve()==Path('/home/ms-admin/data/models/Qwen2.5-14B-Instruct-1M').resolve()
        assert info['context_length']=='2048' and info['max_running_requests']=='32' and info['chunked_prefill_size']=='512'
        assert info['attention_backend']=='dual_chunk_flash_attn' and info['cuda_graph']=='disabled' and info['disable_overlap_schedule']=='1'
        assert info['page_size']=='256'
        assert info['flash_attention_policy']=='local_FA2_fallback_for_dual_chunk; stage_path='+str(flash_stage)
        native.append(metadata(group/'package_versions.txt'))
        paths.update((marker,group/'run_metadata.txt',group/'package_versions.txt'))
        statuses=sorted((group/'cases').glob('*/status.txt'))
        assert {p.parent.name for p in statuses}==set(settings['training_case'].split(','))
        for status in statuses:
            assert status.read_text().strip()=='ok'
            folder=status.parent
            files=[status,folder/'TP0.manifest.json',folder/'TP0.raw_request.jsonl',folder/'autobench_trace.jsonl',folder/'TP0.schedule_batch.jsonl']
            paths.update(files)
            paths.update((folder.parent/(folder.name+'.server.command.txt'),folder.parent/(folder.name+'.benchmark.command.txt'),folder/'benchmark.jsonl'))
            case_files.append((group.name,folder))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    requests=set(); prompts=set(); results=[]
    for repeat,folder in case_files:
        manifest=json.loads((folder/'TP0.manifest.json').read_text())
        server=shlex.split((folder.parent/(folder.name+'.server.command.txt')).read_text())
        client=shlex.split((folder.parent/(folder.name+'.benchmark.command.txt')).read_text())
        def argument(argv,key):
            assert argv.count(key)==1,(folder,key)
            return argv[argv.index(key)+1]
        expected_server={'--device':'cuda','--tp-size':'1','--context-length':'2048',
                         '--mem-fraction-static':'0.60','--max-running-requests':'32',
                         '--chunked-prefill-size':'512','--page-size':'256',
                         '--attention-backend':'dual_chunk_flash_attn'}
        assert argument(server,'-m')=='sglang_simulator.baseline.launch_server'
        for key,value in expected_server.items(): assert argument(server,key)==value,(folder,key)
        assert '--disable-cuda-graph' in server and '--disable-overlap-schedule' in server
        assert Path(argument(server,'--model-path')).resolve()==Path('/home/ms-admin/data/models/Qwen2.5-14B-Instruct-1M').resolve()
        assert Path(argument(server,'--collection-output-dir')).resolve()==folder.resolve()
        assert argument(client,'-m')=='sglang.benchmark.serving'
        assert argument(client,'--dataset-name')=='random-ids'
        assert argument(client,'--random-range-ratio')=='0' and '--profile' in client
        assert int(argument(client,'--seed'))==202609279+int(repeat.split('-')[-1])
        assert '--tokenize-prompt' not in client
        benchmark=[json.loads(line) for line in (folder/'benchmark.jsonl').read_text().splitlines() if line.strip()][-1]
        assert benchmark['server_info']['device']=='cuda'
        assert benchmark['server_info']['page_size']==256 and benchmark['server_info']['attention_backend']=='dual_chunk_flash_attn'
        assert Path(benchmark['server_info']['model_path']).name==model
        assert benchmark['server_info']['disable_cuda_graph'] and benchmark['server_info']['disable_overlap_schedule']
        raw=[json.loads(line) for line in (folder/'TP0.raw_request.jsonl').open() if line.strip()]
        traces=[json.loads(line) for line in (folder/'autobench_trace.jsonl').open() if line.strip()]
        expected=32 if folder.name=='latency_256x128_r1' else 64
        assert len(raw)==len(traces)==manifest['request_count']==benchmark['completed']==int(argument(client,'--num-prompts'))==expected
        ids={item['rid'] for item in raw}
        assert len(ids)==len(raw) and not requests & ids
        requests.update(ids)
        replay={item['metadata']['source_rid']:item for item in traces};assert set(replay)==ids
        for item in raw:
            copied=replay[item['rid']]
            assert len(item['input_ids'])==item['input_length']==copied['prompt_len']
            assert item['input_ids']==copied['prompt'] and len(item['output_ids'])==item['output_length']==copied['output_len']
            prompts.add(hashlib.sha256(json.dumps(item['input_ids'],separators=(',',':')).encode()).hexdigest())
        batches=[json.loads(line) for line in (folder/'TP0.schedule_batch.jsonl').open() if line.strip()]
        assert len(batches)==manifest['schedule_batch_count']
        assert manifest['batch_timing']=='device_synchronized' and manifest['batch_latency_unit']=='seconds'
        for item in batches:
            latency=float(item['iter_latency'])
            assert math.isfinite(latency) and latency>0 and item['request_infos'] and item['forward_mode'] in (1,2)
            assert math.isclose(item['end_timestamp']-item['start_timestamp'],latency,rel_tol=1e-12,abs_tol=1e-12)
            assert {req['rid'] for req in item['request_infos']}<=ids
        results.append({'repeat':repeat,'case':folder.name,'requests':len(raw),'schedule_rows':len(batches),'prefill_rows':sum(r['forward_mode']==1 for r in batches),'decode_rows':sum(r['forward_mode']==2 for r in batches)})
    assert len(requests)==800 and len(results)==15
    keys=('python','torch','torch_cuda','sglang','sgl_kernel_runtime_version','sglang-kernel','sgl_kernel_file','triton','flash_attn_file','SGLANG_SIMULATOR_FORCE_FA2_FLASH_ATTN')
    assert len({tuple(row[key] for key in keys) for row in native})==1
    expected_versions={'python':'3.10.21','torch':'2.9.1+cu126','torch_cuda':'12.6',
                       'sgl_kernel_runtime_version':'0.4.6.post1','sglang-kernel':'0.4.6.post1+cu129',
                       'triton':'3.5.1'}
    assert all(native[0][key]==value for key,value in expected_versions.items())
    assert Path(native[0]['sgl_kernel_file']).parent.parent.name=='minimal-stage-v4'
    assert Path(native[0]['flash_attn_file']).resolve()==flash_stage/'flash_attn/__init__.py'
    assert native[0]['SGLANG_SIMULATOR_FORCE_FA2_FLASH_ATTN']=='1'
    bundle=joblib.load(bundle_path)
    assert bundle['train_rows']==sum(r['schedule_rows'] for r in results)>0
    assert bundle['train_cases']==15 and bundle['forward_modes']==[1,2]
    assert {Path(p).resolve() for p in bundle['source_data_roots']}=={(g/'cases').resolve() for g in groups}
    feature_node=next(node for node in ast.parse((sim/'src/sglang_simulator/time_predictor/ml_features.py').read_text()).body if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='ML_FEATURE_NAMES' for n in node.targets))
    assert tuple(bundle['features'])==ast.literal_eval(feature_node.value)
    params=bundle['model'].get_params()
    assert params['loss']=='quantile' and params['quantile']==.5 and params['random_state']==202609280
    assert params['max_iter']==400 and params['max_leaf_nodes']==63 and params['learning_rate']==.04 and params['l2_regularization']==.1
    assert params['early_stopping'] is True and params['validation_fraction']==.15 and params['n_iter_no_change']==20
    assert 0<int(bundle['model'].n_iter_)<=params['max_iter']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in before.items())
    out.mkdir(parents=True)
    def write_csv(path,rows):
        with path.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_csv(out/'case_counts.csv',results)
    write_csv(out/'source_sha256.csv',[{'path':p,'sha256':h} for p,h in before.items()])
    record={'model':model,'training_requests':len(requests),'unique_training_prompts':len(prompts),'training_cases':len(results),'timing_rows':bundle['train_rows'],'native_versions':{key:native[0][key] for key in keys},'p50_bundle_sha256':before[str(bundle_path)],'actual_p50_iterations':int(bundle['model'].n_iter_),'parameters':params,'input_contract':bundle['input_contract'],'source_files_verified_unchanged':len(before),'no_fit_prediction_gpu_or_measurement_change':True,'scope':'Completed calibration and saved-bundle identity only; the independent native holdout is still collecting, so no held-out score is inferred.'}
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))

if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    script = Path(__file__).resolve()
    execution = {
        "python_argv": [sys.executable, *sys.argv],
        "launch_command": ["env", "PYTHONNOUSERSITE=1", "CUDA_VISIBLE_DEVICES=",
                           "OMP_NUM_THREADS=1", "nice", "-n", "15", sys.executable, str(script)],
        "cwd": str(Path.cwd()),
        "environment": {key: os.environ.get(key) for key in
                        ("PYTHONNOUSERSITE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
        "started_unix": time.time(),
        "source": "Adapts executed qwen25_14b_completed_calibration_20260929_0621/audit.py; exact 1M checkpoint, dual_chunk_flash_attn/page256, unchanged sgl_kernel-v4 plus distinct FlashAttention-v6 import and policy; manifest-derived timing row counts",
        "scope": "Completed Qwen2.5-14B-Instruct-1M calibration only; fit and predict both raise; no GPU work",
        "exit_code": 1,
    }
    try:
        run()
        execution["exit_code"] = 0
    finally:
        execution["finished_unix"] = time.time()
        with (script.parent / "execution.command.json").open("x", encoding="utf-8") as stream:
            json.dump(execution, stream, ensure_ascii=False, indent=2)

