"""Progress fixtures distinguish experimental and supervisory processes."""
import importlib.util
from pathlib import Path

import pytest


SIM = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('expanded_status_fixture', SIM / 'tools/write_pr33824_expanded_status.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SCRIPTS = {
    'comprehensive': 'run_pr33824_comprehensive_predictor_matrix.sh',
    'native': 'run_pr33824_data_tools_same_distribution_matrix.sh',
    'resume': 'resume_pr33824_after_orphan.py',
}


def fake_process(proc_root, pid, argv, state='S'):
    folder = proc_root / str(pid)
    folder.mkdir(parents=True)
    (folder / 'stat').write_text(f'{pid} (python) {state} ' + ' '.join(['0'] * 21))
    (folder / 'cmdline').write_bytes(('\0'.join(argv) + '\0').encode())


@pytest.mark.parametrize('role', SCRIPTS)
def test_exact_actor_detected(tmp_path, role):
    root = tmp_path / 'results'
    proc = tmp_path / 'proc'
    fake_process(proc, 123, ['python', str(SIM / 'scripts' / SCRIPTS[role]), '--results-root', str(root)])
    actors = MODULE.scan_processes(root, proc)
    assert actors[role] == ['123']
    assert sum(len(values) for values in actors.values()) == 1


def test_wrong_root_shell_text_and_zombie_are_ignored(tmp_path):
    root, proc = tmp_path / 'results', tmp_path / 'proc'
    script = str(SIM / 'scripts' / SCRIPTS['native'])
    fake_process(proc, 1, ['bash', script, '--results-root', str(tmp_path / 'other')])
    fake_process(proc, 2, ['bash', '-lc', f'bash {script} --results-root {root}'])
    fake_process(proc, 3, ['bash', script, '--results-root', str(root)], state='Z')
    assert not any(MODULE.scan_processes(root, proc).values())


def test_resolved_root_alias_matches(tmp_path):
    root = tmp_path / 'results'
    root.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    proc = tmp_path / 'proc'
    fake_process(proc, 4, ['bash', str(SIM / 'scripts' / SCRIPTS['native']), '--results-root', str(alias)])
    assert MODULE.scan_processes(root, proc)['native'] == ['4']


@pytest.mark.parametrize('actors, expected', [
    ({'comprehensive': ['1'], 'native': [], 'resume': []}, '运行中'),
    ({'comprehensive': [], 'native': ['2'], 'resume': []}, '原生采集运行中；未检测到总调度进程'),
    ({'comprehensive': [], 'native': ['2'], 'resume': ['3']}, '原生采集运行中；后续阶段等待续跑'),
    ({'comprehensive': [], 'native': [], 'resume': ['3']}, '续跑监控仍存活；未检测到实验进程，请检查续跑状态'),
    ({'comprehensive': [], 'native': [], 'resume': []}, '当前没有实验活进程，请检查日志'),
])
def test_only_supervision_is_not_a_running_experiment(actors, expected):
    assert MODULE.process_state(False, actors) == expected
    assert MODULE.process_state(True, actors) == '全部完成'


def test_status_reports_orphan_without_claiming_completion(tmp_path, monkeypatch):
    (tmp_path / 'run_metadata.txt').write_text('models=model\ntraining_requests_total=800\nheldout_requests_total=1280\n')
    (tmp_path / 'model_manifest.tsv').write_text('model\tmem_fraction_static\nmodel\t0.80\n')
    actors = {'comprehensive': [], 'native': ['2'], 'resume': ['3']}
    monkeypatch.setattr(MODULE, 'scan_processes', lambda root: actors)
    assert not MODULE.write_status(tmp_path)
    text = (tmp_path / 'status.md').read_text()
    assert '原生采集运行中；后续阶段等待续跑' in text
    assert '原生采集矩阵 PID：2' in text and '续跑监控 PID：3' in text
    assert 'ML/AIC 均完成模型 0/1' in text
