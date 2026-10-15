"""Recovery gate fixtures; never start, terminate, or query an actual job."""
import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/resume_pr33824_after_orphan.py'
SPEC = importlib.util.spec_from_file_location('resume_orphan_fixture', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('ready', [False, True])
def test_live_native_always_waits(ready):
    owner = {'pid': 123, 'start_ticks': 456, 'argv': ['original']}
    assert MODULE.decision(owner, dict(owner), ready) == 'wait'


def test_dead_native_without_markers_fails():
    with pytest.raises(RuntimeError, match='without all model completion'):
        MODULE.decision({'pid': 123}, None, False)


def test_dead_native_with_complete_markers_resumes():
    assert MODULE.decision({'pid': 123}, None, True) == 'resume'


@pytest.mark.parametrize('ready', [False, True])
def test_reused_pid_refuses_continuation(ready):
    with pytest.raises(RuntimeError, match='PID was reused'):
        MODULE.decision({'pid': 123, 'start_ticks': 4}, {'pid': 123, 'start_ticks': 5}, ready)


def test_all_models_and_global_marker_required(tmp_path):
    names = [f'model-{i}' for i in range(13)]
    (tmp_path / 'REAL_ML_ALL_MODELS_COMPLETE').write_text('complete\n')
    for name in names[:-1]:
        folder = tmp_path / 'models' / name
        folder.mkdir(parents=True)
        (folder / 'REAL_ML_COMPLETE').write_text('complete\n')
    assert not MODULE.native_complete(tmp_path, names)
    final = tmp_path / 'models' / names[-1]
    final.mkdir()
    (final / 'REAL_ML_COMPLETE').write_text('complete\n')
    assert MODULE.native_complete(tmp_path, names)
    (tmp_path / 'REAL_ML_ALL_MODELS_COMPLETE').write_text('failed\n')
    assert not MODULE.native_complete(tmp_path, names)


def test_missing_process_is_terminal_not_a_time_limit():
    assert MODULE.identity(999999999) is None
