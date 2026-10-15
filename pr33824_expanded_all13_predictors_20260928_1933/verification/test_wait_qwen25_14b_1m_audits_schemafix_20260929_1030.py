"""Readonly regression tests against the executed saved-score audit schema."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

DIRECTORY = Path(__file__).resolve().parent


def load_worker(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load_worker("wait_qwen25_14b_1m_audits_20260929_0852.py")
worker = load_worker("wait_qwen25_14b_1m_audits_schemafix_20260929_1030.py")
score = DIRECTORY / worker.STAGES[0][1]
manifest_path = score / "results/manifest.json"
execution_path = score / "execution.command.json"


class SchemaTests(unittest.TestCase):
    def altered_result(self, change):
        saved = worker.read_json(manifest_path)
        changed = deepcopy(saved)
        change(changed)
        reader = worker.read_json

        def read(path):
            return changed if path.resolve() == manifest_path.resolve() else reader(path)

        with mock.patch.object(worker, "read_json", side_effect=read):
            self.assertFalse(worker.successful_audit(score))

    def test_reproduces_original_false_failure_without_rerunning_audit(self):
        self.assertEqual(worker.read_json(execution_path)["exit_code"], 0)
        self.assertNotIn("model", worker.read_json(manifest_path))
        self.assertFalse(original.successful_audit(score))

    def test_accepts_executed_schema_with_pinned_source_and_exact_model_bundle(self):
        self.assertTrue(worker.successful_audit(score))

    def test_preserves_calibration_schema(self):
        self.assertTrue(worker.successful_audit(DIRECTORY / worker.CALIBRATION))

    def test_rejects_explicit_wrong_model(self):
        self.altered_result(lambda row: row.update(model="Qwen2.5-14B-Instruct"))

    def test_rejects_wrong_bundle(self):
        self.altered_result(lambda row: row.update(bundle_sha256="0" * 64))

    def test_rejects_wrong_test_request_count(self):
        self.altered_result(lambda row: row.update(independent_test_requests=1279))

    def test_rejects_wrong_seed(self):
        self.altered_result(lambda row: row["groups"][0].update(seed=202609280))

    def test_rejects_missing_group(self):
        self.altered_result(lambda row: row["groups"].pop())

    def test_rejects_wrong_case_count(self):
        self.altered_result(lambda row: row["groups"][0]["cases"]["latency_256x128_r1"].update(requests=31))

    def test_rejects_timing_identity_mismatch(self):
        self.altered_result(lambda row: row.update(raw_score_identity_rows_verified=1))

    def test_rejects_wrong_backend(self):
        self.altered_result(lambda row: row["actual_settings_verified"].update(attention_backend="triton"))

    def test_rejects_wrong_page_size(self):
        self.altered_result(lambda row: row["actual_settings_verified"].update(page_size="1"))

    def test_rejects_failed_execution(self):
        reader = worker.read_json

        def read(path):
            row = reader(path)
            return dict(row, exit_code=1) if path.resolve() == execution_path.resolve() else row

        with mock.patch.object(worker, "read_json", side_effect=read):
            self.assertFalse(worker.successful_audit(score))

    def test_rejects_other_execution_target(self):
        reader = worker.read_json

        def read(path):
            row = reader(path)
            if path.resolve() == execution_path.resolve():
                row = dict(row, python_argv=["python", "/wrong/audit.py"])
            return row

        with mock.patch.object(worker, "read_json", side_effect=read):
            self.assertFalse(worker.successful_audit(score))

    def test_rejects_altered_pinned_audit_source(self):
        digest = mock.Mock()
        digest.hexdigest.return_value = "0" * 64
        with mock.patch.object(worker.hashlib, "sha256", return_value=digest):
            self.assertFalse(worker.successful_audit(score))

    def test_native_and_ready_gates_are_identical_source(self):
        import inspect
        for name in ("native_complete", "ready", "process_identity"):
            self.assertEqual(inspect.getsource(getattr(worker, name)), inspect.getsource(getattr(original, name)))


if __name__ == "__main__":
    def forbidden(*args, **kwargs):
        raise AssertionError("Regression tests cannot launch subprocesses or experiments")

    with mock.patch.object(worker.subprocess, "run", side_effect=forbidden):
        unittest.main(verbosity=2)
