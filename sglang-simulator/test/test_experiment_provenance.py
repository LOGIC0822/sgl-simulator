"""Source-provenance tests; no GPU initialization or external repo mutations."""
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "experiment_provenance", ROOT / "tools/capture_pr33824_experiment_provenance.py"
)
assert SPEC and SPEC.loader
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


def test_archive_retains_dirty_collector_and_untracked_sources(tmp_path):
    source = tmp_path / "aic"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    collector = source / "collector/sglang/collect_attn.py"
    collector.parent.mkdir(parents=True)
    collector.write_text("page_size = 64\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    collector.write_text("page_size = 1\n")
    extra = source / "collector/helper.py"
    extra.write_text("# untracked adapter\n")
    sdk = source / "src/aiconfigurator/sdk/operations/elementwise.py"
    sdk.parent.mkdir(parents=True)
    sdk.write_text("# analytic operation\n")
    table = source / "src/aiconfigurator/systems/data/a100/large.json"
    table.parent.mkdir(parents=True)
    table.write_text('{"not_source": true}\n')
    out = tmp_path / "snapshot"
    out.mkdir()
    result = provenance.archive_aic_sources(source, out)
    hashes = json.loads((out / result["sha256_file"]).read_text())
    assert result["dirty"] and result["source_files"] == 3
    assert hashes["collector/sglang/collect_attn.py"] == provenance.digest(collector)
    assert "collector/helper.py" in hashes
    assert "src/aiconfigurator/sdk/operations/elementwise.py" in hashes
    assert "src/aiconfigurator/systems/data/a100/large.json" not in hashes
    assert "+page_size = 1" in (out / result["patch_file"]).read_text()
    with tarfile.open(out / result["archive_file"]) as archive:
        assert archive.extractfile("collector/sglang/collect_attn.py").read() == b"page_size = 1\n"
    assert collector.read_text() == "page_size = 1\n"


def test_archive_clean_checkout_status_is_not_dirty(tmp_path):
    source = tmp_path / "empty-aic"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit",
                    "--allow-empty", "-qm", "fixture"], check=True)
    out = tmp_path / "snapshot"
    out.mkdir()
    result = provenance.archive_aic_sources(source, out)
    assert result["dirty"] is False
    assert result["source_files"] == 0
