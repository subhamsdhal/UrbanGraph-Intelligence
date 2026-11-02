import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "urbangraph.data.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_cli_end_to_end(source_dir: Path, tmp_path: Path, invented_source_grant) -> None:
    # This authority is limited to the exact invented fixture bytes and destination.
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    arguments = (
        "ingest",
        "--dataset",
        "metr-la",
        "--source-dir",
        str(source_dir),
        "--raw-root",
        str(tmp_path / "raw"),
        "--source-reference",
        "synthetic-fixture",
    )
    ingested = _run(*arguments)
    assert ingested.returncode == 0, ingested.stderr
    output = json.loads(ingested.stdout)
    assert output["dataset_id"] == "metr-la"
    assert output["file_count"] == 2
    assert output["verified"] is True
    assert output["reused"] is False
    assert "synthetic-fixture" not in ingested.stdout + ingested.stderr
    directory = str(Path(output["manifest_path"]).parent)
    for command in ("verify", "inspect"):
        result = _run(command, "--snapshot-dir", directory)
        assert result.returncode == 0, result.stderr
        verified = json.loads(result.stdout)
        assert verified["verified"] is True
        assert verified["checked_files"] == 2
        if command == "inspect":
            assert len(verified["files"]) == 2
            assert "source" not in verified
            assert "creator_environment" not in verified
    repeated = _run(*arguments)
    assert repeated.returncode == 0
    assert json.loads(repeated.stdout)["reused"] is True
    (Path(directory) / "payload" / "metr-la.h5").write_bytes(b"corrupt")
    corrupt = _run("verify", "--snapshot-dir", directory)
    assert corrupt.returncode == 1
    assert json.loads(corrupt.stdout)["verified"] is False
    assert _run("inspect", "--snapshot-dir", directory).returncode == 1


@pytest.mark.parametrize("command", ["ingest", "verify", "inspect"])
def test_help(command: str) -> None:
    result = _run(command, "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_cli_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Missing or malformed input must be denied before the CLI probes its path.
    policy = tmp_path / "empty-rights-policy.json"
    policy.write_text(json.dumps({"version": 1, "grants": []}), encoding="utf-8")
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    invalid = _run("ingest", "--dataset", "../invalid", "--source-dir", str(tmp_path))
    assert invalid.returncode == 1
    assert json.loads(invalid.stderr)["level"] == "ERROR"
    assert "Traceback" not in invalid.stderr
    missing = _run("ingest", "--dataset", "metr-la", "--source-dir", str(tmp_path / "missing"))
    assert missing.returncode == 1
    assert "No matching rights grant" in missing.stderr
    assert _run("ingest").returncode == 2
