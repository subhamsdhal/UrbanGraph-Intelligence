import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from . import SnapshotFactory


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "urbangraph.datasets.cli", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


def test_dataset_commands(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    raw = snapshot_factory(traffic_frame)
    for command in ("validate", "profile"):
        result = _run(command, "--snapshot-dir", str(raw.directory), "--json")
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        assert output["dataset_id"] == "metr-la"
        assert output["accepted"] is True
        assert "Traceback" not in result.stderr
    human = _run("validate", "--snapshot-dir", str(raw.directory))
    assert human.returncode == 0 and "VALIDATE: PASS" in human.stdout
    produced = _run(
        "materialize",
        "--snapshot-dir",
        str(raw.directory),
        "--output-root",
        str(tmp_path / "interim"),
        "--json",
    )
    assert produced.returncode == 0, produced.stderr
    output = json.loads(produced.stdout)
    verified = _run(
        "verify",
        "--artifact-dir",
        output["directory"],
        "--snapshot-dir",
        str(raw.directory),
        "--json",
    )
    assert verified.returncode == 0
    assert json.loads(verified.stdout)["verified"] is True


def test_cli_validation_error(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame
) -> None:
    raw = snapshot_factory(traffic_frame.iloc[::-1])
    result = _run("validate", "--snapshot-dir", str(raw.directory), "--json")
    assert result.returncode == 1
    assert json.loads(result.stdout)["structurally_valid"] is False
    assert json.loads(result.stderr)["level"] == "ERROR"
    assert "Traceback" not in result.stderr


def test_cli_policy_failure(snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame) -> None:
    raw = snapshot_factory(traffic_frame)
    result = _run(
        "profile",
        "--snapshot-dir",
        str(raw.directory),
        "--reject-check",
        "negative_values",
        "--json",
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["accepted"] is False


@pytest.mark.parametrize("command", ["validate", "profile", "materialize", "verify"])
def test_dataset_help(command: str) -> None:
    result = _run(command, "--help")
    assert result.returncode == 0 and "usage:" in result.stdout


def test_missing_snapshot_error(tmp_path: Path) -> None:
    result = _run("validate", "--snapshot-dir", str(tmp_path / "missing"), "--json")
    assert result.returncode == 1
    assert "error" in json.loads(result.stdout)
    assert _run("validate").returncode == 2


def test_base_import_does_not_load_data_stack() -> None:
    command = (
        "import sys; import urbangraph; import urbangraph.data; import urbangraph.datasets; "
        "assert not {'numpy','pandas','tables','pyarrow'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False, timeout=10
    )
    assert result.returncode == 0, result.stderr
