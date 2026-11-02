import json
import subprocess
import sys
from pathlib import Path

import pytest

from urbangraph.data import DataError
from urbangraph.data.filesystem import stream_file
from urbangraph.datasets.serialization import canonical_json
from urbangraph.evaluation import artifacts
from urbangraph.evaluation.artifacts import load_plan, publish_plan, verify_plan
from urbangraph.evaluation.contracts import PlanConfig
from urbangraph.evaluation.dataset import identity
from urbangraph.evaluation.preprocessing import FittedStandardizer
from urbangraph.evaluation.windows import WindowCollection


def test_publication_lineage_and_idempotence(
    canonical_dir: Path, raw_snapshot_dir: Path, tmp_path: Path, config: PlanConfig
) -> None:
    before = {p.name: p.read_bytes() for p in canonical_dir.iterdir()}
    result = publish_plan(
        canonical_dir, config, output_root=tmp_path / "plans", raw_snapshot_dir=raw_snapshot_dir
    )
    assert not result.reused
    assert verify_plan(
        result.directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    ).valid
    dataset, split, plan = load_plan(
        result.directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    )
    assert identity(plan) == result.plan_id
    assert (result.directory / "canonical-manifest.json").read_bytes() == (
        canonical_dir / "manifest.json"
    ).read_bytes()
    assert WindowCollection(dataset, split, fold=0, partition="test")[0].sensor_ids == (
        "sensor-Z",
        "001",
    )
    saved = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in result.directory.iterdir()}
    again = publish_plan(
        canonical_dir, config, output_root=tmp_path / "plans", raw_snapshot_dir=raw_snapshot_dir
    )
    assert again.reused and again.plan_id == result.plan_id
    assert saved == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in result.directory.iterdir()
    }
    assert before == {p.name: p.read_bytes() for p in canonical_dir.iterdir()}
    fitted = json.loads((result.directory / "fitted-transforms.json").read_text())["folds"][0]
    restored = FittedStandardizer.from_dict(fitted)
    assert canonical_json(restored.to_dict()) == canonical_json(fitted)
    assert len((result.directory / "plan.json").read_bytes()) < 50000


@pytest.mark.parametrize(
    "corruption", ["missing", "extra", "plan", "scaler", "manifest", "time", "schema", "identity"]
)
def test_corruption_never_repaired(
    canonical_dir: Path, raw_snapshot_dir: Path, tmp_path: Path, config: PlanConfig, corruption: str
) -> None:
    result = publish_plan(
        canonical_dir, config, output_root=tmp_path / "plans", raw_snapshot_dir=raw_snapshot_dir
    )
    if corruption == "missing":
        (result.directory / "plan.json").unlink()
    elif corruption == "extra":
        (result.directory / "extra.txt").write_text("unexpected")
    elif corruption in {"plan", "scaler", "manifest"}:
        name = {
            "plan": "plan.json",
            "scaler": "fitted-transforms.json",
            "manifest": "manifest.json",
        }[corruption]
        (result.directory / name).write_text("{}")
    else:
        manifest = json.loads((result.directory / "manifest.json").read_text())
        key = {"time": "created_at", "schema": "schema_version", "identity": "plan_id"}[corruption]
        manifest[key] = "invalid"
        (result.directory / "manifest.json").write_text(json.dumps(manifest))
    corrupted = {p.name: p.read_bytes() for p in result.directory.iterdir()}
    assert not verify_plan(
        result.directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    ).valid
    with pytest.raises(DataError):
        publish_plan(
            canonical_dir, config, output_root=tmp_path / "plans", raw_snapshot_dir=raw_snapshot_dir
        )
    assert corrupted == {p.name: p.read_bytes() for p in result.directory.iterdir()}


def test_rehashed_scaler_tampering_still_fails(
    canonical_dir: Path, raw_snapshot_dir: Path, tmp_path: Path, config: PlanConfig
) -> None:
    result = publish_plan(
        canonical_dir, config, output_root=tmp_path / "plans", raw_snapshot_dir=raw_snapshot_dir
    )
    fitted_path = result.directory / "fitted-transforms.json"
    fitted = json.loads(fitted_path.read_text())
    fitted["folds"][0]["means"][0] = 9999
    fitted_path.write_text(json.dumps(fitted))
    manifest_path = result.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    size, digest = stream_file(fitted_path)
    for record in manifest["artifacts"]:
        if record["relative_path"] == fitted_path.name:
            record.update(size_bytes=size, sha256=digest)
    manifest_path.write_text(json.dumps(manifest))
    checked = verify_plan(
        result.directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    )
    assert not checked.valid and "training-only" in checked.issues[0]


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_staging_failure_cleanup(
    canonical_dir: Path,
    raw_snapshot_dir: Path,
    tmp_path: Path,
    config: PlanConfig,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = tmp_path / "plans"

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(artifacts, "write_json" if failure == "write" else "_publish", fail)
    with pytest.raises(OSError, match="simulated"):
        publish_plan(canonical_dir, config, output_root=root, raw_snapshot_dir=raw_snapshot_dir)
    assert not list(root.rglob(".staging-*"))
    assert not list(root.rglob(".ingest.lock"))
    assert not list(root.rglob("manifest.json"))


def test_overlap_and_invalid_input(
    canonical_dir: Path, raw_snapshot_dir: Path, tmp_path: Path, config: PlanConfig
) -> None:
    for root in (
        canonical_dir,
        canonical_dir / "evaluation",
        canonical_dir.parents[2],
        canonical_dir.parents[3],
    ):
        with pytest.raises(DataError):
            publish_plan(canonical_dir, config, output_root=root, raw_snapshot_dir=raw_snapshot_dir)
    with pytest.raises((DataError, OSError)):
        publish_plan(
            tmp_path / "missing",
            config,
            output_root=tmp_path / "plans",
            raw_snapshot_dir=raw_snapshot_dir,
        )


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "urbangraph.evaluation.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("protocol", ["benchmark", "rolling"])
def test_cli_complete_flow(
    canonical_dir: Path, raw_snapshot_dir: Path, tmp_path: Path, protocol: str
) -> None:
    created = run_cli(
        "plan",
        "--canonical-dir",
        str(canonical_dir),
        "--raw-snapshot-dir",
        str(raw_snapshot_dir),
        "--protocol",
        protocol,
        "--output-root",
        str(tmp_path / "plans"),
        "--json",
    )
    assert created.returncode == 0, created.stderr + created.stdout
    result = json.loads(created.stdout)
    assert result["verified"] and result["sample_counts"][0]["train"] > 0
    for command in ("verify", "inspect", "windows-preview"):
        completed = run_cli(
            command,
            "--canonical-dir",
            str(canonical_dir),
            "--raw-snapshot-dir",
            str(raw_snapshot_dir),
            "--plan-dir",
            result["directory"],
            "--json",
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        response = json.loads(completed.stdout)
        if command == "windows-preview":
            assert max(response["preview"]["input_positions"]) < min(
                response["preview"]["target_positions"]
            )
    human = run_cli(
        "verify",
        "--canonical-dir",
        str(canonical_dir),
        "--raw-snapshot-dir",
        str(raw_snapshot_dir),
        "--plan-dir",
        result["directory"],
    )
    assert human.returncode == 0 and "VERIFY: PASS" in human.stdout


def test_cli_config_errors_and_help(
    canonical_dir: Path, raw_snapshot_dir: Path, tmp_path: Path, config: PlanConfig
) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    result = run_cli(
        "plan",
        "--canonical-dir",
        str(canonical_dir),
        "--raw-snapshot-dir",
        str(raw_snapshot_dir),
        "--config",
        str(path),
        "--output-root",
        str(tmp_path / "plans"),
        "--json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    bad_preview = run_cli(
        "windows-preview",
        "--canonical-dir",
        str(canonical_dir),
        "--raw-snapshot-dir",
        str(raw_snapshot_dir),
        "--plan-dir",
        json.loads(result.stdout)["directory"],
        "--sample",
        "-1",
        "--json",
    )
    assert bad_preview.returncode == 1 and "error" in json.loads(bad_preview.stdout)
    path.write_text('{"window":{"history":0}}')
    failure = run_cli(
        "plan",
        "--canonical-dir",
        str(canonical_dir),
        "--raw-snapshot-dir",
        str(raw_snapshot_dir),
        "--config",
        str(path),
        "--json",
    )
    assert failure.returncode == 1 and "error" in json.loads(failure.stdout)
    assert run_cli("plan").returncode == 2
    for command in ("plan", "inspect", "verify", "windows-preview"):
        assert run_cli(command, "--help").returncode == 0
    assert (
        run_cli(
            "verify",
            "--canonical-dir",
            str(tmp_path / "missing"),
            "--raw-snapshot-dir",
            str(raw_snapshot_dir),
            "--plan-dir",
            str(tmp_path / "absent"),
            "--json",
        ).returncode
        == 1
    )
