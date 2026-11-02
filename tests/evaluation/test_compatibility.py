"""Historical provenance remains checked while all semantics are reconstructed."""

import json
from pathlib import Path

import pytest

from urbangraph.benchmarks.audit import leakage_audit
from urbangraph.data.filesystem import stream_file
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.evaluation import artifacts
from urbangraph.evaluation.artifacts import code_provenance, publish_plan, verify_plan
from urbangraph.evaluation.contracts import POSITIONAL_PROTOCOL, EvaluationProtocol, PlanConfig

from .test_artifacts_cli import run_cli


def test_historical_plan(
    canonical_dir: Path,
    raw_snapshot_dir: Path,
    tmp_path: Path,
    config: PlanConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = read_json(Path(__file__).parent / "fixtures" / "ug003r-code.json")
    old["dependencies"] = code_provenance()["dependencies"]
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "code_provenance", lambda: old)
        result = publish_plan(
            canonical_dir, config, output_root=tmp_path / "p", raw_snapshot_dir=raw_snapshot_dir
        )
    before = {p.name: p.read_bytes() for p in result.directory.iterdir()}
    assert verify_plan(
        result.directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    ).valid
    assert (
        leakage_audit(result.directory, canonical_dir, raw_snapshot_dir=raw_snapshot_dir)["status"]
        == "PASS"
    )
    assert before == {p.name: p.read_bytes() for p in result.directory.iterdir()}
    # Even recognized old provenance cannot excuse rehashed fitted-state corruption.
    fitted = result.directory / "fitted-transforms.json"
    value = json.loads(fitted.read_text())
    value["folds"][0]["means"][0] += 1
    fitted.write_text(json.dumps(value))
    manifest_path = result.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["artifacts"]:
        if record["relative_path"] == fitted.name:
            record["size_bytes"], record["sha256"] = stream_file(fitted)
    manifest_path.write_text(json.dumps(manifest))
    checked = verify_plan(
        result.directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    )
    assert not checked.valid and "training-only" in checked.issues[0]


@pytest.mark.parametrize("change", ["source", "dependency", "protocol"])
def test_legacy_allowlist_is_narrow(
    canonical_dir: Path,
    raw_snapshot_dir: Path,
    tmp_path: Path,
    config: PlanConfig,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    old = read_json(Path(__file__).parent / "fixtures" / "ug003r-code.json")
    old["dependencies"] = code_provenance()["dependencies"]
    if change == "source":
        old["source_sha256"] = object_dict(old["source_sha256"]) | {"unrecognized.py": "0" * 64}
    elif change == "dependency":
        old["dependencies"] = object_dict(old["dependencies"]) | {"numpy": "unrecognized"}
    else:
        config = PlanConfig(protocol=EvaluationProtocol(POSITIONAL_PROTOCOL, "benchmark"))
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "code_provenance", lambda: old)
        result = publish_plan(
            canonical_dir, config, output_root=tmp_path / "p", raw_snapshot_dir=raw_snapshot_dir
        )
    checked = verify_plan(
        result.directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    )
    assert not checked.valid and "provenance" in checked.issues[0]


def test_positional_cli(canonical_dir: Path, raw_snapshot_dir: Path, tmp_path: Path) -> None:
    before = {p.name: stream_file(p) for p in canonical_dir.iterdir()}
    result = run_cli(
        "plan",
        "--protocol",
        "benchmark-positional",
        "--canonical-dir",
        str(canonical_dir),
        "--raw-snapshot-dir",
        str(raw_snapshot_dir),
        "--output-root",
        str(tmp_path / "p"),
        "--json",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout)
    path = Path(output["directory"])
    assert verify_plan(path, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir).valid
    audit = leakage_audit(path, canonical_dir, raw_snapshot_dir=raw_snapshot_dir)
    assert object_dict(audit["temporal_diagnostics"])["discontinuity_count"] == 0
    assert before == {p.name: stream_file(p) for p in canonical_dir.iterdir()}
    repeated = run_cli(
        "plan",
        "--protocol",
        "benchmark-positional",
        "--canonical-dir",
        str(canonical_dir),
        "--raw-snapshot-dir",
        str(raw_snapshot_dir),
        "--output-root",
        str(tmp_path / "p"),
        "--json",
    )
    assert repeated.returncode == 0 and json.loads(repeated.stdout)["reused"]
