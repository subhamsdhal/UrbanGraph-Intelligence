import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import DataError, RawFileRecord, content_identity, ingest_dataset
from urbangraph.data.filesystem import discover, stream_file
from urbangraph.datasets.governed_canonical import create_canonical
from urbangraph.datasets.serialization import read_json, write_json
from urbangraph.evaluation.artifacts import publish_plan
from urbangraph.evaluation.contracts import PlanConfig
from urbangraph.evaluation.dataset import identity
from urbangraph.graphs.artifacts import build_graph
from urbangraph.graphs.contracts import GraphSpecification
from urbangraph.stgnn.artifacts import publish_run, restore, verify_run
from urbangraph.stgnn.cli import main
from urbangraph.stgnn.contracts import GraphModelConfig, TrainingConfig
from urbangraph.stgnn.runner import run_experiment


def test_graph_research_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grants: list[dict[str, object]] = []
    policy = tmp_path / "invented-stgnn-rights.json"

    def grant(
        source_id: str,
        source: Path,
        digest: str,
        operation: str,
        purpose: str,
        destination: Path | None = None,
    ) -> None:
        now = datetime.now(UTC)
        grants.append(
            {
                "source_id": source_id,
                "source_sha256": digest,
                "source_path": str(source),
                "purpose": purpose,
                "operation": operation,
                "destination_root": str(destination if destination is not None else tmp_path),
                "evidence_reference": "invented-stgnn-integration-fixture-only",
                "status": "approved",
                "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                "retention_days": 1,
            }
        )
        policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
        monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    frame = pd.DataFrame(
        np.arange(600).reshape(300, 2) % 21,
        columns=["a", "b"],
        index=pd.date_range("2024-01-01", periods=300, freq="5min"),
    )
    frame.to_hdf(incoming / "metr-la.h5", key="traffic")
    records = tuple(
        RawFileRecord(relative, *stream_file(incoming / relative))
        for relative in discover(incoming).files
    )
    grant("metr-la", incoming, content_identity(records), "raw_ingest", "local_research")
    raw = ingest_dataset(incoming, "metr-la", raw_root=tmp_path / "raw")
    for operation in ("snapshot_parse", "canonical_create", "canonical_read"):
        grant("metr-la", raw.directory, raw.manifest.snapshot_id, operation, "local_research")
    for operation in ("canonical_graph_create", "canonical_graph_read"):
        grant("metr-la", raw.directory, raw.manifest.snapshot_id, operation, "graph_analysis")
    grant(
        "metr-la",
        raw.directory,
        raw.manifest.snapshot_id,
        "model_train",
        "model_training",
        tmp_path / "stgnn",
    )
    grant(
        "metr-la",
        raw.directory,
        raw.manifest.snapshot_id,
        "model_evaluate",
        "scientific_evaluation",
        tmp_path / "stgnn",
    )
    grant(
        "metr-la",
        raw.directory,
        raw.manifest.snapshot_id,
        "model_publish",
        "model_training",
        tmp_path / "stgnn",
    )
    grant(
        "metr-la",
        raw.directory,
        raw.manifest.snapshot_id,
        "model_compare",
        "benchmark_comparison",
        tmp_path / "stgnn",
    )
    canonical = create_canonical(raw.directory, output_root=tmp_path / "interim")
    plan = publish_plan(
        canonical.directory,
        PlanConfig(),
        output_root=tmp_path / "evaluation",
        raw_snapshot_dir=raw.directory,
    )
    locations = tmp_path / "locations.csv"
    locations.write_text("sensor_id,latitude,longitude\nb,34,-118\na,35,-119\n")
    distances = tmp_path / "distances.csv"
    distances.write_text("from,to,cost\na,a,0\nb,b,0\na,b,2\nb,a,3\n")
    for reference, path in (("invented-locations", locations), ("invented-distances", distances)):
        for operation in ("graph_csv_read", "graph_create", "graph_reuse"):
            grant(reference, path, stream_file(path)[1], operation, "graph_analysis")
    graph = build_graph(
        {
            "specification": GraphSpecification("metr-la").to_dict(),
            "raw_snapshot_dir": str(raw.directory),
            "canonical_dir": str(canonical.directory),
            "locations": str(locations),
            "location_reference": "invented-locations",
            "distances": str(distances),
            "distance_reference": "invented-distances",
            "output_root": str(tmp_path / "graphs"),
        }
    )
    config: dict[str, object] = {
        "canonical_dir": str(canonical.directory),
        "raw_snapshot_dir": str(raw.directory),
        "plan_dir": str(plan.directory),
        "graph_dir": str(graph.directory),
        "locations": str(locations),
        "location_reference": "invented-locations",
        "distances": str(distances),
        "distance_reference": "invented-distances",
        "output_root": str(tmp_path / "stgnn"),
        "model": GraphModelConfig(hidden=4, blocks=1).to_dict(),
        "training": TrainingConfig(
            epochs=1, threads=1, max_train_windows=8, max_validation_windows=4
        ).to_dict(),
        "search_residual": True,
    }
    path = tmp_path / "config.json"
    write_json(path, config)
    training_grant = next(item for item in grants if item["operation"] == "model_train")
    evaluation_grant = next(item for item in grants if item["operation"] == "model_evaluate")
    training_grant["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="not approved"):
        run_experiment(config, dry_run=True)
    training_grant["status"] = "approved"
    original_digest = training_grant["source_sha256"]
    training_grant["source_sha256"] = "0" * 64
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="differs from current purpose-specific"):
        run_experiment(config, dry_run=True)
    training_grant["source_sha256"] = original_digest
    evaluation_grant["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="not approved"):
        run_experiment(config)
    assert not (tmp_path / "stgnn").exists()
    evaluation_grant["status"] = "approved"
    evaluation_grant["source_sha256"] = "0" * 64
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="differs from current purpose-specific"):
        run_experiment(config)
    evaluation_grant["source_sha256"] = original_digest
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    assert main(["train", "--config", str(path), "--dry-run"]) == 0
    result = run_experiment(config)
    run = Path(str(result["directory"]))
    graph_context = {
        key: str(config[key])
        for key in (
            "graph_dir",
            "raw_snapshot_dir",
            "canonical_dir",
            "locations",
            "location_reference",
            "distances",
            "distance_reference",
        )
    }
    checked = verify_run(run, graph_context=graph_context)
    assert checked["valid"] and checked["integrity"] == "pass"
    assert checked["authorship_assurance"] == "unverified"
    assert checked["scientific_validation"] == "not_established"
    assert checked["production_approval"] == "not_established"
    assert main(["compare", "--run-dir", str(run), "--config", str(path)]) == 0
    compare_grant = next(item for item in grants if item["operation"] == "model_compare")
    compare_grant["source_sha256"] = "0" * 64
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    assert main(["compare", "--run-dir", str(run), "--config", str(path)]) == 1
    compare_grant["source_sha256"] = original_digest
    evaluation_grant["source_sha256"] = "0" * 64
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="differs from current purpose-specific"):
        verify_run(run, graph_context=graph_context)
    evaluation_grant["source_sha256"] = original_digest
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    reuse_grant = next(
        item
        for item in grants
        if item["source_id"] == "invented-distances" and item["operation"] == "graph_reuse"
    )
    reuse_grant["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="not approved"):
        verify_run(run, graph_context=graph_context)
    reuse_grant["status"] = "approved"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    inventory = {
        p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob("*") if p.is_file()
    }
    with pytest.raises(DataError, match="completed-run reuse denied"):
        run_experiment(config)
    assert inventory == {
        p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob("*") if p.is_file()
    }
    models = {
        fold: restore(run, metadata, graph)
        for fold, metadata in read_json(run / "models.json").items()
    }
    scalers = read_json(run / "scalers.json")
    payloads = {
        name: read_json(run / f"{name}.json") for name in ("selection", "metrics", "runtime")
    }
    payloads["metrics"] = {"caller_authored": "not a runner score"}
    descriptor = dict(read_json(run / "run.json"))
    descriptor.pop("content_sha256")
    descriptor["runner_name"] = "self-asserted-runner"
    caller_root = tmp_path / "stgnn" / "caller-candidate"
    direct_root = caller_root / "runs"
    context = {
        "raw_snapshot_dir": raw.directory,
        "canonical_dir": canonical.directory,
        "locations": locations,
        "location_reference": "invented-locations",
        "distances": distances,
        "distance_reference": "invented-distances",
    }

    def direct_publish(candidate: dict[str, object] = descriptor) -> Path:
        return publish_run(models, scalers, graph, payloads, candidate, direct_root, **context)

    publish_grant = next(item for item in grants if item["operation"] == "model_publish")
    original_publish = dict(publish_grant)
    for defect in (
        "missing",
        "revoked",
        "expired",
        "wrong-purpose",
        "wrong-destination",
        "wrong-digest",
    ):
        if defect == "missing":
            grants.remove(publish_grant)
        elif defect == "revoked":
            publish_grant["status"] = "revoked"
        elif defect == "expired":
            publish_grant["expires_utc"] = "2020-01-01T00:00:00+00:00"
        elif defect == "wrong-purpose":
            publish_grant["purpose"] = "benchmark_comparison"
        elif defect == "wrong-destination":
            publish_grant["destination_root"] = str(tmp_path / "other")
        else:
            publish_grant["source_sha256"] = "0" * 64
        policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
        with pytest.raises(DataError):
            direct_publish()
        assert not direct_root.exists()
        if defect == "missing":
            grants.append(publish_grant)
        publish_grant.clear()
        publish_grant.update(original_publish)
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    for claim in (
        {"trusted": True},
        {"validated": True},
        {"production_ready": True},
        {"saved_run_trust": {"authorship_assurance": "verified"}},
    ):
        with pytest.raises(DataError, match="trust or endorsement"):
            direct_publish(descriptor | claim)
        assert not direct_root.exists()
    submitted = direct_publish()
    submission = verify_run(submitted, graph_context=graph_context)
    assert submission["integrity"] == "pass"
    assert submission["classification"] == "caller_supplied_unverified"
    assert submission["authorship_assurance"] == "unverified"
    assert submission["scientific_validation"] == "not_established"
    assert submission["production_approval"] == "not_established"
    assert read_json(submitted / "metrics.json") == payloads["metrics"]
    caller_config = config | {"output_root": str(caller_root)}
    with pytest.raises(DataError, match="completed-run reuse denied"):
        run_experiment(caller_config)
    assert read_json(submitted / "metrics.json") == payloads["metrics"]
    legacy = dict(read_json(submitted / "run.json"))
    legacy.pop("saved_run_trust")
    (submitted / "run.json").write_text(
        json.dumps(legacy, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    legacy_id = identity(legacy)
    legacy_manifest = dict(read_json(submitted / "manifest.json"))
    legacy_manifest["run_id"] = legacy_id
    legacy_manifest["files"]["run.json"] = stream_file(submitted / "run.json")[1]
    (submitted / "manifest.json").write_text(
        json.dumps(legacy_manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    legacy_path = submitted.with_name(legacy_id)
    submitted.rename(legacy_path)
    assert (
        verify_run(legacy_path, graph_context=graph_context)["classification"] == "legacy_unknown"
    )
    with pytest.raises(DataError, match="completed-run reuse denied"):
        run_experiment(caller_config)
    assert main(["evaluate", "--run-dir", str(run), "--config", str(path)]) == 0
    groups = read_json(run / "metrics.json")["groups"]
    assert (
        isinstance(groups, dict)
        and len([k for k in groups if k.startswith("common/horizon/")]) == 12
    )
    (run / "model-0.pt").write_bytes(b"corrupt")
    with pytest.raises(DataError, match="checksum"):
        verify_run(run, graph_context=graph_context)
    assert main(["verify-run", "--run-dir", str(run), "--config", str(path)]) == 1
