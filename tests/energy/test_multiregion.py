"""Adversarial regional data, identity, graph and matched forecasting tests."""

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import pytest

from energraph.artifacts import (
    PendingPublication,
    finalize_governed_publication,
    resolve_broker_reference,
)
from energraph.energy_graphs import (
    EnergyGraphSpec,
    build_energy_graph,
    geographic_edges,
    realign,
    statistical_edges,
    verify_energy_graph,
)
from energraph.regional_data import (
    align_series,
    load_regional_plan,
    materialize_regions,
    open_regions,
    regional_plan,
)
from energraph.regions import (
    RegionDefinition,
    RegionRegistry,
    RegionType,
    SourceAreaMapping,
    TemporalValidity,
    country_registry,
)
from urbangraph.data import DataError, ingest_dataset, verify_snapshot
from urbangraph.data.filesystem import stream_file
from urbangraph.evaluation.preprocessing import fit_standardizer


class InventedRights(Protocol):
    path: Path
    grants: list[dict[str, object]]

    def source(self, incoming: Path, dataset_id: str, raw_root: Path) -> None: ...

    def _grant(
        self,
        source_id: str,
        source: Path,
        digest: str,
        operation: str,
        destination: Path,
        purpose: str = "local_research",
    ) -> None: ...


ORDER = ("country:DE", "country:FR", "country:BE")


def fixture(tmp_path: Path, rights: InventedRights) -> tuple[Path, Path, Path]:
    import json

    source = tmp_path / "source"
    source.mkdir()
    n = 2400
    t = np.arange(n, dtype=float)
    table = {
        "utc_timestamp": pd.date_range("2015-01-01", periods=n, freq="h", tz="UTC").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    }
    fields = []
    for j, region in enumerate(ORDER):
        code = region.split(":")[1]
        column = code + "_load_actual_entsoe_transparency"
        table[column] = 100 * (j + 1) + 20 * np.sin(t / 24) + np.cos(t / (j + 3))
        fields.append(
            {
                "name": column,
                "unit": "MW",
                "opsdProperties": {"Region": code, "Variable": "load_actual_entsoe_transparency"},
            }
        )
    pd.DataFrame(table).to_csv(source / "time_series_60min_singleindex.csv", index=False)
    (source / "datapackage.json").write_text(
        json.dumps(
            {
                "version": "2020-10-06",
                "resources": [
                    {"path": "time_series_60min_singleindex.csv", "schema": {"fields": fields}}
                ],
            }
        )
    )
    rights.source(source, "regional-fixture", tmp_path / "raw")
    raw = ingest_dataset(source, "regional-fixture", raw_root=tmp_path / "raw")
    for root, grants in (
        (
            tmp_path / "canonical",
            (
                ("canonicalize", "energy_regional_canonicalize"),
                ("local_research", "derived_commit"),
                ("local_research", "energy_regional_canonical_read"),
            ),
        ),
        (
            tmp_path / "plans",
            (
                ("scientific_evaluation", "energy_regional_plan_create"),
                ("local_research", "derived_commit"),
                ("model_training", "energy_regional_plan_read"),
            ),
        ),
        (
            tmp_path / "graphs",
            (
                ("graph_analysis", "energy_graph_create"),
                ("local_research", "derived_commit"),
                ("graph_analysis", "energy_graph_read"),
            ),
        ),
    ):
        for purpose, operation in grants:
            rights._grant(
                raw.directory.parent.name,
                raw.directory,
                raw.manifest.snapshot_id,
                operation,
                root,
                purpose,
            )
    canonical_result = materialize_regions(raw.directory, tmp_path / "canonical", ORDER)
    assert not isinstance(canonical_result, PendingPublication)
    canonical = canonical_result.directory
    geography = tmp_path / "borders.csv"
    geography.write_text("CNTR_BN_CODE,CNTR_CODE\n1,DE\n1,FR\n2,FR\n2,BE\n")
    for purpose, operation in (
        ("graph_analysis", "energy_graph_create"),
        ("local_research", "derived_commit"),
        ("graph_analysis", "energy_graph_read"),
        ("graph_analysis", "geography_read"),
    ):
        rights._grant(
            "invented-regional-border",
            geography,
            stream_file(geography)[1],
            operation,
            tmp_path / "graphs",
            purpose,
        )
    return canonical, geography, raw.directory


@pytest.mark.parametrize(
    "region", ["country:DE", "country:FR", "country:NL", "country:BE", "country:AT", "country:CZ"]
)
def test_country_registration(region: str) -> None:
    r = country_registry().get(region)
    assert r.region_type == RegionType.COUNTRY and r.eic is None


@pytest.mark.parametrize("kind", list(RegionType))
def test_type_registration(kind: RegionType) -> None:
    r = RegionDefinition(
        kind.value + ":TEST", "Test", "TEST", kind, "synthetic evidence", "fixture", "synthetic"
    )
    assert r.identifier.startswith(kind.value)


def test_time_transition_ambiguity() -> None:
    old = RegionDefinition(
        "bidding-zone:DE_AT_LU",
        "Old",
        "OLD",
        RegionType.BIDDING_ZONE,
        "fixture",
        "synthetic historical mapping",
        "synthetic",
        validity=TemporalValidity(end="2018-10-01T00:00:00+02:00"),
    )
    new = RegionDefinition(
        "bidding-zone:DE_LU",
        "New",
        "NEW",
        RegionType.BIDDING_ZONE,
        "fixture",
        "synthetic historical mapping",
        "synthetic",
        validity=TemporalValidity(start="2018-10-01T00:00:00+02:00"),
    )
    mappings = (
        SourceAreaMapping("source", old.identifier, old.validity, "fixture"),
        SourceAreaMapping("source", new.identifier, new.validity, "fixture"),
    )
    registry = RegionRegistry((old, new), mappings)
    assert registry.resolve("source", pd.Timestamp("2018-09-30T21:59:59Z")) == old
    assert registry.resolve("source", pd.Timestamp("2018-09-30T22:00:00Z")) == new
    ambiguous = RegionRegistry((old, new), mappings + (mappings[1],))
    with pytest.raises(DataError):
        ambiguous.resolve("source", pd.Timestamp("2019-01-01T00:00Z"))
    with pytest.raises(DataError):
        registry.axis((old.identifier, new.identifier))


@pytest.mark.parametrize("bad", ["country:XX", "bidding-zone:DE", "control-area:DE"])
def test_unknown_axis(bad: str) -> None:
    with pytest.raises(DataError):
        country_registry().axis((bad,))


def test_eic_rejected() -> None:
    with pytest.raises(DataError):
        RegionDefinition(
            "country:DE",
            "Germany",
            "DE",
            RegionType.COUNTRY,
            "fixture",
            "country",
            "fixture",
            eic="invented",
        )


def test_alignment_masks_order_dst() -> None:
    index = pd.date_range("2024-03-30", periods=72, freq="h", tz="UTC")
    series = {r: pd.DataFrame({"load": np.arange(72, dtype=float)}, index=index) for r in ORDER}
    series[ORDER[1]] = series[ORDER[1]].iloc[10:].copy()
    f = align_series(series, ORDER[::-1])
    assert tuple(f.columns) == ORDER[::-1] and len(f) == 72
    assert f[ORDER[1]].isna().sum() == 10 and f[ORDER[0]].iloc[0] == 0
    assert pd.DatetimeIndex(f.index).tz_convert("Europe/Berlin").hour.size == 72
    series[ORDER[0]] = series[ORDER[0]].set_axis(index.tz_localize(None), axis=0)
    with pytest.raises(DataError):
        align_series(series, ORDER)


@pytest.mark.parametrize("protocol", ["chronological", "rolling"])
def test_plan_scaler_leakage(
    tmp_path: Path,
    protocol: str,
    invented_research_rights: InventedRights,
    broker_context_factory: Any,
) -> None:
    canonical_dir, _, raw = fixture(tmp_path, invented_research_rights)
    context, raw_ref, policy_approver = broker_context_factory(
        raw,
        invented_research_rights,
        (
            ("canonicalize", "energy_regional_canonicalize"),
            ("local_research", "energy_regional_canonical_read"),
            ("scientific_evaluation", "energy_regional_plan_create"),
            ("model_training", "energy_regional_plan_read"),
        ),
    )
    canonical_pending = materialize_regions(
        raw, canonical_dir.parent, ORDER, broker_context=context, raw_broker_reference=raw_ref
    )
    assert isinstance(canonical_pending, PendingPublication)
    policy_approver(canonical_pending)
    canonical_snapshot = finalize_governed_publication(context, canonical_pending)
    plan_pending = regional_plan(
        canonical_snapshot,
        tmp_path / "plans",
        protocol,
        (24, 48, 24),
        raw_snapshot_dir=raw,
        broker_context=context,
    )
    assert isinstance(plan_pending, PendingPublication)
    policy_approver(plan_pending)
    plan = finalize_governed_publication(context, plan_pending)
    data, split, latency = load_regional_plan(
        canonical_snapshot, plan, raw_snapshot_dir=raw, broker_context=context
    )
    assert split.config.window.history == 168 and split.config.window.horizon == 24
    assert latency[ORDER[1]] == 48 and split.config.features.observed_delay_steps == 48
    scaler = fit_standardizer(data, split, fold=0)
    restored = scaler.inverse_transform(
        scaler.transform(data.frame.to_numpy(), sensor_ids=ORDER), sensor_ids=ORDER
    )
    np.testing.assert_allclose(restored, data.frame.to_numpy())
    train = split.folds[0].partition("train")
    original = scaler.means
    data.frame.iloc[train.stop :] = 999999
    assert fit_standardizer(data, split, fold=0).means == original


@pytest.mark.parametrize("method", ["pearson", "spearman"])
@pytest.mark.parametrize("changed_partition", ["validation", "test"])
def test_train_only_graph(
    tmp_path: Path,
    method: str,
    changed_partition: str,
    invented_research_rights: InventedRights,
    broker_context_factory: Any,
) -> None:
    canonical_dir, _, raw = fixture(tmp_path, invented_research_rights)
    context, raw_ref, policy_approver = broker_context_factory(
        raw,
        invented_research_rights,
        (
            ("canonicalize", "energy_regional_canonicalize"),
            ("local_research", "energy_regional_canonical_read"),
            ("scientific_evaluation", "energy_regional_plan_create"),
            ("model_training", "energy_regional_plan_read"),
        ),
    )
    canonical_pending = materialize_regions(
        raw,
        canonical_dir.parent,
        ORDER,
        broker_context=context,
        raw_broker_reference=raw_ref,
    )
    assert isinstance(canonical_pending, PendingPublication)
    policy_approver(canonical_pending)
    canonical = finalize_governed_publication(context, canonical_pending)
    assert canonical.broker_reference is not None

    plan_pending = regional_plan(
        canonical,
        tmp_path / "plans",
        "rolling",
        (24, 24, 24),
        raw_snapshot_dir=raw,
        broker_context=context,
    )
    assert isinstance(plan_pending, PendingPublication)
    policy_approver(plan_pending)
    plan = finalize_governed_publication(context, plan_pending)
    assert plan.broker_reference is not None

    data, split, _ = load_regional_plan(
        canonical, plan, raw_snapshot_dir=raw, broker_context=context
    )
    train = split.folds[0].partition("train")
    spec = EnergyGraphSpec("statistical", minimum_support=100, correlation=method, k=1)
    before, meta = statistical_edges(data.frame.iloc[train.start : train.stop], spec)
    part = split.folds[0].partition(changed_partition)
    data.frame.iloc[part.start : part.stop] = np.random.default_rng(33).normal(
        size=(part.stop - part.start, 3)
    )
    after, ameta = statistical_edges(data.frame.iloc[train.start : train.stop], spec)
    np.testing.assert_array_equal(before, after)
    assert meta == ameta


def test_region_latency_adversary() -> None:
    from energraph.regional_research import RegionalFeatures, regional_features

    frame = pd.DataFrame(
        {r: np.arange(500, dtype=float) for r in ORDER},
        index=pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC"),
    )
    spec = RegionalFeatures(ORDER[0], ORDER, (24, 48, 72), "M1")
    o = np.array([300], dtype=np.int64)
    h = np.array([24], dtype=np.int64)
    x, names, _ = regional_features(frame, o, h, spec)
    for region, delay in zip(ORDER, spec.latencies, strict=True):
        frame.loc[frame.index[301 - delay :], region] = -99999
    np.testing.assert_array_equal(x, regional_features(frame, o, h, spec)[0])
    assert x[0, names.index("country_FR_load_lag_0")] == 252


@pytest.mark.parametrize(
    "kind", ["identity", "empty", "geographic-proxy", "statistical", "random-control"]
)
def test_graph_artifacts(
    tmp_path: Path,
    kind: str,
    invented_research_rights: InventedRights,
    broker_context_factory: Any,
) -> None:
    _, geography, raw = fixture(tmp_path, invented_research_rights)
    rights = invented_research_rights
    border_source = tmp_path / "invented-border-source"
    border_source.mkdir()
    (border_source / "borders.csv").write_bytes(geography.read_bytes())
    border_source_id = "invented-regional-border-fixture"
    rights.source(border_source, border_source_id, tmp_path / "border-raw")
    border_raw = ingest_dataset(
        border_source,
        border_source_id,
        raw_root=tmp_path / "border-raw",
    )
    border_rights = (
        ("graph_analysis", "energy_graph_create"),
        ("graph_analysis", "energy_graph_read"),
        ("local_research", "derived_commit"),
    )
    for purpose, operation in border_rights:
        rights._grant(
            border_source_id,
            border_raw.directory,
            border_raw.manifest.snapshot_id,
            operation,
            tmp_path / "graphs",
            purpose,
        )
    operations = (
        ("canonicalize", "energy_regional_canonicalize"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_regional_canonical_read"),
        ("scientific_evaluation", "energy_regional_plan_create"),
        ("model_training", "energy_regional_plan_read"),
        ("graph_analysis", "energy_graph_create"),
        ("graph_analysis", "energy_graph_read"),
    )
    broker_context, raw_reference, policy_approver, additional_references = broker_context_factory(
        raw,
        rights,
        operations,
        additional_raw_sources=(
            (
                border_raw.directory,
                (
                    ("graph_analysis", "energy_graph_create"),
                    ("graph_analysis", "energy_graph_read"),
                ),
            ),
        ),
    )
    border_reference = additional_references[border_source_id]
    assert (
        resolve_broker_reference(
            broker_context.broker,
            border_reference.artifact_id,
            purpose="graph_analysis",
            operation="energy_graph_create",
        )
        == border_reference
    )
    canonical_pending = materialize_regions(
        raw,
        tmp_path / "canonical",
        ORDER,
        broker_context=broker_context,
        raw_broker_reference=raw_reference,
    )
    assert isinstance(canonical_pending, PendingPublication)
    policy_approver(canonical_pending)
    canonical = finalize_governed_publication(broker_context, canonical_pending)
    assert canonical.broker_reference is not None

    plan_pending = regional_plan(
        canonical,
        tmp_path / "plans",
        "chronological",
        (24, 24, 24),
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(plan_pending, PendingPublication)
    policy_approver(plan_pending)
    plan = finalize_governed_publication(broker_context, plan_pending)
    assert plan.broker_reference is not None

    spec = EnergyGraphSpec(kind, minimum_support=100, k=1)
    broker_geography = kind in {"geographic-proxy", "random-control"}
    geographic_parent_args = (
        {
            "geography_snapshot": border_raw,
            "geography_artifact_reference": border_reference,
            "geography_relative_path": "payload/borders.csv",
        }
        if broker_geography
        else {}
    )
    graph = build_energy_graph(
        canonical,
        plan,
        tmp_path / "graphs",
        spec,
        raw_snapshot_dir=raw,
        broker_context=broker_context,
        **geographic_parent_args,
    )
    assert isinstance(graph, PendingPublication)
    policy_approver(graph)
    graph = finalize_governed_publication(broker_context, graph)
    assert graph.broker_reference is not None

    again = build_energy_graph(
        canonical,
        plan,
        tmp_path / "graphs",
        spec,
        raw_snapshot_dir=raw,
        broker_context=broker_context,
        **geographic_parent_args,
    )
    assert isinstance(again, PendingPublication)
    policy_approver(again)
    again = finalize_governed_publication(broker_context, again)
    assert again.broker_reference is not None
    assert graph.broker_reference.content_sha256 == again.broker_reference.content_sha256
    graph_context = {
        "canonical": canonical,
        "plan": plan,
        "raw_snapshot_dir": raw,
        "broker_context": broker_context,
    }
    if broker_geography:
        graph_context.update(geographic_parent_args)
    a, desc = verify_energy_graph(graph, expected_order=ORDER, **graph_context)
    assert desc["physical_topology_verified"] is False
    permutation = realign(a, ORDER, ORDER[::-1])
    np.testing.assert_array_equal(permutation, a[::-1, ::-1])
    with pytest.raises(DataError):
        verify_energy_graph(graph, expected_order=ORDER[::-1], **graph_context)
    path = graph.directory / "payload/graph.npz"
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(DataError):
        verify_energy_graph(graph, **graph_context)


def test_geo_evidence_duplicate(tmp_path: Path, invented_research_rights: InventedRights) -> None:
    import json

    source = tmp_path / "g.csv"
    source.write_text("CNTR_BN_CODE,CNTR_CODE\n1,DE\n1,FR\n")
    rights = invented_research_rights
    rights._grant(
        "invented-border-duplicate",
        source,
        stream_file(source)[1],
        "geography_read",
        tmp_path,
        "graph_analysis",
    )
    a, evidence = geographic_edges(
        source, ORDER, source_reference="invented-border-duplicate", destination=tmp_path
    )
    assert a[0, 1] == a[1, 0] == 1 and "NOT verified" in str(evidence)
    source.write_text(source.read_text() + "1,DE\n")
    rights.grants[-1]["source_sha256"] = stream_file(source)[1]
    rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))
    with pytest.raises(DataError):
        geographic_edges(
            source,
            ORDER,
            source_reference="invented-border-duplicate",
            destination=tmp_path,
        )


def test_canonical_corruption(tmp_path: Path, invented_research_rights: InventedRights) -> None:
    canonical, _, raw = fixture(tmp_path, invented_research_rights)
    data = open_regions(canonical, raw_snapshot_dir=raw)
    assert data.sensor_ids == ORDER
    path = canonical / "payload/load.parquet"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(DataError):
        open_regions(canonical, raw_snapshot_dir=raw)


def test_complete_regional_pilot(
    tmp_path: Path,
    invented_research_rights: InventedRights,
    monkeypatch: pytest.MonkeyPatch,
    broker_context_factory: Any,
) -> None:
    import json

    from energraph.regional_research import run_regional, verify_regional_run
    from urbangraph.boosting.contracts import BoostConfig

    _, geography, raw = fixture(tmp_path, invented_research_rights)
    border_source = tmp_path / "invented-border-source"
    border_source.mkdir()
    (border_source / "borders.csv").write_bytes(geography.read_bytes())
    border_source_id = "invented-regional-pilot-border-fixture"
    invented_research_rights.source(border_source, border_source_id, tmp_path / "border-raw")
    border_raw = ingest_dataset(border_source, border_source_id, raw_root=tmp_path / "border-raw")
    border_checked = verify_snapshot(border_raw.directory)
    assert border_checked.valid and border_checked.manifest is not None
    for purpose, operation in (
        ("graph_analysis", "energy_graph_create"),
        ("graph_analysis", "energy_graph_read"),
        ("local_research", "derived_commit"),
    ):
        invented_research_rights._grant(
            border_source_id,
            border_raw.directory,
            border_checked.manifest.snapshot_id,
            operation,
            tmp_path / "graphs",
            purpose,
        )
    operations = (
        ("canonicalize", "energy_regional_canonicalize"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_regional_canonical_read"),
        ("scientific_evaluation", "energy_regional_plan_create"),
        ("model_training", "energy_regional_plan_read"),
        ("graph_analysis", "energy_graph_create"),
        ("graph_analysis", "energy_graph_read"),
        ("model_training", "energy_regional_train"),
        ("saved_run_verification", "energy_regional_verify"),
    )
    geographic_operations = (
        ("graph_analysis", "energy_graph_create"),
        ("graph_analysis", "energy_graph_read"),
        ("model_training", "energy_regional_train"),
        ("saved_run_verification", "energy_regional_verify"),
        ("local_research", "derived_commit"),
    )
    broker_context, raw_reference, policy_approver, additional = broker_context_factory(
        raw,
        invented_research_rights,
        operations,
        additional_raw_sources=((border_raw.directory, geographic_operations),),
    )
    border_reference = additional[border_source_id]
    canonical_pending = materialize_regions(
        raw,
        tmp_path / "canonical",
        ORDER,
        broker_context=broker_context,
        raw_broker_reference=raw_reference,
    )
    assert isinstance(canonical_pending, PendingPublication)
    policy_approver(canonical_pending)
    canonical = finalize_governed_publication(broker_context, canonical_pending)
    plan_pending = regional_plan(
        canonical,
        tmp_path / "plans",
        "chronological",
        (24, 24, 24),
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(plan_pending, PendingPublication)
    policy_approver(plan_pending)
    plan = finalize_governed_publication(broker_context, plan_pending)
    graph_pending = build_energy_graph(
        canonical,
        plan,
        tmp_path / "graphs",
        EnergyGraphSpec("geographic-proxy"),
        geography_snapshot=border_raw,
        geography_artifact_reference=border_reference,
        geography_relative_path="payload/borders.csv",
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(graph_pending, PendingPublication)
    policy_approver(graph_pending)
    graph = finalize_governed_publication(broker_context, graph_pending)
    for source_id, source in (
        (raw.parent.name, raw),
        (border_source_id, border_raw.directory),
    ):
        snapshot = verify_snapshot(source)
        assert snapshot.valid and snapshot.manifest is not None
        for purpose, operation in (
            ("model_training", "energy_regional_train"),
            ("saved_run_verification", "energy_regional_verify"),
            ("local_research", "derived_commit"),
        ):
            invented_research_rights._grant(
                source_id,
                source,
                snapshot.manifest.snapshot_id,
                operation,
                tmp_path / "runs",
                purpose,
            )
    config = BoostConfig(
        trees=4,
        max_depth=2,
        alternative_depth=3,
        num_leaves=7,
        min_child_samples=5,
        threads=1,
        max_train_rows=1200,
        max_validation_rows=600,
        run_seconds=300,
    )
    context = {
        "raw_snapshot_dir": raw,
        "broker_context": broker_context,
        "geography_snapshot": border_raw,
        "geography_artifact_reference": border_reference,
        "geography_relative_path": "payload/borders.csv",
    }
    run_pending = run_regional(canonical, plan, graph, tmp_path / "runs", config, **context)
    assert isinstance(run_pending, PendingPublication)
    policy_approver(run_pending)
    run = finalize_governed_publication(broker_context, run_pending)
    run_report = verify_regional_run(
        run,
        canonical=canonical,
        plan=plan,
        graph=graph,
        raw_snapshot_dir=raw,
        **{k: v for k, v in context.items() if k not in {"broker_context", "raw_snapshot_dir"}},
        broker_context=broker_context,
    )
    assert run_report["selected_models"] == 9
    repeated_pending = run_regional(canonical, plan, graph, tmp_path / "runs", config, **context)
    assert isinstance(repeated_pending, PendingPublication)
    policy_approver(repeated_pending)
    repeated = finalize_governed_publication(broker_context, repeated_pending)
    repeated_report = verify_regional_run(
        repeated,
        canonical=canonical,
        plan=plan,
        graph=graph,
        raw_snapshot_dir=raw,
        **{k: v for k, v in context.items() if k not in {"broker_context", "raw_snapshot_dir"}},
        broker_context=broker_context,
    )
    assert repeated_report["experiment_id"] == run_report["experiment_id"]
    first_descriptor = json.loads(
        (run.directory / "payload/descriptor.json").read_text(encoding="utf-8")
    )
    second_descriptor = json.loads(
        (repeated.directory / "payload/descriptor.json").read_text(encoding="utf-8")
    )
    first_records = first_descriptor["records"]
    second_records = second_descriptor["records"]
    assert len(first_records) == len(second_records) == 9
    for first_record, second_record in zip(first_records, second_records, strict=True):
        assert first_record["population_id"] == second_record["population_id"]
        assert first_record["metrics"] == second_record["metrics"]
    from energraph.uncertainty.population import reconstruct

    population_root = tmp_path / "population-runs"
    fake_descriptor = (
        population_root
        / "energraph-uncertainty-population"
        / "invented"
        / "payload"
        / "descriptor.json"
    )
    fake_descriptor.parent.mkdir(parents=True)
    fake_descriptor.write_text("{}", encoding="utf-8")
    for source_id, source, digest in (
        (raw.parent.name, raw, raw.name),
        ("invented-regional-border", geography, stream_file(geography)[1]),
    ):
        invented_research_rights._grant(
            source_id,
            source,
            digest,
            "energy_population_create",
            population_root,
            "scientific_evaluation",
        )
    original_read_json = __import__(
        "energraph.uncertainty.population", fromlist=["read_json"]
    ).read_json

    def forbid_population_descriptor(path: Path) -> object:
        if "energraph-uncertainty-population" in str(path):
            raise AssertionError("saved population descriptor read before permission")
        return original_read_json(path)

    with monkeypatch.context() as guarded:
        guarded.setattr("energraph.uncertainty.population.read_json", forbid_population_descriptor)
        with pytest.raises(DataError):
            reconstruct(
                canonical.directory,
                plan.directory,
                run.directory,
                population_root,
                0,
                graph=graph.directory,
                raw_snapshot_dir=raw,
            )
    assert fake_descriptor.read_text(encoding="utf-8") == "{}"
    verify_grants = [
        grant
        for grant in invented_research_rights.grants
        if grant["operation"] == "energy_regional_verify"
    ]
    assert verify_grants
    for grant in verify_grants:
        grant["status"] = "revoked"
    invented_research_rights.path.write_text(
        json.dumps({"version": 1, "grants": invented_research_rights.grants})
    )

    def forbidden_descriptor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("saved regional descriptor read before verify permission")

    with monkeypatch.context() as guarded:
        guarded.setattr("energraph.regional_research.read_json", forbidden_descriptor)
        with pytest.raises(DataError):
            run_regional(
                canonical,
                plan.directory,
                graph.directory,
                tmp_path / "runs",
                config,
                **context,
            )


def test_geometry_crosscheck_excludes_ambiguous_global_relations(
    tmp_path: Path, invented_research_rights: InventedRights
) -> None:
    import json

    csv = tmp_path / "relations.csv"
    csv.write_text("CNTR_BN_CODE,CNTR_CODE\n1,DE\n1,FR\n1,IN\n2,DE\n2,FR\n3,BE\n3,FR\n")
    features = [
        {"properties": {"CNTR_BN_CODE": i, "CNTR_CODE": code, "COAS_FLAG": coast}}
        for i, code, coast in [(1, "IN", "T"), (2, "DE", "F"), (3, "BE", "F")]
    ]
    (tmp_path / "CNTR_BN_20M_2020_4326.geojson").write_text(json.dumps({"features": features}))
    geometry = tmp_path / "CNTR_BN_20M_2020_4326.geojson"
    for source, reference in ((csv, "invented-relations"), (geometry, "invented-geometry")):
        invented_research_rights._grant(
            reference,
            source,
            stream_file(source)[1],
            "geography_read",
            tmp_path,
            "graph_analysis",
        )
    adjacency, evidence = geographic_edges(
        csv,
        ORDER,
        source_reference="invented-relations",
        geometry_reference="invented-geometry",
        destination=tmp_path,
    )
    assert adjacency[0, 1] == 1 and adjacency[1, 2] == 1
    assert evidence[0]["geometry_sha256"] is not None
    assert evidence[0]["excluded_global_relation_rows"] == 3
    for item in evidence:
        boundary_ids = item["boundary_ids"]
        assert isinstance(boundary_ids, list)
        assert "1" not in boundary_ids


def test_graph_fold_populations_differ(
    tmp_path: Path,
    invented_research_rights: InventedRights,
    broker_context_factory: Any,
) -> None:
    canonical, _, raw = fixture(tmp_path, invented_research_rights)
    operations = (
        ("canonicalize", "energy_regional_canonicalize"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_regional_canonical_read"),
        ("scientific_evaluation", "energy_regional_plan_create"),
        ("model_training", "energy_regional_plan_read"),
        ("graph_analysis", "energy_graph_create"),
        ("graph_analysis", "energy_graph_read"),
    )
    broker_context, raw_reference, approve = broker_context_factory(
        raw, invented_research_rights, operations
    )
    canonical_pending = materialize_regions(
        raw,
        tmp_path / "canonical",
        ORDER,
        broker_context=broker_context,
        raw_broker_reference=raw_reference,
    )
    assert isinstance(canonical_pending, PendingPublication)
    approve(canonical_pending)
    canonical_snapshot = finalize_governed_publication(broker_context, canonical_pending)
    plan_pending = regional_plan(
        canonical_snapshot,
        tmp_path / "plans",
        "rolling",
        (24, 24, 24),
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(plan_pending, PendingPublication)
    approve(plan_pending)
    plan_snapshot = finalize_governed_publication(broker_context, plan_pending)
    graphs = []
    for fold in (0, 1):
        pending = build_energy_graph(
            canonical_snapshot,
            plan_snapshot,
            tmp_path / "graphs",
            EnergyGraphSpec("statistical", minimum_support=100),
            fold=fold,
            raw_snapshot_dir=raw,
            broker_context=broker_context,
        )
        assert isinstance(pending, PendingPublication)
        approve(pending)
        graphs.append(finalize_governed_publication(broker_context, pending))
    first, second = graphs
    assert first.directory != second.directory


def test_cli_regional_success_failure(capsys: pytest.CaptureFixture[str]) -> None:
    from energraph.cli import main

    assert main(["inspect-regions"]) == 0
    assert "country:DE" in capsys.readouterr().out
    assert (
        main(
            [
                "verify-energy-graph",
                "--directory",
                "nonexistent-eg002-fixture",
                "--canonical",
                "missing-canonical",
                "--plan",
                "missing-plan",
                "--snapshot",
                "missing-source",
            ]
        )
        == 1
    )


def test_direction_realign_and_negative_rejection() -> None:
    from urbangraph.graphs.normalization import normalize

    matrix = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    reordered = realign(matrix, ORDER, ORDER[::-1])
    assert reordered[2, 1] == 2 and reordered[1, 2] == 0
    assert normalize(matrix, "random-walk")[0, 1] == 1
    with pytest.raises(DataError):
        normalize(-matrix, "random-walk")
    with pytest.raises(DataError):
        EnergyGraphSpec("physical")


def test_unsupported_future_flow_feature() -> None:
    from energraph.regional_research import RegionalFeatures

    with pytest.raises(DataError):
        RegionalFeatures("country:DE", ("country:DE", "physical-flow:DE_FR"), (24, 0), "M1")


def test_duplicate_and_long_missing_alignment() -> None:
    index = pd.date_range("2024-01-01", periods=1500, freq="h", tz="UTC")
    series = {r: pd.DataFrame({"load": np.ones(1500)}, index=index) for r in ORDER}
    series[ORDER[1]].iloc[100:1400] = np.nan
    masked = align_series(series, ORDER)
    assert masked[ORDER[1]].isna().sum() == 1300
    with pytest.raises(DataError):
        align_series(series, ORDER, "strict")
    series[ORDER[0]] = pd.concat([series[ORDER[0]], series[ORDER[0]].iloc[:1]])
    with pytest.raises(DataError):
        align_series(series, ORDER)
