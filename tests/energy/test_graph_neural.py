"""Energy graph causality, weighted aggregation and matched-capacity controls."""

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
import torch

from energraph.graph_models import EnergyForecaster, EnergyModelConfig
from urbangraph.data import DataError

REGIONS = ("country:DE", "country:FR", "country:BE")


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


def model(condition: str = "geographic") -> EnergyForecaster:
    adjacency = (
        np.eye(3)
        if condition in {"identity", "temporal"}
        else np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    )
    torch.manual_seed(2026)
    return EnergyForecaster(
        EnergyModelConfig(condition, hidden=4, blocks=2), adjacency, REGIONS, history=12, horizon=4
    )


def inputs() -> torch.Tensor:
    x = torch.randn(2, 12, 3, 2)
    x[..., 1] = 1
    return x


@pytest.mark.parametrize(
    "condition",
    ["temporal", "identity", "geographic", "correlation", "random", "correlation-unweighted"],
)
def test_shape_backward_capacity(condition: str) -> None:
    torch.set_num_threads(2)
    m = model(condition)
    output = m(inputs())
    assert output.shape == (2, 4, 3, 1)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(p.numel() for p in m.parameters()) == sum(
        p.numel() for p in model("temporal").parameters()
    )


@pytest.mark.parametrize("condition", ["temporal", "identity"])
def test_independent_control(condition: str) -> None:
    m = model(condition).eval()
    x = inputs()
    altered = x.clone()
    altered[:, :, 1:, 0] += 100
    torch.testing.assert_close(m(x)[:, :, 0], m(altered)[:, :, 0], rtol=0, atol=0)


def test_neighbors_direction_weights_and_batch_isolation() -> None:
    m = model()
    x = inputs()
    altered = x.clone()
    altered[:, :, 2, 0] += 100
    assert not torch.allclose(m(x)[:, :, 0], m(altered)[:, :, 0])
    z = torch.randn(2, 12, 3, 4)
    torch.testing.assert_close(m.aggregate(z), m.aggregate(z, dense=True))
    torch.testing.assert_close(m.aggregate(z)[:, :, 1], z[:, :, 0])
    torch.testing.assert_close(m(x)[:1], m(x[:1]))
    a = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    weighted = EnergyForecaster(EnergyModelConfig(), np.eye(3), REGIONS)
    graph = EnergyForecaster(EnergyModelConfig("geographic"), a, REGIONS)
    values = torch.tensor([[[1.0], [5.0], [9.0]]])
    assert graph.aggregate(values)[0, 2, 0] == 4
    assert weighted.aggregate(values)[0, 2, 0] == 9


def test_temporal_causality_and_permutation() -> None:
    m = model()
    x = inputs()
    changed = x.clone()
    changed[:, 6:, :, 0] += 30
    torch.testing.assert_close(m.encode(x)[:, :6], m.encode(changed)[:, :6])
    p = [2, 0, 1]
    a = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    permuted = EnergyForecaster(
        m.config, a[np.ix_(p, p)], tuple(REGIONS[i] for i in p), history=12, horizon=4
    )
    state = {k: v for k, v in m.state_dict().items() if k not in {"support", "edges", "weights"}}
    permuted.load_state_dict(state, strict=False)
    torch.testing.assert_close(m(x)[:, :, p], permuted(x[:, :, p]))


@pytest.mark.parametrize("bad", ["shape", "nan", "inf", "mask", "axis"])
def test_invalid_inputs(bad: str) -> None:
    m = model()
    x = inputs()
    if bad == "shape":
        x = x[:, :-1]
    if bad == "nan":
        x[0, 0, 0, 0] = float("nan")
    if bad == "inf":
        x[0, 0, 0, 0] = float("inf")
    if bad == "mask":
        x[0, 0, 0, 1] = 0.5
    with pytest.raises(DataError):
        m(x, regions=REGIONS[::-1] if bad == "axis" else REGIONS)


def test_missing_zero_and_empty_graph() -> None:
    m = EnergyForecaster(
        EnergyModelConfig("geographic"), np.zeros((3, 3)), REGIONS, history=12, horizon=4
    )
    assert torch.isfinite(m(torch.zeros(2, 12, 3, 2))).all()
    x = torch.zeros(2, 12, 3, 2)
    x[..., 1] = 1
    assert torch.isfinite(m(x)).all()
    with pytest.raises(DataError):
        EnergyForecaster(EnergyModelConfig(), -np.eye(3), REGIONS)


def test_seed_reload(tmp_path: Path) -> None:
    a = model()
    b = model()
    x = inputs()
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
    file = tmp_path / "weights.pt"
    torch.save(a.state_dict(), file)
    b.load_state_dict(torch.load(file, weights_only=True))
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_dense_equivalence() -> None:
    m = model()
    x = inputs()
    expected = m(x)
    gpu = m.to("cuda")
    actual = gpu(x.to("cuda")).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    actual.square().mean().backward()


def test_synthetic_training_artifact_and_latency(
    tmp_path: Path,
    invented_research_rights: InventedRights,
    monkeypatch: pytest.MonkeyPatch,
    broker_context_factory: Any,
) -> None:
    import json

    import pandas as pd

    from energraph.artifacts import PendingPublication, finalize_governed_publication
    from energraph.energy_graphs import EnergyGraphSpec, build_energy_graph
    from energraph.graph_binding import bound_graph
    from energraph.graph_research import run_graph_experiment, verify_graph_run
    from energraph.graph_windows import RegionalWindows
    from energraph.regional_data import load_regional_plan, materialize_regions, regional_plan
    from urbangraph.data import ingest_dataset
    from urbangraph.evaluation.preprocessing import fit_standardizer
    from urbangraph.stgnn.contracts import TrainingConfig

    source = tmp_path / "s"
    source.mkdir()
    table: dict[str, object] = {
        "utc_timestamp": pd.date_range("2015-01-01", periods=2400, freq="h", tz="UTC").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    }
    fields = []
    for i, r in enumerate(REGIONS):
        name = r.split(":")[1] + "_load_actual_entsoe_transparency"
        table[name] = 100 * (i + 1) + np.sin(np.arange(2400) / 24) * 10
        fields.append(
            {
                "name": name,
                "unit": "MW",
                "opsdProperties": {
                    "Region": r.split(":")[1],
                    "Variable": "load_actual_entsoe_transparency",
                },
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
    rights = invented_research_rights
    rights.source(source, "invented-neural-regions", tmp_path / "r")
    raw = ingest_dataset(source, "invented-neural-regions", raw_root=tmp_path / "r")
    for root, grants in (
        (
            tmp_path / "c",
            (
                ("canonicalize", "energy_regional_canonicalize"),
                ("local_research", "derived_commit"),
                ("local_research", "energy_regional_canonical_read"),
            ),
        ),
        (
            tmp_path / "p",
            (
                ("scientific_evaluation", "energy_regional_plan_create"),
                ("local_research", "derived_commit"),
                ("model_training", "energy_regional_plan_read"),
            ),
        ),
        (
            tmp_path / "g",
            (
                ("graph_analysis", "energy_graph_create"),
                ("local_research", "derived_commit"),
                ("graph_analysis", "energy_graph_read"),
            ),
        ),
        (
            tmp_path / "out" / "runs",
            (
                ("model_training", "energy_neural_train"),
                ("local_research", "derived_commit"),
                ("saved_run_verification", "energy_neural_verify"),
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
    context, raw_ref, policy_approver = broker_context_factory(
        raw.directory,
        rights,
        (
            ("canonicalize", "energy_regional_canonicalize"),
            ("local_research", "energy_regional_canonical_read"),
            ("scientific_evaluation", "energy_regional_plan_create"),
            ("model_training", "energy_regional_plan_read"),
            ("graph_analysis", "energy_graph_create"),
            ("graph_analysis", "energy_graph_read"),
            ("model_training", "energy_neural_train"),
            ("saved_run_verification", "energy_neural_verify"),
        ),
    )
    canonical_pending = materialize_regions(
        raw.directory,
        tmp_path / "c",
        REGIONS,
        broker_context=context,
        raw_broker_reference=raw_ref,
    )
    assert isinstance(canonical_pending, PendingPublication)
    policy_approver(canonical_pending)
    canonical = finalize_governed_publication(context, canonical_pending)
    plan_pending = regional_plan(
        canonical,
        tmp_path / "p",
        "chronological",
        (24, 12, 0),
        raw_snapshot_dir=raw.directory,
        broker_context=context,
    )
    assert isinstance(plan_pending, PendingPublication)
    policy_approver(plan_pending)
    plan = finalize_governed_publication(context, plan_pending)
    graph_pending = build_energy_graph(
        canonical,
        plan,
        tmp_path / "g",
        EnergyGraphSpec(method="identity"),
        raw_snapshot_dir=raw.directory,
        broker_context=context,
    )
    assert isinstance(graph_pending, PendingPublication)
    policy_approver(graph_pending)
    graph = finalize_governed_publication(context, graph_pending)
    assert all(snapshot.broker_reference is not None for snapshot in (canonical, plan, graph))
    with pytest.raises(DataError, match="semantic"):
        bound_graph(
            canonical,
            plan,
            graph,
            0,
            "correlation",
            raw_snapshot_dir=raw.directory,
            broker_context=context,
        )
    with pytest.raises(DataError, match="fold"):
        bound_graph(
            canonical,
            plan,
            graph,
            4,
            "identity",
            raw_snapshot_dir=raw.directory,
            broker_context=context,
        )
    data, split, latencies = load_regional_plan(
        canonical, plan, raw_snapshot_dir=raw.directory, broker_context=context
    )
    scaler = fit_standardizer(data, split, fold=0)
    windows = RegionalWindows(data, split, 0, "train", scaler, latencies, maximum=4)
    sample = windows[0]
    assert not sample.x[-24:, 0, 1].any()
    assert not sample.x[-12:, 1, 1].any()
    assert sample.x[:, 2, 1].all()
    data.frame.iloc[sample.origin - 23 : sample.origin + 1, 0] = 1e10
    torch.testing.assert_close(sample.x, windows[0].x)
    policy = TrainingConfig(
        epochs=1,
        patience=1,
        batch_size=8,
        max_train_windows=8,
        max_validation_windows=8,
        scaling="per_sensor",
        trial_seconds=60,
    )
    run_pending = run_graph_experiment(
        canonical,
        plan,
        graph,
        tmp_path / "out",
        "identity",
        0,
        policy,
        widths=(4,),
        blocks=1,
        raw_snapshot_dir=raw.directory,
        broker_context=context,
    )
    assert isinstance(run_pending, PendingPublication)
    policy_approver(run_pending)
    run = finalize_governed_publication(context, run_pending)
    assert run.broker_reference is not None
    assert verify_graph_run(
        run,
        canonical=canonical,
        plan=plan,
        graph=graph,
        raw_snapshot_dir=raw.directory,
        broker_context=context,
    )["valid"]
    assert (
        run_graph_experiment(
            canonical,
            plan,
            graph,
            tmp_path / "out",
            "identity",
            0,
            policy,
            widths=(4,),
            blocks=1,
            raw_snapshot_dir=raw.directory,
            broker_context=context,
            saved_run=run,
        )
        == run
    )
    verify_grants = [
        grant for grant in rights.grants if grant["operation"] == "energy_neural_verify"
    ]
    assert verify_grants
    for grant in verify_grants:
        grant["status"] = "revoked"
    rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))

    def forbidden_descriptor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("saved neural descriptor read before verify permission")

    with monkeypatch.context() as guarded:
        guarded.setattr("energraph.graph_research.read_json", forbidden_descriptor)
        guarded.setattr(context.broker, "read_directory_file", forbidden_descriptor)
        with pytest.raises(DataError):
            run_graph_experiment(
                canonical,
                plan,
                graph,
                tmp_path / "out",
                "identity",
                0,
                policy,
                widths=(4,),
                blocks=1,
                raw_snapshot_dir=raw.directory,
                broker_context=context,
                saved_run=run,
            )
    for grant in verify_grants:
        grant["status"] = "approved"
    rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))
    checkpoint = run.directory / "payload/trial-0/state.pt"
    with checkpoint.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(DataError):
        verify_graph_run(
            run,
            canonical=canonical,
            plan=plan,
            graph=graph,
            raw_snapshot_dir=raw.directory,
            broker_context=context,
        )


def test_metric_masks_and_region_scale() -> None:
    from energraph.graph_research import metric_report

    actual = np.ones((2, 24, 3), dtype=np.float64) * 10
    predicted = actual + np.array([1.0, 2.0, 4.0])
    mask = np.ones_like(actual, dtype=bool)
    result = metric_report(actual, predicted, mask, REGIONS, (1.0, 2.0, 4.0))
    assert result["macro_normalized_mae"] == 1
    assert result["macro_mae_MW"] == pytest.approx(7 / 3)
    horizons = result["horizons"]
    assert isinstance(horizons, dict) and len(horizons) == 24
    mask[0, 0, 0] = False
    actual[0, 0, 0] = np.nan
    assert (
        metric_report(actual, predicted, mask, REGIONS, (1.0, 2.0, 4.0))["macro_normalized_mae"]
        == 1
    )
    predicted[0, 0, 0] = np.inf
    with pytest.raises(DataError):
        metric_report(actual, predicted, mask, REGIONS, (1.0, 2.0, 4.0))


def test_cli_reports_models_and_bad_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from energraph.cli import main

    assert main(["graph-models"]) == 0
    assert "implemented" in json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "stgnn-verify",
                "--directory",
                str(tmp_path / "missing"),
                "--canonical",
                str(tmp_path / "canonical"),
                "--plan",
                str(tmp_path / "plan"),
                "--graph",
                str(tmp_path / "graph"),
                "--snapshot",
                str(tmp_path / "raw"),
            ]
        )
        == 1
    )
    assert "error" in json.loads(capsys.readouterr().out)
