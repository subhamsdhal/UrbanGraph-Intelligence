"""Actual synthetic replay checkpoint publication and recovery."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from urbangraph.serving import replay_checkpoint
from urbangraph.serving.local_synthetic import LocalSyntheticEngine
from urbangraph.serving.replay_checkpoint import (
    CheckpointError,
    load_checkpoint,
    resume_checkpoint,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]


def _engine() -> LocalSyntheticEngine:
    return LocalSyntheticEngine.load(
        ROOT / "tests" / "fixtures" / "p3_synthetic_bundle",
        ROOT / "tests" / "fixtures" / "eg006",
    )


def test_checkpoint_recovery_recomputes_actual_forecasts(tmp_path: Path) -> None:
    engine = _engine()
    origins = tuple(engine.data.origins)
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    original = save_checkpoint(engine, (origins[0], origins[0]), first)
    assert original.safe_metrics()["accepted"] == 1
    sequence, recovered = load_checkpoint(engine, first)
    assert sequence == (origins[0], origins[0])
    assert recovered == original
    resumed = resume_checkpoint(engine, first, (origins[1],), second)
    assert resumed.safe_metrics() == {"accepted": 2, "duplicates": 1, "late": 0, "rejected": 0}
    assert len(load_checkpoint(engine, second)[1].forecasts[0]["result"]["predictions"]) == 24  # type: ignore[index]
    assert first.read_bytes() != second.read_bytes()


def test_no_overwrite_and_tamper_rejection(tmp_path: Path) -> None:
    engine = _engine()
    origin = tuple(engine.data.origins)[0]
    path = tmp_path / "checkpoint.json"
    save_checkpoint(engine, (origin,), path)
    before = path.read_bytes()
    with pytest.raises(CheckpointError, match="without overwrite"):
        save_checkpoint(engine, (origin,), path)
    assert path.read_bytes() == before
    path.write_bytes(before.replace(b'"version":1', b'"version":2'))
    with pytest.raises(CheckpointError):
        load_checkpoint(engine, path)


def test_unapproved_origin_never_persisted(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="approved synthetic"):
        save_checkpoint(_engine(), ("external-origin",), tmp_path / "checkpoint.json")
    assert not (tmp_path / "checkpoint.json").exists()


def test_recovery_after_new_engine_load(tmp_path: Path) -> None:
    first_engine = _engine()
    origin = tuple(first_engine.data.origins)[0]
    path = tmp_path / "checkpoint.json"
    expected = save_checkpoint(first_engine, (origin,), path)
    # Reinitialize the real model and independently verify/recompute the saved sequence.
    sequence, recovered = load_checkpoint(_engine(), path)
    assert sequence == (origin,)
    assert recovered == expected


def test_duplicate_field_checkpoint_refused(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_bytes(b'{"document":{},"document":{},"sha256":"forged"}')
    with pytest.raises(CheckpointError, match="Duplicate checkpoint field"):
        load_checkpoint(_engine(), path)


def test_incomplete_checkpoint_is_not_restorable(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_bytes(b'{"document":')
    with pytest.raises(CheckpointError, match="cannot be read"):
        load_checkpoint(_engine(), path)


def test_interrupted_stage_write_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(_fd: int) -> None:
        raise OSError("simulated interrupted flush")

    monkeypatch.setattr(replay_checkpoint.os, "fsync", interrupted)
    origin = tuple(_engine().data.origins)[0]
    destination = tmp_path / "checkpoint.json"
    with pytest.raises(CheckpointError, match="without overwrite"):
        save_checkpoint(_engine(), (origin,), destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".replay-stage-*.json"))


def test_replaced_stage_is_not_unlinked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def replace_before_publication(stage: Path, _destination: Path) -> None:
        stage.rename(tmp_path / "retained-original-stage.json")
        stage.write_bytes(b"unrelated replacement")
        raise OSError("simulated publication failure")

    monkeypatch.setattr(replay_checkpoint, "_publish_no_overwrite", replace_before_publication)
    origin = tuple(_engine().data.origins)[0]
    with pytest.raises(CheckpointError, match="staging path changed"):
        save_checkpoint(_engine(), (origin,), tmp_path / "checkpoint.json")
    assert list(tmp_path.glob(".replay-stage-*.json"))[0].read_bytes() == b"unrelated replacement"
    assert not (tmp_path / "checkpoint.json").exists()


def test_publication_failure_preserves_os_cause_and_cleans_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(_stage: Path, _destination: Path) -> None:
        raise PermissionError(13, "publication denied")

    monkeypatch.setattr(replay_checkpoint, "_publish_no_overwrite", denied)
    origin = tuple(_engine().data.origins)[0]
    destination = tmp_path / "checkpoint.json"
    with pytest.raises(CheckpointError, match="without overwrite") as captured:
        save_checkpoint(_engine(), (origin,), destination)
    assert isinstance(captured.value.__cause__, PermissionError)
    assert not destination.exists()
    assert not list(tmp_path.glob(".replay-stage-*.json"))


def test_concurrent_publication_has_one_winner_and_preserves_checkpoint(tmp_path: Path) -> None:
    engine = _engine()
    origin = tuple(engine.data.origins)[0]
    destination = tmp_path / "checkpoint.json"

    def publish() -> bool:
        try:
            save_checkpoint(engine, (origin,), destination)
        except CheckpointError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(lambda _: publish(), range(2)))
    assert sorted(outcomes) == [False, True]
    assert load_checkpoint(engine, destination)[0] == (origin,)
    assert not list(tmp_path.glob(".replay-stage-*.json"))
