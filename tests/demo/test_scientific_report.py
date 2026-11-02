"""Adversarial checks for the portable synthetic evidence contract."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

import energraph.scientific_report as scientific_report
from energraph.scientific_report import EvidenceError, create_package, verify_package
from energraph.scientific_validation import evaluate_demo


def test_create_verify_and_reproducible_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    assert create_package(first) == create_package(second)
    for name in ("report.json", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert verify_package(first)["status"] == "PASS"
    with pytest.raises(EvidenceError, match="already exists"):
        create_package(first)


@pytest.mark.parametrize(
    "action",
    [
        "missing",
        "extra",
        "corrupt-report",
        "corrupt-manifest",
        "duplicate",
        "nan",
        "overflow",
        "schema",
        "forged",
    ],
)
def test_reject_untrusted_packages(tmp_path: Path, action: str) -> None:
    package = tmp_path / "package"
    create_package(package)
    report = package / "report.json"
    manifest = package / "manifest.json"
    if action == "missing":
        manifest.unlink()
    elif action == "extra":
        (package / "extra").write_text("x")
    elif action == "corrupt-report":
        report.write_bytes(report.read_bytes() + b"x")
    elif action == "corrupt-manifest":
        manifest.write_bytes(manifest.read_bytes() + b"x")
    elif action == "duplicate":
        manifest.write_text('{"schema_version":1,"schema_version":2}')
    elif action == "nan":
        report.write_text('{"x":NaN}')
    elif action == "overflow":
        report.write_text('{"x":1e999}')
    elif action == "schema":
        value = json.loads(manifest.read_text())
        value["schema_version"] = "unsupported"
        manifest.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    else:
        value = json.loads(report.read_text())
        value["forged_provenance"] = True
        report.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(EvidenceError):
        verify_package(package)


def test_reject_symlink(tmp_path: Path) -> None:
    package = tmp_path / "package"
    create_package(package)
    target = package / "report.json"
    target.rename(package / "report-real.json")
    try:
        target.symlink_to(package / "report-real.json")
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this host")
    with pytest.raises(EvidenceError):
        verify_package(package)


def test_publish_collision_preserves_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "package"
    original_publish = scientific_report._publish

    def collide(stage: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owned-by-other").write_text("preserve")
        original_publish(stage, destination)

    monkeypatch.setattr(scientific_report, "_publish", collide)
    with pytest.raises(OSError):
        create_package(output)
    assert (output / "owned-by-other").read_text() == "preserve"
    assert not list(tmp_path.glob(".eg007b-stage-*"))


def test_fresh_evaluation_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "package"
    create_package(output)
    monkeypatch.setattr(scientific_report, "evaluate_demo", lambda: {"different": True})
    with pytest.raises(EvidenceError, match="fresh trusted"):
        verify_package(output)


def _fake_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkout = tmp_path / "checkout"
    (checkout / "src" / "energraph").mkdir(parents=True)
    monkeypatch.setattr(
        scientific_report, "__file__", str(checkout / "src" / "energraph" / "scientific_report.py")
    )
    return checkout


def test_reject_checkout_and_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = _fake_checkout(tmp_path, monkeypatch)
    sibling = tmp_path / "existing-sibling"
    sibling.mkdir()
    attempts = (
        checkout,
        checkout / "inside",
        sibling / ".." / "checkout" / "escaped",
        sibling / ".." / "existing-sibling" / ".." / "checkout" / "nested-escape",
    )
    for output in attempts:
        with pytest.raises(EvidenceError, match="outside the source checkout"):
            create_package(output)
    assert {entry.name for entry in checkout.iterdir()} == {"src"}


def test_similar_sibling_is_valid_external_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_checkout(tmp_path, monkeypatch)
    similar = tmp_path / "checkout-backup"
    similar.mkdir()
    assert create_package(similar / "evidence")["status"] == "PASS"


def test_reject_directory_symlink_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_checkout(tmp_path, monkeypatch)
    link = tmp_path / "link"
    try:
        link.symlink_to(checkout, target_is_directory=True)
    except OSError:
        pytest.skip("Creating directory symlinks is not permitted on this host")
    with pytest.raises(EvidenceError, match="Reparse points and symbolic links"):
        create_package(link / "evidence")
    assert not (checkout / "evidence").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junctions are platform-specific")
def test_reject_windows_junction_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _fake_checkout(tmp_path, monkeypatch)
    link = tmp_path / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(checkout)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0 or not (
        link.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        pytest.skip("Creating Windows junctions is not permitted on this host")
    try:
        with pytest.raises(EvidenceError, match="Reparse points and symbolic links"):
            create_package(link / "evidence")
        assert not (checkout / "evidence").exists()
    finally:
        link.rmdir()


def test_reject_parent_replacement_during_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    original_evaluate = evaluate_demo

    def replace_parent() -> dict[str, object]:
        parent.rename(tmp_path / "former-parent")
        parent.mkdir()
        return original_evaluate()

    monkeypatch.setattr(scientific_report, "evaluate_demo", replace_parent)
    with pytest.raises(EvidenceError, match="changed during evaluation"):
        create_package(parent / "evidence")
    assert not (parent / "evidence").exists()
    assert not list(parent.glob(".eg007b-stage-*"))
