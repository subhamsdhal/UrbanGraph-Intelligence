"""Invented directory bytes exercise the broker tree contract."""

from pathlib import Path

import pytest

from energraph.broker_tree import copy_verified_tree, inspect_tree
from urbangraph.data import DataError


def _invented(root: Path) -> None:
    root.mkdir()
    (root / "descriptor.json").write_bytes(b"{}")
    (root / "trial-0").mkdir()
    (root / "trial-0" / "weights.bin").write_bytes(b"invented weights")


def test_tree_identity_copy_and_tamper(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _invented(source)
    identity = inspect_tree(source)
    assert identity.digest == "47893fe25c773f5a51f2c53ce5f45cc621f3e0586a2724f5d071de55d592f0a0"
    assert identity.directories == ("trial-0",)
    destination = tmp_path / "private"
    copy_verified_tree(source, destination, identity)
    assert inspect_tree(destination) == identity
    with pytest.raises(DataError, match="already exists"):
        copy_verified_tree(source, destination, identity)
    (destination / "trial-0" / "weights.bin").write_bytes(b"tampered")
    assert inspect_tree(destination).digest != identity.digest


def test_tree_rejects_case_aliases(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _invented(source)
    (source / "DESCRIPTOR.JSON").write_bytes(b"alias")
    if "DESCRIPTOR.JSON" not in {item.name for item in source.iterdir()}:
        pytest.skip("Host filesystem collapses case aliases")
    with pytest.raises(DataError, match="case-normalized"):
        inspect_tree(source)


def test_tree_rejects_links(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _invented(source)
    try:
        (source / "shortcut").symlink_to(source / "descriptor.json")
    except OSError:
        pytest.skip("File symlink creation unavailable")
    with pytest.raises(DataError, match="Symlinks"):
        inspect_tree(source)


@pytest.mark.parametrize("name", ["CON.txt", "trailing.", "named:stream"])
def test_tree_rejects_nonportable_names(tmp_path: Path, name: str) -> None:
    source = tmp_path / "source"
    _invented(source)
    try:
        (source / name).write_bytes(b"invented")
    except OSError:
        pytest.skip("Host refuses nonportable name before broker check")
    if name not in {item.name for item in source.iterdir()}:
        pytest.skip("Host filesystem normalizes unsupported name")
    with pytest.raises(DataError, match="nonportable"):
        inspect_tree(source)
