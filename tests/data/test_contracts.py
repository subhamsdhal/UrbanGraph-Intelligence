import hashlib
from dataclasses import replace

import pytest

from urbangraph.data import (
    METR_LA,
    PEMS_BAY,
    DataError,
    DatasetIdentity,
    DatasetSource,
    RawFileRecord,
)
from urbangraph.data.contracts import (
    content_identity,
    validate_path_inventory,
    validate_relative_path,
)


@pytest.mark.parametrize("value", ["metr-la", "pems-bay", "future-city-2", "a", "0"])
def test_extensible_identities(value: str) -> None:
    assert DatasetIdentity(value).dataset_id == value
    assert METR_LA.dataset_id == "metr-la"
    assert PEMS_BAY.dataset_id == "pems-bay"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "METR-LA",
        "../metr-la",
        "/root",
        "a/b",
        "a\\b",
        "a:b",
        "a--b",
        "-a",
        "a-",
        "con",
        "a" * 65,
    ],
)
def test_invalid_identifiers(value: str) -> None:
    with pytest.raises(DataError):
        DatasetIdentity(value)


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/a",
        "../a",
        "a/../b",
        "a/./b",
        "a//b",
        "a/",
        "C:/file",
        "a\\b",
        "a:stream",
        "NUL.txt",
        "CON .txt",
        "nested/COM1",
        "nested/LPT¹.txt",
        "name.",
        "name ",
        "a\x00b",
        "a?b",
        "e\u0301.h5",
    ],
)
def test_unsafe_relative_paths(path: str) -> None:
    with pytest.raises(DataError):
        validate_relative_path(path)


@pytest.mark.parametrize("paths", [("a", "a"), ("A", "a"), ("Dir/a", "dir/b"), ("a", "a/b")])
def test_path_collisions(paths: tuple[str, ...]) -> None:
    with pytest.raises(DataError):
        validate_path_inventory(paths)


def test_content_identity_golden_vector() -> None:
    a = RawFileRecord("a.bin", 3, hashlib.sha256(b"abc").hexdigest())
    # Independent encoding pins the canonical algorithm, not the implementation order.
    canonical = (
        '{"files":[{"relative_path":"a.bin","sha256":"'
        + hashlib.sha256(b"abc").hexdigest()
        + '","size_bytes":3}],"identity_version":"sha256-path-size-v1"}'
    )
    assert content_identity((a,)) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    b = replace(a, relative_path="nested/b.bin")
    assert content_identity((a, b)) == content_identity((b, a))
    assert content_identity((a,)) != content_identity((b,))
    assert content_identity((a,)) != content_identity((replace(a, size_bytes=4),))


@pytest.mark.parametrize("size", [-1, True])
def test_invalid_size(size: int) -> None:
    with pytest.raises(DataError):
        RawFileRecord("a", size, "a" * 64)


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64])
def test_invalid_digest(digest: str) -> None:
    with pytest.raises(DataError):
        RawFileRecord("a", 0, digest)


def test_unknown_source_is_not_invented() -> None:
    assert DatasetSource().reference is None
    assert DatasetSource().licensing_notes is None
    with pytest.raises(DataError):
        DatasetSource(reference=" ")
