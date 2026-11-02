import json
import subprocess
from pathlib import Path

import pytest

from urbangraph import __version__, reproducibility


def test_missing_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    monkeypatch.setattr(reproducibility, "_git", missing)
    info = reproducibility.collect_environment_info(tmp_path)
    assert info.git_status == "unavailable"
    assert info.git_available is False
    assert info.git_commit is None
    assert info.project_version == __version__
    assert info.python_version
    assert info.platform
    assert json.loads(json.dumps(info.to_dict())) == info.to_dict()


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.parametrize("has_commit", [False, True])
def test_repository_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dirty: bool, has_commit: bool
) -> None:
    def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        assert cwd == tmp_path
        responses = {
            ("--version",): (0, "git version 2"),
            ("rev-parse", "--is-inside-work-tree"): (0, "true\n"),
            ("rev-parse", "--verify", "HEAD"): (0, "a" * 40) if has_commit else (128, ""),
            ("status", "--porcelain", "--untracked-files=normal"): (
                0,
                "?? private-filename\n" if dirty else "",
            ),
        }
        code, output = responses[args]
        return subprocess.CompletedProcess(list(args), code, stdout=output)

    monkeypatch.setattr(reproducibility, "_git", git)
    info = reproducibility.collect_environment_info(tmp_path)
    assert info.git_available
    assert info.git_status == ("dirty" if dirty else "clean")
    assert info.git_commit == ("a" * 40 if has_commit else None)
    assert "private-filename" not in json.dumps(info.to_dict())


def test_outside_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            list(args), 0 if args == ("--version",) else 128, stdout=""
        )

    monkeypatch.setattr(reproducibility, "_git", git)
    info = reproducibility.collect_environment_info(tmp_path)
    assert info.git_available
    assert info.git_status == "not-a-repository"
    assert info.git_commit is None


@pytest.mark.parametrize("failure", [OSError("failure"), subprocess.TimeoutExpired("git", 5)])
def test_git_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: Exception) -> None:
    def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        raise failure

    monkeypatch.setattr(reproducibility, "_git", git)
    assert reproducibility.collect_environment_info(tmp_path).git_status == "error"


def test_subprocess_is_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        assert command == ["git", "--version"]
        return subprocess.CompletedProcess(command, 0, stdout="git version 2")

    monkeypatch.setattr(subprocess, "run", run)
    reproducibility._git(tmp_path, "--version")
    assert observed["timeout"] == 5
    assert observed["capture_output"] is True
    assert observed["cwd"] == tmp_path
    assert "shell" not in observed


def test_status_command_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args[0] == "status":
            return subprocess.CompletedProcess(list(args), 128, stdout="")
        output = "true" if "--is-inside-work-tree" in args else "a" * 40
        return subprocess.CompletedProcess(list(args), 0, stdout=output)

    monkeypatch.setattr(reproducibility, "_git", git)
    info = reproducibility.collect_environment_info(tmp_path)
    assert info.git_available
    assert info.git_status == "error"
    assert info.git_commit == "a" * 40


def test_invalid_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("directory")

    monkeypatch.setattr(reproducibility, "_git", git)
    assert reproducibility.collect_environment_info(tmp_path / "absent").git_status == "error"
