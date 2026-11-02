"""Collect bounded, serializable environment metadata without environment secrets."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from urbangraph import __version__

GitStatus = Literal["clean", "dirty", "not-a-repository", "unavailable", "error"]


@dataclass(frozen=True)
class EnvironmentInfo:
    """Snapshot only; this is not a complete experiment provenance manifest."""

    python_version: str
    platform: str
    project_version: str
    git_available: bool
    git_status: GitStatus
    git_commit: str | None

    def to_dict(self) -> dict[str, str | bool | None]:
        """Return JSON-compatible scalars, excluding paths and environment variables."""
        return asdict(self)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
        check=False,
    )


def collect_environment_info(repository: str | Path | None = None) -> EnvironmentInfo:
    """Inspect the given directory (default: cwd); tolerate missing or failing Git.

    Dirty includes untracked files. An unborn repository has no commit. Raw Git
    output, filenames, hostnames and environment variables are never returned.
    """
    cwd = Path.cwd() if repository is None else Path(repository)
    available = False
    status: GitStatus = "unavailable"
    commit: str | None = None
    try:
        probe = _git(cwd, "--version")
        available = probe.returncode == 0
        if available:
            check = _git(cwd, "rev-parse", "--is-inside-work-tree")
            if check.returncode != 0 or check.stdout.strip() != "true":
                status = "not-a-repository"
            else:
                head = _git(cwd, "rev-parse", "--verify", "HEAD")
                commit = head.stdout.strip() if head.returncode == 0 else None
                changes = _git(cwd, "status", "--porcelain", "--untracked-files=normal")
                status = (
                    "error"
                    if changes.returncode != 0
                    else "dirty"
                    if changes.stdout.strip()
                    else "clean"
                )
    except FileNotFoundError:
        # An invalid cwd is an inspection error, not evidence Git is missing.
        status = "unavailable" if cwd.is_dir() else "error"
    except (OSError, subprocess.TimeoutExpired):
        status = "error"
    return EnvironmentInfo(
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        project_version=__version__,
        git_available=available,
        git_status=status,
        git_commit=commit,
    )
