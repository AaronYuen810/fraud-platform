"""Pytest hooks — ensure Feast registry is applied before tests that touch the online store."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _feast_cli() -> str:
    exe = shutil.which("feast")
    if exe:
        return exe
    candidate = _ROOT / ".venv" / "bin" / "feast"
    if candidate.is_file():
        return str(candidate)
    pytest.skip("Feast CLI not found; install dev dependencies and ensure `feast` is on PATH.")


@pytest.fixture(scope="session", autouse=True)
def feast_apply_session() -> None:
    """Register entities and feature views so push/online reads work in tests."""
    subprocess.run(
        [_feast_cli(), "-c", "feature_repo", "apply"],
        cwd=_ROOT,
        check=True,
    )
