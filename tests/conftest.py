"""Shared fixtures. Tests run against the real fixture files in this repo."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_triage.logparse import parse_log
from llm_triage.models import ParsedLog

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def crash_log(repo_root: Path) -> Path:
    return repo_root / "logs" / "run_042_crash.log"


@pytest.fixture(scope="session")
def baseline_config(repo_root: Path) -> Path:
    return repo_root / "configs" / "exp_baseline.yaml"


@pytest.fixture(scope="session")
def failed_config(repo_root: Path) -> Path:
    return repo_root / "configs" / "exp_failed.yaml"


@pytest.fixture(scope="session")
def parsed(crash_log: Path) -> ParsedLog:
    return parse_log(crash_log)
