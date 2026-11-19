"""Report rendering tests: what the tool can already state, and what it cannot."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_triage.report import render_markdown
from llm_triage.triage import TriageBundle, build_bundle


@pytest.fixture
def bundle(crash_log: Path, baseline_config: Path, failed_config: Path) -> TriageBundle:
    return build_bundle(crash_log, baseline_config, failed_config)


def test_report_states_the_evidence(bundle: TriageBundle) -> None:
    report = render_markdown(bundle)
    assert "## Detected failure states" in report
    assert "## Allocator state at abort" in report
    assert "## Config delta vs baseline" in report
    assert "requested: 2.50 GiB on GPU 0" in report


def test_report_marks_memory_relevant_deltas(bundle: TriageBundle) -> None:
    report = render_markdown(bundle)
    assert "| `training.per_device_train_batch_size` | 4 | 8 | +4 (+100%) | yes |" in report
    assert "| `training.gradient_checkpointing` | true | false | — | yes |" in report


def test_report_leaves_judgement_to_devin(bundle: TriageBundle) -> None:
    report = render_markdown(bundle)
    assert "TODO(devin)" in report
    assert "_not computed_" in report


@pytest.mark.xfail(
    reason="root-cause ranking is the open TODO(devin) item — remove this marker when it lands",
    strict=True,
)
def test_report_is_complete(bundle: TriageBundle) -> None:
    report = render_markdown(bundle)
    assert "TODO(devin)" not in report
