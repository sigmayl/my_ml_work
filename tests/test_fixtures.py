"""The committed crash log must be exactly what `llm_triage.fixtures` renders."""

from __future__ import annotations

from pathlib import Path

from llm_triage.fixtures import METRICS, render
from llm_triage.logparse import parse_log


def test_committed_log_matches_renderer(crash_log: Path) -> None:
    assert crash_log.read_text(encoding="utf-8") == render()


def test_rendered_log_parses_to_the_same_evidence(crash_log: Path, tmp_path: Path) -> None:
    rendered = tmp_path / "rendered.log"
    rendered.write_text(render(), encoding="utf-8")

    on_disk = parse_log(crash_log)
    regenerated = parse_log(rendered)
    assert regenerated.kinds == on_disk.kinds
    assert regenerated.oom == on_disk.oom
    assert regenerated.metrics == on_disk.metrics


def test_loss_trend_is_monotonic_until_divergence() -> None:
    losses = [float(row[3]) for row in METRICS[:-1]]
    assert losses == sorted(losses, reverse=True)
    assert METRICS[-1][3] == "nan"
