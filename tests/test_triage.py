"""Bundle assembly tests, plus the two steps that are explicitly not implemented."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_triage.triage import (
    NotImplementedYet,
    TriageBundle,
    build_bundle,
    propose_fix,
    rank_failure_states,
)


@pytest.fixture
def bundle(crash_log: Path, baseline_config: Path, failed_config: Path) -> TriageBundle:
    return build_bundle(crash_log, baseline_config, failed_config)


def test_bundle_collects_log_and_delta(bundle: TriageBundle) -> None:
    assert bundle.parsed.oom is not None
    assert len(bundle.deltas) == 4  # output_dir + three training keys
    assert len(bundle.memory_deltas) == 3


def test_ranking_is_not_implemented(bundle: TriageBundle) -> None:
    with pytest.raises(NotImplementedYet):
        rank_failure_states(bundle)


def test_fix_proposal_is_not_implemented(bundle: TriageBundle) -> None:
    with pytest.raises(NotImplementedYet):
        propose_fix(bundle)


def test_fix_must_preserve_non_memory_keys(bundle: TriageBundle) -> None:
    """Spec for whoever implements propose_fix: only memory-relevant keys may move."""
    untouched = {
        "training.learning_rate",
        "training.lora_r",
        "training.lora_alpha",
        "training.max_seq_length",
        "seed",
    }
    assert not {delta.key for delta in bundle.deltas} & untouched
