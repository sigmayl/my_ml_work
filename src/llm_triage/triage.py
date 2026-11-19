"""Assemble triage evidence from a log and a pair of configs.

Status: the *collection* half of the loop lives here and is tested. The *judgement*
half — ranking competing failure states, mapping a config delta to a cause, and
proposing the patched YAML — is not implemented; see the `TODO(devin)` markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_triage.configdiff import diff_files
from llm_triage.logparse import parse_log
from llm_triage.models import ConfigDelta, FailureState, ParsedLog


class NotImplementedYet(NotImplementedError):
    """Raised by steps that are scoped to Devin (see README § What's left)."""


@dataclass(frozen=True, slots=True)
class TriageBundle:
    """Everything the report needs, minus the causal judgement."""

    parsed: ParsedLog
    deltas: tuple[ConfigDelta, ...]
    baseline_path: str
    failed_path: str

    @property
    def memory_deltas(self) -> tuple[ConfigDelta, ...]:
        return tuple(delta for delta in self.deltas if delta.memory_relevant)


def build_bundle(
    log_path: str | Path, baseline_path: str | Path, failed_path: str | Path
) -> TriageBundle:
    """Parse the log and diff the failed config against the baseline."""
    return TriageBundle(
        parsed=parse_log(log_path),
        deltas=diff_files(baseline_path, failed_path),
        baseline_path=str(baseline_path),
        failed_path=str(failed_path),
    )


def rank_failure_states(bundle: TriageBundle) -> FailureState:
    """Pick the fatal state and demote the rest to symptoms.

    Inputs available here: `bundle.parsed.failures` (with line numbers and steps),
    `bundle.parsed.oom` (allocator state at abort) and `bundle.memory_deltas`.
    """
    # TODO(devin): rank the detected states. OOM aborts the process; a NaN loss on
    # the step before it is a symptom, not the trigger — but the ranking must be
    # derived from the evidence, not hardcoded to this fixture.
    raise NotImplementedYet(
        "rank_failure_states: unimplemented — needs the fatal-vs-symptom ranking rule"
    )


def propose_fix(bundle: TriageBundle) -> dict[str, object]:
    """Return the failed config with the memory-relevant keys reverted.

    Only keys present in `bundle.memory_deltas` may change; everything else
    (lr, lora_*, seed, paths) must be preserved byte-for-byte in value.
    """
    # TODO(devin): decide *which* memory deltas to revert (reverting
    # gradient_accumulation_steps would change the effective batch size, so it is
    # not automatically part of the fix) and emit the patched mapping.
    raise NotImplementedYet("propose_fix: unimplemented — needs the revert-selection rule")
