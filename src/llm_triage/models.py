"""Dataclasses shared by the parser, the config differ and the report renderer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureKind(str, Enum):
    """Failure classes we have actually seen in this project's runs."""

    OOM = "oom"
    """Allocator aborted: `CUDA out of memory`."""

    NON_FINITE_LOSS = "non_finite_loss"
    """Loss went NaN/Inf — usually a symptom, not the trigger."""

    TENSOR_MISMATCH = "tensor_mismatch"
    """Shape bug after a config/patch change."""

    NCCL_TIMEOUT = "nccl_timeout"
    """Collective hang or watchdog abort across ranks."""

    CHILD_FAILURE = "child_failure"
    """`torch.distributed.elastic` worker teardown (non-zero exit)."""

    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StepMetric:
    """One `Step N/M | Loss: x | LR: y` line."""

    step: int
    total_steps: int | None
    loss: float | None
    """`None` when the logged value was not finite (e.g. `nan`)."""
    lr: float | None
    grad_norm: float | None
    line_no: int


@dataclass(frozen=True, slots=True)
class FailureState:
    """A single detected failure signal, with the line that produced it."""

    kind: FailureKind
    message: str
    line_no: int
    step: int | None = None
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class TracebackFrame:
    """One `File "...", line N, in f` frame, in source order."""

    path: str
    line: int
    func: str


@dataclass(frozen=True, slots=True)
class OomReport:
    """Parsed `CUDA out of memory` allocator state."""

    requested_gib: float
    gpu: int
    total_gib: float
    allocated_gib: float
    free_gib: float
    reserved_gib: float | None = None


@dataclass(frozen=True, slots=True)
class ParsedLog:
    """Everything recoverable from one crashed run's stdout/stderr."""

    path: str
    metrics: tuple[StepMetric, ...]
    failures: tuple[FailureState, ...]
    frames: tuple[TracebackFrame, ...]
    oom: OomReport | None = None

    @property
    def last_metric(self) -> StepMetric | None:
        """Last step that logged a loss, finite or not."""
        return self.metrics[-1] if self.metrics else None

    @property
    def kinds(self) -> tuple[FailureKind, ...]:
        """Distinct failure kinds, in order of first appearance."""
        seen: dict[FailureKind, None] = {}
        for failure in self.failures:
            seen.setdefault(failure.kind, None)
        return tuple(seen)

    def of_kind(self, kind: FailureKind) -> tuple[FailureState, ...]:
        return tuple(failure for failure in self.failures if failure.kind is kind)


@dataclass(frozen=True, slots=True)
class ConfigDelta:
    """One changed key between a baseline and a candidate config."""

    key: str
    """Dotted path, e.g. `training.per_device_train_batch_size`."""
    baseline: object
    candidate: object
    numeric_delta: float | None = None
    relative_delta: float | None = None
    memory_relevant: bool = False
