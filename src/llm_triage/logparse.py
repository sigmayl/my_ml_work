"""Regex parsing of `accelerate launch` + DeepSpeed ZeRO-3 training logs.

Design note: this module is deliberately dependency-free (stdlib `re` only) so the
parser can run on the training host where no project deps are installed, and so a
log shipped off the cluster stays readable without the stack that produced it.

Everything here is covered by tests/fixtures/*.log; add a fixture before adding a
pattern, otherwise the pattern rots silently.
"""

from __future__ import annotations

import re
from pathlib import Path

from llm_triage.models import (
    FailureKind,
    FailureState,
    OomReport,
    ParsedLog,
    StepMetric,
    TracebackFrame,
)

_LOG_LINE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} [\d:]+)\]\s+(?P<level>[A-Z]+)\s+-\s+(?P<msg>.*)$"
)
_STEP = re.compile(r"Step (?P<step>\d+)/(?P<total>\d+)")
_LOSS = re.compile(r"Loss:\s*(?P<loss>nan|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)
_LR = re.compile(r"LR:\s*(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_GRAD_NORM = re.compile(
    r"(?:Grad norm|grad_norm)\s*[:=]\s*(?P<gnorm>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

_OOM = re.compile(
    r"CUDA out of memory\. Tried to allocate (?P<req>\d+(?:\.\d+)?) (?P<unit>[KMG]iB) "
    r"\(GPU (?P<gpu>\d+); (?P<total>\d+(?:\.\d+)?) GiB total capacity; "
    r"(?P<alloc>\d+(?:\.\d+)?) GiB already allocated; "
    r"(?P<free>\d+(?:\.\d+)?) GiB free(?:; (?P<reserved>\d+(?:\.\d+)?) GiB reserved)?",
)
_FRAME = re.compile(r'^\s+File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')
_EXCEPTION = re.compile(r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Abort)):\s*(?P<msg>.*)$")
_TENSOR_MISMATCH = re.compile(
    r"(?:size mismatch|shape mismatch|Expected input batch_size|"
    r"cannot reshape tensor of \d+ elements)",
    re.IGNORECASE,
)
_NCCL_TIMEOUT = re.compile(
    r"(?:NCCL[A-Za-z ]*?(?:timed out|timeout)|Watchdog caught collective operation timeout)",
    re.IGNORECASE,
)
_CHILD_FAILURE = re.compile(r"ChildFailedError|Caught exception in worker process")
_RANK = re.compile(r"^\s+rank\s+:\s+(?P<rank>\d+)")
_EXITCODE = re.compile(r"^\s+exitcode\s+:\s+(?P<code>\d+)")

_UNIT_TO_GIB = {"KiB": 1 / 1024**2, "MiB": 1 / 1024, "GiB": 1.0}


def parse_log(path: str | Path) -> ParsedLog:
    """Parse one training log into metrics, failures, frames and allocator state."""
    log_path = Path(path)
    lines = log_path.read_text(encoding="utf-8").splitlines()

    metrics: list[StepMetric] = []
    failures: list[FailureState] = []
    frames: list[TracebackFrame] = []
    oom: OomReport | None = None
    step: int | None = None
    child_seen = False

    for line_no, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        message = _message_of(line)
        step_match = _STEP.search(message)

        if step_match:
            step = int(step_match.group("step"))
            metric = _build_metric(step, int(step_match.group("total")), message, line_no)
            metrics.append(metric)
            if metric.loss is None:
                failures.append(
                    FailureState(
                        kind=FailureKind.NON_FINITE_LOSS,
                        message=message,
                        line_no=line_no,
                        step=step,
                        evidence=line,
                    )
                )
            continue

        if (oom_match := _OOM.search(line)) is not None:
            oom = _build_oom(oom_match)
            failures.append(
                FailureState(
                    kind=FailureKind.OOM,
                    message=line.strip(),
                    line_no=line_no,
                    step=step,
                    evidence=line,
                )
            )
            continue

        if (frame_match := _FRAME.match(raw)) is not None:
            frames.append(
                TracebackFrame(
                    path=frame_match.group("path"),
                    line=int(frame_match.group("line")),
                    func=frame_match.group("func"),
                )
            )
            continue

        if _CHILD_FAILURE.search(line) and not child_seen:
            child_seen = True
            failures.append(
                FailureState(
                    kind=FailureKind.CHILD_FAILURE,
                    message=line.strip(),
                    line_no=line_no,
                    step=step,
                    evidence=line,
                )
            )
            continue

        if _TENSOR_MISMATCH.search(line):
            failures.append(
                FailureState(
                    kind=FailureKind.TENSOR_MISMATCH,
                    message=line.strip(),
                    line_no=line_no,
                    step=step,
                    evidence=line,
                )
            )
            continue

        if _NCCL_TIMEOUT.search(line):
            failures.append(
                FailureState(
                    kind=FailureKind.NCCL_TIMEOUT,
                    message=line.strip(),
                    line_no=line_no,
                    step=step,
                    evidence=line,
                )
            )

    return ParsedLog(
        path=str(log_path),
        metrics=tuple(metrics),
        failures=tuple(failures),
        frames=tuple(frames),
        oom=oom,
    )


def _message_of(line: str) -> str:
    match = _LOG_LINE.match(line)
    return match.group("msg") if match else line


def _build_metric(step: int, total: int, message: str, line_no: int) -> StepMetric:
    loss_match = _LOSS.search(message)
    lr_match = _LR.search(message)
    gnorm_match = _GRAD_NORM.search(message)
    return StepMetric(
        step=step,
        total_steps=total,
        loss=_finite_or_none(loss_match.group("loss")) if loss_match else None,
        lr=float(lr_match.group("lr")) if lr_match else None,
        grad_norm=float(gnorm_match.group("gnorm")) if gnorm_match else None,
        line_no=line_no,
    )


def _finite_or_none(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _build_oom(match: re.Match[str]) -> OomReport:
    reserved = match.group("reserved")
    return OomReport(
        requested_gib=float(match.group("req")) * _UNIT_TO_GIB[match.group("unit")],
        gpu=int(match.group("gpu")),
        total_gib=float(match.group("total")),
        allocated_gib=float(match.group("alloc")),
        free_gib=float(match.group("free")),
        reserved_gib=float(reserved) if reserved else None,
    )
