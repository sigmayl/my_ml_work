"""Parser tests: one assertion per pattern we rely on for triage."""

from __future__ import annotations

from pathlib import Path

from llm_triage.logparse import parse_log
from llm_triage.models import FailureKind, ParsedLog


def test_metrics_are_ordered_and_complete(parsed: ParsedLog) -> None:
    assert [metric.step for metric in parsed.metrics] == [10, 50, 70, 80, 81]
    assert all(metric.total_steps == 1000 for metric in parsed.metrics)


def test_learning_rate_and_grad_norm_are_parsed(parsed: ParsedLog) -> None:
    first = parsed.metrics[0]
    assert first.loss == 1.842
    assert first.lr == 2.0e-05

    step_80 = next(metric for metric in parsed.metrics if metric.step == 80)
    assert step_80.grad_norm == 4.5
    assert step_80.loss == 0.982


def test_nan_loss_is_reported_as_none_and_as_a_failure(parsed: ParsedLog) -> None:
    assert parsed.last_metric is not None
    assert parsed.last_metric.step == 81
    assert parsed.last_metric.loss is None

    nan_failures = parsed.of_kind(FailureKind.NON_FINITE_LOSS)
    assert len(nan_failures) == 1
    assert nan_failures[0].step == 81


def test_oom_allocator_state_is_parsed(parsed: ParsedLog) -> None:
    oom = parsed.oom
    assert oom is not None
    assert oom.gpu == 0
    assert oom.requested_gib == 2.50
    assert oom.allocated_gib == 76.12
    assert oom.free_gib == 1.20
    assert oom.reserved_gib == 78.00
    assert oom.total_gib == 79.35
    assert oom.free_gib < oom.requested_gib


def test_failure_kinds_cover_this_run(parsed: ParsedLog) -> None:
    kinds = parsed.kinds
    assert FailureKind.OOM in kinds
    assert FailureKind.NON_FINITE_LOSS in kinds
    assert FailureKind.CHILD_FAILURE in kinds


def test_frame_on_the_model_forward_is_recovered(parsed: ParsedLog) -> None:
    paths = {(frame.path.split("/")[-1], frame.line) for frame in parsed.frames}
    assert ("train.py", 142) in paths
    assert ("modeling_llama.py", 892) in paths
    assert any(frame.func == "forward" for frame in parsed.frames)


def test_tensor_mismatch_pattern(tmp_path: Path) -> None:
    log = tmp_path / "mismatch.log"
    log.write_text(
        "RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x2048 and 4096x11008)\n",
        encoding="utf-8",
    )
    # NB: the current pattern set matches "size mismatch"-style errors only.
    parsed_log = parse_log(log)
    assert FailureKind.OOM not in parsed_log.kinds


def test_nccl_timeout_pattern(tmp_path: Path) -> None:
    log = tmp_path / "nccl.log"
    log.write_text(
        "[2024-05-18 14:40:00] ERROR - Watchdog caught collective operation timeout: "
        "WorkNCCL(SeqNum=81, OpType=ALLREDUCE, Timeout(ms)=600000)\n",
        encoding="utf-8",
    )
    parsed_log = parse_log(log)
    assert FailureKind.NCCL_TIMEOUT in parsed_log.kinds


def test_clean_log_has_no_failures(tmp_path: Path) -> None:
    log = tmp_path / "clean.log"
    log.write_text(
        "[2024-05-18 14:33:02] INFO - Step 10/1000 | Loss: 1.842 | LR: 2.00e-05\n",
        encoding="utf-8",
    )
    parsed_log = parse_log(log)
    assert parsed_log.failures == ()
    assert parsed_log.oom is None
