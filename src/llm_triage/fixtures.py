"""Regenerate `logs/run_042_crash.log` from a declarative trace table.

The committed log is a **synthetic reproduction** of a real crash (see README § Notes).
Keeping it generated rather than hand-edited means the fixture stays reviewable: the
step/loss table below is the whole story, and `tests/test_fixtures.py` asserts that the
file on disk is exactly what this module renders.

    python -m llm_triage.fixtures --write
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

RUN_ID = "run_042"
CONFIG = "configs/exp_failed.yaml"
TOTAL_STEPS = 1000
OOM_REQUEST_GIB = 2.50
OOM_TOTAL_GIB = 79.35
OOM_ALLOCATED_GIB = 76.12
OOM_FREE_GIB = 1.20
OOM_RESERVED_GIB = 78.00


class MetricRow(tuple):
    """`(ts, level, step, loss, lr, grad_norm)` — `None` means "not logged"."""

    __slots__ = ()

    def __new__(
        cls,
        ts: str,
        level: str,
        step: int,
        loss: str,
        lr: str | None = None,
        grad_norm: str | None = None,
    ) -> MetricRow:
        return super().__new__(cls, (ts, level, step, loss, lr, grad_norm))


PROLOGUE: tuple[str, ...] = (
    "[2024-05-18 14:32:07] INFO - accelerate launch --config_file accelerate_config.yaml "
    "--num_processes 4 train.py --config " + CONFIG,
    f"[2024-05-18 14:32:10] INFO - Starting training {RUN_ID} with {CONFIG}",
    "[2024-05-18 14:32:11] INFO - deepspeed.initialize: zero_optimization.stage=3, "
    "offload=none, bf16=enabled",
    "[2024-05-18 14:32:15] INFO - Loading Llama-3-8B in bfloat16...",
    "[2024-05-18 14:32:58] INFO - [rank0] model loaded (8.03B params, 16.1 GB/rank sharded)",
    "[2024-05-18 14:32:59] INFO - [rank2] NCCL warmup ok, world_size=4, backend=nccl",
)

METRICS: tuple[MetricRow, ...] = (
    MetricRow("14:33:02", "INFO", 10, "1.842", lr="2.00e-05"),
    MetricRow("14:35:12", "INFO", 50, "1.213", lr="2.00e-05"),
    MetricRow("14:36:40", "INFO", 70, "1.104", lr="1.98e-05"),
    MetricRow("14:38:45", "WARNING", 80, "0.982", grad_norm="4.5"),
    MetricRow("14:39:01", "ERROR", 81, "nan"),
)
"""Loss decreases, one grad-norm warning, then the step where it goes non-finite."""

EPILOGUE: tuple[str, ...] = (
    "[2024-05-18 14:39:01] ERROR - torch.distributed.elastic.multiprocessing.api: [ERROR] "
    "Caught exception in worker process:",
    "Traceback (most recent call last):",
    '  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1511, '
    "in _call_impl",
    "    return forward_call(*args, **kwargs)",
    '  File "/workspace/train.py", line 142, in forward',
    "    outputs = self.model(**inputs)",
    '  File "/usr/local/lib/python3.10/dist-packages/transformers/models/llama/modeling_llama.py"'
    ", line 892, in forward",
    "    layer_outputs = decoder_layer(",
    '  File "/usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/partitioned_param.py"'
    ", line 402, in forward",
    "    return self._forward(*args, **kwargs)",
    '  File "/usr/local/lib/python3.10/dist-packages/torch/autograd/function.py", line 552, '
    "in apply",
    "    return super().apply(*args, **kwargs)  # pylint: disable=protected-access",
    f"RuntimeError: CUDA out of memory. Tried to allocate {OOM_REQUEST_GIB:.2f} GiB "
    f"(GPU 0; {OOM_TOTAL_GIB:.2f} GiB total capacity; {OOM_ALLOCATED_GIB:.2f} GiB already "
    f"allocated; {OOM_FREE_GIB:.2f} GiB free; {OOM_RESERVED_GIB:.2f} GiB reserved in total "
    "by PyTorch)",
    "[2024-05-18 14:39:01] ERROR - [rank0] "
    "torch.distributed.elastic.multiprocessing.errors.ChildFailedError:",
    "  train.py FAILED",
    "-----------------------------------------------------------",
    "Root Cause (first observed failure):",
    "[0]:",
    "  time      : 2024-05-18_14:39:01",
    "  host      : node-0",
    "  rank      : 0 (local_rank 0)",
    "  exitcode  : 1",
    "  error_file: <N/A>",
    f"  traceback : RuntimeError: CUDA out of memory. Tried to allocate {OOM_REQUEST_GIB:.2f} GiB",
    "-----------------------------------------------------------",
    "[2024-05-18 14:39:02] INFO - [rank1] NCCL watchdog: aborting collective",
    "[2024-05-18 14:39:02] INFO - Process group destroyed.",
)


def render() -> str:
    """Render the whole log, byte-identical to the committed fixture."""
    lines = list(PROLOGUE)
    lines.extend(_metric_line(row) for row in METRICS)
    lines.extend(EPILOGUE)
    return "\n".join(lines) + "\n"


def _metric_line(row: MetricRow) -> str:
    ts, level, step, loss, lr, grad_norm = row
    line = f"[2024-05-18 {ts}] {level} - Step {step}/{TOTAL_STEPS} | Loss: {loss}"
    if grad_norm is not None:
        return f"{line} | Grad norm: {grad_norm}"
    if lr is not None:
        return f"{line} | LR: {lr}"
    return line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm_triage.fixtures")
    parser.add_argument(
        "--write",
        action="store_true",
        help="overwrite logs/run_042_crash.log instead of printing to stdout",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "logs" / "run_042_crash.log",
    )
    args = parser.parse_args(argv)

    content = render()
    if args.write:
        args.out.write_text(content, encoding="utf-8")
        print(f"wrote {args.out} ({len(content.splitlines())} lines)")
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
