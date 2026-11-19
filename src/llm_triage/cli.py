"""Command line entrypoint: `llm-triage {parse,diff,triage}`.

Exit codes are part of the contract (CI and the Devin automation both rely on them):
  0 — success
  2 — bad input (missing file, unparseable YAML)
  3 — the requested step is not implemented yet (see README § What's left)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_triage import __version__
from llm_triage.configdiff import diff_files, format_value
from llm_triage.logparse import parse_log
from llm_triage.report import render_markdown
from llm_triage.triage import TriageBundle, build_bundle

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNIMPLEMENTED = 3

REPORT_NAME = "triage_report.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-triage",
        description="Triage crashed accelerate + DeepSpeed ZeRO-3 fine-tuning runs.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="parse a run log into failure states")
    parse_cmd.add_argument("log", type=Path)
    parse_cmd.add_argument("--json", action="store_true", help="machine-readable output")
    parse_cmd.set_defaults(handler=_cmd_parse)

    diff_cmd = subparsers.add_parser("diff", help="diff a failed config against a baseline")
    diff_cmd.add_argument("baseline", type=Path)
    diff_cmd.add_argument("failed", type=Path)
    diff_cmd.add_argument("--json", action="store_true", help="machine-readable output")
    diff_cmd.set_defaults(handler=_cmd_diff)

    triage_cmd = subparsers.add_parser("triage", help="build the triage report bundle")
    triage_cmd.add_argument("--log", type=Path, required=True)
    triage_cmd.add_argument("--baseline", type=Path, required=True)
    triage_cmd.add_argument("--failed", type=Path, required=True)
    triage_cmd.add_argument("--out-dir", type=Path, default=Path("."))
    triage_cmd.add_argument(
        "--allow-partial",
        action="store_true",
        help="write the evidence report even though root-causing is unimplemented",
    )
    triage_cmd.add_argument("--json", action="store_true", help="machine-readable output")
    triage_cmd.set_defaults(handler=_cmd_triage)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except OSError as exc:
        print(f"llm-triage: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:
        print(f"llm-triage: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _cmd_parse(args: argparse.Namespace) -> int:
    parsed = parse_log(args.log)
    if args.json:
        print(json.dumps(_parsed_payload(parsed), indent=2))
        return EXIT_OK

    print(f"{parsed.path}: {len(parsed.metrics)} step lines, {len(parsed.failures)} signals")
    for metric in parsed.metrics:
        loss = "nan" if metric.loss is None else f"{metric.loss:.4f}"
        gnorm = f"{metric.grad_norm:g}" if metric.grad_norm is not None else "—"
        print(f"  step {metric.step:>4} loss={loss:>8} grad_norm={gnorm:>6}")
    for failure in parsed.failures:
        print(f"  [{failure.kind.value}] line {failure.line_no}: {failure.message[:96]}")
    return EXIT_OK


def _cmd_diff(args: argparse.Namespace) -> int:
    deltas = diff_files(args.baseline, args.failed)
    if args.json:
        print(json.dumps([_delta_payload(delta) for delta in deltas], indent=2))
        return EXIT_OK

    if not deltas:
        print("configs are identical")
        return EXIT_OK
    for delta in deltas:
        flag = " (memory-relevant)" if delta.memory_relevant else ""
        print(
            f"{delta.key}: {format_value(delta.baseline)} -> {format_value(delta.candidate)}{flag}"
        )
    return EXIT_OK


def _cmd_triage(args: argparse.Namespace) -> int:
    bundle = build_bundle(args.log, args.baseline, args.failed)
    report_path = Path(args.out_dir) / REPORT_NAME
    report_path.write_text(render_markdown(bundle), encoding="utf-8")

    if args.json:
        print(json.dumps(_bundle_payload(bundle, report_path), indent=2))
    else:
        print(f"wrote {report_path}")
        print(
            f"evidence: {len(bundle.parsed.failures)} signals, "
            f"{len(bundle.deltas)} config deltas ({len(bundle.memory_deltas)} memory-relevant)"
        )

    if not args.allow_partial:
        print(
            "llm-triage: root-cause ranking and fix proposal are not implemented "
            "(TODO(devin)); re-run with --allow-partial to accept the evidence-only report",
            file=sys.stderr,
        )
        return EXIT_UNIMPLEMENTED
    return EXIT_OK


def _parsed_payload(parsed: object) -> dict[str, object]:
    from llm_triage.models import ParsedLog

    assert isinstance(parsed, ParsedLog)
    oom = parsed.oom
    return {
        "path": parsed.path,
        "metrics": [
            {
                "step": metric.step,
                "total_steps": metric.total_steps,
                "loss": metric.loss,
                "lr": metric.lr,
                "grad_norm": metric.grad_norm,
                "line_no": metric.line_no,
            }
            for metric in parsed.metrics
        ],
        "failures": [
            {
                "kind": failure.kind.value,
                "message": failure.message,
                "line_no": failure.line_no,
                "step": failure.step,
            }
            for failure in parsed.failures
        ],
        "frames": [
            {"path": frame.path, "line": frame.line, "func": frame.func} for frame in parsed.frames
        ],
        "oom": None
        if oom is None
        else {
            "requested_gib": oom.requested_gib,
            "gpu": oom.gpu,
            "total_gib": oom.total_gib,
            "allocated_gib": oom.allocated_gib,
            "free_gib": oom.free_gib,
            "reserved_gib": oom.reserved_gib,
        },
    }


def _delta_payload(delta: object) -> dict[str, object]:
    from llm_triage.models import ConfigDelta

    assert isinstance(delta, ConfigDelta)
    return {
        "key": delta.key,
        "baseline": delta.baseline,
        "failed": delta.candidate,
        "numeric_delta": delta.numeric_delta,
        "relative_delta": delta.relative_delta,
        "memory_relevant": delta.memory_relevant,
    }


def _bundle_payload(bundle: TriageBundle, report_path: Path) -> dict[str, object]:
    return {
        "log": _parsed_payload(bundle.parsed),
        "deltas": [_delta_payload(delta) for delta in bundle.deltas],
        "baseline_path": bundle.baseline_path,
        "failed_path": bundle.failed_path,
        "report_path": str(report_path),
        "root_cause": None,
        "implemented": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
