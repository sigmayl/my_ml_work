"""Markdown report rendering for a triage bundle."""

from __future__ import annotations

import re

from llm_triage.configdiff import format_value
from llm_triage.models import ConfigDelta
from llm_triage.triage import TriageBundle

ROOT_CAUSE_PLACEHOLDER = (
    "<!-- TODO(devin): replace this section with the ranked root cause: one quoted log\n"
    "     line + one config delta + the mechanism that connects them. -->\n"
    "_not computed_\n"
)

SUMMARY_PLACEHOLDER = (
    "<!-- TODO(devin): exactly three bullets — what broke / why / what changed.\n"
    "     No preamble, no restating the prompt. -->\n"
    "- _what broke_: not computed\n"
    "- _why_: not computed\n"
    "- _what changed_: not computed\n"
)


def render_markdown(bundle: TriageBundle) -> str:
    """Render the evidence sections plus placeholders for the judgement sections."""
    rendered = "\n".join(
        [
            _header(bundle),
            _failure_section(bundle),
            _oom_section(bundle),
            _delta_section(bundle),
            "## Root cause\n",
            ROOT_CAUSE_PLACEHOLDER,
            "## Summary\n",
            SUMMARY_PLACEHOLDER,
        ]
    )
    return re.sub(r"\n{3,}", "\n\n", rendered)


def _header(bundle: TriageBundle) -> str:
    parsed = bundle.parsed
    last = parsed.last_metric
    last_step = f"step {last.step} of {last.total_steps}" if last else "no step logged"
    return (
        f"# Triage report — `{parsed.path.split('/')[-1]}`\n\n"
        f"- log: `{parsed.path}`\n"
        f"- failed config: `{bundle.failed_path}`\n"
        f"- baseline: `{bundle.baseline_path}`\n"
        f"- last logged progress: {last_step}\n"
        f"- detected states: {', '.join(k.value for k in parsed.kinds) or 'none'}\n"
    )


def _failure_section(bundle: TriageBundle) -> str:
    lines = ["## Detected failure states\n"]
    if not bundle.parsed.failures:
        lines.append("_no failure signal matched_\n")
        return "\n".join(lines)
    lines.append("| kind | step | line | message |")
    lines.append("| --- | --- | --- | --- |")
    for failure in bundle.parsed.failures:
        message = failure.message.replace("|", "\\|")
        if len(message) > 110:
            message = message[:107] + "…"
        lines.append(
            f"| `{failure.kind.value}` | {failure.step or '—'} | {failure.line_no} | {message} |"
        )
    lines.append("")
    return "\n".join(lines)


def _oom_section(bundle: TriageBundle) -> str:
    oom = bundle.parsed.oom
    if oom is None:
        return ""
    return (
        "## Allocator state at abort\n\n"
        f"- requested: {oom.requested_gib:.2f} GiB on GPU {oom.gpu}\n"
        f"- total: {oom.total_gib:.2f} GiB, allocated: {oom.allocated_gib:.2f} GiB, "
        f"free: {oom.free_gib:.2f} GiB\n"
        + (f"- reserved: {oom.reserved_gib:.2f} GiB\n" if oom.reserved_gib is not None else "")
        + "\n"
    )


def _delta_section(bundle: TriageBundle) -> str:
    lines = ["## Config delta vs baseline\n"]
    if not bundle.deltas:
        lines.append("_configs are identical_\n")
        return "\n".join(lines)
    lines.append("| key | baseline | failed | Δ | memory-relevant |")
    lines.append("| --- | --- | --- | --- | --- |")
    for delta in bundle.deltas:
        lines.append(_delta_row(delta))
    lines.append("")
    return "\n".join(lines)


def _delta_row(delta: ConfigDelta) -> str:
    change = "—"
    if delta.numeric_delta is not None:
        change = f"{delta.numeric_delta:+g}"
        if delta.relative_delta is not None:
            change += f" ({delta.relative_delta:+.0%})"
    flag = "yes" if delta.memory_relevant else ""
    return (
        f"| `{delta.key}` | {format_value(delta.baseline)} | {format_value(delta.candidate)} "
        f"| {change} | {flag} |"
    )
