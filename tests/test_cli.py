"""CLI contract tests: exit codes and machine-readable output."""

from __future__ import annotations

import json
from pathlib import Path

from llm_triage.cli import EXIT_OK, EXIT_UNIMPLEMENTED, EXIT_USAGE, REPORT_NAME, main


def test_parse_json_is_machine_readable(crash_log: Path, capsys) -> None:
    assert main(["parse", str(crash_log), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["metrics"]) == 5
    assert payload["oom"]["requested_gib"] == 2.5


def test_diff_json_lists_changed_keys(baseline_config: Path, failed_config: Path, capsys) -> None:
    assert main(["diff", str(baseline_config), str(failed_config), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert [row["key"] for row in payload] == [
        "output_dir",
        "training.gradient_accumulation_steps",
        "training.gradient_checkpointing",
        "training.per_device_train_batch_size",
    ]


def test_triage_reports_unimplemented(
    crash_log: Path, baseline_config: Path, failed_config: Path, tmp_path: Path, capsys
) -> None:
    code = main(
        [
            "triage",
            "--log",
            str(crash_log),
            "--baseline",
            str(baseline_config),
            "--failed",
            str(failed_config),
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert code == EXIT_UNIMPLEMENTED
    assert (tmp_path / REPORT_NAME).exists()


def test_triage_allow_partial_succeeds(
    crash_log: Path, baseline_config: Path, failed_config: Path, tmp_path: Path
) -> None:
    code = main(
        [
            "triage",
            "--log",
            str(crash_log),
            "--baseline",
            str(baseline_config),
            "--failed",
            str(failed_config),
            "--out-dir",
            str(tmp_path),
            "--allow-partial",
            "--json",
        ]
    )
    assert code == EXIT_OK
    report = (tmp_path / REPORT_NAME).read_text(encoding="utf-8")
    assert "## Root cause" in report


def test_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    assert main(["parse", str(tmp_path / "nope.log")]) == EXIT_USAGE
