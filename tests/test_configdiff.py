"""Config differ tests: the diff must be key-level, typed and spoiler-free."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_triage.configdiff import diff_configs, diff_files, flatten, format_value, load_config
from llm_triage.models import ConfigDelta


def test_flatten_produces_dotted_keys() -> None:
    flat = flatten({"training": {"lora_r": 16, "nested": {"a": 1}}, "seed": 42})
    assert flat == {"training.lora_r": 16, "training.nested.a": 1, "seed": 42}


def test_diff_is_key_level_and_sorted(baseline_config: Path, failed_config: Path) -> None:
    deltas = diff_files(baseline_config, failed_config)
    # output_dir moves every run and must not be mistaken for a cause.
    assert [delta.key for delta in deltas] == [
        "output_dir",
        "training.gradient_accumulation_steps",
        "training.gradient_checkpointing",
        "training.per_device_train_batch_size",
    ]
    assert [delta.memory_relevant for delta in deltas] == [False, True, True, True]


def test_numeric_deltas_are_signed(baseline_config: Path, failed_config: Path) -> None:
    deltas = {delta.key: delta for delta in diff_files(baseline_config, failed_config)}

    batch = deltas["training.per_device_train_batch_size"]
    assert (batch.baseline, batch.candidate) == (4, 8)
    assert batch.numeric_delta == 4.0
    assert batch.relative_delta == pytest.approx(1.0)

    accum = deltas["training.gradient_accumulation_steps"]
    assert accum.numeric_delta == -4.0
    assert accum.relative_delta == pytest.approx(-0.5)


def test_boolean_flip_has_no_numeric_delta(baseline_config: Path, failed_config: Path) -> None:
    deltas = {delta.key: delta for delta in diff_files(baseline_config, failed_config)}
    ckpt = deltas["training.gradient_checkpointing"]
    assert (ckpt.baseline, ckpt.candidate) == (True, False)
    assert ckpt.numeric_delta is None
    assert ckpt.memory_relevant


def test_unchanged_keys_are_not_reported(baseline_config: Path) -> None:
    assert diff_configs(load_config(baseline_config), load_config(baseline_config)) == ()


def test_missing_key_is_reported_as_null() -> None:
    deltas = diff_configs({"training": {"lora_r": 16}}, {"training": {}})
    assert len(deltas) == 1
    delta: ConfigDelta = deltas[0]
    assert delta.candidate is None
    assert not delta.memory_relevant


def test_bool_is_not_treated_as_int() -> None:
    deltas = diff_configs(
        {"training": {"gradient_checkpointing": True}}, {"training": {"gradient_checkpointing": 1}}
    )
    assert len(deltas) == 1
    assert deltas[0].numeric_delta is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, "true"), (False, "false"), (None, "null"), (16, "16"), (["a"], "[a]")],
)
def test_format_value_matches_yaml_spelling(value: object, expected: str) -> None:
    assert format_value(value) == expected
