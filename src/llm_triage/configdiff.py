"""Key-level YAML config diffing.

A text diff of two YAMLs is useless for triage: one added comment reorders the
whole patch. So configs are flattened to dotted keys and compared structurally,
with numeric deltas kept for the report.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from llm_triage.models import ConfigDelta

MEMORY_RELEVANT_KEYS = frozenset(
    {
        "training.per_device_train_batch_size",
        "training.per_train_batch_size",
        "training.gradient_accumulation_steps",
        "training.gradient_checkpointing",
        "training.max_seq_length",
        "training.num_train_epochs",
    }
)
"""Keys known to move activation memory. Heuristic, not a model of the allocator."""


def flatten(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings to dotted keys; leaves scalars and lists as-is."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def load_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def diff_configs(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    memory_relevant_keys: frozenset[str] = MEMORY_RELEVANT_KEYS,
) -> tuple[ConfigDelta, ...]:
    """Return changed keys, sorted by dotted key, memory-relevant ones flagged."""
    flat_baseline = flatten(baseline)
    flat_candidate = flatten(candidate)

    deltas: list[ConfigDelta] = []
    for key in sorted(set(flat_baseline) | set(flat_candidate)):
        before = flat_baseline.get(key)
        after = flat_candidate.get(key)
        if before == after and type(before) is type(after):
            continue
        numeric_delta, relative_delta = _numeric_delta(before, after)
        deltas.append(
            ConfigDelta(
                key=key,
                baseline=before,
                candidate=after,
                numeric_delta=numeric_delta,
                relative_delta=relative_delta,
                memory_relevant=key in memory_relevant_keys,
            )
        )
    return tuple(deltas)


def diff_files(baseline_path: str | Path, candidate_path: str | Path) -> tuple[ConfigDelta, ...]:
    """Convenience wrapper: load two YAML files and diff them."""
    return diff_configs(load_config(baseline_path), load_config(candidate_path))


def format_value(value: object) -> str:
    """Render a YAML scalar the way the source config would spell it."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value) + "]"
    return str(value)


def _numeric_delta(before: object, after: object) -> tuple[float | None, float | None]:
    if isinstance(before, bool) or isinstance(after, bool):
        return None, None
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None, None
    delta = float(after) - float(before)
    relative = delta / float(before) if before else None
    return delta, relative
