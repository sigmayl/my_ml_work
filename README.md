# LLM Experiment Triage

Tooling for the loop I run after every crashed fine-tuning job: **read the log → diff the config
against the last known-good baseline → rank the failure → propose a patch → hand off a report.**

The upstream lab repo is private (institutional NDA + licensed checkpoints). This is the sanitized
minimal reproducible example (MRE) for that loop, built around one real failure mode: a Llama-3 LoRA
run launched with `accelerate` + DeepSpeed ZeRO-3 that dies at step 81.

> **Status:** the *evidence* half of the loop is implemented and tested (`parse`, `diff`, evidence
> assembly). The *judgement* half — ranking the failure, mapping a config delta to a cause, emitting
> the patched YAML — is stubbed with `TODO(devin)` markers. See [What's left](#whats-left).

## Stack

| Layer | Tooling |
| --- | --- |
| Launcher | `accelerate launch`, single node, 4 processes |
| Sharding | DeepSpeed ZeRO-3 (`configs/ds_z3_config.json`) |
| Model | `meta-llama/Meta-Llama-3-8B`, bf16, LoRA via PEFT |
| Dataset | `tatsu-lab/alpaca`, packed to `max_seq_length: 2048` |
| Package | Python ≥ 3.10, hatchling, ruff, pytest |

## Layout

```text
.
├── accelerate_config.yaml        # launch topology (DEEPSPEED, bf16, 4 GPUs)
├── pyproject.toml                # package metadata, ruff + pytest config
├── Makefile                      # install / lint / test / triage
├── train.py                      # skeleton entrypoint (mirrors the lab's, not runnable here)
├── src/llm_triage/
│   ├── models.py                 # FailureState, StepMetric, ConfigDelta, ParsedLog
│   ├── logparse.py               # regex parser (stdlib only)
│   ├── configdiff.py             # key-level YAML diff, memory-relevant flags
│   ├── triage.py                 # bundle assembly + the two TODO(devin) judgements
│   ├── report.py                 # markdown renderer
│   ├── fixtures.py               # regenerates the committed crash log
│   └── cli.py                    # `llm-triage {parse,diff,triage}`
├── tests/                        # 36 tests, run against the real fixture files
├── configs/
│   ├── exp_baseline.yaml         # last known-good run (run_039)
│   ├── exp_failed.yaml           # run_042 — the one that crashed
│   └── ds_z3_config.json         # ZeRO-3 policy
├── logs/run_042_crash.log        # synthetic reproduction of the failing run
└── .github/workflows/ci.yml      # ruff + format + pytest + CLI smoke test
```

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make test        # 36 passed, 1 xfailed (the xfail is the open judgement path)
make lint        # ruff check + ruff format --check
```

## CLI

```console
$ llm-triage parse logs/run_042_crash.log
logs/run_042_crash.log: 5 step lines, 3 signals
  step   10 loss=  1.8420 grad_norm=     —
  ...
  step   81 loss=     nan grad_norm=     —
  [non_finite_loss] line 11: Step 81/1000 | Loss: nan
  [child_failure] line 12: ... Caught exception in worker process:
  [oom] line 24: RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB (GPU 0; ...

$ llm-triage diff configs/exp_baseline.yaml configs/exp_failed.yaml
output_dir: /workspace/runs/run_039 -> /workspace/runs/run_042
training.gradient_accumulation_steps: 8 -> 4 (memory-relevant)
training.gradient_checkpointing: true -> false (memory-relevant)
training.per_device_train_batch_size: 4 -> 8 (memory-relevant)

$ llm-triage triage --log logs/run_042_crash.log \
    --baseline configs/exp_baseline.yaml --failed configs/exp_failed.yaml --out-dir out/
llm-triage: root-cause ranking and fix proposal are not implemented (TODO(devin));
re-run with --allow-partial to accept the evidence-only report
$ echo $?
3
```

`--json` on any subcommand emits the same data machine-readably. Exit codes are contractual:

| code | meaning |
| --- | --- |
| `0` | success |
| `2` | bad input (missing log, unparseable YAML) |
| `3` | the requested step is still unimplemented (see [What's left](#whats-left)) |

## What's left

| Loop step | Status | Where |
| --- | --- | --- |
| Parse the log into structured metrics + failure states | done | `logparse.py` |
| Detect `CUDA OOM` with allocator numbers, `Loss: nan`, tensor mismatch, NCCL timeout, elastic child failure | done | `logparse.py` |
| Key-level config diff with numeric deltas + memory-relevant flags | done | `configdiff.py` |
| Render evidence (failure table, allocator state, config delta) | done | `report.py` |
| **Rank the detected states — fatal vs symptom** | **`TODO(devin)`** | `triage.rank_failure_states` |
| **Map the config delta to a cause and propose the fix** | **`TODO(devin)`** | `triage.propose_fix` |
| **Fill the root-cause section and the 3-bullet summary** | **`TODO(devin)`** | `report.py` placeholders |
| Open the PR | out of scope for the package | — |

The two stubs raise `NotImplementedYet` (a `NotImplementedError` subclass) with the inputs they
already have spelled out in their docstrings, so the work is scoped rather than open-ended.

## The Devin task

Point Devin at this repo and ask it to close the loop end to end:

1. **Parse** `logs/run_042_crash.log` — reuse `llm_triage.logparse`, don't re-write regexes. It
   already recovers step, loss, LR, grad-norm, the traceback frames and the allocator line.
2. **Rank** the three detected states (`non_finite_loss`, `child_failure`, `oom`). The log contains
   both a NaN loss at step 81 and an OOM on rank 0, so the ranking must be evidence-driven: which
   signal aborted the process, which is a symptom?
3. **Diff** `configs/exp_failed.yaml` against `configs/exp_baseline.yaml` — `llm-triage diff` already
   returns four deltas, one of which (`output_dir`) is per-run noise and must not be blamed.
4. **Root-cause.** Connect the delta to the abort: `per_device_train_batch_size` 4 → 8 doubles
   activation memory while `gradient_checkpointing` goes `true` → `false`; `gradient_accumulation_steps`
   8 → 4 keeps the *effective* batch size constant, so activations are no longer amortized as
   micro-steps. Rank 0 asks for the extra 2.50 GiB with 1.20 GiB free and aborts.
5. **Patch + hand off.** Emit `configs/exp_fixed.yaml` (only memory-relevant keys move; `learning_rate`,
   `lora_*`, `seed`, `max_seq_length` must be preserved), fill `triage_report.md` with the root cause
   and exactly three bullets — *what broke / why / what changed* — then open a PR on
   `devin/triage-run-042`.

### Acceptance criteria

- [ ] Failure ranked as `oom`; `non_finite_loss` recorded as a co-occurring symptom, not the cause.
- [ ] Root cause cites one quoted log line **and** one config delta, plus the mechanism between them.
- [ ] `configs/exp_fixed.yaml` is valid YAML and diffs against the failed config only in
      memory-relevant keys.
- [ ] `triage_report.md` has exactly three bullets, no preamble, no restating the prompt.
- [ ] `make test` and `make lint` still pass; the `xfail` marker on `test_report_is_complete` is gone.
- [ ] Work lands on a branch and a PR is opened — no direct commits to `main`.

## Fixtures

`logs/run_042_crash.log` is **synthetic**: it is rendered from the step table in
`src/llm_triage/fixtures.py`, not copied off the cluster (no proprietary data, weights or run IDs are
included). `tests/test_fixtures.py` asserts the file on disk is byte-identical to what the renderer
produces, so the fixture can't drift. Regenerate with:

```bash
python -m llm_triage.fixtures --write
```

Known limitation, deliberately left in: `parse` matches `size mismatch`-style tensor errors but not
`mat1 and mat2 shapes cannot be multiplied`. `tests/test_logparse.py` pins that gap rather than
papering over it.
