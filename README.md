# LLM Experiment Triage (MRE for Devin)

Minimal reproducible example (MRE) for an autonomous **experiment triage loop**: parse a crashed
training log, diff the run config against the last known-good baseline, deduce the root cause,
propose a patch, and write a handoff report.

The upstream lab repo is private (institutional NDA + licensed checkpoints), so this repo carries a
**sanitized, synthetic reproduction** of one real failure mode: a Llama-3 LoRA run launched with
`accelerate` + DeepSpeed ZeRO-3 that dies at step 81. No proprietary data, weights, or logs are
included — only the config shape, the launch topology, and the traceback class.

## Stack

| Layer | Tooling |
| --- | --- |
| Launcher | `accelerate launch` (multi-GPU, single node) |
| Sharding | DeepSpeed ZeRO-3 (`configs/ds_z3_config.json`) |
| Model | `meta-llama/Meta-Llama-3-8B`, bf16, LoRA (PEFT) |
| Dataset | `tatsu-lab/alpaca` (packed, `max_seq_length: 2048`) |

## Repo layout

```text
.
├── accelerate_config.yaml     # launch topology (num_processes, mixed precision)
├── configs/
│   ├── exp_baseline.yaml      # last known-good run
│   ├── exp_failed.yaml        # run_042 — the one that crashed
│   └── ds_z3_config.json      # ZeRO-3 sharding policy
├── logs/
│   └── run_042_crash.log      # stdout/stderr of the failed run
└── train.py                   # skeleton entrypoint (not required to execute)
```

## The Devin task

Point Devin at this repo and prompt it to complete the loop end to end:

1. **Parse** `logs/run_042_crash.log`. Extract step index, loss trajectory, grad-norm warning, and
   the terminal traceback. Use regex / structured parsing rather than eyeballing — the goal is a
   reusable triage step, not a one-off read.
2. **Classify** the failure state. The log contains both a `Loss: nan` and a
   `RuntimeError: CUDA out of memory`, so it must distinguish the *primary* fatal error from the
   secondary symptom and rank them.
3. **Diff** `configs/exp_failed.yaml` against `configs/exp_baseline.yaml` (key-level YAML diff,
   numeric deltas, not a text diff).
4. **Root-cause.** Correlate the diff with the traceback: the memory footprint changed because
   `per_device_train_batch_size` was doubled from 4 → 8 while `gradient_checkpointing` went
   `true` → `false`; `gradient_accumulation_steps` was halved 8 → 4 to hold the effective batch
   size constant, which means activations are no longer amortized across micro-steps while
   checkpointing is gone. Rank 0 allocates the extra 2.50 GiB and aborts.
5. **Patch + report.** Emit `configs/exp_fixed.yaml` reverting the memory-relevant keys, plus
   `triage_report.md` with exactly three bullets: what broke, why (evidence lines quoted), what
   changed. Then open a PR against this repo on a `devin/triage-run-042` branch.

### Acceptance criteria

- [ ] Failure classified as `OOM` with `NaN loss` recorded as a co-occurring symptom, not the cause.
- [ ] Config diff is key-level and numeric (`4 -> 8`, `true -> false`, `8 -> 4`).
- [ ] Root cause cites at least one quoted log line and one config key.
- [ ] `configs/exp_fixed.yaml` is a valid YAML superset of the failed config (only memory-relevant
      keys changed; `learning_rate`, `lora_*`, `max_seq_length` untouched).
- [ ] `triage_report.md` is exactly three bullets, no preamble, no restating the prompt.
- [ ] Changes land on a branch and a PR is opened — no direct commits to `main`.

## Notes

- `train.py` is a skeleton kept for path-faithfulness with the traceback; it is **not** expected to
  run in this repo (no checkpoints, no GPU allocator to reproduce the abort).
- The log timestamps/steps are synthetic but the shape (bf16 load, ZeRO-3 init, grad-norm warning
  before divergence, `torch.distributed.elastic` worker teardown) mirrors the observed run.
