---
name: Training run triage
about: Structured triage request for a crashed fine-tuning run
title: "[triage] run_XXX — <failure class>"
labels: triage
assignees: ""
---

**Run id / config**
- run: `run_XXX`
- config: `configs/exp_XXX.yaml`
- baseline: `configs/exp_baseline.yaml`
- log: `logs/run_XXX_crash.log`

**Observed**
<!-- last loss / grad norm before abort, terminal traceback class -->

**Suspected**
<!-- config keys that moved since baseline -->

**Requested output**
- [ ] `configs/exp_fixed.yaml`
- [ ] `triage_report.md` (3 bullets: what broke / why / what changed)
- [ ] PR on `devin/triage-run-XXX`
