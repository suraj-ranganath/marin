# Grugformer + MANO Run Log

This file is the canonical log for all `grugformer_mano` runs.
Append-only: do not rewrite prior entries except for factual corrections.

## Entry Format

Use this exact section structure for each new run:

```
## Run <YYYY-MM-DD>-<NN>

### Metadata
- Date:
- Owner:
- Objective:
- Status: `success` | `partial` | `failed`
- Run Type: `toy` | `dry` | `full`
- W&B:
- Results File:
- Code Reference:

### Config
- Model:
- Optimizer:
- Hardware:
- Dataset:
- Key Env:
- Command:

### Outcomes
- C4-EN BPB:
- Model FLOPs:
- Model FLOPs/Token:
- Model Size:
- Tokens Trained:
- Training Time:
- Notes:

### Tradeoffs
- What was intentionally simplified:
- Impact on comparability:
- Risks introduced:

### To Reach Full Submission
- [ ] ...
- [ ] ...

### Best Practices Learned
- ...
```

## Run 2026-02-21-01

### Metadata
- Date: 2026-02-21
- Owner: Suraj Ranganath
- Objective: Validate an end-to-end `grugformer + MANO` 30m pipeline on available GPU and ensure W&B + speedrun results generation work.
- Status: `success` (toy validation)
- Run Type: `toy`
- W&B: https://wandb.ai/suranganath-uc-san-diego/marin/runs/grugformer_mano_30m-9706b9
- Results File: `marin_store/checkpoints/speedrun/grugformer_mano_30m-9706b9/speedrun_results.json`
- Code Reference: `experiments/speedrun/grugformer_mano/grugformer_mano_speedrun.py`

### Config
- Model: Grugformer 30m preset (`hidden_dim=128`, `intermediate_dim=448`, `layers=4`, `heads=2`, `kv_heads=2`)
- Optimizer: MANO (`lr=3e-3`, `wd=0.1`, `beta1=0.9`, `beta2=0.95`, grad clip `1.0`)
- Hardware: 1x GPU (`RTX A5000`), pinned with `CUDA_VISIBLE_DEVICES=0`
- Dataset: `fineweb-edu-10M` pretokenized cache
- Key Env: `SR_SIZE=30m`, `SR_NUM_STEPS=2`, `SR_USE_10M_DATASET=1`, `SR_FORCE_BATCH_AXIS_DATA=1`, `SR_MAX_EVAL_BATCHES=2`
- Command: `.venv/bin/python -m experiments.speedrun.grugformer_mano.grugformer_mano_speedrun`

### Outcomes
- C4-EN BPB: `3.390134334564209`
- Model FLOPs: `1.811536674816e12`
- Model FLOPs/Token: `36855808.0`
- Model Size: `33,784,960` params
- Tokens Trained: `16,384`
- Training Time: `12.075248216278851s`
- Notes: End-to-end completed: training, eval logging to W&B, and speedrun results JSON generation.

### Tradeoffs
- What was intentionally simplified:
- Used `10M` cache instead of full `10B` speedrun training data.
- Used only `2` train steps for fast validation.
- Capped eval with `SR_MAX_EVAL_BATCHES=2` to keep runtime short.
- Impact on comparability:
- This run is not leaderboard-comparable to full submissions; BPB is from a very short training budget and reduced eval coverage.
- FLOP/token and model size are valid, but total model FLOPs/tokens are toy-scale.
- Risks introduced:
- Overestimates near-term readiness if interpreted as full-run quality.
- May hide long-run instabilities that appear at full train/eval scale.

### To Reach Full Submission
- [ ] Remove toy limits: unset `SR_MAX_EVAL_BATCHES` and increase `SR_NUM_STEPS` to full budget.
- [ ] Train on full speedrun data path (disable `SR_USE_10M_DATASET`).
- [ ] Keep W&B run clean and non-resumed for final reporting.
- [ ] Run on stable dedicated GPU resources (and/or expanded GPU allocation).
- [ ] Confirm final `speedrun_results.json` from full run and report `eval/paloma/c4_en/bpb` vs full model FLOPs.

### Best Practices Learned
- On GPU + explicit mesh Grug runs, set batch mapping to `data` (`SR_FORCE_BATCH_AXIS_DATA=1` path in this experiment).
- Pin visible GPUs (`CUDA_VISIBLE_DEVICES`) to avoid accidental multi-GPU clique/rendezvous issues.
- Keep a fast validation path (`SR_MAX_EVAL_BATCHES`) for smoke tests before expensive full runs.
- Always verify the generated speedrun JSON and W&B summary match the intended run config.

## Run 2026-02-21-02

### Metadata
- Date: 2026-02-21
- Owner: Suraj Ranganath
- Objective: Extend the toy run duration while keeping the same Grugformer + MANO setup and W&B tracking.
- Status: `success`
- Run Type: `toy`
- W&B: https://wandb.ai/suranganath-uc-san-diego/marin/runs/grugformer_mano_30m_long35-98ced0
- Results File: `marin_store/checkpoints/speedrun/grugformer_mano_30m_long35-98ced0/speedrun_results.json`
- Code Reference: `experiments/speedrun/grugformer_mano/grugformer_mano_speedrun.py`

### Config
- Model: Grugformer 30m preset (`hidden_dim=128`, `intermediate_dim=448`, `layers=4`, `heads=2`, `kv_heads=2`)
- Optimizer: MANO (`lr=3e-3`, `wd=0.1`, `beta1=0.9`, `beta2=0.95`, grad clip `1.0`)
- Hardware: 1x GPU (`RTX A5000`), pinned with `CUDA_VISIBLE_DEVICES=1`
- Dataset: `fineweb-edu-10M` pretokenized cache
- Key Env: `SR_SIZE=30m`, `SR_NUM_STEPS=35`, `SR_SEQ_LEN=1024`, `SR_BATCH_SIZE_30M=8`, `SR_USE_10M_DATASET=1`, `SR_FORCE_BATCH_AXIS_DATA=1`, `SR_MAX_EVAL_BATCHES=8`, `SR_RUN_NAME_SUFFIX=long35`
- Command: `.venv/bin/python -m experiments.speedrun.grugformer_mano.grugformer_mano_speedrun`

### Outcomes
- C4-EN BPB: `2.459099531173706`
- Model FLOPs: `3.170189180928e13`
- Model FLOPs/Token: `36855808.0`
- Model Size: `33,784,960` params
- Tokens Trained: `286,720`
- Training Time: `48.06831410480663s`
- Notes: Successful extended smoke test, but still far below full-run duration/budget.

### Tradeoffs
- What was intentionally simplified:
- Used `10M` cache rather than full speedrun training data.
- Kept low step count (`35`) and capped eval batches (`8`) for quick turnaround.
- Impact on comparability:
- Not comparable to full leaderboard runs due token budget and reduced eval coverage.
- Risks introduced:
- May overestimate quality trends if interpreted as full-budget behavior.

### To Reach Full Submission
- [ ] Remove toy constraints and run substantially higher train-token budget.
- [ ] Switch from `fineweb-edu-10M` to full speedrun training dataset.
- [ ] Remove eval cap for final comparable metrics.

### Best Practices Learned
- Explicitly set `SR_SEQ_LEN` and `SR_BATCH_SIZE_30M` to avoid accidental oversized defaults on GPU.

## Run 2026-02-21-03

### Metadata
- Date: 2026-02-21
- Owner: Suraj Ranganath
- Objective: Push to a longer toy run (`1000` steps) and measure C4-EN BPB vs FLOPs trend.
- Status: `success`
- Run Type: `toy`
- W&B: https://wandb.ai/suranganath-uc-san-diego/marin/runs/grugformer_mano_30m_long1000-80934f
- Results File: `marin_store/checkpoints/speedrun/grugformer_mano_30m_long1000-80934f/speedrun_results.json`
- Code Reference: `experiments/speedrun/grugformer_mano/grugformer_mano_speedrun.py`

### Config
- Model: Grugformer 30m preset (`hidden_dim=128`, `intermediate_dim=448`, `layers=4`, `heads=2`, `kv_heads=2`)
- Optimizer: MANO (`lr=3e-3`, `wd=0.1`, `beta1=0.9`, `beta2=0.95`, grad clip `1.0`)
- Hardware: 1x GPU (`RTX A5000`), pinned with `CUDA_VISIBLE_DEVICES=1`
- Dataset: `fineweb-edu-10M` pretokenized cache
- Key Env: `SR_SIZE=30m`, `SR_NUM_STEPS=1000`, `SR_SEQ_LEN=1024`, `SR_BATCH_SIZE_30M=8`, `SR_USE_10M_DATASET=1`, `SR_FORCE_BATCH_AXIS_DATA=1`, `SR_MAX_EVAL_BATCHES=8`, `SR_RUN_NAME_SUFFIX=long1000`
- Command: `.venv/bin/python -m experiments.speedrun.grugformer_mano.grugformer_mano_speedrun`

### Outcomes
- C4-EN BPB: `1.9428470134735107`
- Model FLOPs: `9.05768337408e14`
- Model FLOPs/Token: `36855808.0`
- Model Size: `33,784,960` params
- Tokens Trained: `8,192,000`
- Training Time: `153.4451215080917s`
- Notes: Clear BPB improvement versus shorter toy runs; still not full submission scale.

### Tradeoffs
- What was intentionally simplified:
- Used `10M` cache and kept eval cap at `8` batches.
- Impact on comparability:
- Still toy-scale compute and data; relative trends are useful but leaderboard parity is limited.
- Risks introduced:
- Long-horizon stability and full-eval behavior remain untested.

### To Reach Full Submission
- [ ] Run with full speedrun training data.
- [ ] Remove eval cap for final submission-grade measurement.
- [ ] Increase compute budget to target full speedrun comparability.

### Best Practices Learned
- A `1000`-step run is useful for directional signal, but wall-time remained too short for a longer benchmark-style report.

## Run 2026-02-21-04

### Metadata
- Date: 2026-02-21
- Owner: Suraj Ranganath
- Objective: Launch a `10000`-step run to hit ~20-30 minute duration.
- Status: `failed`
- Run Type: `toy`
- W&B: none (failed before W&B init)
- Results File: none
- Code Reference: `experiments/speedrun/grugformer_mano/grugformer_mano_speedrun.py`

### Config
- Model: Grugformer 30m preset (default seq/batch path)
- Optimizer: MANO
- Hardware: 1x GPU (`RTX A5000`)
- Dataset: `fineweb-edu-10M`
- Key Env: `SR_SIZE=30m`, `SR_NUM_STEPS=10000`, `SR_USE_10M_DATASET=1`, `SR_FORCE_BATCH_AXIS_DATA=1`, `SR_MAX_EVAL_BATCHES=8`, `SR_RUN_NAME_SUFFIX=long10000`
- Command: `.venv/bin/python -m experiments.speedrun.grugformer_mano.grugformer_mano_speedrun`

### Outcomes
- C4-EN BPB: n/a
- Model FLOPs: n/a
- Model FLOPs/Token: n/a
- Model Size: n/a
- Tokens Trained: n/a
- Training Time: n/a
- Notes: Failed immediately with `wandb.errors.UsageError: No API key configured`.

### Tradeoffs
- What was intentionally simplified:
- Relied on ambient shell auth state.
- Impact on comparability:
- No run data generated.
- Risks introduced:
- Prevented training start and delayed longer-duration experiment.

### To Reach Full Submission
- [ ] Always set `WANDB_API_KEY` explicitly in launch command or env bootstrap.
- [ ] Add preflight check for W&B auth before launching long jobs.

### Best Practices Learned
- Treat W&B auth as required launch configuration, not an implicit environment assumption.

## Run 2026-02-21-05

### Metadata
- Date: 2026-02-21
- Owner: Suraj Ranganath
- Objective: Retry `10000`-step run with W&B auth enabled.
- Status: `failed`
- Run Type: `toy`
- W&B: https://wandb.ai/suranganath-uc-san-diego/marin/runs/grugformer_mano_30m_long10000-62975e
- Results File: none
- Code Reference: `experiments/speedrun/grugformer_mano/grugformer_mano_speedrun.py`

### Config
- Model: Grugformer 30m preset with defaults (`seq_len=2048`)
- Optimizer: MANO
- Hardware: 1x GPU (`RTX A5000`)
- Dataset: `fineweb-edu-10M`
- Key Env: `SR_SIZE=30m`, `SR_NUM_STEPS=10000`, `SR_USE_10M_DATASET=1`, `SR_FORCE_BATCH_AXIS_DATA=1`, `SR_MAX_EVAL_BATCHES=8`, `SR_RUN_NAME_SUFFIX=long10000` (default batch path -> `train_batch_size=128`, `train_seq_len=2048`)
- Command: `.venv/bin/python -m experiments.speedrun.grugformer_mano.grugformer_mano_speedrun`

### Outcomes
- C4-EN BPB: n/a
- Model FLOPs: n/a
- Model FLOPs/Token: n/a
- Model Size: n/a
- Tokens Trained: n/a
- Training Time: n/a
- Notes: Failed with GPU OOM (`Allocator ran out of memory trying to allocate 32.00GiB`).

### Tradeoffs
- What was intentionally simplified:
- Used default batch/sequence settings in the script.
- Impact on comparability:
- No submission metrics produced.
- Risks introduced:
- Large default shape (`128 x 2048`) is not viable on single A5000 for this setup.

### To Reach Full Submission
- [ ] Force safe GPU settings (`SR_SEQ_LEN=1024`, `SR_BATCH_SIZE_30M=8`) for this environment.
- [ ] Add a launch profile per hardware class to avoid accidental OOM defaults.

### Best Practices Learned
- Safe explicit GPU overrides are required for reproducible long runs in this config.

## Run 2026-02-21-06

### Metadata
- Date: 2026-02-21
- Owner: Suraj Ranganath
- Objective: Complete a longer 30m Grugformer+MANO run meeting a ~20-30 minute wall-time target.
- Status: `success`
- Run Type: `toy`
- W&B: https://wandb.ai/suranganath-uc-san-diego/marin/runs/grugformer_mano_30m_long10000_safe-dab5c8
- Results File: `marin_store/checkpoints/speedrun/grugformer_mano_30m_long10000_safe-dab5c8/speedrun_results.json`
- Code Reference: `experiments/speedrun/grugformer_mano/grugformer_mano_speedrun.py`

### Config
- Model: Grugformer 30m preset (`hidden_dim=128`, `intermediate_dim=448`, `layers=4`, `heads=2`, `kv_heads=2`)
- Optimizer: MANO (`lr=3e-3`, `wd=0.1`, `beta1=0.9`, `beta2=0.95`, grad clip `1.0`)
- Hardware: 1x GPU (`RTX A5000`), pinned with `CUDA_VISIBLE_DEVICES=1`
- Dataset: `fineweb-edu-10M` pretokenized cache
- Key Env: `SR_SIZE=30m`, `SR_NUM_STEPS=10000`, `SR_SEQ_LEN=1024`, `SR_BATCH_SIZE_30M=8`, `SR_USE_10M_DATASET=1`, `SR_FORCE_BATCH_AXIS_DATA=1`, `SR_MAX_EVAL_BATCHES=8`, `SR_RUN_NAME_SUFFIX=long10000_safe`, `WANDB_API_KEY=<set>`
- Command: `.venv/bin/python -m experiments.speedrun.grugformer_mano.grugformer_mano_speedrun`

### Outcomes
- C4-EN BPB: `1.7023742198944092`
- Model FLOPs: `9.05768337408e15`
- Model FLOPs/Token: `36855808.0`
- Model Size: `33,784,960` params
- Tokens Trained: `81,920,000`
- Training Time: `1163.2630535541102s`
- Notes: End-to-end success; executor wall-time was `1381.99s` (~`23m02s`), which satisfies the 20-30 minute target.

### Tradeoffs
- What was intentionally simplified:
- Still used `fineweb-edu-10M` and capped eval batches at `8`.
- Impact on comparability:
- Stronger than short toy runs, but still not full submission comparable due data/eval limits.
- Risks introduced:
- Reported BPB is directionally useful but not a final leaderboard-quality submission metric.

### To Reach Full Submission
- [ ] Train on full speedrun training dataset (disable `SR_USE_10M_DATASET`).
- [ ] Remove eval cap (`SR_MAX_EVAL_BATCHES`) for full comparable evaluation.
- [ ] Keep long-run stable settings and scale hardware/token budget to full run target.

### Best Practices Learned
- For A5000, `SR_SEQ_LEN=1024` + `SR_BATCH_SIZE_30M=8` is a stable launch profile for longer Grugformer+MANO runs.
- `SR_RUN_NAME_SUFFIX` prevents accidental cache reuse when iterating step budgets.
