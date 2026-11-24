# SALTED playground

Work on the personal fork `zekunlou/SALTED`. Every branch an agent creates must begin with `codex/`. Protect `master` on both the fork and upstream. Never open a pull request to `andreagrisafi/SALTED`. Before pushing, inspect the current branch and the push URL. Push only the named `codex/` branch to the personal fork.

## Leading an experiment

1. Read the experiment notes in `playground/README.md`. Record the starting commit, the user’s objective, the scientific outputs to preserve, and a time or trial budget.
2. Run the baseline before changing SALTED. Select a development case and reserve a second relevant case for independent confirmation. Keep inputs, thresholds and thread counts fixed across a comparison.
3. Give each worker one hypothesis, a bounded set of files and a `codex/` branch in its own worktree. Ask for the patch commit, exact commands, baseline and candidate measurements, and what the result implies. Run competing timing experiments sequentially on this laptop.
4. Inspect the patch and reproduce the result. Run the relevant unit and integration tests, including a real multi-rank run for MPI changes. Compare performance across repeated runs and check numerical outputs against the reference. Record failed hypotheses and inconclusive results alongside adopted changes.
5. Summarize the decision, scientific error, runtime and peak memory with their units. Keep raw data, generated models and machine-specific paths local.

Use the installed Arbor skill for campaigns with competing hypotheses. For one change, use the [experiment brief](playground/experiment.md) to record the assignment, evidence and decision.
