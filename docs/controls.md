# Optional controls and sensitivity runs

These configurations are delivered but are **not executed in the first pilot**.
Each new run receives a separate persisted budget and output directory. Prepare
and push its configuration before model execution. No control uses humanize or
SWE-bench tasks for policy training.

| Configuration | Intervention | Interpretation |
|---|---|---|
| `single_agent.toml` | Disable delegation during collection and evaluation | Joint effect of the changed training and inference harness |
| `repair_only.toml` | Freeze the first pilot's validated bug corpus; update solver actions only | Removes adaptive injection and injector gradients; corpus dependence remains |
| `procedural.toml` | Use deterministic test-detected AST mutations in training repositories | Changes the task generator and exposes original tests; not an isolated delegation ablation |
| `no_delegation_eval.toml` | Evaluate frozen/final pilot adapters without child sessions | Inference-time delegation intervention under the same team token cap |
| `seed_1.toml`, `seed_2.toml` | Repeat self-play with another initial seed | Small sensitivity check; not a substitute for a powered study |

For each training configuration, use a new directory:

```bash
uv run --no-sync selfplay preflight --config configs/single_agent.toml --run runs/single-agent
uv run --no-sync selfplay pilot --config configs/single_agent.toml --run runs/single-agent
uv run --no-sync selfplay report --run runs/single-agent --output reports/single-agent --site site/single-agent
```

Replace the configuration and run directory consistently for other training
controls. Repair-only reads `runs/pilot/artifact_records`; if the pilot produced
no valid bugs, it fails explicitly rather than silently changing the control.
Procedural training uses the same mutation operator implementation as the
diagnostic generator on **different repositories**, validating all selected
clean tests before admitting a mutant. This is a separately labeled control,
not the SSR test-weakening environment.

For the evaluation-only intervention:

```bash
uv run --no-sync selfplay evaluate --config configs/no_delegation_eval.toml --run runs/no-delegation-eval
uv run --no-sync selfplay report --run runs/no-delegation-eval --output reports/no-delegation-eval --site site/no-delegation-eval
```

It reads the initial and final completed checkpoint from `checkpoint_run` and
does no optimization. Baseline and final evaluation always use the same
delegation setting within a comparison. Each condition needs its own report;
do not merge conditions by selecting their best task outcome.
