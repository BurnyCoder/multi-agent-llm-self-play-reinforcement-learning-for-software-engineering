# Working on this research project

Preserve the user's accepted plan and `reports/preregistration.md`. Work only in this checkout and the explicitly configured task data directory. Use WSL2 and the project-local `.venv`; do not run Windows `uv` against the Linux environment. Do not run concurrent dependency installers.

Keep model/data downloads and their caches on the external `D:\AI\multi-agent-ssr` drive. Verify that `/mnt/d` is mounted before preparation or loading. Do not recreate files on C: or C-backed WSL storage when their originals are on D:; reconnect and mount the drive instead. Preserve unique run records when removing newly created duplicates.

Read original papers, official documentation and source implementations before changing algorithms; cite them in the relevant modules and reports. Prefer small reusable library-backed components. Keep full model prompt/output logs, exact action tokens and policy versions. Withhold clean reference source, reference fixes, injection diffs and original Git history from repair agents. The inverse test-weakening patch is intentionally their task specification ([SSR §2.4](https://arxiv.org/html/2512.18552v3#S2.SS4); `rollouts.repair_prompt`). Never mount host credentials in agent containers.

Research root and child agents must run asynchronously; a primary is instructed to continue independent work while children run. All use one shared model and adapter. GPU inference is serialized; explicit edits also use a lock (`agents.AgentHarness`). No parameter updates during collection. Each eligible nonzero-advantage action is scored once per optimization pass in its own context; zero-advantage actions are recorded but skipped (`learning.update_policy`, [CWM Appendix C](https://arxiv.org/html/2510.02387v1#A3)). These are implementation requirements, not claims that every recorded run completed learning.

Write behavior/math tests before fixes where meaningful; inspect the full user-facing pipeline and logs. Commit functional units, push fixes, and perform one focused final review. Preflight and research model execution require committed/pushed code and manifests. Do not run the expanded controlled study in this first delivery.

All results must come from saved records. Regenerate artifacts with the report command. Keep positive, negative and inconclusive outcomes; record exclusions and pre/post-hoc changes. No fabricated values or unsupported improvement claims. The paper author is Libor "Burny" Burian; the requested AI statement belongs only in the paper.
