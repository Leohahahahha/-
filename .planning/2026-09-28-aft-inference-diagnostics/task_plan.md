# AFT inference diagnostics

User approved the prior in-chat design: optional attention export, modality summaries and paired normalized-history perturbations; deployment AI handoff documentation; retain dual-GPU training. This is a bounded extension of existing model/policy/serving flows, not a new expert architecture.

## Phases

1. Recover previous changes and verify CPU baseline — complete.
2. Test-first opt-in attention capture without parameter-tree or default inference changes — complete.
3. Test-first policy diagnostics, fixed-noise perturbations, export/plot and server options — complete.
4. Deployment handoff, incident log, dual-GPU instructions, regression verification and Git commits — complete (code 2e4a9c8; documentation saved alongside this plan).
5. Diagnose user-provided 14:49 training log, XLA memory warning and live progress without restarting training — complete; current dual-GPU run has real updates, old 4gpu-named run failed on missing adjacency assets. Incident57 records evidence and verification boundaries.
6. Diagnose missing W&B charts: identify current run/project, inspect sync evidence and read-only cloud history — complete; approved host query confirms running and uploaded step300 in tabero-aft/fwut42ps, not legacy tabero-vtla. Incident58 updated.
7. Compare physical-signal versus latent prediction for future-feedback gates; inspect primary papers/current shear semantics and record scoped recommendation — complete; representative N0/MoSS/TA-VLA methods verified, direct396/6 baseline recommended with calibrated residual/time/reference caveats. Incident59 updated, no architecture/training change.
8. Explain current A/T/F joint attention versus legacy concatenated action/wrench token; verify mask and record code locations — complete; actual CPU mask assertions passed, incident60 records private weights/sharedattention, all-horizon reads and supervised7:13 semantics. No model/training changes.

## Constraints

- Work only in linked worktree on 部署代码兼容版本; leave dirty original training branch untouched.
- No training, robot calls, external upload or dependency changes without need.
- Default-off diagnostics must preserve standard predictions and checkpoint parameter names.
- Attention describes reads, not causal contribution. Perturb only normalized history, retain future A/T/F generation.
- Capture head-averaged attention at one configured denoising step; report explicit layer/query/token layout.
- Handoff consumers must not feed diagnostic output into robot control.

## Errors

Logged in progress.md: mistaken dummy-depth and mean-history fixtures corrected; capture failure/orphan/layer bounds reproduced and fixed. Broader project suite has 9 unrelated/environment failures, with 5 reproduced on baseline d734786 and 3 tokenizer cache cases recovered using existing cache. FAST asset absent. First view_image used a pytest temp path already pruned; later fresh-path inspection succeeded. No unsafe integration/merge is performed while full suite remains red.
