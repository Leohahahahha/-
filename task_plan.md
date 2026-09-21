# Task Plan: Tabero action-only LoRA smoke training

## Active task: enable 6D wrist-wrench prediction for the whiteboard pair (2026-09-15)
- Phase 49: Trace the existing 13D Tabero action/force objective and audit both datasets' `wrist_wrench[6]` fields. Status: complete.
- Phase 50: Add an explicit `7D action + 6D wrist_wrench` training transform, independent normalization, force-enabled configs and collision-safe run names. Status: complete.
- Phase 51: Extend CPU transform/config/batch/launcher tests and update the paired runbook plus incident log. Status: complete.
- Phase 52: Provide new single-line dry-run/train/resume/log commands without launching either GPU training job. Status: complete.
- Do not launch training, load model weights, upload to W&B, run inference, connect to a service, or actuate the robot; the user will start training.

## Active task: paired whiteboard action-definition training (2026-09-15)
- Phase 44: Audit both compact datasets, action/task contracts, episode counts, fields and comparable split candidates. Status: complete.
- Phase 45: Add two isolated tactile-LoRA configs initialized from published Tabero 49999 with identical 20k/4k/W&B hyperparameters. Status: complete.
- Phase 46: Add a fail-fast sequential preparation/training launcher with separate run, log, asset and checkpoint paths. Status: complete.
- Phase 47: Run CPU audits/config/batch/launcher tests without starting model training, then update the incident log. Status: complete.
- Phase 48: Provide single-line unattended launch, monitoring, resume/failure and artifact commands. Status: complete.
- Do not launch either 20,000-step training run, load model weights for inference, connect services, or actuate the robot; the user will start training.

## Active task: v3 tactile LoRA 20k training handoff (2026-09-09)
- Phase 31: Audit the v3 dataset, next-state action contract, split sizes, and task text metadata. Status: complete.
- Phase 32: Add an isolated v3 tactile-LoRA configuration initialized only from the published Tabero checkpoint, with W&B online logging and a 20,000-update schedule. Status: complete.
- Phase 33: Add a guarded user-operated launcher, normalization preparation path, CPU tests, and exact run/resume/log commands. Status: complete.
- Phase 34: Replace raw axis-angle subtraction with an invertible SO(3) relative-pose transform for the v3 profile and recompute statistics. Status: complete.
- Phase 35: Quantify compact-gap and deployment-rate mismatches, add explicit provenance/guards where possible, and distinguish unresolved data limitations. Status: complete.
- Phase 36: Diagnose the user-running v3 SO(3) 20k loss plateau from local logs and structured metrics without touching the process. Status: complete.
- Do not launch training, load pretrained parameters for inference, upload to W&B, or contact/actuate the robot; the user will run the final commands.

## Active investigation: v3 checkpoint versus inference-code contract (2026-09-10)
- Phase 41: Locate the v3 configuration and trace the exact training/inference SO(3) transforms in the current branch. Status: complete.
- Phase 42: Compare checkpoint/norm provenance with the runtime configuration contract and determine failure modes. Status: complete.
- Phase 43: Update the incident log and report deployment severity plus the required remediation/verification boundary. Status: complete.
- Static/source inspection and CPU contract tests only; do not load model weights, run inference/training, connect services, or actuate the robot.

## Active investigation: repeated K=2 real-robot grasp failures (2026-09-10)
- Phase 37: Validate all five JSONL files, termination modes, model/checkpoint/config provenance and actual command counts. Status: in progress.
- Phase 38: Quantify per-run raw predictions, saturated commands, tracking, cumulative motion, K=2 timing/truncation and cross-run consistency. Status: pending.
- Phase 39: Compare live behavior with the exact offline evaluation target/metric/split and identify supported causes versus hypotheses. Status: pending.
- Phase 40: Update the incident log and provide a prioritized remediation/verification sequence. Status: pending.
- Read-only log/dataset/evaluation analysis plus documentation only; do not connect to ROS/HTTP, run the model, train, or actuate hardware.

## Active review: external FR3 deployment branch (2026-09-07)
- Phase 25: Read the specified branch and diff deployment/model contracts against the trained workspace. Status: complete.
- Phase 26: Run branch CPU compatibility tests and inspect checkpoint normalization/shape provenance. Status: complete.
- Phase 27: Record the authoritative future deployment data contract and report blockers/limits. Status: complete.
- Review only; do not modify the external branch, load the model, use a GPU, connect ROS/HTTP, or actuate hardware.

## Active task: real FR3 deployment code (2026-09-03)
- Phase 22: Trace attached collector/converter/controller and current policy contracts. Status: complete.
- Phase 23: Implement portable observation, inference and guarded HTTP execution with deployment guide. Status: complete.
- Phase 24: Run CPU contract/failure-path tests and provide manual launch commands. Status: complete.
- Code and offline tests only in this task; do not connect to or actuate hardware or launch model workloads.

## Active analysis: tactile versus RGB+state offline evaluation (2026-09-07)
- Phase 25: Verify that evaluation datasets, seeds, horizons and sampling settings are comparable. Status: complete.
- Phase 26: Compare aggregate, per-episode, per-horizon and paired per-anchor errors; inspect six plots. Status: complete.
- Phase 27: Assess what the evidence supports about tactile input and state the experimental limitations. Status: complete.
- Read-only artifact analysis; preserve checkpoints, outputs and source files.

## Active analysis: new-prompt tactile checkpoint (2026-09-07)
- Phase 28: Verify new evaluation completeness, pairing and training provenance. Status: complete.
- Phase 29: Compare new prompt against old tactile and RGB+state results overall, per episode and per horizon; inspect plots. Status: complete.
- Phase 30: Judge whether the task-text change produced a meaningful improvement and document limits. Status: complete.
- Read-only artifact analysis; no training, inference or robot access.

## Active investigation: raw state/action boundaries (2026-09-03)
- Phase 19: Inspect attached raw events, evaluation arrays and conversion provenance. Status: complete.
- Phase 20: Quantify temporal alignment, startup/end discontinuities and rotation representation; trace training/evaluation transforms. Status: complete.
- Phase 21: Write evidence-backed diagnosis and concrete remediation guidance. Status: complete (available-evidence diagnosis; exact acquisition-versus-converter localization requires missing action events/source).
- Read-only data analysis; preserve datasets, model weights and existing code changes. No training or model inference needed.

## Goal
Continue the existing Tabero LoRA checkpoint on the A6000 workstation using RGB, state and marker-motion inputs, supervising seven action dimensions. Latest user instruction (2026-09-03) reserves training for manual operation because GPU availability changed. Do not launch training. Keep the failing experiment's 26 train / 3 validation split for diagnosis. No robot connection or actuation.

## Next Step
Finish the isolated v3 tactile 20k configuration and validate it on CPU. Then hand off one-line preparation/login/training/resume/log commands for user operation. Preserve raw data, prior runs, and real-robot deployment changes.

## Current Phase
Phase 35 complete: v3 audit/assets, SO(3) relative action transform, independent tactile 20k profile, W&B/manual launcher and CPU checks are ready. No training, pretrained inference, W&B login/upload, or robot operation was performed by the agent.

## Phases

### Phase 1: Confirm contracts and execution resources
- [x] Record source commit, current GPU availability and dependency setup.
- [x] Check checkpoint architecture, tactile reference-frame semantics and action transforms.
- **Status:** complete

### Phase 18: Separately trained no-tactile baseline
- [x] Inspect initialization, input transforms, checkpoint filtering and normalization compatibility.
- [x] Add opt-in RGB+state-only configuration, manual training launcher and matching offline evaluation.
- [x] Test on CPU and document exploratory versus matched retraining comparisons; preserve existing tactile path.
- **Status:** complete

### Phase 2: Implement minimal compatible training changes
- [x] Add explicit local dataset root and disjoint episode selectors.
- [x] Add action-only LoRA configuration, strict checkpoint restoration and validation metrics.
- [x] Add data audit, reproducible split and regression tests (suite authored, not executed).
- **Status:** complete

### Phase 3: Code verification
- [x] Stop environment provisioning; preserve partial environment/cache.
- [x] Run available static and lock-consistency checks without model execution.
- [x] Record regression suite unrun due to missing dependencies.
- **Status:** complete

### Phase 4: Real-robot interface and future smoke instructions
- [x] Document input contract, absolute 7D output, normalization and inverse transforms.
- [x] Document robot-side safety checks and axis-angle / startup-label risks.
- [x] Provide future opt-in audit, download, training and offline-inference commands; do not run them.
- **Status:** complete

### Phase 5: Review and handoff
- [x] Review diffs and test availability; record checks and limitations (no model metrics exist).
- [x] Clearly distinguish prepared code from proven pipeline reproduction, robot performance or tactile benefit.
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|---|---|
| Preserve raw dataset and existing GPU processes | No destructive cleanup, relabeling or resource reclamation authorized. |
| Keep checkpoint architecture, mask force/padding loss | Avoid reinitializing the learned output head or tactile modules. |
| Start with a bounded small-batch single-GPU smoke run | Shared workstation and small dataset; no long training requested. |
| Maintain planning files in repository | Explicit user request for planning-with-files. |
| User update: code only, no training; deployment target real FR3 | Supersedes earlier authorization to run a smoke training job. Stop provisioning and do not run model workloads. |
| Latest update: installation and unit tests permitted, training remains user-operated | Resume environment installation/testing only; deliver manual commands. |

### Phase 18: Diagnose FR3 shadow-deployment logs
- [x] Parse both JSONL runs and correlate them with the console log.
- [x] Separate ROS/DDS discovery warnings from policy-output and latency failures.
- [x] Quantify independent gripper, translation, rotation and workspace guard violations.
- **Status:** complete

### Phase 19: Add asynchronous action-chunk diagnostics
- [x] Preserve the observation state with every returned action chunk.
- [x] Log each complete action chunk once and link control ticks with a chunk ID.
- [x] Record action[0], selected action[k], current/observation states and physical distance decompositions.
- [x] Record checkpoint-local normalization metadata hashes and update the deployment guide.
- [x] Run deployment/service CPU tests and lint checks.
- **Status:** complete

### Phase 6: Finish isolated dependencies and execute non-training tests
- [x] Finish frozen dependency install (225 packages).
- [x] Run action-only / episode-selection / strict-restore regression tests on CPU (29 tests passed in selected suite).
- [x] Validate CLI argument parsing without calling train.main.
- **Status:** complete

### Phase 7: Manual training handoff
- [x] Record verified environment and test results.
- [x] Update instructions and provide ordered commands, artifact paths and caveats.
- **Status:** complete

### Phase 8: Tactile encoder LoRA adaptation
- [x] Inspect encoder, existing adapters and checkpoint restoration contracts.
- [x] Add zero-output-initialized tactile LoRA without changing pretrained base parameter paths.
- [x] Add new configuration; only new tactile LoRA leaves may be absent from source checkpoint.
- **Status:** complete

### Phase 9: Regression tests and manual handoff
- [x] Test initialization equivalence, gradients, trainable scope and strict restoration with new adapters (49 selected tests passed; 2 deselected).
- [x] Update real-robot runbook and exact user-operated commands.
- **Status:** complete

## Errors Encountered
| Error | Attempt | Resolution |
|---|---|---|
| New-prompt `metrics.jsonl` contains a NUL-filled line from the interrupted/resumed run | 1 | Parse valid JSON lines individually; verified 13 validation records, finite updates, duplicate pre-resume log records and finalized checkpoint 2999. |
| System SciPy binary incompatible with active NumPy during paired metric script | 1 | Re-run the read-only analysis with the project's pinned virtual-environment Python. |
| Prior sandbox could not see GPU or network | 1 | Use approved escalated commands for GPU/network checks and training. |
| Snap uv cannot run inside sandbox | 1 | Approved host invocation works; environment setup running there. |
| Optional ~/.local/bin/uv absent | 1 | Use /snap/bin/uv already installed. |
| uv sync --frozen: lockfile missing workspace member tabero-vtla | 1 | Existing lockfile is inconsistent with pyproject; resolve/update with uv sync and review resulting diff. |
| Hatch cannot locate package tabero_vtla | 1 | Project renamed but source remains src/openpi; declare wheel package explicitly. |
| PyAV 14.4.0 source build requires missing FFmpeg development libraries | 1 | Pin compatible 14.2.0 with verified CPython 3.11 Linux wheels; no system packages changed. |
| Ruff initial new-file style/import warnings | 1 | Correct imports/assertions and format new files. |
| Narrow Ruff CLI selection reenabled F722 jaxtyping false positives | 1 | Honor project ignore; existing unused pi0_fast import is baseline, not added here. |
| Tactile test import ordering after adding parameter-count check | 1 | Ruff sorted the new imports; full new-file lint and targeted core checks pass. |
| Baseline test constructed Observation before normal batching | 1 | Correct fixture to perform array conversion and batch insertion; production pipeline already does so. |
| New baseline test Ruff PT018 assertions | 1 | Split compound assertions; no runtime change. |
| Comparison-test Ruff PT011 broad raises assertion | 1 | Specify expected mismatch error messages. |
| Force config probe defaulted to read-only `~/.cache/openpi` | 1 | Re-run with the project cache `OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi`. |
| Real loader probe tried to create Hugging Face locks in read-only `/data` cache | 1 | Use an isolated `/tmp` Hugging Face cache and restrict the probe to episode 0. |
| Combined regression exposed a stale fake marker reference row | 1 | Correct the test fixture to the already-enforced fixed reference grid; production code was unchanged. |

### Phase 10: Investigate native crash in manual audit/batch checks
- [x] Isolate the failing command and native dependency using CPU-only subprocesses.
- [x] Reproduce with continued echo; verify script-file execution, helper imports and episode 0 reads without training.
- [x] Document the cause, evidence, launcher and scope of verification.
- **Status:** complete
- Full data preparation/batch checking remains user-operated; no system component was changed.
- Error found: interactive Bash continuation aborts in injected `/usr/lib/libpayload.so`; malformed ownership of fallback `loginName="-"` causes invalid free. Script-file execution passed; permanent repair belongs to the host component maintainer.

### Phase 11: Post-smoke checks and multi-GPU guidance
- [x] Inspect data coverage, sharding, checkpoint loading and LR/evaluation semantics.
- [x] Validate proposed CLI arguments without constructing a model or entering training.
- [x] Prepare single-line manual commands, evaluation limits and data-risk caveats.
- **Status:** complete

### Phase 12: Numerical-failure diagnosis and protection
- [x] Confirm latest training authority, available GPU/disk resources and original failure evidence.
- [x] Add per-step finite checks and preserve pre-update healthy state on failure; 40 selected CPU regression tests pass.
- NaN reproduction/localization deferred to user-operated training; root cause is not established.
- **Status:** complete (protection only; root-cause reproduction deferred)

### Phase 13: Manual multi-GPU handoff (supersedes agent execution)
- [x] Adapt launcher to explicitly selected GPU count and verify without training (8 dry-run/invalid-input cases).
- [x] Document one-line launch and log-viewing commands; explain checkpoints and unresolved NaN risk.
- **Status:** complete

### Phase 14: Training result analysis
- [x] User completed real_fr3_recovery_20260903_112020; verified 3000 updates and finalized checkpoint 2999.
- [x] Reported validation improvement 1.0688 -> 0.3350, minimum 0.2923 at unsaved update 2250; no real-robot success claim.
- **Status:** complete

### Phase 15: Offline action evaluation
- [x] Implement raw-observation inference, strict checkpoint/assets handling, physical metrics and episode-end masking.
- [x] Save summary, per-frame/chunk predictions and trajectory/error plots; no robot connectivity.
- [x] 59 selected CPU tests pass; CLI/help/shell/lint checks pass. Approved CPU-only boundary reads pass on real validation data.
- [x] Document one-line user-operated inference commands, outputs and limitations.
- **Status:** complete

### Phase 16: Interpret completed offline evaluation
- [x] Locate completed artifacts and verify evaluation coverage/settings.
- [x] Inspect all three figures and quantify episode-level, horizon and baseline results.
- [x] Prepare Chinese explanation of plotting inputs, metric definitions, observed behavior and limits.
- **Status:** complete

### Phase 17: Tactile ablation feasibility
- [x] Check raw tactile coordinates, encoder handling and existing evaluation controls.
- [x] Define paired input-ablation comparisons, metrics and interpretation limits.
- [x] Report necessary code changes; do not modify training/evaluation code or run a model in this feasibility turn.
- **Status:** complete

### Phase 19: Compare v3 checkpoints 4000 and 20000
- [x] Verify paired evaluation provenance, arrays and completion status.
- [x] Compare first action, full valid chunk, per-episode, per-horizon, safety diagnostics, long tails and latency.
- [x] Reconcile physical metrics with the training validation-loss curve and update the incident log.
- **Status:** complete

### Phase 20: Synchronous FR3 deployment with saturating limits
- [x] Convert the client from asynchronous chunk selection to blocking inference and action[0].
- [x] Saturate finite prediction bounds while retaining stop conditions for measured/system faults.
- [x] Add raw/bounded/limited diagnostics and exact boundary/controller regression tests.
- [x] Complete final tests and review the intended commit scope.
- [x] Commit and push the deployment change (`e5b4f21`).
- **Status:** complete

### Phase 21: Deploy v3 checkpoint 20000
- [x] Diagnose v1 config versus v3 checkpoint asset mismatch.
- [x] Verify checkpoint 20000 asset_id, required normalization keys, and hash.
- [x] Commit v3 config, SO(3) transforms, actionable server error, tests, and documentation.
- [x] Push commit `badb9ef` to `fix/fr3-tactile-shadow-deploy`.
- **Status:** complete

### Phase 22: Diagnose repeated K=2 real-robot grasp failures
- [x] Validate the five JSONL logs, metadata, completeness, and comparability.
- [x] Quantify physical motion, command acceptance, tracking, saturation, K=2 use, timing, terminal pose, and gripper behavior.
- [x] Reconcile the live traces with v3 demonstrations and the saved offline-evaluation artifact.
- [x] Update the incident overview and detailed diagnosis.
- [x] Preserve the diagnosis locally without committing or pushing, per the user's latest instruction.
- **Status:** complete

### Phase 23: Inference-host rewrite handoff
- [x] Inventory the current model, sensor, transport, timing, action, safety, and logging contracts from code/config.
- [x] Document confirmed deployment failures and clearly label remaining hypotheses.
- [x] Write a Chinese handoff with rewrite architecture, acceptance tests, staged rollout, and operator commands.
- [x] Update the incident overview/details and verify Markdown structure, links, numeric anchors, and whitespace.
- [x] Preserve the handoff locally without committing or pushing, per the user's latest instruction.
- **Status:** complete

### Phase 24: Whiteboard 12000-step offline evaluation
- [x] Inventory the four user-selected checkpoints and bind each to its matching config and label source.
- [x] Extend recorded-observation evaluation to 7D action plus 6D K-frame wrist-wrench outputs.
- [x] Add a sequential four-checkpoint runner and comparison CSV/JSON generator.
- [x] Run CPU contract checks on validation episodes 1, 7 and 17 for all four checkpoints.
- [x] Run full GPU inference for the four available 12000-step checkpoints.
- **Status:** complete

### Phase 25: Mixed full-finetune 20000-step offline evaluation
- [x] Verify finalized next-state and sent-command checkpoint assets at step 20000.
- [x] Run CPU contract checks and full 1001-anchor GPU evaluations for both label sources.
- [x] Compare mixed full-finetune next-state 20000 against LoRA32 next-state 12000 under the same protocol.
- [x] Record action, wrench, horizon, episode, baseline and output-bound findings in the incident log.
- **Status:** complete

### Phase 26: Aligned next-state prediction overlay
- [x] Verify Full-FT 20k and LoRA32 12k arrays use identical episodes, frames, targets, states and valid masks.
- [x] Overlay both models and recorded first actions for episodes 1, 7 and 17.
- [x] Include per-frame position and SO(3) rotation errors and export PNG/PDF artifacts.
- [x] Overlay recorded and predicted K-frame wrench components plus force/torque errors.
- **Status:** complete
