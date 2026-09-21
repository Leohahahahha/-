# Progress Log

## 2026-09-15: paired whiteboard action-definition training
- Started file-planned implementation for two isolated 20k tactile-LoRA experiments and a fail-fast sequential launcher. No training, model inference, GPU allocation, W&B upload, service connection, or robot action has been started.
- Recovered the existing plan and dirty-worktree state. Next: audit both datasets and reuse only the established v3 hyperparameters, not its dataset-specific assets.
- Read both metadata contracts. They are paired 39-episode/12,126-frame whiteboard exports with identical observation schema and task text but distinct next-state versus sent-command action semantics. Noted the new 640-pixel front width and 98 compacted candidate steps for validation.
- Inspected the v3 config/preparation/launcher implementation and isolated the settings that should remain identical. A first paired scan attempted to materialize both complete tactile columns and produced no result within 30 seconds; no files changed. Next attempt will compare lightweight columns only, while formal per-dataset audit handles tactile/video integrity.
- Lightweight paired state/action audit completed in 1.5 seconds. Inputs and episode lengths align exactly and next-state labels satisfy their contract. Sent-command labels contain much larger target/state offsets and 161 adjacent position jumps above 50 mm; these will be retained as the requested action definition but explicitly audited, not silently corrected.
- Updated split requirement: draw exactly three episode IDs once with a fixed seed and share that split across both experiments.
- Fixed the random split reproducibly with seed 42: validation IDs `(1,7,17)`, 1,001 frames; train has 11,125 frames. This split is now the contract for both configs.
- Added the two whiteboard configs with independent dataset roots/assets and identical model/training settings, plus metadata checks that bind each config to its declared action source and temporal offset. No normalization or training was run.
- Added the fail-fast paired launcher with automatic CPU preparation, separate output namespaces, ordered GPU training, W&B grouping, stage status, and safe resume behavior. Corrected output-root propagation for both preparation and training before testing.
- Added paired-config and launcher tests, formatted touched Python, verified shell syntax, and passed 28 focused CPU tests in 38.22 seconds. Formal data preparation/full video decode and both GPU runs remain user-operated.
- Localized sent-command discontinuities: every episode is affected, and an episode-1 example toggles commanded z between the 0.6 m ceiling and the measured-height region. Kept labels intact as requested and will document the confounding risk in the incident log/runbook.
- Actual-path dry-run passed. Full Ruff surfaced known baseline config warnings and one new pre-JAX import-order warning; added a scoped file directive for the intentional import timing instead of changing unrelated baseline code.
- Reviewed the full diff after tests and caught excessive formatter churn in `config.py`. Preparing an atomic reverse patch for that file followed by the minimal configuration insertion; no user-owned dirty file will be reverted.
- Standard reverse-diff conversion was rejected before editing. Logged the failure and switched to an atomic full-file update through `apply_patch`; baseline is safe because `config.py` was clean at task start.
- Restored `config.py` to its clean task-start content atomically and reapplied only the paired config block. Reverification passes: 30 CPU tests, both Ruff scopes, Bash syntax and whitespace; the config diff is now isolated.
- Added the Chinese runbook and README link, plus incident 30 and overview entry with the sent-command confounder and explicit validation limits. Final handoff uses only single-line commands, tmux, live GPU selection, pair/status logs and `--resume`; both 20k trainings remain unstarted.

## 2026-09-10: v3 checkpoint versus inference-code contract
- Started a static deployment-contract investigation after the inference host reportedly lacked the v3 config and used component-wise rotvec addition. No model, GPU, network service, ROS connection, training process, or robot was touched.
- Read the repository instructions and recovered the file-based plan. Next: locate config registration and trace forward/inverse action transforms plus runtime selection.
- Confirmed the current A6000 branch has the v3 config and paired SO(3) transforms. Traced policy construction: checkpoint-local stats are used, then inference output is unnormalized and passed through the config-selected inverse transform. This proves code/config provenance is independent from matching parameter/stat hashes.
- Quantified legacy-inverse damage on all v3 labels without loading a model: one-step p95/max orientation error is 0.410/1.445 degrees; valid horizon-49 p95/max rises to 4.348/6.672 degrees. Position and gripper reconstruction are not affected by this particular difference.
- Located the implementation commit (`badb9ef`) and confirmed current HEAD includes it. Existing incident history says one inference-host checkout previously passed v3 checkpoint loading and shadow, so the newly reported missing config points to checkout/path drift rather than bad checkpoint bytes.
- Ran a no-model CPU config probe and 22 focused tests; the current checkout selects the correct SO(3) inverse and rejects an unknown config name. No inference-host runtime was accessed, so actual host checkout/process provenance remains unverified.
- Compared documentation with the client handshake. Found that config/asset/norm provenance is logged but not enforced, so a customized legacy server can evade the current client guard. This raises the issue from a simple startup/version problem to a possible silent orientation-semantic mismatch.
- Verified the exact predecessor of implementation commit `badb9ef`: it lacks both the v3 config and SO(3) transform classes and uses only component-wise delta/absolute transforms. Prepared to document this as a version-drift recurrence without overwriting the prior incident resolution.
- Recorded current source hashes and Git ancestry. A combined documentation patch failed validation atomically due to malformed patch structure; retried as smaller context-verified patches, with no partial change from the failed attempt.
- Added incident 29 and its overview row with the two distinct failure modes, quantitative SO(3) impact, supported version-drift hypothesis, proposed fail-closed fix, and explicit no-host/no-model/no-robot verification boundary. `git diff --check` passes; Phase 41-43 complete.

## 2026-09-09: live v3 loss diagnosis
- Read the attached excerpt and recovered the project plan. Identified the active log as `v3_touch_so3_20k_run1.log`.
- Began read-only curve/optimizer diagnosis; no signal, checkpoint, GPU allocation, training state, W&B run, or source code has been changed.
- Parsed all structured metrics through update 2000 and established a continuing downward train trend. Validation improved deterministically from 2.80060 at initialization to 0.333902 at update 1000 and 0.313991 at update 2000.
- Verified from source that each update resamples Gaussian flow noise and Beta diffusion time, displayed training rows average only ten updates, and validation reuses fixed per-batch RNGs. With global batch 4 and LR still near its `1e-5` peak, the observed 0.16-0.30 short-window variation is expected and is not a plateau.
- Confirmed all finite/update guards remain healthy and tactile LoRA has 779,008 trainable parameters. No training state, GPU process, checkpoint, source code, or W&B run was changed during this diagnosis.

## 2026-09-09: v3 tactile LoRA 20k preparation
- Read the planning-with-files skill and recovered the existing project plan/history.
- Inspected current training/W&B behavior and the v3 metadata contract. Confirmed the requested long run needs new train-only normalization assets rather than reusing v1 assets.
- Began an isolated v3 configuration and launcher design; no training, pretrained inference, W&B login/upload, dataset mutation, or robot operation has occurred.
- Added `pi0_lora_tabero_v3_touch_20k`, exact completed-update checkpoint numbering, W&B scalar/image-upload separation, explicit W&B finish, and the guarded new/resume launcher.
- Generated the v3 audit and train-only normalization assets under `/data/yanghaojun/outputs`; full video decoding, next-state labels, split, task indices and corrected tactile rolling-history checks passed.
- Real CPU train/validation batch transformation passed. New v3 unit suite passed 5/5; a wider selected suite had 44 passes and 2 pre-existing branch failures (stale tactile reference fixture and absent debug config), neither caused by the v3 changes.
- Added the one-line command runbook and README link. Actual W&B authentication/upload and the 20k GPU training run remain user-operated.
- User requested addressing the remaining axis-angle, compact-timing, and deployment-rate mismatches before training. Reopened implementation for a v3-only SO(3) relative action transform; existing configs must retain legacy behavior.
- Added and tested invertible `RelativePoseActions` / `AbsolutePoseActions`, enabled only for the v3 profile, and regenerated the v3 normalization/audit assets.
- Corrected launcher normalization hash and documented compact timing / deployment-rate limitations. Real transformed train and validation batches pass after the SO(3) change.
- Final focused suite passed 43 tests; a wider run passed 87 tests with only 3 known stale tactile-reference fixtures from the branch. Lint, formatting for new/touched focused files, shell syntax, and whitespace checks pass.

## 2026-09-07: New-prompt tactile checkpoint 2000 evaluation
- Verified the user-completed 693-anchor output and inspected all three episode plots.
- Ran exact paired physical-unit comparisons against same-run 2999, old tactile 2999 and RGB+state 2999.
- Found a rotation-outlier benefit at 2000 but broader regressions in position and chunk rotation; validation loss alone selected the wrong general-purpose checkpoint.

## 2026-09-07: Per-run checkpoint selection
- Enumerated finalized saved checkpoints and matched them to validation records for all three runs.
- Identified per-run candidates and separated validation-loss selection from completed physical-unit offline evaluation evidence.

## 2026-09-07: Review user-modified FR3 deployment branch
- Exact hashes match for pi0.py, pi0_config.py, tactile_encoder.py, policy_config.py and weight_loaders.py, so branch metadata/validation additions do not change checkpoint parameter structure. IPC and conversion JSON hashes also match exactly.
- Checkpoint 2999 is finalized and contains params/assets/train_state. Norm stats provide state/actions [7] and tactile_prefix [396]. Branch config resolves action_horizon=50, action_dim=32, supervised_action_dim=7, tactile_prefix_dim_in=3564, history=8, TCN LoRA rank=16; server tactile contract validation passes.
- Audited all 6,655 converted dataset frames: every marker tensor is [9,198,2] and every frame-0 reference equals the branch's fixed left-then-right grid.
- Branch-only deployment/server suite passes 34 tests. Expanded selected suite reports 53 passed / 2 failed: both failures use an obsolete all-zero tactile fixture in tabero_offline_test.make_sample, now rejected by the deliberate reference-grid check. This is a test-fixture defect, not a live observation or checkpoint incompatibility.
- One inspection initially failed because the JSON has a norm_stats wrapper; corrected parser. Another used system Python without PyArrow; reran with the verified environment. One config probe tried the default unwritable cache and one imported the local scripts package from the wrong cwd; reran with OPENPI_DATA_HOME and branch cwd.
- Read GitHub branch fix/fr3-tactile-shadow-deploy at commit 24e6370 into /tmp for read-only review after connector file inspection and approved network clone.
- Branch changes the ZED input to reliable compressed transport, adds per-stream QoS, URL overrides/validation, tactile summaries, and exact [9,198,2] float32/reference/history metadata checks on client and server.
- Parameter-bearing model architecture/loading paths appear unchanged from the local trained workspace; verify exact hashes, checkpoint assets, and CPU tests next.

## 2026-09-07: New-prompt tactile evaluation analysis
- Located and validated the complete new-prompt evaluation artifact set.
- Confirmed 693-anchor evaluation completed with finite results. Pairing/provenance and detailed comparisons are in progress.
- Verified exact old/new prompt metadata and closely matched old/new tactile training configurations. Handled the interrupted/resumed log by ignoring one NUL record and duplicate abandoned-branch metric rows.
- Completed paired first-action, valid-chunk, per-episode, per-horizon and diagnostic comparisons against old tactile and RGB+state checkpoints.
- Generated a three-model overlay and official paired comparison artifacts. Final finding: consistent orientation gain, slight immediate position gain, but worse gripper and long-horizon position; mixed rather than comprehensive improvement.

## 2026-09-07: Tactile ablation result analysis
- Began read-only paired analysis of completed tactile and RGB+state evaluation artifacts.
- Located complete artifact sets for both evaluations; no new model inference or training will be run.
- Verified matched evaluation selection and identical state/action normalization values. Identified differing initialization as a causal-comparison limitation.
- Initial paired script failed on the system SciPy/NumPy binary mismatch; switching to the pinned project environment.
- Re-ran with the pinned environment; paired arrays and metrics verified. Inspected all six source plots and generated a direct overlay comparison plot.
- Ran the repository's official CPU comparison script successfully; wrote comparison JSON/text to the visualization workspace. Analysis concludes no demonstrated tactile benefit in the available offline results, with training-initialization and evaluation limitations recorded.

## 2026-09-03: Real robot deployment implementation
- Final verification: 50 CPU tests passed in 8.11s; new authored Python files pass Ruff/format checks, byte compilation and git diff --check. IPC and training conversion metadata copies match source SHA256 exactly. Final enable handling latches received release/heartbeat gaps after arming.
- Task complete as code/manual handoff. No model inference/training, live ROS, HTTP robot access, external deployment or hardware actuation occurred. Existing training/data files and checkpoints were preserved.
- First controller suite had one test-only floating-point timestamp comparison failure (0.19999999999998863 versus 0.2); changed the assertion to approximate comparison. 26 client tests then passed.
- Server inspection found a wrong tactile dimension attribute before runtime; corrected modality detection to the actual tactile_streams field. Added moved-checkpoint tests with real data configs/statistics and mocked model creation for both tactile and no-tactile models.
- 50 CPU tests passed in 7.41s, including existing policy inverse-transform coverage. CLI help works without ROS/model initialization. Exact parity with functions extracted from the supplied converter passed for 12 tactile histories and quaternion canonicalization.
- Completed Chinese host setup, model/client launch, shadow, enable heartbeat, checkpoint/tokenizer portability, and HTTP hold limitations guide. Preserved IPC attachment and conversion metadata byte-for-byte.
- Implemented CPU observation/action contracts, exact IPC decoder integration, ROS image acquisition/history sampler, bounded HTTP/WebSocket clients, async inference and guarded 10 Hz execution loop, and checkpoint-local policy server. No endpoints contacted.
- Initial Ruff fixes addressed import order/format only. ROS docs fetch was denied and one Franka docs URL returned 404; source-specific implementation uses attached code and installed library interfaces.
- User supplied six collection/control/conversion files and requests deployment code for another host. Restored planning context and preserved existing dirty worktree.
- Attached online adapter depends on unavailable observation_builder/image_utils; inspect actual converter before choosing live observation path.

## 2026-09-03: Raw state/action boundary investigation
- User supplied original robot_state_events.jsonl and asked whether the large boundary offsets originate in collection or model code.
- Confirmed raw attachment and episode_000004.png are readable. No AGENTS.md found in project/skills search or checked ancestor locations.
- Existing workspace contains substantial earlier changes; preserve them. Will analyze raw data and saved predictions on CPU.
- CPU-only NumPy/PyArrow probe confirms NPZ targets/states equal Parquet and episode-4 raw pose/finger mapping matches to float32 precision. Previous-episode final target held through frames 0..10 causes ~423 mm offset; end behavior is distinct.
- Read current action query and forward/inverse transforms; no next-state label construction occurs in model loading. Asked asynchronously for raw action events and collection/conversion scripts.
- One notes patch failed because findings heading was `Findings & Decisions`; retried with correct context. Missing source raw directory means converter/acquisition code unavailable locally.
- Finished reproducible read-only audit in diagnostics/state_action_20260903; all 693 saved chunks (34,650 slots including padding) match per-episode Parquet labels and valid masks. Raw episode-4 full pose/gripper mapping verified.
- Actual production transform classes with checkpoint stats round-trip all saved targets with max XYZ error 0.0000152 mm. No model constructed or inference performed.
- 28/29 episodes have first-label position gap >50 mm; 14/29 reuse all seven previous-last values exactly. Found 638 state/action rotvec branch-mismatch frames despite physical rotation <10 degrees.
- Saved summary JSON, episode-4 frame CSV, inspected boundary plot and Chinese report distinguishing confirmed pre-training target issue, remaining acquisition/conversion uncertainty, and real model errors near episode end.

## 2026-09-03: Separately trained RGB+state baseline
- User prefers retraining without tactile input rather than inference-time substitution. Preparing independent opt-in configuration and manual commands; training/inference remain user-operated.
- Need preserve all 26/3 split, action representations and existing datasets/results; verify source initialization because prior successful tactile run warm-started a local 100-update tactile smoke checkpoint.
- Added opt-in RGB+state and matched tactile-retrain profiles, both initialized from published Tabero; fixed batch4/3000 updates. Baseline drops only the explicitly enumerated removed tactile encoder checkpoint leaves and rejects shared-weight mismatches.
- Added strict real-FR3 no-tactile adapter and optional offline modality handling (default tactile unchanged). Baseline uses no tactile observations/encoder/tokens; normalization reuses verified train-only shared stats.
- Added manual launcher with 1/2/4-GPU divisibility checks, fixed batch, published-only source, normalization hash guard and read-only dry-run. No training or inference invoked.
- First selected CPU suite: 70 passed, 1 test-fixture failure from calling Observation before the normal scalar-to-array batching step. Correcting fixture (not weakening production type checks). Full model parameter shapes traced abstractly on CPU; baseline has no tactile leaves and all shared parameter shapes match.
- Initial new-test lint found four compound assertions; split them. Existing model/evaluation paths otherwise passed the selected regression run.
- Added CPU-only comparison CLI and synthetic alignment/sign/unit/padding tests, plus full single-line training/evaluation/logging/comparison guide in docs/tabero_rgb_state_baseline.md.
- Expanded selected CPU suite passed 83 tests (14 existing dependency warnings) in19.25s. Train CLI --help passes without entering main. Additional modality-parametrized strict-policy factory checks are being added before final handoff.
- Comparison-test lint requested narrower raises matching; added explicit error-message pattern. Verified existing train-only normalization provenance and matching full SHA256 against checkpoint assets.
- Final CPU suite: 93 passed,14 existing dependency deprecation warnings,29.07s. Covers both launch variants/GPU counts, actual training CLI argument parsing without main, no-tactile abstract shapes, strict restore, absent/invalid sensor-field isolation, policy inverse transforms and paired result comparisons.
- New Python files and offline changes pass Ruff/format checks; affected core files pass targeted Ruff checks; shell syntax and git diff --check pass. Added README link to the dedicated guide. No raw data, original checkpoints or existing offline results changed; no actual baseline training, pretrained inference, GPU use or robot connection occurred.

## 2026-09-03: Tactile ablation feasibility
- User asks whether inference without tactile information can be compared with the completed tactile-conditioned evaluation.
- Restored planning files and dirty-worktree context. Will inspect current implementation and recommend a controlled comparison; no code changes, training, pretrained inference or robot communication in this feasibility turn.
- No AGENTS.md found by the home-directory file search (rg exit 1 means no matches).
- Confirmed CLI requires tactile inputs and offers no ablation switch. Conversion metadata provides a fixed sampled pixel reference grid; TCN does not subtract it, so raw zeros are not zero shear.
- Verified completed baseline manifest and independent per-frame sampling noise. Defined reference-grid input replacement versus true branch masking as distinct interventions and documented comparison/causal limits.
- Only planning notes changed. No evaluation/training source edits, unit tests, model loads, data/checkpoint changes or robot actions performed. Implementation can follow user confirmation.

## 2026-09-03: Completed offline evaluation interpretation
- User requests analysis of three completed episode evaluations and explanation of how the figures were generated.
- Read evaluator and plotting source; locating user-produced artifacts. No training or pretrained inference is needed.
- Located offline_2999_20260903_125340, verified complete manifest/all source hashes, read summaries and visually inspected all3 plots. Computed supplemental baseline/startup/rotation/gripper statistics from saved NPZ only.
- Main conclusion: broad translation trends and whole-chunk positional improvement exist, but first-action errors do not beat the state-hold baseline consistently, severe physical rotation errors persist, gripper overshoot exists, and measured145 ms inference exceeds a100 ms synchronous control interval. Not ready for unguarded robot execution.
- Recorded panel meanings, precise units, target/state distinction and open-loop-versus-rollout limitation. No training/inference rerun, artifact rewrite or control changes.

## 2026-09-03: Offline evaluation implementation
- User approved adding an offline evaluation script and CPU unit tests; actual model inference remains user-operated.
- Read planning-with-files and restored project state. Existing policy can load checkpoint assets and invert normalization/delta transforms; no offline evaluation CLI exists yet.
- Will preserve original observations/labels, mask episode-end targets, report rotation geodesic errors rather than raw axis-angle component errors, and avoid robot connections.
- First multi-file patch failed atomically due to a policy_config docstring context mismatch; inspected the actual line and applied separate patches. No user changes were reverted.
- Added inference-independent metric/evaluation module with fixed per-anchor noise, label-free copied observations, padding cross-checks, state-hold baseline, streamed CSV, NPZ/JSON and headless per-episode plots. Added optional strict JAX restore to policy factory.
- Added CLI and single-GPU shell launcher: checkpoint-local assets only, local/offline resource mode, finalized checkpoint check, validation-only episode selection and fresh output directory enforcement. check-only mode reads boundary samples on CPU without loading a model.
- First offline CPU suite: 17 passed, 1 failed. Installed SciPy rejects a read-only broadcast view in the one-anchor baseline rotation calculation; fixed with explicit writable copies. Initial Ruff reported test style/private-alias issues, now corrected.
- Expanded CPU regression suite passed 58 tests. Real-data check-only uncovered DataConfig's train/validation overlap guard when changing the dataset episodes selector; corrected by validating held-out membership first, then clearing validation_episodes on the dataset-only copy. Added a regression test; original training config stays unchanged.
- Corrected check-only then reached HF dataset loading, but sandbox denied its normal cache lock write under /data. Requested an approved CPU-only host retry; model loading remains disabled.
- Added full manual offline runbook: single GPU, full 693-frame validation, exact output/log locations, optional bounded sampling, artifact/metric definitions, failure behavior and real-robot limitations.
- Approved host check-only completed successfully: episode 4 frames 0/221, episode 14 frames 0/221, episode 24 frames 0/248; shapes, raw observations and 50-to-1 valid action masks checked. Report at /tmp/tabero-offline-check.sVimMR/report_host. Only normal HF cache locks were needed; source data/checkpoint unchanged.
- Final selected CPU suite: 59 passed, 6 existing dependency deprecation warnings, 11.79 seconds (19 new offline cases plus 40 prior regressions). Plot output, strict factory wiring, inverse transforms, noise pairing, label isolation and failure artifacts are covered using synthetic predictions, not pretrained inference.
- CLI --help, shell syntax, new/core policy Ruff checks, formatting and git diff --check passed. No trained model was loaded, no GPU inference/training or robot communication performed. Actual inference performance/results are unverified until the user runs the launcher.

## Session continuation: agent authorized to train
- Read planning-with-files and recovered project files. Latest user overrides prior manual-only training restriction; agent will diagnose, repair, train and analyze without robot execution.
- Fresh GPU check approved: all four A6000 cards idle; sufficient /data space and RAM. Preserving all existing edits and checkpoints; scope initially 26 training / 3 validation episodes for controlled diagnosis.
- Implemented per-update input/loss/gradient/update/optimizer/candidate-parameter finite checks and conditional preservation of old state. Invalid updates stop immediately, dump the exact transformed batch, identify first bad gradient leaf, and save the healthy state. They are never silently skipped.
- Initial targeted CPU suite passed 39 tests. Added a real train_step miniature rejection test and a bounded four-GPU launcher with more frequent healthy checkpoint saves; raw data and source checkpoint unchanged.

## Session continuation: post-smoke and multi-GPU instructions
- User requests next checks and multi-GPU operating instructions, not automatic training. Read sharding, optimizer schedule, data-loader epoch/drop-last behavior and policy loading APIs; no training code changed.
- Recorded user's successful run and committed checkpoint, superseding historical "no training yet" status for that user-run experiment. Agent still has not launched training.
- Confirmed CLI --help exposes fsdp-devices, batch-size, LR schedule overrides and weight-loader.params-path. Preparing argument-only parsing verification and manual commands.
- Proposed 4-GPU/global-batch-8/1000-step weight-only warm-start command parsed successfully on CPU, including LR decay 1000/warmup 50, 26/3 split, strict loader and no resume. No model, optimizer or GPU workload was instantiated. Four-GPU runtime/performance remain untested.
- Prepared checkpoint reload-only command and staged 4-GPU 20-step test / conditional 1000-step trial guidance. Offline action-quality evaluation is not yet implemented; user must not interpret reload as physical-policy validation. Only planning files updated.

## Session continuation: native crash investigation
- User reports `free(): invalid pointer` and VS Code terminal exit when running preparation followed by batch inspection.
- Read scripts and existing project context. Audit output directory is absent; this suggests preparation did not finish, but the original failing process is not yet identified.
- Batch checker already forces `JAX_PLATFORMS=cpu` before importing JAX. No training is needed for diagnosis.
- Both helper `--help` commands succeeded. Episode 0 read-only CPU Parquet/video check succeeded for all 201 frames and expected tensor shapes.
- Read existing Bash core and inspected native stack/library metadata. Identified invalid free of static fallback login-name string in UHS-related `/usr/lib/libpayload.so`, `record_cmdline_core`, source line 677.
- Reproduced failure using harmless continued echo in isolated interactive PTY: single line exit 0; continued line and bracketed paste both SIGABRT. Identical command in a script file passed, establishing the supported workaround.
- Added `scripts/run_tabero_checks.sh`, detailed `docs/tabero_terminal_crash.md`, and runbook pointer. Launcher syntax/help and whitespace checks passed.
- Diagnosis complete. No training, full audit, tokenizer download, model inference or actual transformed batch was run. No system Bash/library/hook changes were made; underlying host component repair remains the administrator/maintainer's responsibility.

## Session continuation: tactile encoder LoRA
- Read planning-with-files and restored project planning state. User approved the proposed new tactile LoRA variant, not training execution.
- Preserving prior modifications and baseline; will add opt-in zero-output adapters, selective strict restoration, CPU tests and manual commands.
- Inspected NNX TCN, training initialization/shape placeholders, strict checkpoint loader and inference reconstruction. Implementation will retain base kernel/bias paths and add rank-16 adapters to eight tactile linear maps; original configs remain rank zero.
- Implemented opt-in NNX LoRA linear, prefix rank/alpha config plumbing, new sensor-adaptation config and strict allowlist for new adapters only. No changes to raw sensor fields, action labels or existing baseline.
- First CPU run: tactile_encoder_test.py + tabero_smoke_test.py, 33 passed in 19.91s (one dependency deprecation warning). Small synthetic modules only; no optimizer, pretrained-model inference or training loop.
- Full Ruff checks on new tests and targeted core checks pass. Writing manual runbook with config-specific norm-stat preparation and matching inference config.
- Added startup tactile-adapter count logging and an abstract shape-count test (779,008 new trainable parameters / 65,239,040 frozen tactile parameters). No full model allocation.
- Expanded selected CPU suite: 49 passed, 2 deselected, 13 dependency deprecation warnings in 22.52s. No failed tests. Import-order lint after adding count test was auto-corrected; new-file and targeted core Ruff checks pass.
- New train configuration `--help` exits 0 without entering training. Updated README/runbook with both configs, strict initialization expectations, config-specific normalization, GPU selection and same-config inference requirement.
- Planning phases 8/9 complete. No installation, checkpoint download, real-data preparation, full-policy inference, training, robot connection, dataset modifications or GPU work was performed in this continuation.

## Session: 2026-09-02

### Phase 1: Discovery
- Status: in_progress
- Read the full planning-with-files skill and its three planning templates.
- No previous planning files or tracked code changes were present.
- Created task_plan.md, findings.md and progress.md before implementation.
- Prior conversation's read-only evidence was recorded separately from unverified user handoff risks.

## Test Results
No implementation tests or training have run yet.

## Error Log
No new errors in this implementation session yet.

## 2026-09-08: attached shadow-run analysis
- Read-only parsed `shadow_006.jsonl`, `shadow_007.jsonl` and `a.txt`; no robot commands or model inference were run.
- Confirmed the dominant failure is unsafe checkpoint-2000 action output, with small gripper-bound overshoot masking many large pose errors. ROS/DDS warnings did not prevent observations.
- Identified a runtime-age enforcement gap for already accepted chunks; do not enable `--execute` from these results.

## 2026-09-08: action-chunk diagnostic logging
- Modified the tracked `fix/fr3-tactile-shadow-deploy` baseline locally; changes are not pushed.
- Added one-time full chunk records, per-tick action[0]/action[k] distance records and policy normalization metadata records.
- Verification: Ruff passed; `test_deployment.py` plus `serve_tabero_test.py` passed 35/35 on CPU; `git diff --check` passed.
- Parsed the first user-run enriched log and verified its normalization hash against local checkpoint assets.
- Added per-stream timestamp skew/age error details without changing timing thresholds. Updated verification: 36/36 selected CPU tests pass.
- Parsed duplicate complete 30-second logs and separated async amplification from intrinsic action[0] outliers. Added explicit missing-stream startup diagnostics; selected CPU suite now passes 37/37.
- Matched the shadow target-to-state metric to checkpoint-2000 offline offsets 2/3. Confirmed a remaining roughly 1.6--2x mean live distance increase, old/new prompt mismatch, and at least 38.7-degree live-orientation shift from all three held-out trajectories.

## 2026-09-09: v3 checkpoint 4000 versus 20000
- Read-only verified both complete evaluation artifact sets and exact pairing of all 690 anchors, saved targets/states/masks and evaluation settings.
- Computed paired first/chunk, per-episode, per-horizon, guard and long-tail comparisons and inspected episode 4/24 plots. Checkpoint 20000 wins all overall mean physical metrics, while episode-24 long-horizon position and one deep-horizon rotation outlier remain concerns.
- Read the full training validation curve: 4000 has minimum proxy loss, which conflicts with the fixed-seed physical-action ordering. Recommended 20000 for guarded shadow and multi-seed/intermediate-checkpoint evaluation before stronger claims.
- Updated the required interview incident overview and added a complete checkpoint-selection incident entry. No model inference, training, shadow or robot action was run.
- Confirmed the tactile startup KeyError was caused by the bridge not being launched; user asked to close that line of investigation.
- Audited all 6655 labels: stale episode prefixes are widespread, while post-prefix target lead remains nonzero. Confirmed the exporter copies raw actions and does not introduce the stale prefix.
- Updated the FR3 deployment prompt to the exact task string from dataset metadata. JSON parsing, whitespace validation, and the selected deployment/service CPU suite pass (37 passed).
- Compared the last shadow start against all 29 training episodes: no training state is within 38.65 degrees. The next live step is a correctly aligned, corrected-prompt shadow run; execute remains disabled.
- Audited `/data/yanghaojun/datasets/tabero_lerobot_compact_v3` against v1. Verified every v3 state/action row, quantified position/rotation/gripper continuity, and saved a reproducible JSON report.
- Confirmed v3 removes all >50 mm one-step position labels and cross-episode stale action reuse. Identified 114 remaining axis-angle branch crossings and a deployment-rate mismatch affecting 41.1% of v3 one-step translations.
- Created `docs/tabero_interview_incident_log.md` with dated incident names, evidence, status, fixes, validation limits and interview framing; added a repository instruction to update it for future issues.
- Initial combined regression run produced 52 passes and 2 tokenizer failures because `OPENPI_DATA_HOME` was unset and the managed environment cannot create `~/.cache/openpi`; rerunning against the existing `/data/yanghaojun/cache/openpi` asset path.
- The corrected-cache rerun passed all 54 selected SO(3), v3 profile, deployment and service tests in 13.02 seconds.

- Confirmed GPU 2/3 currently idle. Started isolated Python 3.11 dependency setup under /data/yanghaojun/envs/tabero-smoke with frozen lockfile.
- Inspected tactile TCN implementation: configured no-difference path consumes all nine frames, without baseline subtraction.
- uv snap fails in sandbox; approved host invocation works (uv 0.12.6). Optional ~/.local/bin/uv does not exist; using installed snap uv.
- Frozen dependency install failed because the repository lockfile does not include the current tabero-vtla workspace name. Python 3.11.16 and empty environment were successfully created; retry by resolving the project lockfile, not bypassing dependencies.
- Implemented config, strict loader, action-only loss, dataset selectors and deterministic held-out evaluation with local metrics.jsonl.
- Second installation attempt resolved dependencies but failed packaging the renamed project; corrected Hatch configuration. Downloads are cached for retry.
- Added read-only Parquet/video audit with train-only normalization, pinned checkpoint downloader and regression tests.
- Source compile checks passed. PyAV source build failed; switched to compatible wheel release 14.2.0.
- Wheel-based dependency installation now progressing. Added bounded evaluation sampled across all three held-out episodes and local finite-metric checks.
- Found and fixed LeRobot v0.3.3 non-contiguous episode query-index bug locally; added boundary regression test and unused-column pruning.

## User scope update
- User explicitly requested code changes only, no training now; deployment target is real FR3 hardware, not simulation.
- No training, model inference, checkpoint download or robot actuation has run.
- Dependency installation was in progress; stopping that exact installer while preserving completed downloads and the isolated environment.
- Remaining work changed to implementation review, static/dependency-light checks and real-robot contract documentation. Actual data audit/normalization/model validation deferred until requested.
- Installer session 48866 was interrupted successfully (exit 130). Completed Python install and package cache remain recoverable/reusable; nothing was deleted.

## Code-only handoff
- Added dedicated real-FR3 input/output adapters and policy metadata; no robot controller, connection, simulator or actuation code was added.
- Added docs/tabero_real_robot_smoke.md with exact action units, inverse normalization/delta path, episode split, opt-in future commands and robot safety boundary. README links to it.
- Preserved original configs' objectives and disabled evaluation by default outside the new config.
- Ten modified/new Python modules pass AST parsing and compileall on Python 3.11.16.
- New scripts and regression-test file pass full project Ruff checks. Core edited files pass E4/E7/E9/F checks honoring F722 annotation exceptions and excluding baseline F401 imports.
- pyproject.toml and uv.lock parse successfully; workspace package name and PyAV 14.2.0 pin agree.
- git diff --check passed.
- Full pytest suite NOT RUN: pytest, jax, flax, torch, lerobot and pyarrow are not installed in the interrupted environment. No runtime compatibility or model convergence claim is made.
- Dataset audit/normalization helper NOT RUN; checkpoint downloader NOT RUN; training/inference NOT RUN; no datasets or labels edited.
- No model checkpoint or validation metrics were produced. GPU availability observations are not reservations.

### Files changed
- Core: pyproject.toml, uv.lock, scripts/train.py, src/openpi/models/{pi0.py,pi0_config.py}, src/openpi/training/{config.py,data_loader.py,weight_loaders.py}, src/openpi/policies/libero_policy.py.
- New: scripts/{prepare_tabero_smoke.py,download_tabero_checkpoint.py}, src/openpi/training/tabero_smoke_test.py, docs/tabero_real_robot_smoke.md.
- Documentation: README.md, task_plan.md, findings.md, progress.md.

## Session continuation: user-operated training
- User permits dependency installation and unit tests; final training must be launched by the user.
- Resuming isolated environment installation. No training/model inference/robot actuation will be started.
- Will validate regression tests and command-line parsing, then provide manual run instructions.
- Added scripts/check_tabero_batch.py for user-operated, CPU-only batch inspection and strict-checkpoint positive/shape/extra-key tests. Frozen environment installation is downloading remaining large CUDA/Torch dependencies.
- Download progress: PyAV 14.2.0, PyArrow, SciPy, OpenCV-headless and several CUDA components completed. No install/build error in the resumed attempt so far.
- Runbook now includes tmux activation, per-command CPU checks, explicit GPU selection and artifact paths. Additional code changes pass Ruff and diff whitespace checks.
- JAX CPU/CUDA plugin packages, Triton and cuSolver completed downloading; the installer is still fetching larger Torch/CUDA packages. No training/test loop has been started yet.
- cuBLAS, cuDNN, cuFFT, cuSparse and NCCL downloads completed; remaining large package is PyTorch. Existing caches are reused, no environment outside /data/yanghaojun was replaced.
- Frozen environment install completed successfully: 225 packages installed, Python 3.11.16, JAX/JAXlib 0.5.3, Flax 0.10.2, PyTorch 2.7.1, LeRobot 0.3.3, PyAV 14.2.0. Installation exit 0.
- Started CPU-only selected unit tests (excluding tokenizer downloads and training-loop tests) and train CLI --help parsing only.
- CPU tests completed: 29 passed, 2 deselected, 13 dependency deprecation warnings in 18.10s. Files: tabero_smoke_test.py, transforms_test.py, normalize_test.py, lora_test.py; `-k not tokenize`. No optimizer/training loop or pretrained model executed.
- train.py local-smoke --help exited 0. Documented flags parse correctly.
- Latest nvidia-smi: all 4 cards busy. GPU 0/1 use 22037/22264 MiB; GPU 2/3 use 47957/47837 MiB of 49140 MiB. No GPU workload was launched by this task; user must reselect a permitted card when training.
- Audit, batch-check and download helper `--help` commands exit successfully without running their operations. Exact package versions, split IDs and strict checkpoint path verified by importing configuration only.
- Final Ruff checks for helper scripts/tests and git diff --check pass. Manual training handoff ready; no training, checkpoint download, real-data audit, normalization or robot actuation performed.

## 2026-09-03: Manual GPU-selection handoff after NaN
- Latest user reserves training for manual operation again. No GPU training was launched by the agent; only CPU synthetic tests.
- Added per-update finite-state checks, rejected-update state preservation and failure batch/report persistence in train.py; root cause of prior NaN remains unproven.
- Launcher now requires an explicit CUDA_VISIBLE_DEVICES list, uses global batch 2 x selected GPU count, retains 3000 updates and 26/3 split, and prints exact log paths. Dry-run does not initialize JAX or write outputs.
- Prior CPU process handle expired (unknown session); reran tests. Result: 39 passed, 1 new integration-test failure because its TinyModel did not inherit BaseModel. Correcting the test fixture, not disabling type checks.
- An apply_patch with an incorrect progress.md context failed atomically; reapplied with the verified context.
- Corrected TinyModel to inherit the actual BaseModel contract. Selected CPU suite now passes: 40 tests, 6 dependency deprecation warnings, 14.97 seconds. Synthetic optimizer updates only; no full Tabero/GPU training.
- Shell syntax, eight GPU-list dry-run/rejection cases, new numerics-module Ruff checks and git diff --check passed. Dry-run confirms 1/2/3/4 GPU batch and validation-batch mapping without accessing GPU devices or creating output directories.
- Added manual-launch/log-viewing runbook section; recorded this scope change in planning files. Actual training and NaN reproduction are not performed.

## 2026-09-09: synchronous deployment and saturating guard
- Replaced the asynchronous future/chunk-index loop with one blocking inference per observation and fixed execution to action[0].
- Replaced rejection of finite prediction bounds with explicit workspace, target-distance, SO(3), gripper-position, and rate saturation. Tracking and system faults remain fail-closed.
- Added raw/bounded/limited action and `limits_applied` logging; updated deployment and incident documentation.
- Final selected deployment/service/SO(3)/v3 suite passed 55/55 after adding exact-boundary coverage. Ruff, format, JSON parsing and `git diff --check` also pass.
- Committed the scoped deployment changes as `e5b4f21` and pushed them to `deploy/fix/fr3-tactile-shadow-deploy`. Unrelated local training, diagnostics, and planning files were not included.

## 2026-09-09: v3 checkpoint 20000 deployment mismatch
- Diagnosed the inference-host failure as v1 config `pi0_lora_tacfield_local_tactile_lora_smoke` requesting `local/tabero_lerobot_compact_v1` assets from a v3 checkpoint.
- Verified checkpoint 20000 contains `local/tabero_lerobot_compact_v3/norm_stats.json`, hash `3179b7ee7553cac64f3ac8a40897ed620b78b2ee354f9a297534f88cc66a0aed`, and the required action/state/tactile keys.
- Added and tested the v3 config/SO(3) deployment path plus an actionable asset mismatch error. Selected CPU suite passed 57/57.
- Committed as `badb9ef` and pushed to `deploy/fix/fr3-tactile-shadow-deploy`.

## 2026-09-10: Began five-run K=2 real-robot failure analysis
- Restored repository planning context and confirmed the working tree has only the pre-existing untracked planning/diagnostic files.
- Completed integrity and execution analysis of all five logs.
- Verified identical model/deployment metadata and near-identical, demonstration-aligned initial poses.
- Quantified successful command delivery, large physical motion, low tracking error, roughly 6 Hz control, 80.5% K=2 second-step use, 62.2% rate saturation, divergent terminal poses, and one run without meaningful gripper closure.
- Next: verify the saved offline-evaluation artifact and document why its open-loop averages do not establish closed-loop grasp success.
- Located the matching v3-touch checkpoint-20000 offline result directory; inspecting its metadata and metrics next.
- Verified its recorded-observation/open-loop scope, first-action versus full-chunk metrics, hold-current-state baseline, and inference latency. The small first-step number does not measure task success and is weaker than the no-motion baseline on mean error.
- Confirmed evaluation coverage is only held-out episodes 4/14/24 and that deployment logs do not preserve sensor payloads needed to verify visual/tactile semantic distribution shift.

## 2026-09-10: Inference-host rewrite handoff
- User requested a handoff document for rewriting the inference-host deployment code, including the currently observed deployment failures.
- Reusing the verified five-run and offline-evaluation findings; next step is to inventory exact current code/config contracts before writing the document.
- Verified the branch revision, runtime configuration, prediction saturation order, safety thresholds, K=1/K=2 scheduling, v3 model identity, prompt, and action contract.
- Read the current observation, transport, server, conversion, and control-loop implementations. Captured exact image crop, tactile history/layout, HTTP unit conversions, WebSocket protocol, metadata handshake, warmup/deadman semantics, and shutdown behavior for the handoff.
- Confirmed the split Python environments, minimal robot-side dependencies, GitHub remote/branch, and relevant implementation baseline commits.
- Wrote the Chinese rewrite handoff and linked it from README.
- Updated incident 17/27 and added incident 28 with five-run and offline-evaluation evidence.
- Next: validate Markdown structure, local links, numeric consistency and the exact commit scope, then commit and push only README plus the two documentation files.
- Documentation validation passed: 456 lines, balanced code fences, README link present, required hashes/evidence present, and `git diff --check` clean. No runtime code changed, so no model/ROS/robot tests were run.
- Git author is configured. Ready to commit only README, the new handoff, and the incident log; planning/diagnostic files remain untracked and excluded.
- User requested local storage only. The attempted Git mutation was rejected before execution; HEAD remains `53370e6`. The handoff, README link, and incident-log update remain local and uncommitted.

## 2026-09-15: Enabled wrist-wrench prediction for paired whiteboard training
- Traced the repository's original 13D Tabero objective and confirmed force supervision needs a combined `7D action + 6D wrench` target plus 13D train-only normalization; changing metadata alone would not train the force slots.
- Audited all 12,126 wrench rows in both datasets: finite float32 and bit-identical between exports. Recorded the user's N/N·m and K-frame contract without adding an implicit unit or frame transform.
- Added dedicated action+wrench input/output transforms, SO(3)-safe data config, force-specific paired config names, dual sequence queries, 0.1 wrench loss, force-aware audit/batch checks and collision-safe assets/checkpoints.
- Extended the model server and FR3 client contract so a force checkpoint returns/logs `wrist_wrench [50,6]` while only `actions [50,7]` reaches the motion safety path. Predicted wrench is not used for force control.
- Real episode-0 read-only loader integration confirmed `[50,7]` actions and `[50,6]` wrench concatenate exactly to float32 `[50,13]`.
- First combined regression found one stale smoke fixture whose fake marker row 0 violated the already-existing reference-grid assertion; corrected only the fixture. Final selected CPU suite: 106 passed in 21.80 seconds.
- Ruff checks, narrow config syntax/import checks, shell syntax, force-pair dry-run and `git diff --check` pass. No GPU training, checkpoint loading, W&B upload, inference service, network connection or robot actuation was performed.

## 2026-09-17: Implemented four-checkpoint whiteboard offline evaluation
- Added `scripts/eval_tabero_whiteboard_offline.py` with strict 13D whiteboard config validation and checkpoint-local normalization.
- Extended `src/openpi/policies/tabero_offline.py` to record and score `wrist_wrench[50,6]` alongside the existing absolute 7D action outputs.
- Added `scripts/run_tabero_whiteboard_offline_12k.sh` for the four user-selected checkpoints and `scripts/summarize_tabero_whiteboard_offline.py` for one comparison CSV/JSON.
- CPU check-only validation succeeded for all four checkpoints on episode 1/7/17 boundary samples. Wrench-specific tests passed 2/2 and whiteboard config suites passed 26/26.
- Initial sandboxed GPU probing could not reach the driver; later host-authorized access exposed four idle A6000 GPUs and full inference completed.

## 2026-09-18: Evaluated mixed full-finetune step 20000
- Verified finalized next-state and sent-command step-20000 checkpoints and their checkpoint-local normalization assets.
- CPU boundary checks passed for both; full GPU evaluation completed for all 1001 validation anchors per model with no non-finite outputs.
- Saved complete summaries, predictions and action/wrench plots under `/data/yanghaojun/outputs/offline_eval/offline_whiteboard_full_ft_mixed20k_{next,sent}_20260918`.
- Compared next-state mixed full-finetune 20k against LoRA32 next-state 12k. Mixed full-finetune wins every reported mean action and wrench metric and reduces gripper out-of-range outputs from 186 to 14.
- Updated the incident overview and detailed evidence. No shadow or robot command was run.
- Added a reusable aligned-overlay script and exported a single three-episode figure comparing recorded first actions, mixed Full-FT 20k and LoRA32 12k, with position and SO(3) error panels.
- Audited next-state wrench metrics in detail: Full-FT 20k has lower mean force/torque error on all three episodes and all 50 horizons, while the valid-chunk force P95 remains essentially tied with LoRA32 12k.
- Added a reusable six-component wrench overlay and exported a single figure for episodes 1, 7 and 17 with recorded targets and both checkpoint predictions.
