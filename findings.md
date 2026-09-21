# Findings & Decisions

## Paired whiteboard action-definition training (2026-09-15)
- User supplied two compact whiteboard datasets: `test2_tabero_next_state_compact` labels actual reached next state, while `test2_tabero_sent_command_compact` labels the teleoperation command sent to the robot. They require independent normalization assets and checkpoints; all other training settings should match the existing v3 tactile 20k profile.
- Requested execution is sequential and unattended: the sent-command run must start only after the next-state run exits successfully. Formal model training remains user-operated.
- Preserve existing dirty README/incident updates, diagnostics, handoff and planning files; changes for this task must be additive and isolated.
- Both datasets report LeRobot v2.1, 39 episodes, 12,126 frames, 10 Hz, 78 videos, 7D state/action, marker history `[9,198,2]`, wrist wrench retained but unsupervised, and identical task text: `Pick up the yellow whiteboard eraser and use it to erase the black X mark from the whiteboard.`
- The next-state metadata declares `actions[t] = state[t+1]` and omits the terminal source observation. The sent-command metadata declares synchronized absolute teleoperation command at the current source frame and omits the same terminal row, enabling paired anchors.
- Both exports exclude raw episode 19, compact 35/39 output episodes, and report 98 missing candidate steps. Front RGB is now `[520,640,3]` rather than the prior v3 crop `[520,390,3]`; wrist RGB remains `[480,640,3]`. Real decoding/preprocessing must be exercised before handoff.
- Use exactly the same held-out episode IDs for both models and compute independent train-only normalization statistics. Reusing either v3 assembly stats or one whiteboard dataset's action stats would invalidate the action-definition comparison.
- Existing `run_tabero_v3_touch_20k.sh` is intentionally hard-coded to the assembly v3 config, asset path/hash, 26/3 split, 172 validation batches and v3 log description. It must not be called unchanged for the whiteboard pair.
- The established reusable settings are: published Tabero 49999 params, tactile TCN LoRA rank/alpha 16, action-only 7D loss, true SO(3) relative actions, global batch 4, workers 4, seed 42, warmup 500, cosine `1e-5 -> 1e-6`, 20,000 updates, validation every 1,000, saves/retention every 4,000, W&B online and image upload off.
- Initial all-column paired Parquet scan did not finish within the 30-second tool yield and returned no captured output, likely due to materializing both full tactile histories. Do not repeat that expensive comparison unchanged; split lightweight state/action pairing from each dataset's existing full tactile/video audit.
- Lightweight paired audit passed: all 39 episode lengths and every state/timestamp/frame/episode/global/task index match exactly. For next-state, every nonterminal action exactly equals the following state. With held-out IDs `(4,14,24)`, lengths would be 272/343/274 = 889 validation frames and 11,237 train frames, but the user subsequently requested a freshly randomized fixed 3-episode validation split.
- The label distributions are materially different, as intended but also risky: next-state target-to-state position p50/p95/max is `5.21/14.50/46.74 mm` with zero adjacent jumps above 50 mm. Sent-command target-to-state is `40.78/380.97/473.38 mm`; adjacent action jumps have p95/max `27.51/451.68 mm` and 161 rows exceed 50 mm. Next-vs-sent target position difference is p50/p95/max `33.63/379.03/472.07 mm`.
- Sent-command SO(3) target-to-state p50/p95/max is `2.47/4.22/5.77°`, versus next-state `0.195/0.648/2.315°`. Both require the true SO(3) transform; raw branch-like action-vector jumps occur in both exports.
- User explicitly requests exactly three randomly selected validation episodes out of 39. Select them once with a recorded seed and use the identical IDs in both configs; do not sample at each launch.
- Deterministic draw `random.Random(42).sample(range(39), 3)` selects validation episodes `(1,7,17)`, with lengths 274/315/412 = 1,001 frames. Both experiments will train on the other 36 episodes / 11,125 frames and evaluate 250 full batches = 1,000 frames at global batch 4; one tail frame is omitted consistently.
- Added isolated configs `pi0_lora_tabero_whiteboard_next_state_20k` and `pi0_lora_tabero_whiteboard_sent_command_20k` by replacing only dataset/split/metadata/eval-size fields on the established v3 tactile 20k profile. They therefore retain the same published-49999 loader, model/LoRA, optimizer, seed-independent defaults, save schedule and W&B behavior while using distinct repo/asset IDs.
- Preparation now verifies each config against both `info.json` and conversion `action_source_mode`, plus the declared state-step offset, before auditing or writing statistics. This prevents accidentally preparing sent-command data under the next-state config or vice versa.
- Added `run_tabero_whiteboard_pair_20k.sh`: new mode prepares both independent stats, trains next-state first, requires its finalized `20000/params`, then starts sent-command. `set -euo pipefail` prevents the second run after any first-run/audit/GPU failure. It writes per-model logs, a pair log and an atomic stage/status file.
- The pair launcher supports `--resume`: completed runs are skipped, incomplete runs require both a numeric checkpoint and `wandb_id.txt`, then reuse the original W&B identity. Missing/ambiguous partial state fails closed rather than overwriting.
- Added an explicit preparation assets-base override and matching train CLI base-directory overrides so `TABERO_OUTPUT_ROOT` cannot cause stats/checkpoints to be read from a different root than the launcher reports. GPU probe and both train processes explicitly set `JAX_PLATFORMS=cuda`; preparation explicitly uses CPU.
- Added CPU contracts for both dataset/source/split mappings, inherited model/optimizer/20k schedule, independent asset IDs and ordered dry-run command generation. The launcher dry-run creates no output directory and emits next-state before sent-command.
- Formatting, shell syntax and a focused suite covering new configs, existing v3/SO(3) behavior and server loading all pass: 28 tests in 38.22 seconds. No model parameters were constructed or trained.
- Sent-command outliers are not isolated to one bad episode: all 39 episodes contain at least one adjacent position-command jump above 50 mm, totaling 161. Per-episode maxima range from 138.8 to 451.7 mm, often near similar mid-trajectory frames but sometimes at multiple phases.
- Concrete episode 1 evidence: at frames 118-119, measured z is about 0.142 m while commanded z is exactly 0.600 m; at frame 120 the command z jumps to 0.148 m, producing the 451.7 mm adjacent command discontinuity. This looks like command saturation/gating/reset behavior rather than ordinary one-step tracking lag, but raw collection/control logs are required to prove the mechanism.
- Per the requested experiment, retain these sent-command labels unchanged. Interpret the result as a comparison of the two exported label streams, not a clean causal comparison of idealized "executed state" versus "intended command" semantics; the outliers are a major confounder.
- Real-path launcher dry-run succeeds and emits the two independent commands in the required order with the correct 20k/4k/W&B/SO(3) settings. An unrestricted Ruff pass reported five pre-existing `config.py` findings plus one new test import-order warning required by pre-JAX CPU environment setup; fix only the new-file warning and validate core config with the repository's established narrow rules.
- Diff review found that running Ruff format on the monolithic legacy `config.py` mechanically reformatted about 300 unrelated lines around pre-existing noncanonical style. Revert only this file's complete task diff through `apply_patch`, then reapply the isolated whiteboard block without formatting the legacy file; other task/user changes remain untouched.
- First automated reverse-hunk conversion failed atomically because the custom patch parser did not accept the standard reverse-diff hunk context. The file remained unchanged. Use a single full-content `apply_patch` update from current content to the known-clean HEAD version, then reinsert only the new block.
- Excess formatter churn was successfully removed with an atomic full-content `apply_patch` restoration followed by the isolated 81-line whiteboard block. `config.py` now has no unrelated task diff. Focused verification still passes 30/30 tests in 13.27 seconds; new-file Ruff, narrow core Ruff, shell syntax and `git diff --check` all pass.

## V3 checkpoint versus inference-code contract (2026-09-10)
- User reports that an inference-host checkout lacks the handoff's v3 config and still reconstructs orientation by component-wise rotvec addition, while checkpoint and normalization hashes match. Need distinguish the current workstation branch from the inference-host checkout and verify whether configuration/transform code is serialized in either asset (it ordinarily is not).
- Initial repository state is branch `fix/fr3-tactile-shadow-deploy` with existing user changes in `README.md`, `docs/tabero_interview_incident_log.md`, `diagnostics/`, and a handoff document. Preserve these changes and perform static inspection only.
- The current A6000 branch does contain `pi0_lora_tabero_v3_touch_20k` in `src/openpi/training/config.py`; it sets `use_so3_relative_actions=True`. `TaberoTacFieldDataConfig.create()` consequently installs `RelativePoseActions` for inputs and `AbsolutePoseActions` for outputs.
- The correct inverse in `AbsolutePoseActions` is implemented as `R_action = R_state * R_relative`; the legacy `AbsoluteActions` still performs component-wise addition and is deliberately retained for older profiles. `policy_config.create_trained_policy()` executes Unnormalize before the selected data-config output transform, so the runtime config—not the checkpoint tree or normalization JSON—selects the inverse geometry.
- If the inference-host checkout genuinely lacks the v3 config, `--config pi0_lora_tabero_v3_touch_20k` should fail at config lookup. If it instead substitutes a legacy config or bypasses/copies asset validation, parameter/stat hashes can still match while the output interpretation is wrong; matching hashes do not cover source/config semantics.
- Geometry-only replay on all 6,626 v3 one-step labels shows that replacing `R_state * R_relative` with rotvec addition produces nonzero physical orientation error: median `0.05596°`, p95 `0.41044°`, max `1.44486°`; 2,493/6,626 exceed `0.1°`. This is deterministic transform error, not model uncertainty.
- The error grows with chunk horizon on valid expert targets because every horizon is relative to the anchor state. At offsets 1/2/4/9/19/49 the p95 wrong-inverse errors are `0.766/1.061/1.431/1.821/2.441/4.348°`, with a selected-horizon maximum `6.672°`. Actual erroneous model outputs can be worse and cannot be bounded from label replay alone.
- The current registry explicitly appends the v3 profile and config lookup is name-based. Thus a checkout that truly lacks this definition cannot faithfully construct the v3 policy even if it possesses byte-correct checkpoint assets.
- Git provenance: current HEAD is `53370e642c72ac468194ef773bb20cf8ba8d0403`; the v3 deployment configuration was introduced by commit `badb9ef` (`Add v3 checkpoint deployment configuration`). A host checkout predating or not containing `badb9ef`, or a service launched from a different project directory, explains the reported absence.
- Existing incident entry 23 records an earlier inference-host v1/v3 mismatch that first failed on the v1 asset id, then reported successful v3 config/normalization loading and 150 synchronous shadow inferences after the fix. The new report therefore cannot be assumed to describe that same verified checkout/runtime; current host path/HEAD/process command must be re-established.
- CPU config probe resolves the current v3 profile to asset id `local/tabero_lerobot_compact_v3`, input transforms `[TaberoActionOnlyInputs, RelativePoseActions]`, and output transforms `[AbsolutePoseActions, TaberoActionOnlyOutputs]`. A missing name raises `ValueError` before policy construction.
- Focused CPU verification passed 22/22 tests across transforms, v3 configuration/checkpoint schedule, and Tabero service asset/strict-restore contracts. This verifies the current source tree only; it does not validate the reported inference-host checkout or a live process.
- Server metadata carries config name, asset id, norm/conversion hashes and action/tactile contracts, but there is no Git commit or transform-source fingerprint in the handshake. A checkout with copied metadata plus legacy transform code can therefore evade hash-only provenance unless config semantics are separately asserted.
- Correction after tracing the client: `examples/fr3_deploy/core.py::validate_metadata()` does **not** currently assert `config`, `asset_id`, `norm_stats_sha256`, or `training_action_representation`. The client only logs the first three after accepting the handshake. The handoff document says they must be checked, so implementation is weaker than the documented gate.
- Consequence: normal current server startup with a v1 config and unmodified v3 checkpoint layout fails safely on asset-id mismatch, and an absent v3 config name fails safely at lookup. But an old/bespoke server that points the legacy transform at copied v3 stats can pass the client's present external-contract checks and silently return component-added orientations. This is a real deployment compatibility defect, not proof that any already-recorded run used the wrong transform.
- Direct inspection of `badb9ef^` confirms the older tree contains only `DeltaActions`/`AbsoluteActions`, no v3 config name, `use_so3_relative_actions`, `RelativePoseActions`, or `AbsolutePoseActions`. The reported host symptoms exactly match a pre-`badb9ef` source baseline, but host HEAD/path evidence is still required before calling that the proven host root cause.
- Incident log already contains user-owned updates through issue 28. Add a separate issue 29 for recurrence via source-checkout drift and the handshake gap, preserving the historically verified resolution of issue 23.
- Current clean committed source fingerprints are transforms `1ce86f3c...57f68`, config `134fd4c8...745f4`, policy construction `a3310af1...3f13`, server `0740c067...b98c7`, and client core `2e111726...e9d72`. Current HEAD `53370e6` contains `badb9ef` as an ancestor.
- One combined `apply_patch` attempt failed atomically because an update-file marker followed a bare hunk marker. No file content changed from that attempt; split patches are being applied with verified contexts.

## Live v3 SO(3) 20k loss diagnosis (2026-09-09)
- User-provided excerpt covers approximately updates 550-1260. Individual 10-update training averages fluctuate roughly 0.20-0.45, with a visible late-window cluster near 0.24-0.27; validation at update 1000 is 0.333902 over 688 held-out frames.
- Every finite/update guard in the excerpt is healthy (`inputs/loss/grads/updates/optimizer/candidate_params=1`, `update_applied=1`, no bad gradient leaf). This is not a NaN, rejected-update, or optimizer-stall symptom.
- `tactile_loss=0` is expected for this action-only profile: tactile is an input and its encoder/LoRA receives gradients through `action_loss`; there is deliberately no tactile/force prediction target.
- The active read-only run is `/data/yanghaojun/outputs/logs/v3_touch_so3_20k_run1.log`.
- Full structured metrics through update 1671 disprove a plateau: mean train loss falls from 2.306 (updates 1-100) to 0.379 (251-500), 0.313 (501-750), 0.272 (751-1000), 0.247 (1001-1250), 0.230 (1251-1500), and 0.218 (1501-1671). Last 10/25/50 logged-window means are 0.216/0.222/0.227.
- Validation improved from 2.80060 at initialization to 0.333902 at update 1000 and 0.313991 at update 2000. Evaluation reuses a deterministic per-batch RNG at every checkpoint, so this comparison is not explained by changing validation noise/time samples.
- Run provenance is correct: GPUs 0 and 2, global batch 4, published Tabero 49999 initialization, SO(3)-matched norm hash `3179...a0aed`, W&B online, 500-update warmup and 20k cosine schedule. No checkpoint is expected before update 4000.
- The apparent 0.24 plateau is short-window variance: each displayed row averages only ten updates with global batch 4, while the flow-matching target is regenerated from random Gaussian noise and a Beta-distributed diffusion time on every update. The loss is normalized-space flow-velocity MSE over 50 horizons x 7 action dimensions, not a physical position/rotation error.
- The LR reaches `1e-5` at update 500 and remains about `9.87e-6` at update 2000, so this run is still near peak LR and only 10% complete. Do not restart or retune from these data; reassess the deterministic validation trend around updates 3000 and 4000, then use the saved 4000 checkpoint for offline physical-unit evaluation.

## v3 tactile 20k training preparation (2026-09-09)
- User requests a fresh tactile-input LoRA run on `/data/yanghaojun/datasets/tabero_lerobot_compact_v3`, initialized from the published Tabero checkpoint, for 20,000 updates with checkpoints every 4,000 updates and W&B logging.
- v3 reports 29 episodes / 6,626 frames at 10 Hz with the same state `[7]`, action `[7]`, and marker-motion `[9,198,2]` tensor shapes. Its important semantic change is next-observation supervision: `actions[t] = state[t+1]` within an episode and the terminal source frame is omitted.
- The intended prompt in `tasks.jsonl` is `Align the black circular component with the receiving hole on the gray circular component and insert it to complete the assembly.` Initial inspection found stale acquisition text in `episodes.jsonl`; determine whether the loader's authoritative task mapping is sufficient and add an auditable consistency check before training.
- Existing Tabero tactile-LoRA configuration inherits `wandb_enabled=False` from the smoke profile. The new configuration must explicitly enable it and must not warm-start from any local smoke/recovery run.
- Preserve all existing dirty deployment files. This task will add isolated training files/configuration and CPU tests only; the user retains authority to start training and external W&B upload.
- Full read-only audit succeeded and generated v3-specific train-only stats. Split is 5,936 train / 690 validation frames; all non-terminal `actions[t] == state[t+1]` checks have zero error; no episode has an adjacent action-position jump over 5 cm.
- Correct tactile contract check: index 0 is the fixed reference and indices 1..8 are rolling coordinates. Across all episodes, reference-static, history-shift, and episode-start repeat errors are all exactly zero. The earlier whole-9-frame shift diagnostic was invalid and was corrected before finalizing the audit report.
- All 29 Parquet task indices are 0 and therefore use the desired string from `tasks.jsonl`. All 29 `episodes.jsonl` summary strings remain stale; this is reported but does not affect `PromptFromLeRobotTask`, which reads `dataset_meta.tasks`.
- Axis-angle representation risk remains: the audit counts 114 branch-like adjacent jumps whose physical rotations are small. The v3 next-state relabeling fixes startup position targets but does not unwrap the axis-angle representation.
- After enabling the v3-only SO(3) action transform, regenerated normalization SHA256 is `3179b7ee7553cac64f3ac8a40897ed620b78b2ee354f9a297534f88cc66a0aed`; minimum std values are finite/non-degenerate. The maximum action-stat std fell from about 1.91 under raw rotvec subtraction to 0.0712.
- Real CPU train and validation batch transforms passed: actions `[4,50,32]`, tactile prefix `[4,9,396]`, two active image views `[4,224,224,3]`, finite normalized values, and tokenized task text `[4,48]`.
- New profile uses global batch 4, warmup 500, cosine `1e-5 -> 1e-6` through 20,000 updates, full-ish validation every 1,000 updates, and retained checkpoints exactly at 4k intervals through 20k. W&B logs config/scalars by default; camera data requires a separate opt-in.
- Implemented paired v3-only transforms: training uses relative XYZ and `R_state^-1 * R_action` as a shortest SO(3) rotvec while keeping gripper absolute; inference composes `R_state * R_relative` and returns the existing absolute 7D real-robot contract. Legacy configs retain component subtraction.
- All 114 raw rotvec branch-like transitions remain visible in audit provenance but no longer become 2π action targets. Maximum physical one-step rotation is 1.22521 degrees; direct transform/round-trip and real train/validation batch checks pass.
- Compact timing is not reconstructable from v3: 12 episodes / 18 missing candidate steps are recorded. The audit also confirms 2,726/6,626 (41.14096%) one-step translations exceed 2 mm. Expert targets are deliberately not clipped to the provisional 0.02 m/s robot guard; the existing guard rate-limits commands and rejects tracking lag, while a strict re-export should split sequences at gaps.

## New-prompt tactile checkpoint 2000 offline result (2026-09-07)
- Completed evaluation is exactly paired with new-prompt 2999: same 693 anchors, recorded targets/state/masks, normalization hash, seed 42, ten denoise steps and horizon 50.
- 2000 first-action means: 34.2263 mm position, 6.9766 deg rotation, 1.3072 mm gripper. Valid-chunk: 53.3202 mm, 14.1850 deg, 2.4956 mm.
- Versus same-run 2999, 2000 worsens first position by 1.376 mm and gripper by 0.031 mm, but improves mean first rotation by 0.960 deg. Its median rotation is slightly worse (2.667 vs 2.552 deg), while RMSE/max improve strongly (18.759/160.176 vs 24.034/175.463), so the mean benefit comes from reducing large rotation outliers, especially episode 24.
- 2000 has lower first-position error on only 28.9% of anchors and lower first-rotation error on 43.7%; it is not a uniform per-frame improvement.
- Valid chunk versus 2999: position worsens 1.785 mm, rotation worsens 1.610 deg, gripper improves 0.202 mm. Episode 24 chunk rotation improves 1.967 deg, but episode 4/14 worsen 5.782/1.505 deg.
- Against RGB+state 2999, checkpoint 2000 is better only in first-action rotation (6.977 vs 8.468 deg); position/gripper and all valid-chunk means are tied or worse (chunk rotation 14.185 vs 14.019 deg).
- Diagnostics versus new-prompt 2999: gripper out-of-range count improves 11,665 -> 10,727, while >50 mm chunk jumps worsen slightly 66 -> 69. Latency is not comparable because GPU load changed during the run.
- Selection implication: validation-loss minimum at 2000 did not translate to best general physical-unit action errors. Keep 2999 as the broader new-prompt checkpoint; retain 2000 only when reducing first-action rotation outliers, particularly episode-24-like cases, is the dominant objective.

## Best checkpoint candidates across three runs (2026-09-07)
- Saved checkpoints for every run are 500/1000/1500/2000/2500/2999.
- Old-prompt tactile: recorded validation minimum is unsaved step 2250 (0.29234); best saved candidate is final 2999 (validation 0.33503).
- RGB+state: recorded minimum is unsaved step 2750 (0.28149); best saved candidate is 2500 (0.30344), ahead of 2999 (0.31037).
- New-prompt tactile: best recorded and saved candidate is 2000 (0.32094), ahead of 2500 (0.32543) and 2999 (0.32826).
- Only the three 2999 checkpoints have matched physical-unit offline action evaluation. Candidate 2000/2500 selection remains provisional until those checkpoints receive the same evaluation.
- Among evaluated 2999 weights, new-prompt tactile is best for first-action position/rotation and chunk rotation; RGB+state is best for chunk position/gripper; old-prompt tactile has no compelling overall advantage.

## 2026-09-07 external deployment branch contract
- Treat this branch contract as the source of truth for subsequent real-FR3 deployment code in this task. Separate model-facing data invariants from ROS transport: compressed/raw topics and QoS describe this live wiring, while the decoded arrays and units below define checkpoint compatibility.
- Static compatibility conclusion for checkpoint real_fr3_recovery_20260903_112020/2999: compatible with config pi0_lora_tacfield_local_tactile_lora_smoke. Parameter-bearing code is byte-identical; rank/history/action shapes and checkpoint-local stats align. Added policy metadata is runtime validation only and does not add checkpoint leaves.
- Remaining gates: fix the two stale all-zero offline test fixtures; verify actual publisher QoS and decoded ZED/D405 resolutions on the inference host; run shadow with the real GPU checkpoint. No claim of task success or safe execution follows from interface compatibility.
- Candidate authoritative branch: Leohahahahha/Tabero_serve, fix/fr3-tactile-shadow-deploy, commit 24e6370.
- Live topics/config: compressed ZED `/zed/zed_node/rgb/color/rect/image/compressed` (reliable, depth 1); raw D405 `/camera/d405/color/image_raw` (reliable, depth 1); packed DM-Tac left/right `/dmtac/{left,right}/packed_frame` (best effort, depth 1), shear_depth/schema 3; reliable Bool enable.
- Model observation remains keys image, wrist_image, state, prompt, tactile_marker_motion. Tactile is strictly float32 [9,198,2], frame 0 fixed left-then-right reference grid, frames 1..8 rolling current coordinates; same converter sampling and early repeat padding.
- Model/action representation remains current trained contract: absolute XYZ m + axis-angle rad + single-finger m; no wrench output. HTTP conversion remains XYZW quaternion plus total gripper width = 2*finger.

## New-prompt tactile evaluation (2026-09-07, in progress)
- New completed output: `/data/yanghaojun/outputs/offline_eval/offline_real_fr3_new_prompt_2999`, checkpoint `real_fr3_new_prompt_20260907_113415/2999`.
- Evaluation covers the same episode lengths 222/222/249 and 693 anchors with stride 1, seed 42, ten denoise steps, horizon 50, identical dataset-info and normalization hashes as the old tactile evaluation.
- Initial aggregate first-action means are 32.8504 mm position, 7.9365 deg rotation and 1.2764 mm gripper; valid-chunk means are 51.5349 mm, 12.5745 deg and 2.6976 mm.
- Prompt changed from hardware/acquisition metadata (`fr3 zed DM-Tac W SDK 0.1.4 raw teleop via hilserl franka_server`) to the meaningful instruction `Align the black circular component with the receiving hole on the gray circular component and insert it to complete the assembly.` Backup metadata confirms the exact prior text.
- New and old tactile runs used the same smoke99 weights, seed 42, batch 4, 3000-step LR schedule, split and model architecture. New run was interrupted near step 2500 and resumed from the last finalized checkpoint; NUL/doubled log entries reflect abandoned work after checkpoint 2400, while final checkpoint follows one restored 3000-step state path.
- Versus old-prompt tactile, new-prompt first action improves position by 0.478 mm (1.4%) and rotation by 1.758 deg (18.1%), but worsens gripper by 0.098 mm (8.3%). Valid chunk worsens position by 0.756 mm and gripper by 0.298 mm, while improving rotation by 2.607 deg (17.2%).
- Per episode first-action rotation improves consistently: 4 by 1.156 deg, 14 by 0.723 deg, 24 by 3.217 deg. First-action position improves by 0.822/0.411/0.231 mm. Gripper worsens on 4/24 and is nearly tied on 14.
- Versus RGB+state, new-prompt tactile first action is better by 0.517 mm position and 0.532 deg rotation but worse by 0.083 mm gripper. Valid chunk is worse by 1.932 mm position and 0.339 mm gripper, better by 1.445 deg rotation.
- Training validation: after step 0, new prompt is lower than old prompt at 11/12 recorded checkpoints, average 0.34446 versus 0.36514. Final is 0.32826 versus 0.33503; however old prompt's isolated best 0.29234 beats new prompt best 0.32094. RGB+state final remains lower at 0.31037.
- New prompt result is a real, consistent orientation improvement but not a broad action improvement. Constant one-task language provides no within-dataset task-discrimination signal; it mainly changes a global conditioning embedding.
- Diagnostics: new prompt has 11,665 gripper chunk outputs outside 0–42.5 mm versus 10,202 old and 9,231 RGB-only; first moves over 50 mm are 59 versus 52/65, and within-chunk position jumps over 50 mm are 66 versus 65/59.
- Saved three-model paired plot at `/home/yanghaojun/.codex/visualizations/2026/09/03/01a065ac-d11c-7593-a727-515aa0b0b11c/new_prompt_three_model_comparison.png`; official new-touch/RGB comparison JSON is in sibling `new_prompt_touch_vs_rgb_state_metrics`.

## Tactile versus RGB+state comparison (2026-09-07, in progress)
- User completed a 3000-step no-tactile model evaluation at `/data/yanghaojun/outputs/offline_eval/offline_rgb_state_20260906_201654_2999` and requests comparison with the prior tactile model evaluation.
- Both output directories contain manifest, summary, predictions CSV/NPZ and episode 4/14/24 plots. Comparison must first verify matched evaluation settings and distinguish model input effect from training/checkpoint differences.
- Evaluation selection is matched: dataset-info hash, 693 anchors, episodes and lengths, stride 1, no cap, seed 42, ten denoise steps, horizon 50 and 7D output. State/action normalization arrays are exactly equal; the norm-file hash differs only because tactile stats contain an additional tactile-prefix entry.
- Training is not a strict single-factor ablation: tactile run loaded its earlier local tactile 99 checkpoint, while RGB+state loaded the published Tabero 49999 checkpoint. Both then used batch 4, seed 42, the same split/LR profile and 3000 updates.
- System Python's SciPy binary is incompatible with its active NumPy; use the project environment for paired SO(3) calculations.
- Official paired comparison passed all alignment checks. Overall first action means: touch/no-touch position 33.3286/33.3675 mm, rotation 9.6942/8.4681 deg, gripper 1.1781/1.1934 mm. Valid-chunk means: 50.7787/49.6027 mm, 15.1816/14.0192 deg, 2.3995/2.3585 mm.
- Touch first-action position advantage is only 0.0389 mm (0.12%) and gripper advantage 0.0152 mm (1.28%); no-touch leads rotation by 1.226 deg and all three valid-chunk means. Touch has lower paired first-action error on only 47.0% position, 50.6% rotation and 39.7% gripper anchors.
- Per-episode effects reverse: touch position is 1.08 mm better on episode 14 but 0.51/0.40 mm worse on 4/24; touch rotation is 0.16 deg better on 14 but 0.44/3.16 deg worse on 4/24. No stable modality benefit across episodes.
- Excluding first 15 frames yields touch/no-touch first-position 18.966/19.331 mm, but touch rotation remains worse 8.914/7.933 deg and gripper is essentially tied. Startup data discontinuities dominate large position spikes.
- Validation flow loss also does not favor touch: final 0.33503 touch versus 0.31037 no-touch; best recorded 0.29234 at step 2250 versus 0.28149 at step 2750. Initial values are not comparable because initialization differs.
- Conclusion: these artifacts do not show useful tactile contribution. They also cannot prove tactile is intrinsically useless due to non-matched initialization, only three held-out trajectories, one run/seed, correlated open-loop samples and absence of task-success/contact-specific evaluation.
- Saved direct paired-error overlay at `/home/yanghaojun/.codex/visualizations/2026/09/03/01a065ac-d11c-7593-a727-515aa0b0b11c/tactile_vs_rgb_state_offline_comparison.png` and official comparison JSON/text under sibling `tactile_vs_rgb_state_metrics`.

## 2026-09-03 deployment contracts
- User supplied dmtac_w_ipc.py during implementation. Verified schema 3 = shear_depth, payload 921600 bytes, shear starts at byte 0 with 614400 bytes, encoding 8UC1. Bundled exact attachment and use its metadata functions; latest http_client attachment is byte-identical to earlier one.
- Current dataset conversion metadata confirms front source 540x960, crop [350,0,740,520], wrist 480x640, 10 Hz, shear_scale=1, source tactile schema 3, and original generic training task prompt. Copied exact conversion metadata into portable deployment directory.
- Attached converter: actions are absolute XYZ meters, canonical axis-angle radians, single-finger meters; HTTP /pose expects XYZ+XYZW quaternion and /move_gripper expects total width = 2*finger. Never repeat VR gain/R_map or add current pose after policy inverse transforms.
- Collector defaults: two ROS2 RGB streams and DM-Tac packed Image topics. Supplied online_policy_adapter only yields old flattened tactile keys, not the trained [9,198,2] field.
- Converter samples rounded linspace 9x11 grid in each 240x320 shear field, concatenates left/right coordinates, adds shear*scale, prefixes fixed grid, pads eight-frame history with first frame at 10 Hz.
- Bridge emits persistent last-action heartbeat even while disabled; startup does not clear previous effective action, consistent with earlier stale startup-label diagnosis. Deployment must initialize independently from measured state.
- Portable inference must explicitly load checkpoint-local assets before data config creation; existing configs contain original host absolute asset paths.
- Supplied packed decoder imports dmtac_w_ipc not attached. Do not guess binary offsets; use that installed collection module's layout or explicit 32FC2 shear topics.

## 2026-09-03: Raw state/action boundaries — verified evidence
- Final audit artifacts: diagnostics/state_action_20260903/{analyze.py,summary.json,episode_000004_frames.csv,episode_000004_boundaries.png,report.md}. 693 chunks/34,650 slots match saved labels and boundaries exactly; forward/inverse production transforms give XYZ max residual 0.0000152 mm.
- 14 episodes reuse all7 previous-last target exactly; 28/29 start with >50mm target/state gap. 638 frames across full dataset have rotvec delta >3rad but physical difference <10deg, including 29 in episode4.
- Attachment has 10,014 successful robot_state rows and no action events. Every converted XYZ state has a raw feedback match within float32 precision (<0.000031 mm).
- Episode 4 NPZ state and first target equal Parquet exactly. Plot reads these arrays directly.
- Episode 4 frames 0..10 hold XYZ [0.6882121,-0.02900319,0.23382697] m, identical to previous episode's final target; current state is near [0.306393,0.000291,0.414352] m. Position gap ~423.359 mm; target changes ~420.1 mm at frame 11 while measured state stays near reset pose.
- Episode 4 maps consecutively to zero-based raw rows 1228..1449 (JSONL lines 1229..1450). Quaternion-to-rotvec SO(3) difference <0.0000072 degrees; gripper_width/2 agrees exactly. There is a 27.72-second gap from previous episode's final raw state to reset pose. No timing compaction/missing frames in episode 4.
- Current loader queries actions at t+0..49, not state[t+1]. DeltaActions subtracts anchor state for first six dimensions; policy Unnormalize then AbsoluteActions restores absolute targets.
- Episode 4 frame 0 recorded target vs state =423.359 mm; prediction vs target =418.906 mm, but prediction vs state =8.678 mm. Last frame recorded target vs state =3.632 mm, prediction vs target =32.640 mm; startup and end have different error sources.
- Excluding first 11 frames, episode-4 prediction/target mean position error=18.685 mm; first 11 alone=391.368 mm. In interior frames 25..211, action[t] is closest to state[t+3] among shifts 0..7 (4.565 mm mean), not state[t+1] (9.684 mm). This estimates combined alignment/control lag, not proven command latency.
- Rotation-vector branch changes exist: state frame12->13 Euclidean rotvec jump ~6.283 rad with physical rotation only 0.086 degrees. Existing DeltaActions does direct component subtraction. This is a separate learning-representation risk, not the cause of 423 mm XYZ mismatch.
- Missing source raw directory /media/enine/Extreme SSD/hj/test and action-event/converter files prevent exact localization between acquisition cache and conversion carry-forward. Asynchronous request sent.

## Separate no-tactile training design (2026-09-03)
- Existing TaberoNoTactDataConfig/Inputs implement two RGB views but lack the strict real-FR3 action-only interface. Reuse their mapping with opt-in strict 7D adapters, preserving old configs.
- Existing completed tactile recovery initialized local smoke99 after 100 tactile updates; a new baseline from published Tabero cannot be called a strictly matched comparison to it. Provide optional matched tactile rerun from published weights, without launching either.
- No-tactile model should instantiate no tactile encoder/token and consume no tactile/depth/wrench columns. Published checkpoint loading must explicitly discard only the known removed prefix encoder; all shared leaves remain strict and missing LoRA is not allowed.
- Keep action transforms/labels unchanged for this comparison; changing startup/rotation preprocessing only in the baseline would confound the comparison.
- Confirmed original train-only split_provenance records 26 train/3 held-out,5962/693 frames and horizon50. The train asset norm_stats SHA256 is identical to checkpoint2999 asset SHA256 7491dc0f...; new profiles reuse these stats without local learned-weight initialization.
- Implemented pi0_lora_tabero_rgb_state and optional pi0_lora_tabero_rgb_state_touch. Shared fixed global batch4,3000 updates,seed42,LR100warmup/1e-5peak/1e-6end. Launcher only accepts GPU counts dividing4 (1,2,4); changing GPUs does not change effective batch. Single-GPU batch4 VRAM not measured.
- CPU abstract full-model parameter shapes confirm no tactile parameters in baseline and all shared shapes identical. Strict restore exception is limited to 16 known original tactile kernel/bias paths; local tactile adapters and unrelated extra/missing/shape errors are rejected.
- Offline evaluator supports either explicit real-FR3 modality while keeping old tactile defaults and metric/noise logic; CPU comparison script verifies paired settings/arrays/shared stats and reports per-episode/overall physical error differences, without claiming matched training or statistical significance.

## Tactile ablation feasibility (2026-09-03)
- Current offline CLI has no tactile ablation option. Both raw policy observation validation and the FR3 input adapter require the marker field; simply omitting it fails before model inference.
- Current TCN receives marker-coordinate history without reference subtraction. Raw all-zero coordinates are not a zero-shear/no-contact reference and could introduce a severe input distribution shift.
- Inference-time replacement tests the existing checkpoint's sensitivity to information removal, not the performance of a separately trained RGB+state model. Preserve checkpoint, raw labels, normalization, frames, denoising count and per-anchor noise when comparing.
- Conversion metadata explicitly records fixed grid x/y indices (11 by 9), left/right ordering, reference=pixel coordinates and current=reference+sampled shear. A fixed zero-shear grid repeated nine times removes sample-specific tactile information while preserving nominal geometry; it is not verified measured no-contact calibration and still passes through the encoder.
- A true tactile-branch ablation needs deliberate token/attention-path handling with matching preprocessing and strict checkpoint restoration. Raw zeros, zero encoder outputs, removing tokens and a reference-grid substitute are different interventions and must not share an ambiguous 'no touch' label.
- Existing baseline manifest is complete: checkpoint2999, episodes4/14/24,693anchors, seed42,denoise10,horizon50, stride1. Existing sampling_noise hashes seed/episode/frame independently, enabling paired input-ablation noise. Baseline results can be reused only while unchanged inputs/targets/masks/settings/checkpoint assets are verified.
- Proposal: start with real tactile versus fixed reference-grid (no sample-dependent shear), optionally add branch masking and mismatched real history as distinct diagnostics. Report paired error deltas and prediction changes, separately for each episode/first action/valid chunk; whole-sequence and reviewed phase subsets must be distinguished without changing original labels.
- Interpretation: degradation supports dependence under this intervention, not guaranteed sensor benefit; unchanged outputs do not prove touch useless in general; improved errors warrant mismatch/noise/overfitting checks, not blanket removal. With only three episodes, correlated frame counts are not independent evidence of real-robot task success.

## Offline plot analysis in progress (2026-09-03)
- Located completed run offline_2999_20260903_125340: 693 anchors, episodes 4/14/24 lengths 222/222/249, seed 42, denoise 10, GPU 1, 30975 valid anchor/offset pairs and 3675 padded targets excluded.
- First-action overall means: position 33.329 mm, SO(3) 9.694 degrees, single-finger 1.178 mm; hold-state baseline 28.690 mm, 0.929 degrees, 1.199 mm. Position median 14.612 mm, p95 116.674 mm; rotation max 171.928 degrees. Not ready for direct robot execution despite qualitative position trends.
- Whole-chunk position mean 50.779 mm versus baseline 82.647 mm; rotation 15.182 versus 1.217 degrees, gripper 2.400 versus 3.018 mm. Do not equate first-action and overlapping-chunk metrics.
- Inspected episode 4 and 14 plots: orange follows broad XYZ trends but has spikes. Large initial position errors coincide with discontinuous recorded targets while state stays near starting pose. Episode 14 has lower first-action errors than 4. Rotation spikes persist despite geodesic evaluation; these are not merely plotting +/-pi axis-angle differences.
- Inspected episode 24: most unstable rotation and finger predictions; sizable isolated X deviation near frame 175 and rotation spike near frame 221. Full first-action means (position mm / rotation deg / finger mm): ep4 37.151 / 7.735 / 1.045; ep14 26.728 / 6.779 / 0.707; ep24 35.805 / 14.040 / 1.717.
- NPZ-derived target position jumps: ep4 frame11=420.098 mm with state movement0.00477 mm; ep14 frame8=277.986 mm with state0.00303 mm; ep24 frame11=390.305 mm with state0.00179 mm. Strong evidence of target discontinuities, not actual robot jumps; underlying collection semantics still require review.
- Sensitivity analysis only (no cropping/file changes): excluding first20 frames of each episode leaves model position mean18.561 mm vs hold12.540 mm and rotation7.536 deg vs hold0.834 deg. Initial labels do not explain all poor first-action metrics.
- First-action rotation >90 degrees on24/693 anchors (ep4=4, ep14=5, ep24=15); representative ep14 frame15 has167.594 degree error while ground-truth/state rotations are physically close. Axis-angle training representation is a plausible factor, not a proven sole root cause.
- First-action finger exceeds42.5 mm on265/693 anchors; maximum43.081 mm. Whole valid chunks exceed on10202/30975 pairs, maximum43.587 mm. Do not mix pair counts with independent observation counts; over-limit magnitudes are small but require explicit robot-side guards.
- Actual subsequent policy calls average145.441 ms (~6.876 Hz), p95146.933 ms; first call including compilation18.034 s. Excludes sensors/video/network/control. Cannot sustain synchronous10 Hz re-inference as measured; chunking/asynchrony requires separate validated deployment design.
- Manifest status complete, 693-anchor coverage, and all three recorded source hashes verified against current files. No retraining, inference or raw artifact modifications were performed for this analysis.
- Plot implementation: `src/openpi/policies/tabero_offline.py:186`, one six-panel figure per episode. All curves use chunk offset 0 across recorded frames; X/Y/Z/finger show target, prediction and recorded state, followed by SO(3) and position errors.
- Each inference receives recorded RGB/wrist RGB/state/tactile history/task, with labels excluded. Chunk metrics use valid episode-boundary masks and compare against a hold-current-state baseline.
- A guessed launcher name `scripts/run_tabero_offline_eval.sh` did not exist; the documented launcher is `scripts/run_tabero_offline.sh`.

## Completed training and offline evaluation scope
- User-run real_fr3_recovery_20260903_112020 completed 3000 updates on GPUs 0,1, batch 4; final checkpoint 2999 has params/assets/train_state and committed metadata.
- Validation (692 frames): 1.068809 at start, 0.292337 minimum at update 2250, 0.335033 final. No 2250 checkpoint exists. Recorded updates were finite; root cause of earlier run's NaN still unproven.
- create_trained_policy uses checkpoint-local norm stats and reverses normalization then delta actions. Raw observations must be passed only once through these transforms; action labels must not enter inference or be mutated by DeltaActions.
- Existing inference output is [50,7] absolute xyz/axis-angle/single-finger meters. Offline evaluation is open-loop recorded-observation prediction, not simulated or physical rollouts.
- Offline runner uses raw LeRobot samples with actions_is_pad checked against episode bounds. Entire policy.infer call is timed through host synchronization, excluding dataset decoding and reporting first-call compilation separately.
- Opt-in strict_params prevents the policy factory from silently dropping extra JAX adapter leaves; other callers retain current defaults.
- Real-data CPU boundary checks passed after respecting DataConfig's disjoint split invariant and allowing the loader's cache lock outside the sandbox. Validation lengths are 222, 222, 249 (693 total); action padding matches 50 valid targets at each start and 1 at each final frame.
- Added 19 offline CPU regression cases; 59 selected tests pass overall. Actual pretrained action sampling, GPU memory/latency and physical task performance were deliberately not executed or claimed.

## Agent-operated recovery (latest authority)
- User now explicitly authorizes actual training and result analysis. Announced using the failing run's 26/3 split for diagnosis; no real robot access.
- Fresh host query: four A6000 cards idle (12-13 MiB used), no compute processes. /data has 582 GiB available, RAM 477 GiB available.
- Failed run `real_fr3_4gpu_3000_20260902_203541`: validation update 250 loss 0.42061, logged update 251 finite, aggregate update 261 NaN in loss/grad_norm/param_norm. No checkpoint saved (first scheduled save at loop index 500). Initial weights were the original smoke 99/params, not four-GPU check 19/params.
- Current train_step applies optimizer updates before the host finite check; logs aggregate every 10 steps. Need detect exact failing update and retain healthy state, without silently skipping bad batches.

## Successful user run and multi-GPU follow-up
- User completed run `real_fr3_tactile_lora_r16_20260902_195449`: 100 updates, 24-frame validation loss 2.3630917 -> 1.0073332, checkpoint 99 committed. Prior read-only verification confirmed source Tabero 86 leaves including 20 existing LoRA leaves; 16 new tactile adapter leaves initialized. This supersedes older preparation-only status entries below.
- JAX code uses a single process and mesh `(device_count // fsdp_devices, fsdp_devices)`. With 4 visible GPUs and fsdp_devices=1, parameters replicate and global batch 8 shards to 2 examples/GPU. No torchrun; data loader explicitly rejects multiple JAX processes.
- Global batch 8 yields 745 complete batches from 5962 train anchors; drop_last=True omits 2 anchors per shuffled pass. 1000 updates roughly 1.34 dataset passes, not a declaration of convergence. Original 100-step run already sampled from all 26 training trajectories, not a single trajectory.
- Existing validation drops incomplete batches. batch=8, eval_num_batches=86 evaluates 688/693 frames, not the full validation set; previous batch=2, eval_num_batches=12 used only 24. Metrics across these sample/noise settings are not strictly comparable.
- Longer runs need matching lr_schedule.decay_steps (old default 100). A new experiment with weight_loader.params_path pointing to smoke 99/params is a weight-only warm start with fresh optimizer/schedule, not --resume.
- No standalone offline action-evaluation script exists. Policy creation can check checkpoint reload without robot connection; physical action errors/rotation geodesics/tactile ablations require offline evaluation implementation or explicit diagnostic code, not an invented CLI.
- User's data audit reports nonzero tactile history-shift and initial-repeat errors, plus action startup/rotation jumps. Nonzero history errors do not alone prove corruption: confirm acquisition frequency, reference semantics and export contract before altering data or long training.

## Native terminal crash investigation (2026-09-02)
- Both helper scripts import and exit successfully with `--help` in the installed environment.
- Read-only CPU probe of episode 0 passed: 201 Parquet rows, expected finite state/action/tactile shapes, both videos decoded to 201 frames.
- Neither the requested audit directory nor output assets exists. Preparation has not completed.
- `/var/crash/_usr_bin_bash.1009.crash` records `/usr/bin/bash`, signal 6, cwd this repository, dated 15:23 today; the reported incident's exact time is not yet correlated.
- Installed Bash is 5.1.16, Ubuntu package 5.1-6ubuntu1.1. VS Code ptyhost logged missing pty 5 at 18:24, without a native stack.
- GNU has an upstream report of Readline display-buffer corruption causing the same allocator message: https://lists.gnu.org/archive/html/bug-bash/2022-09/msg00049.html . This alone does not prove the current trigger.
- Actual local core disproves the Readline hypothesis for this crash: `record_history_cmdline -> record_cmdline_core (payload_x86_64_norm.c:677) -> free -> malloc_printerr("free(): invalid pointer") -> SIGABRT`.
- Faulting code is `/usr/lib/libpayload.so` (`/lib/libpayload.so` on merged /usr). Debug symbols identify a `uhs/agent/go/clib/bashInjector/staticInjector/normInjector` build path and `/var/lib/UHS/historyOrder.log`; exact vendor/installation provenance is unknown.
- `dpkg -V bash` reports `/bin/bash` content checksum mismatch; the payload library is not owned by a dpkg package. Both files have Aug 28 15:19 timestamps.
- Frame 9's `loginName` is `"-"` at an address in the library's read-only segment. Disassembly shows it is passed directly to `free` at the failing site: a non-heap fallback string is being freed.
- Do not alter or disable the host command-recording component. Gather a minimal reproduction and recommend maintainer repair.
- Minimal PTY probe reproduced the exact error without Python or VS Code: single-line echo exited 0; backslash/newline echo and bracketed-paste variant both terminated with SIGABRT (-6). A `bash --noprofile --norc -i` child was sufficient; the payload was not disabled.
- The identical continued echo in a noninteractive script file exited 0. Added `scripts/run_tabero_checks.sh` to execute the normal checks from a file with one terminal invocation, CPU-only environment, faulthandler, timestamped logs and stop-on-failure behavior.
- Launcher syntax/help and `git diff --check` passed. Full preparation/batch inspection was not launched, and no system fixes were applied.

## Tactile adaptation follow-up
- User approved adding backbone LoRA + tactile encoder LoRA while retaining the frozen tactile baseline; training remains user-operated.
- Identical marker tensor shape does not establish matching sensor geometry, orientation, scale, sensitivity or reference-frame semantics. Train-only normalization does not resolve those differences.
- Current strict LoRA filter freezes the plain-linear tactile TCN. Its approximately 65.24M base parameters should remain pretrained/frozen; new zero-output adapters should permit task adaptation without force targets.
- Newly introduced tactile adapter leaves cannot exist in the original checkpoint. Restoration must permit only these new leaves while continuing to require every pretrained base and backbone-LoRA leaf.
- TCN contains eight linear maps: six temporal kernels, one first-block residual projection and one output projection. Rank 16 adds 779,008 parameters (16 leaves), versus 65,239,040 frozen tactile base parameters.
- Training restores into a ShapeDtypeStruct tree, strips placeholders, then initializes the model and overlays checkpoint arrays. Allowed missing adapters must retain placeholders until initialization; they must not synthesize zero arrays for both LoRA factors.
- Design: opt-in nnx.Linear subclass retains kernel/bias paths; A is random, B is zero, scale alpha/rank. Old rank-zero factory returns nnx.Linear unchanged. New config permits only exact known tactile LoRA paths, and requires original backbone LoRA.
- New `pi0_lora_tacfield_local_tactile_lora_smoke` uses rank/alpha 16, unchanged data split, loss, LR and baseline. Strict loader accepts either all new adapters absent (published checkpoint) or all present (adapted checkpoint); partial presence is rejected to avoid resetting learned adapters.
- Assets are configuration-specific (`assets_base_dir/config_name`), so the new run must run preparation with its own `--config`; baseline statistics directory is not automatically reused.
- Abstract shape test verifies 779,008 tactile LoRA and 65,239,040 tactile base parameters from the actual configuration. No full-sized weights were allocated.
- New configuration CLI `--help` succeeds. Selected CPU suite is now 49 passed, 2 tokenizer tests deselected; pretrained restoration and actual convergence still untested.

## Requirements
- Current A6000 workstation only; pretrained NathanWu7/pi0_lora_tacfield_tabero, continuing existing LoRA.
- Inputs: image, wrist_image, state, tactile_marker_motion. Ignore tactile_depth and wrist_wrench.
- Seven absolute end-effector target dimensions (xyz, axis-angle, single-finger position in meters), not seven joint angles.
- 29 episodes / 6655 frames; split 26 train and 3 validation by whole episode. Train-only new normalization.
- Do not confuse finite training loss with physical task success or causal tactile benefit.

## 2026-09-08 shadow deployment diagnosis
- Both runs passed server metadata/conversion validation and continuously produced predictions, so ROS image/tactile subscription, HTTP state reads and WebSocket inference were functioning.
- `shadow_006`: 115 evaluated ticks, 3 independently valid and 112 invalid. Independent guard failures: 110 gripper, 48 translation, 4 rotation; one late result terminated the run.
- `shadow_007`: 298 evaluated ticks, 8 independently valid and 290 invalid. Independent guard failures: 276 gripper, 151 translation, 12 rotation and 10 workspace. Guard ordering masks 139 translation failures behind the earlier gripper check.
- Measured single-finger position stayed at 42.5 mm. Median predicted positions were 42.6725 mm (`006`) and 42.6340 mm (`007`), showing small unbounded model overshoot rather than a total-width/single-finger factor-of-two mismatch. Clamping this alone would expose many pose failures.
- Translation error had p50 40.7/50.2 mm and maxima 107.4/409.3 mm for `006`/`007`; these are substantive stochastic policy-output excursions, not rounding tolerance.
- Tactile tensors remained finite `[9,198,2]` with nonzero motion summaries. No sensor age/skew/shape/schema error appeared.
- The controller accepted a new chunk about every two 10 Hz ticks and selected chunk indices 2/3. `shadow_006` eventually received a candidate older than the 0.35 s acceptance limit. `shadow_007` completed 30 s, but reused accepted chunks at ages up to 0.391 s because the age check currently runs only when a future completes.
- CycloneDDS `NetworkInterfaceAddress` deprecation and missing type-hash messages are startup compatibility warnings in this run; discovery and samples still worked.

## 2026-09-08 asynchronous chunk diagnostics
- Training action timestamps begin at the observation timestamp and advance at 0.1 s for this 10 Hz dataset. Runtime selects floor(observation_age*10), so shadow can compare a future absolute action against a robot that deliberately did not follow earlier actions.
- New `inference_chunk` rows store the accepted/rejected result, observation state and full 50x7 output once per completed inference. New `control_tick` rows store action[0], selected action[k], current state and SO(3)-aware physical distance decompositions, linked by `chunk_id`.
- `selected_vs_previous_tick` plus `chunk_switched_since_previous_tick` isolates block-transition discontinuity from within-block trajectory motion.
- New `policy_metadata` rows retain checkpoint, config, normalization-stat hash and conversion hash. This identifies which assets were used but does not alone establish that the assets came from the intended training dataset.
- Deployment/service selected suite passes: 35 CPU tests. Ruff and whitespace checks pass. No ROS, model inference, network service or robot command was run.

## 2026-09-08 first enriched shadow log
- The inference host used checkpoint `/home/enine/tabero_deploy/checkpoints/2000`; its normalization hash `7491dc...acb7` exactly matches both local checkpoint-2000 training copies. The conversion hash also matches. A copied normalization-file mismatch is ruled out for this run.
- 15 chunks and 29 control ticks were captured over 2.8 s after warmup. All chunks arrived at about 269-271 ms and were accepted; runtime alternated indices 2/3.
- Action[0] position distance from its observation was 11.1-28.5 mm (median 20.3), with zero samples above the 50 mm target guard. Selected action[2/3] was 7.8-66.8 mm (median 35.7), with 6/29 above 50 mm. Shadow/future-index mismatch therefore explains a material part of position rejection.
- Current state differed from the inference observation by only 0.0007-0.0051 mm, confirming the arm was effectively stationary. Selected-vs-action0 median position difference was 17.4 mm.
- Consecutive chunks received almost identical observation states (median 0.002 mm apart), yet action[0] changed by median 14.6 mm/max 34.0 mm. At chunk switches the selected target changed by median 23.1 mm/max 50.7 mm, versus 5.2/15.8 mm within a chunk. Stochastic block replacement is a separate continuity concern.
- 14/15 chunks had action[0] gripper above 42.5001 mm. This is independent of asynchronous index selection and remains the dominant first guard failure, although the overshoot is small.
- The run stopped because multimodal capture timestamps exceeded the configured 100 ms skew after 2.8 s. The old message does not identify the stream; enriched per-stream age diagnostics were added for the next run.

## 2026-09-08 complete 30-second enriched shadow log
- The two supplied JSONL attachments are byte-identical. One contains 149 accepted chunks and 297 complete control ticks over 29.626 s; its final pasted line is truncated, but the terminal completed without a traceback.
- Unique chunk action[0] position error: median 23.18 mm, p95 61.35 mm, max 142.86 mm; 13/149 exceed 50 mm. Selected action[2/3]: median 43.93 mm, p95 90.53 mm, max 377.38 mm; 130/297 exceed 50 mm. Async future-index selection materially worsens the distance, while action[0] itself still has unsafe outliers.
- Observation state changed only a median 0.0021 mm from the current measured state, confirming shadow stationarity. Selected-vs-action0 median was 19.44 mm.
- At chunk switches, selected targets changed by median 31.35 mm and max 349.13 mm, compared with 6.51 mm median within a chunk. Nearly identical adjacent observations (median 0.0022 mm apart) produced action[0] values changing by median 19.97 mm and max 131.84 mm.
- Action[0] independently violates: gripper 133/149, translation 13/149, rotation 4/149 and workspace 6/149. Guard ordering hides many pose failures behind the gripper error. Only 2/297 selected ticks passed all checks.
- All 149 chunks arrived in 239-265 ms and were accepted; the control loop alternated indices 2/3. Left/right tactile summaries remained finite and nonzero.
- The earlier `stopped: "'left'"` is a startup KeyError from a missing left tactile first frame. The subsequent complete run proves the stream recovered. Added a startup helper that lists all missing streams explicitly.
- Both shadow start records send the old acquisition-description prompt `fr3 zed DM-Tac W SDK 0.1.4 raw teleop via hilserl franka_server`. That is an input-contract mismatch if the live server is the new-prompt checkpoint 2000 evaluated offline with the assembly instruction.
- The shadow guard distance is prediction-to-current-state, while offline `position_mm` is prediction-to-recorded-action. For a like-for-like prediction-to-state comparison at offsets 2/3, checkpoint-2000 offline medians are 15.0/15.8 mm and means 26.2/28.6 mm; shadow medians are 40.7/50.2 mm and means 45.5/54.4 mm. Thus metric mismatch explains part, but not all, of the observed gap.
- Runtime never used offset 0: ages were mainly 244--391 ms, selecting offsets 2/3. Offline prediction-to-state distance grows with horizon, and the known interior label/control alignment is closest around state[t+3], which can amplify live target jumps.
- The live measured pose was essentially fixed at `[0.30304, 0.01729, 0.41308]` m and rotvec `[2.2632,-2.1413,0.0039]`. Its orientation is at least 38.7 degrees from every held-out offline state, while XYZ is 15.8 mm from the nearest held-out state. The live starting orientation is therefore a strong state-distribution shift or frame/TCP-definition mismatch.
- Checkpoint-2000 offline already produced 10,727/34,650 gripper values above 42.5 mm and target-to-state jumps above 50 mm on 60/693 offset-0 anchors. The live failure is worse and more persistent, but it is an amplification of defects visible offline rather than a deployment-only unit conversion error.

## 2026-09-08 label and live-start follow-up
- Tactile startup absence was user-confirmed: the bridge had not been launched. The later complete run has finite nonzero left/right tensors, so no further missing-tactile investigation is needed.
- Across all 29 episodes, the constant stale-action prefix has median 10 frames and maximum 21 frames. After removing each detected prefix, recorded action-to-state position offset is p50 6.96 mm, p95 46.76 mm and max 91.31 mm; 3.714% exceed 50 mm. Current live action[0] remains worse at p50 23.18 mm, p95 61.35 mm and max 142.86 mm.
- The LeRobot exporter converts each source `action.pose7` row directly; it has no cross-episode forward fill. The stale values therefore already exist in raw episode Parquet. Exact localization requires the imported common collector that implements `record_episode`, which is absent from the current repository and attachments.
- Deployment used the acquisition-description prompt, while dataset `meta/tasks.jsonl` contains `Align the black circular component with the receiving hole on the gray circular component and insert it to complete the assembly.` Updated the default deployment config and docs to the exact training task.
- Across all 6655 training states, the live start orientation is at least 38.65 degrees away. Training episode starts have median XYZ `[0.306427, 0.000291, 0.414326]` m and mean rotvec `[2.900868, -1.201307, -0.005462]`, with orientation spread p95 0.143 degrees/max 0.510 degrees. Correcting this start/TCP mismatch is required before another shadow comparison.

## 2026-09-09 compact-v3 continuity audit
- V3 metadata states `actions[t] = state[t+1]`; element-level comparison confirms this for every row of all 29 episodes. V3 states exactly equal v1 states excluding each terminal row, and v3 has 6626 = 6655 - 29 frames.
- V1 action/state position p50/p95/p99/max is 7.484/64.509/423.359/456.515 mm with 527 frames above 50 mm. V3 is 1.163/14.718/18.511/45.501 mm with zero frames above 50 mm.
- V1 has 28/29 episode starts above 50 mm and 14 cross-episode exact action reuses. V3 has zero of either. The stale position-label issue is fixed.
- V3 still has 114/6626 same-frame 2pi rotvec component branch crossings despite a physical one-step rotation maximum of only 1.225 degrees. All 29 episodes are affected. Current component-wise DeltaActions remains incompatible with smooth SO(3) semantics.
- On 26 training episodes, v1 action0 has 482/5962 position targets above 50 mm and max 456.515 mm. V3 action0 has zero above 50 mm and max 45.501 mm. This makes v1 corruption a plausible major cause of learned position outliers, but a controlled retrain/eval is required to establish causality.
- V3 retains 12 compacted episodes/18 missing source intervals. Its largest 45.501 mm transition occurs in compacted episode 17. Exact association with a missing interval cannot be proven from rewritten uniform timestamps.
- 2726/6626 (41.1%) v3 one-step targets exceed 2 mm per 100 ms, while deployment limits translation to 0.02 m/s = 2 mm per 100 ms. Async execution would frequently lag unless control timing/rate is reconciled.

## 2026-09-09 v3 checkpoint 4000 versus 20000 offline evaluation
- Both evaluations completed and are exactly paired: 690 anchors from episodes 4/14/24, identical episode/frame, state, target and valid arrays, v3 dataset/norm hashes, seed 42, ten denoise steps and horizon 50.
- Checkpoint 20000 improves all six overall mean physical metrics versus 4000. First-action position/rotation/gripper improve 5.652->4.623 mm, 0.1704->0.1632 deg and 0.553->0.442 mm. Valid-chunk metrics improve 41.559->38.622 mm, 1.2616->1.2265 deg and 1.575->1.405 mm.
- Paired first-action 20000 win rates are 72.5% position, 60.0% rotation and 67.0% gripper. Valid-slot win rates are 56.5%, 52.0% and 60.6%, so the chunk improvements are real in this fixed evaluation but less uniform.
- Position improves at every inspected horizon offset; offset 49 improves by 6.15 mm. Per-episode first-action means improve for all three episodes and all modalities. Chunk position improves by 6.39/7.43 mm on episodes 4/14 but worsens by 4.05 mm on episode 24; episode-24 chunk position P95 worsens 106.20->124.06 mm.
- Checkpoint 20000 reduces first target-to-state mean/max 4.716/40.854->3.389/22.855 mm, gripper out-of-range values 58->47, and within-chunk max position step 48.32->31.90 mm. Both have zero first or within-chunk position jumps above 50 mm and no nonfinite outputs.
- Long-tail caveats: checkpoint 20000 has a 7.856 deg deep-horizon rotation maximum versus 5.487 deg for 4000, at episode 24 frame 122 offset 42. It also has an isolated episode-24 frame-29 offset-49 position error 179.79 mm versus 81.65 mm for 4000, although overall >100 mm position slots fall from 2010 to 1614.
- Hold-current-state is still better for the one-step target on all first-action means (3.627 mm, 0.1264 deg, 0.0806 mm), while checkpoint 20000 is substantially better over the full valid chunk than hold (38.62 vs 73.29 mm position, 1.226 vs 1.601 deg rotation, 1.405 vs 1.861 mm gripper).
- Training validation loss gives the opposite ordering: minimum 0.29023 at 4000 and 0.44452 at 20000 after a sustained post-4000 increase. This establishes degradation of the normalized flow-matching proxy, not degradation of this fixed-seed sampled physical metric. Multi-seed inference and intermediate 8k/12k/16k offline evaluations remain necessary to assess robustness.
- Selection: among the two evaluated checkpoints, use 20000 for the next guarded shadow because it wins the broader physical metrics; retain 4000 as fallback and do not infer real-robot success from open-loop evaluation.

## Prior Read-only Findings
- Repository: /home/yanghaojun/Tabero-VTLA; dataset: /data/yanghaojun/datasets/tabero_lerobot_compact_v1.
- Dataset is LeRobot v2.1, 10 Hz, 29 Parquet files / 58 mp4v videos; state/actions [7], tactile_marker_motion [9,198,2].
- Current config pi0_lora_tacfield_tabero uses pi0_base initialization, 13 effective dimensions (7 control + 6 force), padded model width 32; needs explicit change.
- Current loader lacks root/episode selection; train.py lacks held-out evaluation.
- Default LoRA filter freezes non-LoRA LLM parameters but not all other modules. Must choose and report trainable scope explicitly.
- Prior GPU query: four A6000 48GB cards, two with about 26GiB free; recheck before allocation.
- Prior inspection found no project environment/checkpoint; system Python 3.10, project requires >=3.11.
- Hugging Face metadata lists checkpoint step 49999 with params and train_state. params tree has 86 leaves, including 20 LoRA and 16 tactile-encoder leaves plus backbone. It is not adapter-only. Metadata write_shape is a shard shape, not necessarily global tensor shape.

## User Handoff: Risks to Recheck
- 28/29 episodes reportedly have early large action-position jumps; do not silently crop or relabel.
- Axis-angle branch jumps near pi can generate misleading component losses.
- Local tactile tensor is eight historical frames plus current; checkpoint config enables reference-frame handling. Verify semantic compatibility before training.
- Task text is a generic acquisition description, not a meaningful manipulation instruction.

## Resources
- https://huggingface.co/NathanWu7/pi0_lora_tacfield_tabero
- src/openpi/training/config.py, data_loader.py, weight_loaders.py
- src/openpi/models/pi0.py, pi0_config.py, tactile_encoder.py
- scripts/train.py, compute_norm_stats.py

## Implementation Discovery
- Source commit: 1ed9cf44c05bc63fa3b3dbc0ca83ce9dbc8b7b2e.
- GPU 2 and 3 are now idle (~12 MiB each); GPU 0/1 actively used. Prefer GPU 2 after a fresh check.
- Available RAM ~429 GiB. Use /data for environment, cache and artifacts to avoid filling home filesystem.
- TCN has_reference_frame=True, diff_from_reference=False passes all 9 frames unchanged; no subtraction or baseline removal occurs. Local rolling 9-frame history can preserve checkpoint tensor architecture; temporal meaning differs from a fixed baseline and will be documented.
- Current DeltaActions subtracts first six state components; no SO(3) unwrap/geodesic conversion in this path. Keep established transform for initial pipeline smoke and report branch risk, do not relabel raw data.
- A strict LoRA-only freeze filter can freeze every non-LoRA parameter, including vision/tactile/head. This satisfies the user's LoRA request, reduces memory and preserves learned tactile representation.
- Implemented smoke split validation=(4,14,24), training=all other episodes. Validation samples evenly span concatenated held-out episodes; bounded smoke evaluation is not a full benchmark.
- New supervised_action_dim=7 branch computes only labeled flow-matching error and does not evaluate fictitious force targets; architecture remains 32D.
- Strict checkpoint loading rejects missing, extra or shape-mismatched leaves. Non-LoRA parameters all frozen.
- Existing pyproject package name tabero-vtla did not match src/openpi and Hatch wheel autodiscovery failed; explicit package selection added.
- PyAV 14.4.0 source build failed (missing libavdevice/libavfilter headers). PyPI https://pypi.org/project/av/14.2.0/ lists a CPython 3.11 manylinux x86_64 wheel; pinned 14.2.0, which satisfies LeRobot >=14.2.0, without touching system FFmpeg.
- Important upstream LeRobot v0.3.3 bug: get_episode_data_index builds subset-relative offsets, but _get_query_indices indexes them with original episode_index. Non-contiguous 26/3 selection would misindex/cross episodes. Added local subclass translating original IDs to subset positions, plus synthetic boundary regression test.
- Prune unused columns from the loaded Arrow view so action-chunk queries do not materialize huge depth arrays. Original Parquet remains unchanged.
- User superseded execution authorization: code only, final target real FR3. No inference/training/download of model weights occurred; installer stopped (exit 130).
- New dedicated real-robot adapters validate state/history shapes and output finite float32 [horizon,7]. They do not command hardware or replace robot-side workspace/rate/collision/deadman checks.
- Full new-file Ruff checks initially found import order/test style warnings, corrected. Core Ruff selection falsely flagged existing jaxtyping annotations (F722, already ignored by project config) and an existing unused pi0_fast import; use project ignore rules for core checks.
- Latest user instruction permits completing dependency installation and unit tests, but reserves training for manual operation. Added standalone CPU batch inspection command (no model construction) for the manual runbook.
- tmux is already installed at /usr/bin/tmux; documented user-created persistent SSH session. CPU test environment flag must stay per-command so training is not accidentally forced onto CPU.
- Latest GPU snapshot after installation: all four cards have active jobs; GPU 0/1 ~22GiB used of 48GiB, GPU 2/3 ~47GiB used. Do not tell user GPU 2 is still free. No GPU model workload was launched by this task.
- train.py pi0_lora_tacfield_local_smoke --help exits 0; CLI accepts --exp-name, --batch-size, --num-train-steps and other documented flags without creating a run.

## 2026-09-03: NaN and manual recovery
- The failed 3000-step run has no saved checkpoint; the first save was scheduled at loop index 500. Its step-250 validation loss was finite (~0.42061), followed by NaN in the step-260 aggregate. This does not localize the first invalid update or establish its cause.
- Confirmed the earlier healthy 99/params and subsequent four-GPU 19/params directories still exist. The recovery launcher defaults to the former (same source as the failed run); it starts a fresh optimizer/schedule, not exact resume.
- Current config still uses 26 training episodes and validation episodes (4, 14, 24), not all 29 for optimization. With 1/2/3/4 GPUs, global batches are 2/4/6/8; bounded validation covers 692/692/690/688 frames of 693 respectively.
- GPU availability has changed; no old snapshot should be treated as a reservation. Manual selection is required, and the launcher checks that JAX sees exactly the selected GPU count before training.

## 2026-09-09: Synchronous FR3 control and prediction saturation
- Finite model predictions are saturated at the XYZ workspace, target translation radius, SO(3) target rotation, single-finger position, and per-cycle translation/rotation/gripper rate limits.
- Measured workspace/gripper violations, command-tracking error, non-finite values, stale sensors/results, deadman loss, and HTTP failures remain stop conditions because rewriting a prediction cannot repair those faults.
- The deployment loop performs one blocking inference per fresh observation and always selects action[0]. The full 50-step chunk remains in JSONL for diagnosis only.
- With `control_period_sec=0.10`, inference slower than 100 ms lowers the actual update rate; no catch-up commands are emitted. Result age remains capped at 350 ms and enable is rechecked before actuation.
- Logs preserve raw, bounded, and rate-limited actions plus every applied limit so saturation does not hide model quality problems.

## 2026-09-10: Repeated K=2 real-robot failure investigation
- User reports five unchanged K=2 deployments all failed to grasp or reach the target despite small-looking offline errors.
- Inputs are five complete-file attachments named `sync_k2_execute_003.jsonl` through `007.jsonl`; analysis must distinguish control/timing effects from perception/task/model generalization and from offline metric mismatch.
- All five traces have identical deployment metadata: checkpoint `20000`, config `pi0_lora_tabero_v3_touch_20k`, v3 assets, prompt, conversion hash, and normalization hash. Software/model consistency is verified; physical scene equality is not logged.
- Starts are demonstration-aligned: nearest v3 episode-start position `0.17–1.47 mm` and rotation `0.09–0.53°`. The earlier start-pose OOD diagnosis does not apply to these five runs.
- The robot moved and accepted all pose commands: net displacement `310–353 mm`, path length `511–727 mm`, and `5,875/5,875` pose commands returned `ok`. Tracking stayed below stop thresholds (position p95 `5.0–5.8 mm`, max `6.8–8.3 mm`; rotation max `1.37°`).
- Effective control frequency was `5.97–6.18 Hz` versus dataset `10 Hz`. Action 1 was executed for `2,620/3,255 = 80.5%` usable chunks; `634` chunks were truncated when action 1 became stale. This mismatch can alter the learned timing, but no paired K=1/K=2 same-scene test proves it caused failure.
- `3,653/5,875 = 62.2%` of control ticks were rate-saturated, mainly translation, so the executed path often differed from raw policy magnitude.
- Final states were `104–188 mm` and `10–20°` from the nearest expert terminal pose. Nearly identical starts led to final Cartesian positions differing by as much as about `111 mm`.
- Live trajectories left the demonstrated state manifold: nearest-demo-position p95 `25–53 mm`; final nearest-any-demo-state distance `19–57 mm` and rotation `8–20°`. This is evidence for compounding closed-loop error.
- Run 005 never meaningfully closed its gripper: single-finger position stayed `40.63–42.50 mm`, while v3 demonstrations reach per-episode minima `27.52–34.64 mm`. Other runs closed further and still failed, so this is one failure mode rather than the sole cause.
- Raw action-0 displacement was only `3.6–4.6 mm` on average, but action-49 mean displacement was `95–174 mm` with maxima `334–359 mm`; a small first-action statistic does not characterize the full horizon.
- Located the exact checkpoint-20000 offline artifact at `/data/yanghaojun/outputs/offline_eval/offline_v3_touch_step20000_full_20260909_200510/summary.json`; the older `offline_real_fr3_new_prompt_2999` results are a different model/run and must not be mixed into this diagnosis.
- The matching artifact is complete and explicitly labels itself a recorded-observation open-loop evaluation rather than a rollout/task-success measurement. It covers `690` anchors and `30,825` valid action targets. First-action mean error is `4.623 mm / 0.163° / 0.442 mm`, but full-valid-chunk mean error is `38.622 mm / 1.226° / 1.405 mm`, with position p95 `101.286 mm` and max `179.788 mm`.
- At action offset 1—the second K=2 action—offline mean/p95 position error is already `8.209/23.566 mm`; at offset 14 it is `32.129/82.312 mm`. Horizon error grows materially even under expert observations.
- The hold-current-state baseline beats the policy on the first action in all three mean metrics (`3.627 mm / 0.126° / 0.081 mm`). Therefore the small action-0 error mostly reflects slow 100 ms expert dynamics and is not evidence of useful task progress. The policy improves over hold on the full chunk, but that still does not test recovery from its own errors.
- Offline policy-call latency was about `148 ms` after compilation and explicitly excludes video loading, robot and network time; live end-to-end inference is about `192 ms`, explaining the observed control-rate reduction.

## 2026-09-10: Inference-host rewrite handoff inventory
- Current branch is `fix/fr3-tactile-shadow-deploy`, HEAD `53370e6`; the new handoff must describe the K=2 implementation present at this revision.
- Authoritative runtime config is `examples/fr3_deploy/config.json`: ZED compressed RGB, D405 raw RGB, packed left/right tactile, 100 ms control period, 250 ms sensor age, 100 ms sensor skew, 350 ms result age, and fail-closed enable/HTTP/tracking checks.
- Current prediction processing saturates finite outputs in this order: XYZ workspace, 50 mm target translation, 0.35 rad target rotation, single-finger `[0,42.5] mm`, then 20 mm/s translation, 0.10 rad/s rotation, and 20 mm/s total gripper-width rate. Measured-state/system violations remain errors.
- Current client supports only `actions_per_inference` 1 or 2; K=2 schedules indices 0/1 at 100 ms and re-reads measured state plus all guards before each action.
- Authoritative production config/checkpoint contract is `pi0_lora_tabero_v3_touch_20k` with v3 assets, exact assembly prompt, action horizon 50, and 7D absolute XYZ + axis-angle + single-finger action semantics.
- Model input contract from current code: front RGB must be `uint8 [540,960,3]` and is cropped by ROI `[350,0,740,520]` to `[520,390,3]`; wrist RGB is `uint8 [480,640,3]`; state is finite `[7]`; prompt is exact; tactile is contiguous `float32 [9,198,2]` with row 0 fixed left-then-right reference and rows 1–8 rolling frames sampled from two `240x320x2` shear maps.
- HTTP state must provide `pose=[x,y,z,qx,qy,qz,qw]`, total `gripper_width` in metres, and `stamp.to_sec`; policy state/action stores axis-angle and single-finger metres, so quaternion conversion and width division/multiplication by two are mandatory exactly once.
- WebSocket request is msgpack-numpy observation data; response must contain finite `actions [50,7]`. Server sends metadata first and the client validates protocol, representation, dimensions, dataset FPS, tactile layout, conversion SHA, and tactile usage.
- Runtime starts with a discarded warmup inference. Execute mode then requires a fresh `/tabero/enable` true heartbeat after warmup and throughout control. Command HTTP calls are bounded and never retried; shutdown only sends a best-effort measured-pose hold, which is not an emergency stop.
- Current RGB/tactile JSONL records only sensor stamps indirectly and a tactile summary; it does not store the synchronized sensor payload. The rewrite should make diagnostic sensor capture an explicit bounded option.
- Robot-side Python requirements are deliberately small (`numpy`, `scipy`, `requests`, `opencv-python`, `websockets`, `msgpack`) and use ROS system packages; the model server keeps the repository JAX/Python 3.11 environment. A rewrite should preserve this process/environment boundary even if both run on one host.
- Current GitHub deployment remote is `deploy` at `Leohahahahha/Tabero_serve`, branch `fix/fr3-tactile-shadow-deploy`. Relevant baselines are `badb9ef` (v3 config), `7489594` (K=2 synchronous execution), and `53370e6` (K=2 timing record).
- Added `docs/tabero_fr3_inference_rewrite_handoff.md` with the frozen contract, recommended module/process boundaries, state machine, K=1/K=2 semantics, exact safety behavior, five-run evidence, offline-metric explanation, missing diagnostics, acceptance sequence, host commands, and delivery checklist.
- Updated the interview incident overview, corrected the latest start-pose and K=2 statuses, and added incident 28. Remaining hypotheses are explicitly separated from verified findings.
- The artifact evaluates only held-out episodes `4`, `14`, and `24` (`690` anchors total), from a 29-episode dataset. This is a narrow expert-state validation distribution.
- Deployment JSONL records topics/metadata, state, complete action chunks, bounded commands, tracking and a tactile marker, but not the actual front/wrist pixels or tactile tensors. Therefore camera crop/exposure/object placement and tactile semantic equivalence cannot be checked from these five files and remain hypotheses.

## 2026-09-17: Four-checkpoint whiteboard offline evaluation
- The four requested finalized checkpoints are rank-32 sent-command 12000, rank-32 next-state 12000, full-parameter SGD next-state 12000, and rank-16 next-state 12000 from the 20k schedule.
- Validation episodes are fixed at `(1,7,17)` with lengths `274/315/412`, giving 1001 recorded-observation anchors per model.
- All four model/data/checkpoint contracts pass CPU boundary checks for finite `actions[50,7]`, `wrist_wrench[50,6]`, episode identity and terminal padding.
- Sent-command and next-state models require their own matching target datasets. Their raw absolute errors cannot isolate architecture quality because the action-label distributions differ materially.
- The new evaluator reports action position/rotation/gripper and K-frame force/torque errors, baselines, horizon curves, per-episode plots and reproducibility hashes. No wrench prediction is sent to the robot.
- Initial sandboxed GPU probing could not reach the driver; later host-authorized access found four idle A6000 GPUs and the full evaluations completed successfully.

## 2026-09-18: Mixed full-finetune 20k versus LoRA32 12k
- Both mixed full-finetune step-20000 checkpoints completed a 1001-anchor GPU evaluation on held-out episodes `(1,7,17)` with seed 42, 10 denoise steps and matching checkpoint-local stats.
- For next-state, mixed full-finetune 20k improves over LoRA32 12k in all mean action metrics: first action position/rotation/gripper by 11.5%/12.0%/19.3%, and valid chunk by 18.3%/12.8%/25.2%.
- Next-state wrench force/torque L2 improves by 25.9%/21.5% on the first prediction and 15.4%/15.7% across the valid chunk. Gripper out-of-range predictions fall from 186/46375 to 14/46375.
- The wrench advantage is consistent in mean error: Full-FT 20k beats LoRA32 12k on force and torque for every episode and every one of the 50 prediction horizons. The valid-chunk force P95 is nearly tied at 10.873 versus 10.895 N, so rare high-force-error cases remain despite the lower mean and maximum.
- The improvement holds on all three episodes for both first-action and chunk action metrics; episode 7 shows the largest chunk position improvement, from 55.674 mm to 35.772 mm.
- Mixed full-finetune trains every parameter leaf, while LoRA32 trains 51,545,600/3,354,833,168 parameters (1.536%), including 1,558,016 tactile LoRA parameters. The comparison also differs in steps and LR schedule, so it does not isolate tuning scope as the cause.
- Training validation loss for the mixed run is lowest around 4k/8k and worsens through 20k. This suggests possible late overfitting, but only same-run 4k/8k physical evaluations can select the best checkpoint.
- Direct NPZ alignment is exact: episode, frame and valid arrays are equal, and target/state values have zero maximum absolute difference. This permits pointwise overlays without interpolation or resampling.
- The combined figure uses `prediction[:, 0]` and `target[:, 0]`, matching the evaluator's first-action definition; it visualizes recorded-observation open-loop predictions rather than a closed-loop rollout.
- The aligned wrench figure likewise uses the first predicted wrench at each observation, overlays all six K-frame components, and adds per-frame force/torque L2 errors for the two checkpoints.

## 2026-09-15: Whiteboard wrist-wrench prediction
- The initial paired configs were intentionally action-only: they queried only `actions`, supervised the first 7 model slots and assigned zero loss to force/padding. A retained Parquet field alone did not enable force prediction.
- Both whiteboard exports contain finite float32 `wrist_wrench [12126,6]`; the arrays are exactly equal between exports. Global min/max are `[-8.646,-11.335,-6.167,-2.104,-2.822,-0.786]` and `[10.021,6.878,12.954,2.149,3.834,1.282]`.
- The user confirms the order is `Fx,Fy,Fz,Tx,Ty,Tz`, force units are N, torque units are N·m and all components are expressed in the K frame. The dataset JSON does not contain this fact, so config/audit provenance labels it as a user-confirmed external contract.
- Dedicated transforms concatenate horizon-aligned dataset rows into a 13D target and later split inference output into `actions [50,7]` plus `wrist_wrench [50,6]`. XYZ/SO(3) relative conversion touches only the pose slots; gripper and wrench remain unchanged.
- The force configs retain the Tabero architecture and tactile TCN LoRA, restore published Tabero 49999 strictly, use `effective_action_dim=13`, a 0.1 wrench loss weight and zero padding loss. Their distinct `force` names prevent accidental reuse of earlier 7D assets or checkpoints.
- A real episode-0 LeRobot read produced `actions [50,7]` and `wrist_wrench [50,6]` float32 and the adapter produced `[50,13]` exactly. The robot-side deployment path now validates/logs predicted K-frame wrench but sends only the separately decoded 7D action through the existing safety controller.
- Selected CPU regression result: 106 passed in 21.80 seconds. No pretrained parameters, GPU model, W&B session, training, inference server, network service or robot command was started.
