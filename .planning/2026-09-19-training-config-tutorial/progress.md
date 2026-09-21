# Progress Log

## 2026-09-21 A/F/T consultation
Inspected local Pi0/Gemma/adapters and public MoT/mask/target-noising code. Recorded proposal and limits in incident 43; no model edits or training. A documentation patch used an out-of-order context and failed; reapplied with monotonic file locations.

## Session: 2026-09-19

### Current Status
- **Phase:** Complete
- **Started:** 2026-09-19

### Actions Taken
- Read the complete planning-with-files skill instructions.
- Restored the existing project planning state and inspected the dirty-worktree summary before starting.
- Created and activated the isolated plan `2026-09-19-training-config-tutorial`.
- Defined a source-driven beginner curriculum and durable documentation target; no training or model workload was started.
- Inspected the central config dataclasses, whiteboard LoRA/full-FT factories, Pi0 freeze-filter interface, optimizer factories, training-loop consumption path and recent Git history.
- Established the core teaching distinction: architecture, trainable scope, optimizer/precision and runtime orchestration are separate layers even when one named config bundles them.
- Verified exact backbone and tactile LoRA construction, initialization, scaling and freeze-filter semantics.
- Verified the two-GPU full-finetune launcher profile selection, CLI override path, sequential execution, logging, checkpoint and completion behavior.
- Completed the source call-chain and Git-diff audit for config, precision, optimizer and training-loop changes.
- Chose the lesson sequence: mental model -> Python syntax -> central config -> LoRA -> full tune -> precision/optimizer -> runtime overrides -> safe inspection and exercises.
- Added the durable 21-section Chinese tutorial `docs/tabero_training_config_tutorial.md`.
- CPU-instantiated the mixed-BF16 full-finetune config and verified batch=2, steps=12000, FSDP=2, AdamW FP32 moments, mixed parameter overrides and 36/3 episode split.
- Ran the documented launcher `--dry-run` with a 20,000-step override and verified that both generated commands use 20,000 steps, 2-device FSDP, global batch 2, AdamW 2e-5 -> 2e-6 and W&B.
- Ran focused configuration, launcher and tactile-LoRA unit tests; all 46 passed.
- Ran `git diff --check`; no whitespace errors. Existing unrelated incident-log and untracked artifact changes were preserved.

### Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| CPU config introspection | Effective fields match documented mixed-BF16 full-tune config | All key fields matched | PASS |
| Launcher dry run | CLI target steps override named-config default without training | Both commands used 20000 steps | PASS |
| Focused pytest suite | Config, launcher and tactile LoRA behavior remains valid | 46 passed in 13.63s | PASS |
| `git diff --check` | No whitespace errors | No output/errors | PASS |

### Errors
| Error | Resolution |
|-------|------------|
| Direct execution of `init-session.sh` returned permission denied | Re-ran it via `bash`; isolated planning files were created successfully. |
| Initial findings patch targeted a heading that did not exist | Re-read the active planning files and applied the update under the actual heading. |
# 2026-09-19: Low-LR continuation audit

- Re-read the planning skill and active plan before making the continuation decision.
- Verified the completed next-state run configuration and the finalized `20000` checkpoint.
- Confirmed that the final checkpoint includes full train state and that W&B ended cleanly.
- Inspected source resume semantics: total steps are absolute, and the optimizer schedule counter is restored.
- Found that blindly changing cosine decay to 40,000 on `--resume` is not a valid phase restart.
- Inspected validation evidence: the recorded validation action loss is materially worse at 20,000 than around 7,000, so recommended continuation must include an overfitting warning and early comparison checkpoints.
- Encountered missing `jq`; no installation is necessary because metrics can be parsed with existing Python.
- Selected a separate stage-2 run initialized from the original `20000/params`, with fresh AdamW moments and cosine LR `2e-6 -> 2e-7`; this leaves the original model and W&B run untouched.
- Verified the complete CLI with Tyro without starting training. Parsed values include two-device FSDP, global batch 2, 20,000 stage-2 updates, no warmup, W&B enabled, and image upload disabled.
- Verified schedule values at stage-2 steps 0/4000/8000/12000/16000/20000: approximately `2.00e-6/1.83e-6/1.38e-6/8.22e-7/3.72e-7/2.00e-7`.
- Updated `docs/tabero_interview_incident_log.md` overview and added incident 42 with diagnosis, hypotheses, solution, validation scope, and interview summary.
- Codex's isolated execution namespace could not access the NVIDIA driver, so actual GPU availability must be checked in the user's terminal before launch. No GPU training was started.

# 2026-09-21: Sensory expert literature consultation

- Applied oral-paper-skill to full-text MoT component ablations and scoped MoSS/TA-VLA evidence. Recorded matched comparison plan, WAM claim limits and simpler synchronous runtime interface in incident49; no model implementation or experiments.

- Reviewed user slow/fast diagram; inspected TA-VLA actual history MLP and N0 shared-attention/mask implementation. Documented transfer boundaries and prediction-under-action-intervention limitation in incident48. No model edits or training performed.

- Follow-up: checked direct shear generation against current Pi0 flow sign and primary-paper evidence. Revised first baseline to direct396; latent remains an optional experiment. Documented incident46 and attention-cost equivalence at fixed token count/width. No model changes or training.

- Checked MoSS loss definitions, CGP latent sizes, N0-TWAM expert configuration and local GQA shape constraints.
- Recorded conditional dimension recommendations and incident45; no model edits or training performed.
- One supplementary PDF failed to open; conclusions use accessible primary sources instead.

# 2026-09-20: Completed stage-2 validation-regression diagnosis

- Verified that the requested stage-2 next-state run completed normally and finalized all five planned checkpoints.
- Parsed both runs' JSONL metrics and aligned stage-2 step 0 with the source stage-1 step 20,000 checkpoint.
- Confirmed exact validation baseline reproduction, excluding an obvious source-checkpoint, split, or normalization mismatch.
- Quantified a 5.14% stage-2 validation action-loss regression and a 40.30% regression from the overall stage-1 minimum, while sampled training loss remained lower than the preceding stage.
- Found no numerical failure or checkpoint error. The strongest supported diagnosis is overfitting/generalization degradation, with validation-split variance retained as a limitation because only three validation episodes exist.
- Confirmed that no stage-2 physical offline evaluation artifact exists yet, so stage-2 20k must not replace the physically evaluated stage-1 20k candidate solely because it is newer.
- Updated incident 42 and the overview table with the completed training evidence, diagnosis limits, and checkpoint-selection recommendation.
