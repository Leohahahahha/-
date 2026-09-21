# Progress Log

## Session: 2026-09-16

### Current Status
- **Phase:** Complete — awaiting user-operated GPU launch
- **Started:** 2026-09-16

### Actions Taken
- Read the planning-with-files instructions, created an isolated plan and recorded the user constraints.
- Inspected the dirty-worktree summary and preserved all existing changes.
- Began auditing model freeze, sharding, optimizer and whiteboard config paths; no training was launched.
- Confirmed that a strict Tabero-49999-compatible full tune must retain the checkpoint's Gemma LoRA architecture while setting the freeze filter to `nnx.Nothing`; selected tactile rank 0 so the pretrained base TCN is trained directly without random adapter leaves.
- Selected two-device FSDP and global batch 2 as the conservative 2x48 GiB starting topology; abstract memory sizing is next.
- Abstract sizing completed: 3.353B trainable parameters and about 18.8 GiB per GPU for sharded parameters plus Adam state, before gradients/activations.
- Added two isolated full-parameter configs and a two-GPU sequential launcher with strict Tabero 49999 restore, direct full-TCN adaptation, 12k schedule, 4k atomic checkpoints, W&B grouping and resume support. No training was started.
- Added focused CPU contracts for all-parameter filters, strict initialization, dataset/assets isolation, two-GPU enforcement, hyperparameters and sequential execution. The first focused run passed 6/6; the new test file only needs formatter normalization.
- Formatted the new test and completed the combined old/r32/full whiteboard suite: 26/26 pass. Bash parse, Python compile, Ruff format, diff whitespace, real launcher dry-run and exact abstract trainable-count checks all pass.
- Updated the repository incident overview and added incident 34 with the full-finetune memory/initialization diagnosis, fix and unverified GPU boundary.
- Host preflight confirms GPUs 1 and 2 are free, GPU 0 continues the LoRA run, and the proposed full-FT pair/log/checkpoint paths do not exist. No full-FT training, W&B run, normalization write or robot operation was started.
- Diagnosed the user's real launch. Full-FT W&B run `htwauln8` was created and uploaded configuration plus validation step 0, but the first training update OOMed on both GPUs before any checkpoint. The pair correctly recorded `failed/train_next_state`, exit code 1.
- Reconciled LoRA status: next-state successfully finalized checkpoint 12000 and synced W&B, then sent-command automatically started and reached finite training updates. The apparent W&B crash/stale chart is caused by resuming the same run ID below its already-recorded step 12001; W&B ignored the recomputed 8000–12000 history.
- Added a scheduled, gradient-clipped stateless SGD optimizer and two new collision-safe full-FT SGD configs. Updated the two-GPU launcher to use them, LR `1e-5→1e-6`, and a 0.97 XLA memory fraction. The failed AdamW configs/run remain intact for provenance; no new GPU training was started.
- Verified both new configs abstractly: all 3,353,275,152 parameters remain trainable, optimizer state is 4 bytes, and sharded per-device persistent state is about 6.254 GiB. Combined whiteboard suite passes 26/26; compile, format, Bash parse, dry-run and diff checks pass. GPUs 1/2 and the new `full_ft_sgd_whiteboard_20260916` paths are free.
- Audited the real stateless-SGD run through logged step 990. Updates remained finite but five windowed action-loss means stayed around 1.76--1.85 and gradients were continuously clipped from norms near 14--15 to 1.0; no step-1000 validation was committed, so this is an early optimization warning rather than proof of non-convergence.
- Assessed three-GPU AdamW abstractly: per-device persistent state is 16.910 GiB, with 4.418 GiB replication overhead because several large tensors are not divisible by three. A real first-step fit remains unverified and high-risk.
- Detected a separate runtime interruption: the SGD process disappeared after step 990 and `nvidia-smi` lost communication with the driver. No Python traceback or numeric checkpoint exists; driver reset/failure is a hypothesis pending host/Xid evidence.
- Confirmed the same host interruption ended LoRA sent-command near step 4811. Its complete step-4000 checkpoint is recoverable; SGD has no checkpoint because it stopped before the first 4000 save.
- Reconciled a later full-SGD pair run: next-state completed 4000/8000/12000, sent-command then launched and failed at step-0 validation with NaN action/total loss before any SGD update. W&B run `88cd9jmz` is reporting the local failure correctly.
- Verified the sent-command norm stats and sampled train/validation loader batches are finite on CPU. The first cache attempt failed only because the sandbox could not create a lock in `/data`; rerunning with `/tmp/tabero-hf-check` succeeded.
- Scanned all 500 validation batches for both full-SGD configs. Every loader tensor is finite, and sent-command action magnitudes are comparable to the successful next-state case. The remaining failure is inside model forward computation or the unhealthy GPU runtime; no GPU replay was attempted while `nvidia-smi` remains unavailable.
- Compared the peer's reported 33 GiB two-card DDP setup with this run. The optimizer scalar hyperparameters mostly match and learning rate cannot explain fit; this JAX full tune stores all trainable leaves in FP32, making a DDP lower bound about 49.968 GiB/card before activations. Documented the required peer-side dtype/state/timing checks.
- Recorded the user's confirmation that the peer used mixed precision; retained the distinction between autocast-only compute precision and low-precision persistent training state.
- Reviewed the peer AI's code-level clarification: most parameters are stored BF16, small numerically sensitive modules FP32, and no explicit FP32 master copy is used. This materially confirms the low-precision persistent-state explanation for 33 GiB DDP.
- Added a generic config-owned trainable-parameter dtype policy and two isolated mixed-BF16 full-AdamW whiteboard configs. Transformer and checkpoint LoRA leaves default to BF16 while norms, vision embedding, state/action/time projections, action output and the full tactile TCN remain FP32; the existing full-FP32 configs are preserved.
- Replaced BF16 global-norm accumulation with FP32 loss/norm/clip reductions while preserving each gradient leaf's storage dtype. Added startup summaries for every parameter and optimizer-state dtype rather than inspecting only one leaf.
- Extended the existing two-GPU FSDP sequential launcher with an explicit `mixed_bf16_adamw` profile while preserving its legacy `fp32_sgd` default. The new profile continues strict Tabero-49999 initialization, next-state-before-sent-command ordering, W&B, independent assets and atomic 4k/8k/12k checkpoints.
- Abstract two-device verification reports 3.284B BF16 and 69.4M FP32 parameters. After the user's peer-side fit confirmation, the main profile now explicitly uses FP32 gradients and both FP32 Adam moments: per-device parameters are 3.199 GiB, moments 12.508 GiB and gradients 6.267 GiB, for a 21.975 GiB static subtotal before activations. The previous 12.797 GiB mixed-state layout remains an isolated low-memory fallback.
- Added a state-dtype wrapper because Optax only directly exposes the first-moment dtype; CPU contracts verify both first and second moments remain FP32 after an update and BF16 parameter leaves are cast back to BF16.
- Separated learning-rate schedules by storage precision: mixed BF16 AdamW uses `2e-5 -> 2e-6` to reduce the risk of updates disappearing during BF16 write-back, while the selectable full-FP32 AdamW profile retains `2e-6 -> 2e-7`.
- Completed 42 relevant CPU tests plus Bash, format, lint and whitespace checks. Real mixed-precision and full-FP32 dry-runs print the expected two FSDP commands; no GPU training, normalization generation, W&B run or robot operation was started.

### Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| New launcher Bash parse | Valid syntax | Exit 0 | PASS |
| Config/test Python compile | Valid Python | Exit 0 | PASS |
| Focused full-FT contracts | Config and sequential launcher behavior | 6 passed in 6.05s | PASS |
| Ruff format check | Both touched Python files formatted | New test would be reformatted | FIX PENDING |
| Final Ruff format check | Both touched Python files formatted | 2 files already formatted | PASS |
| Git diff whitespace | No whitespace errors | Exit 0 | PASS |
| Real launcher dry-run | Ordered two-GPU commands with no output writes | Correct next-state then sent-command commands | PASS |
| Abstract config trainable count | Every parameter trainable in both configs | 3,353,275,152 each | PASS |
| Combined whiteboard suite | Preserve old/r32 behavior and verify full-FT path | 26 passed in 5.94s | PASS |
| Host launch preflight | GPUs 1/2 free and names collision-free | Both GPUs ~48.5 GiB free; all four paths free | PASS |
| Real two-GPU full-FT first update | Fit and apply a finite update | OOM: 37.53 GiB computation plus failed 18.76 GiB allocation | FAIL |
| Real two-GPU SGD early run | Apply finite full-parameter updates | Finite through logged step 990; no clear loss trend and no step-1000 validation | PARTIAL |
| Three-device AdamW abstract sizing | Quantify persistent FSDP state | 16.910 GiB/device, including 4.418 GiB replication overhead | PASS |
| Runtime/driver health after SGD step 990 | Process and NVIDIA driver remain available | Process absent; `nvidia-smi` cannot communicate with driver | FAIL |
| Full sent-command validation loader finite scan | 500 batches contain only finite tensors | 500/500 finite; action range -5.394 to 4.120 | PASS |
| Full next-state validation comparison | Establish a matched successful range | 500/500 finite; action range -5.307 to 4.825; step-12000 validation finite | PASS |
| Full-FT W&B initialization | Upload config and metrics | Run `htwauln8`, validation step 0 uploaded | PASS |
| LoRA repaired checkpoint | Atomically commit step 12000 | Finalized in 43.60 seconds | PASS |
| LoRA sequential transition | Start sent-command after next-state success | Run `rp8qw8sz` active with finite updates | PASS |
| Mixed precision parameter grouping | Requested module families receive BF16/FP32 storage | 3,283,869,040 BF16 and 69,406,112 FP32 parameters; actual FP32 paths audited | PASS |
| Mixed AdamW FSDP abstract state | Quantify main and fallback two-card layouts | Main: 21.975 GiB/device static including FP32 gradients; fallback: 12.797 GiB/device | PASS |
| FP32 gradient/Adam implementation | Both moments and gradients stay FP32 while BF16 parameter leaves remain BF16 after update | Focused dtype/update contract passes | PASS |
| Mixed launcher profile | Generate ordered two-GPU commands without starting work | Correct two configs, FSDP=2, AdamW schedule, W&B and 12k saves | PASS |
| Real mixed two-GPU first update | Compile, fit and apply a finite update | Not run by assistant | PENDING |

### Errors
| Error | Resolution |
|-------|------------|
| New test file did not initially match Ruff formatting | Run scoped Ruff formatter, then repeat checks and tests. |
