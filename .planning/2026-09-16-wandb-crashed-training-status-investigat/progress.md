# Progress Log

## Session: 2026-09-16

### Current Status
- **Phase:** In progress — one-GPU recovery handoff
- **Started:** 2026-09-16

### Actions Taken
- Read the full planning-with-files instructions and restored existing repository planning context.
- Created an isolated plan for the W&B crash investigation.
- Confirmed the working tree already contains unrelated user changes; preserve them.
- Located pair/run artifacts and inspected process, checkpoint, W&B and status timestamps.
- Established that only the first run started and that update 16000 did not finalize.
- Found the W&B parent-exit event at 04:05:33 and a roughly 2.5-hour gap after the incomplete checkpoint save activity.
- Confirmed from host GPU process state that no Tabero training remains active.
- Reconciled last display step 16070, metrics step 16071, validation step 16000, and last finalized checkpoint 12000.
- Updated `docs/tabero_interview_incident_log.md` overview and added incident 32 with diagnosis, evidence, hypotheses, recovery boundary, verification scope and interview explanation.
- Did not restart/stop a process, alter checkpoints, load a model, run inference, or touch robot services.
- Follow-up started after the user confirmed the training was attached to an ordinary SSH/VS Code terminal rather than tmux.
- Verified from source/config metadata that the original checkpoint is replicated over two GPUs (`2 x 1` mesh), the launcher supports one GPU with global batch 4, and stale Orbax tmp cleanup is disabled.
- Confirmed GPU 2 is currently the only fully idle A6000 (48,538 MiB free); GPUs 0/1/3 are occupied by other users.
- A first isolated cross-device Orbax probe failed before creating its synthetic checkpoint; no real training artifact was opened or changed.
- Inspected saved `_sharding`: original run used two data-parallel replicas with empty partition specs. The synthetic Orbax probe was inconclusive and was interrupted without touching repository or training artifacts.
- Ran a real restore-only probe on idle GPU 2. It recognized finalized steps 4000/8000/12000 but failed safely on the two-GPU-to-one-GPU sharding mismatch before training or writes.
- Read VS Code remote-agent lifecycle logs and correlated disconnect, three-hour grace disposal, pair closure, and W&B parent exit. Network/session termination is now supported rather than merely hypothesized.
- Began a student-guided configuration task; no source edit or training is authorized for the assistant in this phase.
- Inspected current config inheritance, hardcoded Gemma ranks, tactile-rank fields, paired launcher magic numbers and paired CPU tests.
- Selected the controlled experiment: new tactile-only rank32/alpha32 configs, one-GPU batch2, 12k cosine decay, validation 500 batches, checkpoints at 4k/8k/12k; existing r16/20k configs remain unchanged.
- Reviewed the user's pasted `config.py` diff and new launcher. Confirmed that neither the r32/12k config additions nor the launcher substitutions have been made yet. User asked the assistant to handle the repetitive launcher variables, completion checks, and checkpoint-output text; no training is authorized.
- Updated only `scripts/run_tabero_whiteboard_pair_r32_12k.sh`: corrected its usage path, constrained it to the intended single-GPU experiment, introduced shared 12k/batch2/eval500/save4k variables, selected collision-safe r32 config names, wired all train flags to those variables, replaced both `/20000/params` checks, and corrected checkpoint output to 4k/8k/12k. `config.py` remains student-owned and unchanged in this step.
- Reviewed the user's subsequent `config.py` implementation. Rank/alpha, batch and step parameters are present, but found a blocking duplicate `eval_num_batches` keyword; exact parse/test confirmation is next. No source correction or training was performed.
- Confirmed the blocker with Python compilation and Ruff at `config.py:1829`; AST-only parsing was insufficient and is not used as the final syntax verdict. The user should remove only the obsolete `eval_num_batches=250` at current line 1826, retain `1_000 // batch_size`, and format the two new calls before the next review.
- User removed the duplicate keyword and reported `config.py compile OK`. Direct inspection confirms the retained dynamic validation count and correctly formatted r32 registrations. Whole-file Ruff formatting also changed many unrelated legacy layouts; these are currently preserved pending semantic tests and an explicit cleanup decision.
- Inspected existing paired CPU tests. They continue to exercise only the old 20k configs/launcher; the new r32/12k path currently has no dedicated contract test.
- Imported both new configs on CPU and verified rank/alpha 32, batch2, 12k steps/decay, eval500, save/keep4k, correct dataset IDs, and separate config-scoped asset directories. Existing paired suite passes 10/10. A dedicated r32/12k launcher/config test will now be added; no model construction or training occurred.
- Added a dedicated CPU-only r32/12k contract test file covering both config objects, asset isolation, dry-run flags/order, and the real launcher's requirement for finalized step 12000 before reporting both runs complete. Combined old/new suite passes 15/15.
- Inspected checkpoint plumbing and the installed Orbax API. Confirmed a supported per-manager synchronous switch exists; scoped implementation and a temporary save/finalization test are next.
- The first synchronous-manager implementation failed its bounded synthetic save test. Faulthandler showed the main thread blocked inside Orbax `Checkpointer.save`; further isolation found the callback's worker-thread JAX call also hangs. Interrupted only temporary pytest/processes and recorded the failed approach instead of treating it as verified.
- Chose the revised implementation: keep async Orbax, make the assets callback current-thread and signal-free, and block after each save only for the new configs. Actual large checkpoint persistence cannot be validated in this sandbox and remains a launch-time verification boundary.
- Implemented the revised checkpoint path and added mock/current-thread contracts. Focused old/new whiteboard suite now passes 19/19.
- Broader relevant suite produced 63 passes and two unrelated legacy failures: missing `debug` config in `scripts/train_test.py`, and a baseline fixture that violates the already-enforced fixed tactile reference-grid contract.
- Confirmed proposed pair name `whiteboard_force_r32_12k_20260916` and both run paths are currently free. Sandboxed `nvidia-smi` cannot access the host driver, so the user must choose the GPU from a host-terminal query.
- Investigated the user's live-status question. The first r32 next-state run exited at the final 12000 checkpoint with a temporary-directory `FileExistsError`; 4000/8000 are complete, 12000 is not, and sent-command did not start. No training process remains. This exposed a race in the just-added callback implementation; repair and resume verification are in progress.
- Host tmux inspection confirms the pane is stopped at the traceback, but a later host-level process check found that `scripts/train.py` remains resident in `futex_wait_queue`, owns about 34 GiB on GPU 2, and performs no GPU compute. It is a post-exception shutdown hang, not active training; the parent shells are waiting for it and the status file therefore remains stale. Corrected the callback to evaluate JAX process identity on the current thread while deferring the filesystem callback behind Orbax's contracted directory-creation signal. Focused suite now passes 20/20.
- User reports three GPUs are available. Current run must nevertheless resume on one GPU: launcher enforces one, global batch2 is not divisible by three, and the one-device checkpoint cannot be restored to a three-device topology by the current restore implementation. A three-GPU run would require a fresh start and batch/config/topology changes.
- User interrupted the residual process. Host verification now finds no matching trainer and GPUs 0/1/2 fully free; the launcher trap correctly changed the pair status to `failed/train_next_state`, exit code 130. Finalized recovery points remain 4000/8000 and the incomplete 12000 temporary directory is unchanged.
- User launched the one-GPU resume in tmux. Host evidence confirms checkpoint 8000 restored successfully in 5.05 seconds, PID 662729 is active on GPU 0, GPU utilization reached 87%, and pair status is `running/train_next_state`. The tmux server has PPID 1 and its own session, so ordinary SSH/VS Code disconnection no longer owns the training lifetime.

### Test Results
| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Process/GPU inspection | Determine whether training remains active | No matching process or GPU owner | PASS |
| Checkpoint atomicity inspection | Identify highest complete recovery point | 12000 complete; 16000 temporary only | PASS |
| Log/metrics/W&B reconciliation | Explain crashed state and last progress | Async save stall + abnormal parent exit; exact termination trigger unknown | PASS |
| r32/12k launcher `bash -n` | Valid Bash syntax | Exit 0 | PASS |
| r32/12k launcher dry-run | Print two sequential single-GPU 12k commands without training | Both commands use batch2, eval500, decay12k, save/keep4k and W&B | PASS |
| ShellCheck | Static shell lint | Tool is not installed on the workstation | NOT RUN |
| Focused `git diff --check` | No whitespace errors in launcher/incident/planning edits | Exit 0 | PASS |
| `config.py` Python compilation after user edit | Module must compile | Duplicate `eval_num_batches` at line 1829 | FAIL |
| Ruff format check after user edit | File must parse and conform | Refuses to parse duplicate keyword | FAIL |
| `config.py` Python compilation after correction | Module must compile | `config.py compile OK` | PASS |
| New r32/12k config-object probe | All intended values and isolated assets | Both configs match rank32/batch2/12k/eval500/save4k | PASS |
| Existing whiteboard CPU suite | Preserve old r16/20k behavior | 10 passed in 5.28s | PASS |
| Combined old/new whiteboard CPU suites | Preserve old path and verify new config/launcher | 15 passed in 5.11s | PASS |
| Direct synchronous Orbax manager probe | Finalize a tiny composite checkpoint | Timed out in contracted-signal path; approach rejected | FAIL |
| Revised focused checkpoint/config/launcher suite | Callback current-thread, save wait contract, old/new configs and launchers | 19 passed in 5.73s | PASS |
| Broader relevant regression suite | Detect adjacent regressions | 63 passed; 2 unrelated existing-contract failures | PARTIAL |
| First real r32 checkpoint sequence | Commit 4k/8k/12k before second run | 4k and 8k complete; 12k temp-dir race; process exited | FAIL |
| Revised contracted-signal callback suite | Defer callback until Orbax directory exists without worker-thread JAX | 20 passed in 6.25s | PASS |
| Host residual-process check | Distinguish active training from exception-time shutdown hang | PID 556792 is sleeping on a futex, GPU utilization 0%, traceback is terminal output | PASS |

### Errors
| Error | Resolution |
|-------|------------|
| `shellcheck` is not installed | Recorded as NOT RUN; Bash parser and dry-run still passed, and no dependency install was needed for this scoped edit. |
