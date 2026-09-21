# Task Plan: Diagnose W&B `crashed` training status

## Goal
Determine whether either paired whiteboard training job is still running, establish the last completed update/checkpoint, identify the supported crash cause from local evidence, and record the incident without changing training state.

## Next Step
User interrupts the post-traceback residual process in the existing tmux session, then resumes the same pair on one free GPU from finalized step 8000 and verifies the new step 12000 atomically commits before sent-command starts.

## Current Phase
Phase 8

## Phases

### Phase 1: Runtime discovery
- [x] Identify relevant run names and paths
- [x] Check live processes and GPU use without changing them
- [x] Locate latest logs, checkpoints, and W&B local state
- **Status:** complete

### Phase 2: Failure diagnosis
- [x] Extract last healthy training/evaluation/checkpoint events
- [x] Locate traceback, signal, exit, OOM, or launcher transition evidence
- [x] Distinguish W&B status from actual process and sequential-job state
- **Status:** complete

### Phase 3: Repository incident record
- [x] Update the incident overview table
- [x] Add evidence, cause/hypotheses, fix, and verification scope
- **Status:** complete

### Phase 4: Verification
- [x] Reconcile process, log, checkpoint, and W&B evidence
- [x] Ensure no training/process/checkpoint state was changed
- **Status:** complete

### Phase 5: Delivery
- [x] State whether training stopped and exactly where
- [x] Explain recovery/resume action if needed
- **Status:** complete

### Phase 6: One-GPU resume and network-disconnect follow-up
- [x] Verify the launcher permits one GPU while preserving global batch and checkpoint identity
- [x] Inspect Orbax restore/temporary-directory behavior without loading or mutating the checkpoint
- [x] Separate W&B API connectivity from SSH terminal lifetime using the observed timeline
- [x] Update incident evidence with the user's no-tmux fact; withhold the unsafe current resume command until resharding is fixed
- **Status:** complete

### Phase 7: Guided new one-GPU LoRA experiment
- [x] Identify exact config/launcher/test locations and inheritance
- [x] Recommend a one-GPU batch size and clearly state memory/effective-batch tradeoffs
- [x] Specify collision-safe new config/run names, 12k/4k schedule, and tactile LoRA rank change
- [x] Give student-owned edit and verification steps; do not edit source or launch training
- **Status:** complete

### Phase 8: User-operated training handoff
- [x] Add scoped checkpoint wait/callback reliability changes
- [x] Add and run r32/12k config/launcher/checkpoint CPU contracts
- [x] Confirm the proposed pair/run output names do not exist
- [ ] User launches in tmux and verifies the first real 4000 checkpoint
- **Status:** in_progress

### Phase 9: One-GPU recovery after final-save race
- [x] Identify the final-save `FileExistsError` and last finalized checkpoint
- [x] Repair the callback ordering and pass focused CPU contracts
- [x] Distinguish the resident post-exception process from active training
- [x] User interrupts the residual process
- [x] User resumes from step 8000
- [ ] Verify finalized step 12000 and automatic sent-command launch
- **Status:** in_progress

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Read-only investigation | User asked for status and cause; do not restart, stop, or alter training. |
| Treat W&B `crashed` as a run-level clue, not proof of all launcher state | A sequential launcher may continue or a process may remain alive despite stale UI state. |
| Assistant may edit the repetitive r32/12k launcher plumbing, while the user edits the learning-focused model config | User explicitly delegated completion checks/output and requested exact file/line guidance for their remaining code change. |
| Retain Orbax async manager but wait after every new-run save | Installed Orbax synchronous composite mode deadlocks in bounded probes; immediate waiting preserves its async signal protocol while preventing overlap and surfacing save failure at the checkpoint boundary. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| Sandboxed `nvidia-smi` could not access the driver | Re-ran the read-only query on the host; no Tabero process owns a GPU. |
| Kernel/system journal was not readable for this user | Explicitly leave external signal/OOM termination trigger unproven. |
| Sandboxed final `nvidia-smi` could not access the driver | Require the user to run the provided host-side query and substitute the confirmed free GPU index; do not guess. |
