# Task Plan: Two-GPU sequential whiteboard full-parameter fine-tuning

## Goal
Add safe, isolated two-GPU full-parameter fine-tuning configs and a user-operated sequential launcher for the next-state and sent-command whiteboard datasets, without disturbing the running LoRA experiment.

## Next Step
User launches a new collision-safe `mixed_bf16_adamw` pair on exactly two GPUs and verifies initialization, the first finite AdamW update and real peak memory; no training is started by the assistant.

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] Interpret full fine-tuning as updating all model parameters rather than using only LoRA adapters
- [x] Preserve two datasets, validation episodes 1/7/17, tactile input, 13D action+wrench supervision and sequential execution
- [x] Audit memory, sharding, initialization and existing runtime constraints
- [x] Document in findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Define model/filter, optimizer, batch and schedule
- [x] Define collision-safe config, asset, checkpoint, W&B and launcher names
- **Status:** complete

### Phase 3: Implementation
- [x] Add full-parameter configs
- [x] Add sequential two-GPU launcher
- [x] Add focused CPU contract tests
- [x] Update required incident log
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Run syntax, format, config and launcher dry-run tests
- [x] Verify no training or robot operation was started
- [x] Document test results
- **Status:** complete

### Phase 5: Delivery
- [x] Review exact diff and runtime caveats
- [x] Give one-line user launch and monitoring commands
- [x] Summarize important code changes
- **Status:** complete

### Phase 6: First real GPU launch diagnosis
- [x] Confirm W&B initialization/upload and locate the full-FT run ID
- [x] Locate the exact failure boundary and determine whether any update/checkpoint exists
- [x] Reconcile LoRA W&B `crashed` display with local process/checkpoint state
- [x] Update the required incident record
- [x] Select and verify a lower-memory full-parameter implementation
- [ ] User verifies the real two-GPU first update and peak memory
- [x] Audit the real SGD early-loss trend and gradient clipping behavior
- [x] Estimate three-GPU AdamW FSDP residency and replication overhead
- [x] Separate the step-990 process termination from optimizer convergence
- [x] Locate the sent-command full-SGD W&B failure boundary
- [x] Determine whether step-0 NaN originates in loader tensors or model forward loss
- [ ] Reproduce the first non-finite model-forward batch after GPU-driver recovery
- **Status:** in_progress

### Phase 7: Configurable mixed-precision full fine-tuning

- [x] Design a generic config-owned parameter dtype policy with a full-FP32 fallback
- [x] Implement the requested Tabero BF16/FP32 module grouping without changing LoRA runs
- [x] Keep FP32 loss/norm/clip reductions and make full-gradient storage independently configurable
- [x] Incorporate user-reported fit evidence by adding a main FP32-gradient/FP32-Adam-state profile and preserving the low-memory profile
- [x] Verify parameter and Adam-state dtypes and abstract two-card memory
- [x] Add focused unit tests and update the incident record
- [x] Convert step counts to dataset epochs and recommend a training duration
- [ ] User verifies the real two-GPU compile, first update and peak memory
- **Status:** implementation_complete_runtime_pending

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Build new configs/launcher rather than mutate LoRA configs | Keeps the currently running LoRA A/B experiment and its assets/checkpoints reproducible. |
| User starts training | User explicitly asked the assistant to modify code and provide commands, not to launch the new training. |
| Do not resume the failed full-FT pair | It failed before step 0 completed and has no numeric checkpoint; resume would not provide a valid recovery point. |
| Do not resume the failed sent-command SGD run | It has no numeric checkpoint; use a new experiment name after isolating the step-0 model-forward NaN. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
