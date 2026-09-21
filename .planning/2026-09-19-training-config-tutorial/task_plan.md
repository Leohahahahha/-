# Task Plan: Tabero-VTLA training configuration tutorial

## Goal
Create a beginner-friendly, source-linked Chinese tutorial that teaches how this repository represents LoRA and full fine-tuning configurations, how a config reaches the training loop, what was customized for Tabero whiteboard training, and the Python syntax needed to read and safely modify those configs.

## Next Step
If requested, run matched offline physical evaluation for stage-1 4k/8k and all stage-2 checkpoints before selecting a real-robot candidate.

## Current Phase
Complete

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent: teach rather than merely provide commands
- [x] Identify constraints: beginner in Python, deep learning, VLA and training; start with configs
- [x] Map current config call chain and modification history
- [x] Document source facts in findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Define lesson order from Python syntax to config semantics
- [x] Define a durable tutorial document and review checklist
- **Status:** complete

### Phase 3: Implementation
- [x] Write `docs/tabero_training_config_tutorial.md`
- [x] Cover LoRA, full fine-tuning, filters, dtype policy, optimizer, data split, checkpoint and W&B configs
- [x] Add small exercises and safe modification recipes
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Verify every symbol/config name/path against current source
- [x] Verify example introspection commands without starting training
- [x] Document checks in progress.md
- **Status:** complete

### Phase 5: Delivery
- [x] Review the tutorial at beginner reading level
- [x] Give the first guided lesson and next study step
- **Status:** complete

### Phase 6: Next-state low-learning-rate continuation
- [x] Verify the 20,000-step checkpoint and complete training-state save
- [x] Inspect the validation trend and distinguish technical resumability from expected model quality
- [x] Select a schedule/optimizer continuation method that preserves the original run
- [x] Verify the exact one-line command without launching training
- [x] Record the training decision in the repository incident log and deliver the command
- **Status:** complete

### Phase 7: Diagnose completed stage-2 validation regression
- [x] Locate and verify the completed stage-2 run, checkpoints, and configuration
- [x] Compare train/validation curves across the original and stage-2 runs
- [x] Check for numerical failure, data/split drift, or optimizer-reset transients
- [x] Determine the strongest supported diagnosis and checkpoint recommendation
- [x] Update the incident log and deliver evidence-backed conclusions
- **Status:** complete

## Decisions Made
### Phase 8: A/F/T architecture consultation (2026-09-21)
- [x] Inspect current modality routing and public reference.
- [x] Define VLM visibility, conditioning, temporal targets and joint flow.
- [x] Record architecture evidence and implementation limits.
- **Status:** complete

### Phase 9: Sensory expert sizing consultation (2026-09-21)
- [x] Verify primary literature and local attention interfaces.
- [x] Separate representation dimensions from expert widths.
- [x] Record recommendations and validation limits without model changes.
- **Status:** complete

### Phase 10: Direct shear versus latent generation
- [x] Check source contracts and primary-paper comparison evidence.
- [x] Explain denoising updates, attention costs and experimental limits.
- [x] Update investigation records; no model implementation or training.
- **Status:** complete

### Phase 11: Slow/fast system interface and reference implementation review
- [x] Verify torque-history and multimodal-attention references.
- [x] Identify slow-layer interfaces and transfer limits.
- [x] Record findings without changing model code or running training.
- **Status:** complete

### Phase 12: Expert-versus-head evidence and synchronous inference semantics
- [x] Inspect primary ablation evidence using oral-paper-skill.
- [x] Scope WAM terminology and simplify synchronous interface recommendations.
- [x] Record bounded conclusions and matched experiment proposal.
- **Status:** complete

## Architecture decisions
| Decision | Rationale |
|----------|-----------|
| Teach from the repository's real call chain | The user needs to understand code they will maintain, not a generic training lecture. |
| Write a durable Chinese tutorial under `docs/` | The user explicitly asked to record important knowledge for later review. |
| Do not start or modify training runs | This task is educational and source/documentation focused. |
| The user remains the training operator | We may inspect, test, and prepare commands, but must not launch the GPU training run. |
| Start a separate low-LR stage from checkpoint parameters | This preserves the original run and avoids misusing a restored 20k schedule counter; the tradeoff is intentionally resetting Adam moments. |
| Diagnose the completed stage 2 as strong overfitting evidence, not numerical failure | Fixed validation worsened while sampled train fit improved; the source baseline reproduced exactly and all finite checks passed. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| `init-session.sh` was not executable | Invoked the skill-provided script explicitly with `bash`; plan creation succeeded. |
| `jq` is not installed while extracting validation JSONL | Use the standard-library Python/awk path for read-only metric extraction; no dependency install is needed. |
| `nvidia-smi` cannot communicate with the driver from the Codex execution namespace | Do not infer workstation GPU health from this isolated namespace; require the user to run the provided GPU preflight in the actual training terminal. |
