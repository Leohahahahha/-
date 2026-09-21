# Findings & Decisions

## 2026-09-21: Expert/head comparison and synchronous execution
- MoT v2 section3.5 Figure14 directly ablates modality untying: FFN specialization improves heldout text/image metrics, adding QKV helps, LN untying negligible in Chameleon. Not a robot A/F/T result, and FLOPs/total params/walltime must not be conflated.
- MoSS section4.3 separates staged training and auxiliary prediction; these component results do not isolate user's same-budget shared-backbone3-head versus3-expert comparison. TA-VLA joint projection baseline confirms multioutput need not mean independent/no-interaction predictions.
- Proposed matched control: same modality tokens, masks, losses and shared hidden interface; tied versus untied expert weights isolates specialization more cleanly. Actual asymmetric1024/512/256 comparison is a deployed-system comparison with parameter/initialization differences, requiring reported params, latency, memory and seeds.
- WAM naming can be explicitly scoped as contact-centric joint action/physical-observation predictive model, not claim full world state or counterfactual rollout. DreamZero uses aligned visual/action prediction, so acknowledge nonvisual adaptation.
- Synchronous runtime: training padding/missing-data mask remains useful, runtime fixedH output mask can be implicit; plan_id optional host-generated logging; timeout/staleness is executor policy not learned prediction. Timing anchor or exact synchronization contract still required. A fast action intervention changes expected T/F regardless of concurrent versus synchronous scheduling; same-tick observation was caused by earlier actions.
- oral-paper-skill selected practices: meaningful comparisons, evidence-to-claim matching, bounded conditions. Skill corpus is abstract-derived guidance, not performance evidence.
- Primary leads: MoT https://arxiv.org/html/2411.04996v2 section3.5 modality-decoupling ablation; MoSS https://arxiv.org/html/2604.23272v1 section4.3 training/prediction ablations; DreamZero https://arxiv.org/abs/2602.15922 for visual WAM terminology. Need distinguish modality specialization evidence from exact A/F/T3-head comparator.

## 2026-09-21: Slow/fast architecture reference review
- Verified TA-VLA actual src/openpi/models/pi0.py:167–185 two Linear layers with swish; :205–209 flattens history into one token; :176–178 concatenates futureeffort into action projection dimensions. This is same JAX/Flax openpi lineage, more directly adaptable than N0's PyTorch/Wan modules. Forcewidth256 with eight6D frames can use48→512→256 to follow the 2*width pattern; previous48→128→256 remains a smaller variant, not exact TA-VLA.
- TA-VLA A.4 uses 10 frames sampled over past2seconds including current; concatenates14D effort history into140D, then MLP. A.5 expands action projections to predict torque over H50, not a separate torque expert. A.10 reports MLP18/20 vs RNN10/20 vs attention17/20 on one button task with obs+obj; not evidence of universal superiority on6D wrench. Official code link resolves https://github.com/ZZongzheng0918/TA-VLA; exact encoder implementation not yet audited.
- N0 mot.py is PyTorch/Wan-based. OwnQKV, sharedattention, split/output/FFN is reusable design, not drop-in JAX code. model.py frame/noise masks distinguish video/tactile even-frame versus action odd-frame schedule; tactile expert lacks direct text cross-attention by default. User wants same-slot A/F/T joint denoising and all experts reading VLM, so masks/routes must change.
- Slow-fast interface must include plan_id, observation anchor, per-slot physical timestamps, representation/norm/frame provenance and expiration. Fast residual action changes can invalidate the baseline trajectory's sensory expectation; prediction discrepancy is not itself a corrective direction or calibrated safety detector. 10Hz predictive labels do not establish high-rate dynamics. Diagram's rotvec component addition should not override existing SO(3) composition contract.
- TA-VLA primary https://arxiv.org/html/2509.07962v1 supports decoder-side torque history summarized into one token and auxiliary future torque prediction; input is joint torque rather than user's 6D K-frame wrench.
- N0-TWAM references under review: https://raw.githubusercontent.com/neoteai/N0-TWAM/main/n0_twam/models/mot.py and model.py. Need distinguish shared-attention structure from exact temporal mask/scheduler and implementation framework.

## 2026-09-21: Direct shear investigation
- Local pi0.py:381–382 uses x_tau=tau*noise+(1-tau)*actions and target=noise-actions; :528 updates x_tau+dt*v_tau with negative dt. Predicted vector field is not the updated observation and is not physical marker velocity.
- With one token per future instant and fixed hidden512, direct396 and latent64 have identical Transformer sequence lengths and attention interfaces. Two linear weight matrices differ by 2*512*(396-64)=339968 parameters, excluding biases; latent compression does not intrinsically reduce attention cost in this design.
- CGP III-D uses a VAE plus U-Net denoiser; TableVI compares latent sizes and KL regularization for reconstruction, not matched direct396 versus latent64 shear control. It cannot establish latent superiority for DM-Tac or this Transformer tokenizer.
- Revised first-baseline recommendation: normalized396D shear-space flow, H matching actions, history TCN retained with old input contract, separate2048→512 condition adapter. Latent remains optional ablation, not an attention requirement.
- MoSS equation5 specifies future physical observations as flow targets, without a specified VAE bottleneck. It supports sensor-space generation as an architectural option, not a 396D DM-Tac performance guarantee.
- Primary comparison sources: https://arxiv.org/html/2604.23272v1 and https://arxiv.org/html/2603.05687v3. Need separate evidence for latent modeling from claims of superiority over direct shear.

## 2026-09-21: Sensory expert sizing sources
- MoSS https://arxiv.org/html/2604.23272v1: sensory losses sum before lambda; appendix compares 0.1/0.5/1.0. Sensors are 30D AnySkin and 7D joint torque. All streams read VLM; asymmetric widths are our extension, not a verified MoSS claim.
- CGP https://arxiv.org/html/2603.05687v3: Table V uses 32D latent for simulated tactile arrays and 80D total for four Digit360 sensors (20 per sensor). This supports compact representations, not a DM-Tac optimum.
- N0-TWAM https://github.com/neoteai/N0-TWAM/blob/main/n0_twam/configs/twam_base_cfg.py: action/tactile hidden1024 with shared3072 attention interface.
- Local GQA uses eight query heads, one KV head, head_dim256 across existing experts; each stream projects its own width to/from this interface. Proposed A/T/F widths1024/512/256, tactile latent64 total, wrench6 physical channels are untested defaults.

## Requirements
- User is a beginner in VLA, deep learning and Python and wants teacher-style guidance.
- Begin with training configuration, especially the actual LoRA and full-finetuning changes in this repository.
- Explain where code lives, what was changed, why it changes training behavior, and the Python syntax used.
- Record important knowledge in a persistent review document.
- Use source-linked examples and avoid assuming familiarity with dataclasses, inheritance, filters or CLI overrides.

## Research Findings
- Current branch is `fix/fr3-tactile-shadow-deploy` at commit `5c765e6` (`Add mixed-precision Tabero training and deployment safeguards`). Training-code changes are committed; the current dirty tracked change is only the incident log, so the tutorial must preserve unrelated untracked artifacts.
- `TrainConfig` in `src/openpi/training/config.py` is the central immutable configuration record. It groups model, data, weight loader, optimizer, learning-rate schedule, freeze filter, precision policy, batch/step/eval/checkpoint/W&B options and FSDP topology.
- `TrainConfig.trainable_filter` is computed as `nnx.All(nnx.Param, nnx.Not(self.freeze_filter))`. Therefore the freeze filter, not the config name, is the final authority for whether a parameter leaf is updated.
- Configs are registered in `_CONFIGS`, looked up by name, and parsed through Tyro. The shell command can override non-suppressed dataclass fields after selecting a named base config.
- Whiteboard configs are built in layers: a shared whiteboard factory defines model/data/13D action+wrench/split/default schedule; LoRA configs call it with a tactile LoRA rank; full-finetune configs derive from it, remove the added tactile LoRA overlay, set `freeze_filter=nnx.Nothing`, apply a precision policy and choose AdamW/SGD/FSDP settings.
- `scripts/train.py` consumes the config rather than hard-coding an experiment: it builds the mesh, data loaders, train state and optimizer; applies parameter/gradient dtype policies; filters trainable leaves; runs JIT train/eval steps; logs W&B/JSONL; and saves checkpoints.
- `src/openpi/training/optimizer.py` represents schedules and optimizers as dataclasses with a `create()` method. Project additions include FP32 norm reduction, explicit Adam moment dtype and stateless SGD.
- The latest commit contains the relevant mixed-precision safeguards, so `git show 5c765e6^..5c765e6` can distinguish those project additions from older upstream code. Earlier project commits added the Tabero/FR3 configs and SO(3) action handling.
- LoRA has two independent control axes. The architecture axis creates LoRA leaves through `Pi0Config` variants containing `"lora"` or through `tactile_prefix_lora_rank > 0`; the optimization axis uses `freeze_filter` to decide which leaves can update.
- `Pi0Config.get_freeze_filter()` freezes the original LLM leaves while excluding paths matching `.*lora.*`. The tactile LoRA implements `base + (alpha / rank) * x @ A @ B`; A starts small-random and B starts at zero, preserving the pretrained function initially while allowing B to receive gradients.
- The released Tabero checkpoint already contains backbone LoRA leaves. Whiteboard full fine-tuning therefore preserves that backbone architecture for strict checkpoint compatibility, sets the extra tactile LoRA rank to zero, and sets `freeze_filter=nnx.Nothing` so every existing parameter leaf—including pretrained backbone LoRA leaves and base tactile TCN leaves—is trainable.
- `scripts/run_tabero_whiteboard_pair_full_ft_2gpu_12k.sh` maps `TABERO_FULL_FT_PROFILE` to one of four named configs, converts environment inputs into CLI overrides, and runs next-state before sent-command. Effective precedence is dataclass defaults < shared factory < concrete named config < CLI override.
- The launcher uses two FSDP devices, global batch 2, save interval 4000, eval interval 1000 and W&B scalar logging. Completion is correctly checked at `$run_dir/$target_steps/params`, although its informational checkpoint text is hard-coded to `{4000,8000,12000}` and can be stale for a non-12000 target.
- The shared whiteboard factory uses 39 episodes, holds out episode IDs `(1, 7, 17)`, predicts a 13D target (7D action + 6D K-frame wrench), uses SO(3)-correct relative pose transforms, enables task prompts, warms up 500 updates, and evaluates `1000 // batch_size` validation batches.
- The r32 LoRA configs pass `tactile_lora_rank=32`, `alpha=32`, batch 2 and 12,000 updates into the shared factory. The default r16 pair uses 20,000 updates and batch 4.
- The full-tune factory starts from the same data contract but sets tactile rank 0, strict Tabero-49999 loading, `freeze_filter=nnx.Nothing`, no EMA, FSDP over two devices, and AdamW whose moment dtype follows the precision policy.
- The mixed BF16 policy stores most trainable Transformer parameters in BF16 but uses path-regex overrides to keep normalization affine leaves, the visual input embedding, state/action/time projections, action output head, and tactile encoder in FP32. A derived policy additionally casts all gradients and optimizer states to FP32.
- `scripts/train.py` applies parameter dtypes after loading model structure, computes the scalar loss in FP32, differentiates only `config.trainable_filter`, optionally casts gradients before clipping/optimizer update, rejects non-finite updates, prints parameter and optimizer dtype summaries, and saves/evaluates according to config.
- `num_train_steps` counts optimizer updates, not epochs. With global batch size B and N training frames, an approximate epoch count is `steps * B / N`; repeated sequence sampling and action chunks mean it should be treated as a sampling-equivalent estimate.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Separate four layers: model, data, optimization, runtime | Most confusion comes from treating a config name as if it were the training algorithm itself. |
| Explain LoRA vs full tune through `freeze_filter` and parameter leaves | In this codebase the architecture name and the trainable parameter set are related but distinct. |
| Teach configuration precedence explicitly | A learner cannot safely reason about an effective run by reading only the named config when the launcher supplies CLI overrides. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Skill initializer lacked execute permission | Used `bash .../init-session.sh` rather than changing skill file permissions. |
| First findings patch expected a nonexistent `## Source findings` heading | Re-read the active plan files and patched the existing `## Research Findings` section. |

## Resources
- `src/openpi/training/config.py`: central config types and named experiment registry.
- `src/openpi/models/pi0_config.py`: model architecture options and LoRA freeze-filter construction.
- `scripts/train.py`: config consumer and training loop.
- `src/openpi/training/optimizer.py`: learning-rate and optimizer factories.
- `scripts/run_tabero_whiteboard_pair_full_ft_2gpu_12k.sh`: runtime overrides and sequential orchestration.
# 2026-09-19 next-state continuation audit

## 2026-09-21 A/F/T consultation
- Local Gemma attention supports separate expert Q/K/V projections; Pi0 currently instantiates VLM/action only, tactile enters prefix, and wrench is an action target.
- Public sources inspected: neoteai/N0-TWAM main mot.py, model.py, train.py. Default tactile skips direct text cross-attention; masks encode V/T then action, distinct from proposed simultaneous A/F/T.
- Proposed: three experts read VLM K/V, observed touch/wrench remain outside VLM, noisy future tokens interact mutually; common noise time tau and independent noise with common physical slot k.
- Current next-state target requires explicit +1 force/tactile response alignment; same row is not same physical target time. Latent tactile prediction needs a stable encoder target and reconstruction contract.

- The mixed-BF16/FP32-state next-state run completed normally at checkpoint `20000`; its checkpoint contains `params`, `assets`, and full `train_state` (including AdamW moments).
- The saved run used two-device FSDP, global batch size 2, AdamW with FP32 moments, warmup 500, and cosine learning rate `2e-5 -> 2e-6` over 20,000 total updates.
- `scripts/train.py` treats `num_train_steps` as an absolute final update count. A true in-place resume that adds 20,000 updates therefore needs `num_train_steps=40000`, not `20000`.
- The optimizer schedule counter is part of the restored train state. Merely overriding `decay_steps=40000` would evaluate a new cosine curve at counter 20,000 and can raise the learning rate unexpectedly; the current schedule has no phase-offset field.
- A separate stage-2 run initialized from `20000/params` can restart the low-LR schedule safely, preserves the original run/checkpoints, and gives unambiguous W&B comparison, but resets Adam moments. This is safer than mutating schedule semantics during an in-place resume.
- Validation action loss reached its best recorded value near step 7,000 (`~0.2224`) and was `~0.2968` at step 20,000. The later validation proxy worsened while training loss stayed low, so another 20,000 updates should be treated as an experiment, not assumed to improve deployment quality.
- The prior offline action/wrench evaluation established that the 20,000-step mixed full-finetune checkpoint outperformed the 12,000-step LoRA baseline, but it did not compare intermediate full-finetune checkpoints (such as 8,000) against 20,000. Real-robot benefit remains unverified.

# 2026-09-20 completed stage-2 overfitting audit

- The low-LR stage-2 run completed normally and finalized checkpoints at 4k/8k/12k/16k/20k. No NaN, OOM, rejected update, or checkpoint commit failure occurred; all finite indicators remained 1.
- Stage-2 validation at step 0 exactly equals stage-1 step 20,000 (`action_loss=0.296793699`, `loss=0.072622128`). This confirms the intended source checkpoint, validation split, normalization, and deterministic validation baseline were reproduced.
- Stage-2 validation action loss then rose to `0.312046736` at step 20,000, a 5.14% regression relative to its starting checkpoint. Every saved stage-2 checkpoint is worse on validation action loss than the source 20k checkpoint.
- Across the complete training history, validation action loss was best at stage-1 step 7,000 (`0.222407714`) and reaches `0.312046736` after cumulative 40,000 updates, a 40.30% increase from that minimum.
- Meanwhile sampled training-window action loss dropped from stage-1's final 4k-window mean `0.075131` to stage-2 means `0.070933`, `0.065465`, and `0.061962` over its first three 4k windows. Training fit improved while validation worsened, which is the characteristic generalization-gap pattern of overfitting.
- Stage-2's last two sampled train windows rise slightly (`0.063671`, `0.068131`), but remain below the stage-1 final-window mean; batch sampling/noise prevents treating those non-fixed train metrics as a monotonic train-set evaluation.
- The evidence strongly supports overfitting to the 36 training episodes / validation-split drift, not numerical divergence. Because there are only three validation episodes and flow-matching validation loss is a proxy, it does not by itself prove real-robot task success is worse.
- Do not automatically select stage-2 20k for deployment or continue training further. Preserve the stage-1 20k checkpoint as the current physically evaluated candidate, and compare stage-1 intermediate plus stage-2 checkpoints using the same offline physical metrics before any guarded real-robot test.
