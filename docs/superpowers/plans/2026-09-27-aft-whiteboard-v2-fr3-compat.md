# A/F/T Whiteboard v2 FR3 Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train-ready A/F/T π0 slow-expert configuration for the filtered whiteboard v2 dataset and a backward-compatible FR3 shadow/deployment inference interface with K-frame force history.

**Architecture:** Keep the slow model and legacy FR3 RGB/state/marker pipeline intact. Derive train/validation adjacency from source mapping, add a dedicated AFT serving metadata path backed by checkpoint-local assets, and augment the current `/getstate` sample with an eight-frame K-frame wrench history. Validate all three AFT outputs before the existing action guard can see them.

**Tech Stack:** Python 3.11, NumPy, PyArrow, JAX/Flax NNX, LeRobot v2.1, pytest, existing FR3 HTTP/WebSocket client. Use `/data/yanghaojun/envs/tabero-smoke/bin/python`; no new dependency or GPU training.

**Spec:** `docs/superpowers/specs/2026-09-27-aft-whiteboard-v2-fr3-compat-design.md` (approved, commit `f51e305`).

## Global Constraints

- Work only on Git branch `部署代码兼容版本` in the isolated worktree; do not add or commit the dirty `fix/fr3-tactile-shadow-deploy` checkout.
- Do not start model training, connect to a live robot, send HTTP pose/gripper commands, or claim shadow/real-robot validation.
- Keep existing front/wrist RGB, 7D state, task text, marker extraction/crop, `TargetGuard`, enable heartbeat, and legacy Pi0 serving behavior unchanged except explicit AFT dispatch.
- AFT has H=50, future shear `[50,198,2]`, future K-frame wrench `[50,6]`, and history `[8,6]` with bool mask `[8]` sampled at 10 Hz. Use `/getstate` `force` then `torque` from `K_F_ext_hat_K`; never `external_wrench_base`.
- Whiteboard prompt is exactly `Pick up the yellow whiteboard eraser and erase the X-shaped mark on the whiteboard.`
- New config: train episodes 1–38; validation episode 0; global batch 2; FSDP 2; workers 4; full AdamW with FP32 gradient/moments, mixed BF16/FP32 parameters, no EMA; 30,000 updates; warmup 500; LR `2e-5→2e-6`; eval every 1,000 with 174 batches; save/keep every 6,000; W&B on, images off; seed 42.
- Train-only normalization; do not treat compacted output timestamps or filtered rows as proof of physical 100 ms adjacency. No fast expert or feedback gate.
- Each task is test-first and separately committed. Documentation must distinguish CPU checks from GPU, shadow, and robot validation.

## File Structure

| File | Responsibility |
|---|---|
| `src/openpi/training/aft_data.py` | Source-mapping-aware episode adjacency while retaining old unfiltered audit behavior. |
| `scripts/prepare_aft.py`, `src/openpi/training/aft_assets.py` | One audited adjacency source for preparation/training; bind mapping hashes into checkpoint assets. |
| `src/openpi/training/aft_configs.py` | Add the single whiteboard v2 π0 full-FT 30k AFT preset without changing eight existing presets. |
| `scripts/serve_tabero.py` | Dedicated AFT load/metadata path; legacy Pi0 path unchanged. |
| `examples/fr3_deploy/core.py` | Pure K-frame wrench/history and AFT metadata/output contracts, independently CPU-testable. |
| `examples/fr3_deploy/transport.py` | One HTTP state+wrench acquisition and strict WebSocket three-output validation. |
| `examples/fr3_deploy/observations.py`, `examples/fr3_deploy/run.py` | Add AFT force input to existing 10 Hz sample and record predictions without changing robot control. |
| `docs/aft_slow_expert_code_guide.md`, `docs/tabero_interview_incident_log.md` | Explain actual module I/O and record evidence/remaining validation scope. |

## Review Focus

1. A mapping with removed frames but no mapping file or wrong `frames` count must fail before stats/training, not assume all edges true (Task 1 test).
2. A source timestamp jump from 0.1 s to 0.3 s despite consecutive compact output rows must mask future targets and reset force prehistory (Task 1 test).
3. Missing, NaN, or non-monotonic `/getstate` K-frame force/stamp must reject the AFT sample before WebSocket send (Task 4 test).
4. Wrong conversion hash or assembly prompt with AFT whiteboard checkpoint must fail before model inference/robot command, while old metadata remains accepted (Task 3 test).
5. Missing/nonfinite/wrong-shaped AFT shear or wrench prediction must be rejected before `Chunk` reaches `TargetGuard` (Task 5 test).

---

### Task 1: Audited v2 episode edges and source-bound assets

**Files:** Modify `src/openpi/training/aft_data.py`, `scripts/prepare_aft.py`, `src/openpi/training/aft_assets.py`; test `src/openpi/training/aft_data_test.py`, `scripts/prepare_aft_test.py`, `src/openpi/training/aft_assets_test.py`.

**Interfaces:** Produce `audit_episode_edges(root: str | Path, conversion: dict, episode: int, length: int, supplied: dict | None = None) -> np.ndarray[bool]`, `source_mapping_path(root, episode) -> Path`, and a private `is_contiguous(current: dict, following: dict) -> bool` for one 10 Hz source transition. Existing `audit_edges` remains for old no-mapping datasets. `AFTDataset` and `prepare_aft.main` both call the new function.

- [ ] **Step 1: Write failing synthetic tests.** Use a four-row mapping with source frames `0,1,3,4`, source timestamps `0,.1,.3,.4`, and a report `frames=6, removed_supervised_samples=1, missing_candidate_steps=0`, terminal omitted. Assert edges `[True,False,True]`; `build_episode_windows(rows, 1, contiguous_edges=edges, action_state_step_offset=1)` masks the cross-gap action/sensor; anchor 2 repeats its own first force value. In a second no-filter compact fixture, use consecutive source frame indices but timestamps `0,.1,.3,.4` with `missing_candidate_steps=1` and require edge 1 false. Assert missing mapping, wrong report count, duplicate output index, backwards action-source row, and a supplied all-true adjacency are rejected; an action-source row pointing to a filtered future row must produce a false edge. Add a manifest test that changes a mapping file after preparation and expects `validate_assets(checkpoint_assets, model, data, check_source=True)` to reject its SHA256.

```python
edges = audit_episode_edges(tmp_path, conversion, 0, 4)
np.testing.assert_array_equal(edges, [True, False, True])
with pytest.raises(ValueError, match="mapping|adjacency"):
    audit_episode_edges(tmp_path, conversion, 0, 4, {"0": [True, True, True]})
```

- [ ] **Step 2: Run red tests.** `JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest -q src/openpi/training/aft_data_test.py scripts/prepare_aft_test.py src/openpi/training/aft_assets_test.py`; new tests must fail on the missing audit function/hash behavior, not because of fixture syntax.
- [ ] **Step 3: Implement the minimal mapping audit.** Require `frames == length + terminal_omission + removed_supervised_samples`. Require mapping `source_episode_index` to match the report and output indices to equal `range(length)`; source row/frame indices and timestamps must increase. An edge is true only when the next retained source row/frame equals the current row's recorded action-source row/frame, source row/frame increments by one, and both source/action timestamps agree within a 10 Hz tolerance of `[0.095,0.105]` seconds. Compare any supplied flags exactly; if conversion reports removed samples but mapping is absent, raise. Keep old `audit_edges` fallback only for unfiltered no-mapping datasets.

```python
path = source_mapping_path(root, episode)
if not path.exists():
    if report.get("removed_supervised_samples", 0):
        raise ValueError(f"episode {episode}: filtered source mapping missing")
    return audit_edges(conversion, episode, length, supplied)
rows = json.loads(path.read_text())["rows"]
edges = np.asarray([is_contiguous(rows[i], rows[i + 1]) for i in range(length - 1)], dtype=bool)
if supplied is not None and str(episode) in supplied and not np.array_equal(edges, supplied[str(episode)]):
    raise ValueError(f"episode {episode}: supplied adjacency differs from source mapping")
return edges
```

- [ ] **Step 4: Wire preparation, assets, and loader.** `prepare_aft` audits train+val with the new function, prints counts of true/false edges, writes per-episode source-mapping SHA256 to manifest, and still computes stats from train episodes only. `AFTDataset` recomputes/compares edges with checkpoint adjacency; `validate_assets(check_source=True)` checks recorded mapping hashes if present, while `check_source=False` remains usable without the dataset on inference host.
- [ ] **Step 5: Run green tests and a read-only real-data edge audit.** Re-run the three test modules above. Call `audit_episode_edges` for all 39 v2 episodes without creating assets; verify count, shapes and at least one false edge in episode 0 and in a compacted episode. Do not infer dataset correctness from 39 successful file opens alone.
- [ ] **Step 6: Commit only Task 1 files.** `git add src/openpi/training/aft_data.py src/openpi/training/aft_data_test.py src/openpi/training/aft_assets.py src/openpi/training/aft_assets_test.py scripts/prepare_aft.py scripts/prepare_aft_test.py` then `git commit -m "fix: audit filtered AFT source adjacency"`.

### Task 2: Whiteboard v2 AFT full-FT 30k config

**Files:** Modify `src/openpi/training/aft_configs.py`; test `src/openpi/training/aft_configs_test.py`.

**Interfaces:** Register `aft_pi0_whiteboard_20260922_v2_next_state_full_30k` through `make_aft_configs()`; it must use the Task 1 audited data path and existing AFT weight loader. Existing eight config names remain unchanged.

- [ ] **Step 1: Write failing config assertions.** Test the new name, dataset root, train/val split, H=50, history 8, full trainability, Tabero 49999 init, prompt, `batch_size=2`, FSDP2, workers4, seed42, W&B, 30k schedule, 1k/174 evaluation, 6k saves, FP32 gradient/moments, AdamW betas/eps/clip, and distinct assets/checkpoint paths. Assert all eight old names still resolve.

```python
c = config.get_config("aft_pi0_whiteboard_20260922_v2_next_state_full_30k")
assert c.data.base_config.episodes == tuple(range(1, 39))
assert c.data.base_config.validation_episodes == (0,)
assert (c.num_train_steps, c.save_interval, c.keep_period) == (30_000, 6_000, 6_000)
assert (c.lr_schedule.warmup_steps, c.lr_schedule.peak_lr, c.lr_schedule.decay_lr) == (500, 2e-5, 2e-6)
```

- [ ] **Step 2: Run red test.** `JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest -q src/openpi/training/aft_configs_test.py::test_whiteboard_v2_aft_full_30k_config`; expected failure is unknown config name.
- [ ] **Step 3: Build the new preset from existing AFT π0 next-state full.** Use `dataclasses.replace` in `aft_configs.py`, not the dirty training-branch config. Preserve AFT model, strict Tabero 49999 loader, sensor FP32 overrides and `nnx.Nothing` full-tuning filter; replace only dataset/split, run settings and provenance metadata. Ensure `assets_dirs` resolves to `/data/yanghaojun/outputs/assets/<new-config-name>` so prepared assets live in its `aft/` child.

```python
base = next(x for x in result if x.name == "aft_pi0_next_state_full")
name = "aft_pi0_whiteboard_20260922_v2_next_state_full_30k"
root = "/data/yanghaojun/datasets/whiteboard_20260922_v2_tabero_next_state_filtered"
result.append(dataclasses.replace(base, name=name,
    data=dataclasses.replace(base.data, repo_id="local/whiteboard_20260922_v2_tabero_next_state_filtered",
        base_config=dataclasses.replace(base.data.base_config, root=root,
            episodes=tuple(range(1, 39)), validation_episodes=(0,))),
    batch_size=2, fsdp_devices=2, num_workers=4, num_train_steps=30_000,
    save_interval=6_000, keep_period=6_000, eval_interval=1_000, eval_num_batches=174,
    lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=2e-5,
        decay_steps=30_000, decay_lr=2e-6), seed=42,
    policy_metadata={**base.policy_metadata, "dataset_version": "whiteboard_20260922_v2_tabero_next_state_filtered",
        "task_prompt": "Pick up the yellow whiteboard eraser and erase the X-shaped mark on the whiteboard."}))
```

- [ ] **Step 4: Run green tests and static config print.** Run `aft_configs_test.py`; use `get_config(name)` on CPU to print all compared fields and ensure `model` is `AFTConfig`, not legacy 13D `Pi0Config`. Do not run a training step.
- [ ] **Step 5: Commit.** `git add src/openpi/training/aft_configs.py src/openpi/training/aft_configs_test.py` then `git commit -m "feat: configure whiteboard v2 AFT full finetuning"`.

### Task 3: AFT checkpoint serving and metadata dispatch

**Files:** Modify `scripts/serve_tabero.py`, `examples/fr3_deploy/core.py`, `examples/fr3_deploy/test_deployment.py`; create `scripts/serve_tabero_test.py`.

**Interfaces:** `load_policy` detects `AFTConfig` before old `supervised_action_dim` checks. A private `create_aft_served_policy(config, checkpoint: Path, manifest: dict, conversion_path: Path, denoise_steps: int) -> tuple[BasePolicy, dict]` loads the policy and emits metadata. New AFT metadata includes `architecture="aft"`, `prediction_layout="separate_aft_actions_shear_wrench"`, `predicts_wrench=True`, `predicts_tactile=True`, H=50, exact prompt, existing FR3 action/tactile contract, K-frame wrench units/order, and conversion/norm hashes. Old metadata branch remains byte-compatible.

- [ ] **Step 1: Write red serving/metadata tests.** Build a fake checkpoint directory with `params/` and `assets/aft/{manifest.json,norm_stats.json,adjacency.json}` matching the new config; mock only the network parameter restoration, not asset validation. Confirm AFT `load_policy` does not call `AFTDataConfig.create` or access the absent training dataset, accepts exact conversion hash and rejects a changed conversion/hash, wrong manifest config, missing shear stats, or wrong prompt. Assert legacy metadata tests still pass.

```python
policy, meta = serve_tabero.load_policy(NAME, checkpoint, conversion_path, 10)
assert meta["architecture"] == "aft"
assert meta["prediction_layout"] == "separate_aft_actions_shear_wrench"
assert meta["use_tactile"] is True
with pytest.raises(ValueError, match="prompt"):
    core.validate_metadata(meta, use_tactile=True,
        conversion_sha256=meta["conversion_sha256"], prompt="old assembly text")
```

- [ ] **Step 2: Run red tests.** `JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest -q scripts/serve_tabero_test.py examples/fr3_deploy/test_deployment.py`; fail on AFT type/metadata gate, not on importing ROS.
- [ ] **Step 3: Implement an explicit AFT service branch.** Call `validate_assets(checkpoint / "assets/aft", config.model, config.data.base_config, check_source=False)`; require manifest config/name and `file_hash(conversion_path) == manifest["conversion_sha256"]`; load checkpoint-local stats and assert dimensions state7/actions7/tactile_prefix396/shear396/wrench6; call `policy_config.create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": denoise_steps}, strict_params=True)`. Declare tactile required explicitly because AFTConfig inherits Pi0's empty `tactile_streams`. Build truthful separate-head metadata; do not reuse old 13D `prediction_layout`. Keep old `load_policy` branch unchanged.

```python
if isinstance(config.model, AFTConfig):
    manifest = validate_assets(checkpoint / "assets/aft", config.model,
        config.data.base_config, check_source=False)
    if manifest.get("config") != config_name or file_hash(conversion_path) != manifest.get("conversion_sha256"):
        raise ValueError("AFT checkpoint/config/conversion mismatch")
    return create_aft_served_policy(config, checkpoint, manifest, conversion_path, denoise_steps)
```

- [ ] **Step 4: Extend client metadata validation only for `architecture="aft"`.** Require the exact AFT layout, tactile/wrench shapes and units, force-history length 8, task prompt equality, and conversion hash. Leave absent/legacy `architecture` on the old validator path, including its 13D prediction-layout check.
- [ ] **Step 5: Run green tests and commit.** Run `scripts/serve_tabero_test.py` plus `examples/fr3_deploy/test_deployment.py`; then `git add scripts/serve_tabero.py scripts/serve_tabero_test.py examples/fr3_deploy/core.py examples/fr3_deploy/test_deployment.py` and `git commit -m "feat: serve AFT with checkpoint-local FR3 contract"`.

### Task 4: Atomic live K-frame force history input

**Files:** Modify `examples/fr3_deploy/core.py`, `examples/fr3_deploy/transport.py`, `examples/fr3_deploy/observations.py`, `examples/fr3_deploy/run.py`; test `examples/fr3_deploy/test_deployment.py`.

**Interfaces:** Add `wrench_from_http(payload) -> np.ndarray[6]`, `ForceHistory.append(wrench, stamp) -> tuple[np.ndarray[8,6], np.ndarray[8] bool]`, `add_aft_force_input(data: dict, history: ForceHistory, wrench: np.ndarray, stamp: float) -> dict`, and `RobotHttp.read_state_with_wrench(max_age) -> tuple[state7, wrench6, stamp]`. Keep `read_state(max_age) -> tuple[state7, stamp]` unchanged. `LiveObservations(config, conversion, *, use_tactile: bool, use_force: bool = False)` adds AFT fields only when true.

- [ ] **Step 1: Write red pure tests with the user-confirmed `/getstate` payload.** Assert order and dtype `force + torque`, not `external_wrench_base`; first eight-slot history repeats current wrench with mask `[False]*7+[True]`, next sample shifts it, a >0.15 s gap resets it, and nonmonotonic/missing/nonfinite values raise. Fake `RobotHttp.post` and the wall clock to assert one HTTP read supplies state, wrench and acquisition timestamp. Verify old `read_state` still works without force fields.

```python
history = core.ForceHistory()
values, valid = history.append(np.arange(6, dtype=np.float32), 10.0)
assert values.shape == (8, 6)
np.testing.assert_array_equal(valid, [False] * 7 + [True])
with pytest.raises(ValueError, match="force|torque"):
    core.wrench_from_http({"force": [1, 2, float("nan")], "torque": [0, 0, 0]})
```

- [ ] **Step 2: Run red tests.** `JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest -q examples/fr3_deploy/test_deployment.py::test_k_frame_force_history_contract`; failure must identify the missing parser/history, not camera/ROS imports.
- [ ] **Step 3: Implement one acquisition and 10 Hz sampling.** Refactor `RobotHttp` internal read to validate `stamp.to_sec` once; the new rich method extracts finite `force[3]` and `torque[3]` from that same payload. In `LiveObservations.poll_state`, store `(state, stamp)` and `(wrench, stamp)` atomically under the existing lock for AFT. In `sample_loop`, include `force` in timing keys only for AFT, call pure `add_aft_force_input` at each 100 ms sample, reset its `ForceHistory` whenever marker history is reset, and write `force_history` plus `force_history_mask` into data. `run.py` enables this only after AFT metadata validation.

```python
if self.use_force:
    state, wrench, stamp = self.reader.read_state_with_wrench(self.config["max_sensor_age_sec"])
    with self.lock:
        self.frames["state"] = (state, stamp)
        self.frames["force"] = (wrench, stamp)
else:
    state, stamp = self.reader.read_state(self.config["max_sensor_age_sec"])
    with self.lock:
        self.frames["state"] = (state, stamp)
```

```python
def add_aft_force_input(data, history, wrench, stamp):
    values, valid = history.append(wrench, stamp)
    return {**data, "force_history": values, "force_history_mask": valid}
```

- [ ] **Step 4: Run green tests, add a CPU-only sample-assembly test, and commit.** Assert legacy sample keys unchanged and AFT sample adds exactly `[8,6]` float32 plus `[8]` bool; check no stale sample reaches policy. Run `examples/fr3_deploy/test_deployment.py`; then commit the four source files plus test with message `feat: add synchronized K-frame wrench history to FR3 observations`.

### Task 5: Three-output transport, logs, and operator handoff

**Files:** Modify `examples/fr3_deploy/core.py`, `examples/fr3_deploy/transport.py`, `examples/fr3_deploy/run.py`, `docs/aft_slow_expert_code_guide.md`, `docs/tabero_interview_incident_log.md`; test `examples/fr3_deploy/test_deployment.py`.

**Interfaces:** Extend `Chunk` with optional `tactile_shear` after existing optional `wrist_wrench`; AFT `RemotePolicy.infer` requires/validates actions `[H,7]`, shear `[H,198,2]`, wrench `[H,6]` and force history before sending. Old policy accepts its previous payload/output unchanged. The run log adds predicted shear and current observed K-frame wrench, without altering control actions.

- [ ] **Step 1: Write red transport/log tests.** For AFT metadata, omit force history or make its current mask false and assert no WebSocket send; return missing/NaN/wrong-shaped shear or wrench and assert `ValueError` before chunk creation. Return valid three-head output and assert preserved arrays. Reuse the fake synchronous `control_loop` fixture to assert JSONL contains predicted shear/current observed wrench and no extra robot command; legacy logging test still passes.

```python
policy.metadata = {"architecture": "aft", "use_tactile": True,
    "action_horizon": 50, "predicts_wrench": True, "predicts_tactile": True}
sample.data["force_history"] = np.zeros((8, 6), np.float32)
sample.data["force_history_mask"] = np.r_[np.zeros(7, bool), True]
policy.receive = lambda timeout: {"actions": actions,
    "tactile_shear": np.zeros((50, 198, 2), np.float32),
    "wrist_wrench": np.zeros((50, 6), np.float32)}
assert policy.infer(sample).tactile_shear.shape == (50, 198, 2)
```

- [ ] **Step 2: Run red tests.** `JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest -q examples/fr3_deploy/test_deployment.py::test_remote_aft_three_output_contract`; expected failure is absent AFT validation/field.
- [ ] **Step 3: Implement strict AFT transport and logging.** Validate force input before `ws.send`; validate all three finite output arrays after receive. Append tactile field to `Chunk` so positional legacy construction remains compatible. In `run.py`, log full predicted shear and current observed wrench for AFT only; keep action prefix, enable heartbeat, age checks and `TargetGuard` byte-for-byte behavior. Compare configured prompt with metadata before constructing live observations.

```python
if self.metadata.get("architecture") == "aft":
    history = np.asarray(sample.data.get("force_history"))
    mask = np.asarray(sample.data.get("force_history_mask"))
    if history.shape != (8, 6) or mask.shape != (8,) or mask.dtype != bool or not mask[-1]:
        raise ValueError("AFT requires valid force_history[8,6] and bool mask[8]")
    if not np.isfinite(history).all():
        raise ValueError("AFT force_history must be finite")
```

- [ ] **Step 4: Update documentation and incident status with evidence only.** In the AFT code guide, list new config name, source-mapping edge rules, force source/shape/mask, server-client metadata, three output shapes, exact whiteboard prompt, train-only asset path and user-run single-line prepare/train/log commands. Update incident 55's fix/status/verification fields: mark implementation and CPU checks completed only if observed; leave GPU, W&B, shadow and robot validation pending.
- [ ] **Step 5: Run full targeted CPU verification.** Run `JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest -q src/openpi/training/aft_data_test.py src/openpi/training/aft_assets_test.py src/openpi/training/aft_configs_test.py src/openpi/policies/aft_policy_test.py scripts/prepare_aft_test.py scripts/serve_tabero_test.py examples/fr3_deploy/test_deployment.py`; run `git diff --check`. Run 39-episode read-only audit and, if practical, CPU preparation into a unique `/tmp` destination; never start `scripts/train.py`.
- [ ] **Step 6: Commit and final audit.** `git add examples/fr3_deploy/core.py examples/fr3_deploy/transport.py examples/fr3_deploy/run.py examples/fr3_deploy/test_deployment.py docs/aft_slow_expert_code_guide.md docs/tabero_interview_incident_log.md` then `git commit -m "feat: validate and log AFT three-head FR3 inference"`. Confirm `git status --short --branch` is clean and list all commit hashes. Do not push or deploy without a separate user request.
