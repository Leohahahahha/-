import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.aft_config import AFTConfig
from openpi.training import config
from openpi.training.data_loader import DataLoaderImpl


def test_loader_preserves_aft_targets():
    c = AFTConfig(paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=3)
    obs, targets = c.fake_obs(), c.fake_act()
    batch = {**obs.to_dict(), **targets.to_dict()}
    loader = DataLoaderImpl(config.DataConfig(aft_enabled=True), [batch])
    observation, target = next(iter(loader))
    assert target.shear.shape == (1, 3, 396)
    assert not hasattr(observation, "shear")


def test_eval_reports_all_modalities():
    from scripts.train import eval_step
    from openpi.models.aft import AFTModel

    class SmallAFT(AFTModel):
        def __init__(self):
            self.p = nnx.Param(jnp.array(1.0))

        def compute_loss(self, *args, **kwargs):
            return self.p.value, dict(action_loss=self.p.value, tactile_loss=self.p.value, wrench_loss=self.p.value)

    graph, params = nnx.split(SmallAFT())
    from types import SimpleNamespace

    result = eval_step(jax.random.key(0), SimpleNamespace(model_def=graph, params=params), (None, None))
    assert "wrench_loss" in result and "tactile_loss" in result


def test_cpu_optimizer_and_checkpoint_roundtrip(tmp_path):
    import optax
    from openpi.training import checkpoints
    from scripts.train import train_step
    from openpi.models import model as model_module
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training.utils import TrainState

    small = AFTConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        vision_variant="mu/16",
        image_resolution=(32, 32),
        tactile_width=32,
        force_width=16,
        action_horizon=3,
        max_token_len=2,
        dtype="float32",
    )
    network = small.create(jax.random.key(0))
    graph, params = nnx.split(network)
    c = dataclasses.replace(
        config.get_config("aft_pi0_next_state_lora"),
        model=small,
        freeze_filter=nnx.Not(PathRegex(r"(tactile_expert|force_expert)/.*")),
    )
    tx = optax.adamw(1e-5)
    state = TrainState(
        step=jnp.array(0),
        params=params,
        model_def=graph,
        tx=tx,
        opt_state=tx.init(params.filter(c.trainable_filter)),
        ema_decay=None,
    )
    batch = small.fake_obs(), small.fake_act()
    updated, metrics = train_step(c, jax.random.key(1), state, batch)
    assert int(updated.step) == 1
    assert np.isfinite(metrics["wrench_loss"])
    assert all(
        x.dtype == jnp.float32
        for x in jax.tree.leaves(updated.opt_state)
        if hasattr(x, "dtype") and x.dtype.kind == "f"
    )
    saver, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoints", keep_period=1, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(saver, updated, DataLoaderImpl(config.DataConfig(), []), 1, wait_until_finished=True)
        restored_state = checkpoints.restore_state(saver, updated, DataLoaderImpl(config.DataConfig(), []), 1)
        assert int(restored_state.step) == 1
        for original, loaded in zip(
            jax.tree.leaves(updated.opt_state), jax.tree.leaves(restored_state.opt_state), strict=True
        ):
            np.testing.assert_array_equal(original, loaded)
    finally:
        saver.close()
    restored = small.load(model_module.restore_params(tmp_path / "checkpoints/1/params"))
    updated_model = nnx.merge(updated.model_def, updated.params)
    before = updated_model.sample_predictions(jax.random.key(2), batch[0], num_steps=1)
    after = restored.sample_predictions(jax.random.key(2), batch[0], num_steps=1)
    for key in before:
        np.testing.assert_allclose(before[key], after[key], rtol=1e-5, atol=1e-5)
