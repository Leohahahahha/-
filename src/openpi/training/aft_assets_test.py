import dataclasses
import hashlib
import json

import pytest

from openpi.training.aft_assets import validate_assets, model_contract
from openpi.training.config import get_config


def fixture_assets(tmp_path):
    c = get_config("aft_pi0_next_state_full")
    (tmp_path / "norm_stats.json").write_text("{}")
    (tmp_path / "adjacency.json").write_text("{}")
    manifest = dict(
        model_contract=model_contract(c.model),
        source_root=c.data.base_config.root,
        action_source_mode="next-state",
        train_episodes=list(c.data.base_config.episodes),
        validation_episodes=list(c.data.base_config.validation_episodes),
        norm_stats_sha256=hashlib.sha256(b"{}").hexdigest(),
        adjacency_sha256=hashlib.sha256(b"{}").hexdigest(),
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return c


def test_assets_reject_wrong_dataset_and_tampering(tmp_path):
    c = fixture_assets(tmp_path)
    validate_assets(tmp_path, c.model, c.data.base_config, check_source=False)
    with pytest.raises(ValueError, match="dataset|mode"):
        validate_assets(tmp_path, c.model, get_config("aft_pi0_sent_command_full").data.base_config, check_source=False)
    (tmp_path / "adjacency.json").write_text('{"0":[]}')
    with pytest.raises(ValueError, match="hash"):
        validate_assets(tmp_path, c.model, c.data.base_config, check_source=False)


def test_assets_reject_wrong_model_or_split(tmp_path):
    c = fixture_assets(tmp_path)
    with pytest.raises(ValueError, match="model"):
        validate_assets(
            tmp_path, dataclasses.replace(c.model, action_horizon=3), c.data.base_config, check_source=False
        )
    with pytest.raises(ValueError, match="model"):
        validate_assets(
            tmp_path, dataclasses.replace(c.model, image_resolution=(128, 128)), c.data.base_config, check_source=False
        )
    with pytest.raises(ValueError, match="split"):
        validate_assets(tmp_path, c.model, dataclasses.replace(c.data.base_config, episodes=(0,)), check_source=False)
