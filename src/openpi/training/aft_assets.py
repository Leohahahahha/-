"""One asset contract shared by training, offline evaluation and synchronous policy."""

import hashlib
import json
import pathlib


def file_hash(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def model_contract(model):
    contract = {
        key: getattr(model, key)
        for key in (
            "pi05",
            "action_horizon",
            "action_dim",
            "tactile_width",
            "force_width",
            "paligemma_variant",
            "action_expert_variant",
            "tactile_prefix_diff_from_reference",
            "vision_variant",
            "max_token_len",
            "force_history_frames",
        )
    }
    contract["image_resolution"] = list(model.image_resolution)
    contract["compute_dtype"] = model.dtype
    return contract


def validate_assets(directory, model, data, *, check_source=True):
    directory = pathlib.Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("model_contract") != model_contract(model):
        raise ValueError("AFT model contract mismatch")
    if manifest.get("source_root") != str(data.root):
        raise ValueError("AFT dataset root mismatch")
    mode = "next-state" if data.aft_action_state_step_offset else "sent-command"
    if manifest.get("action_source_mode") != mode:
        raise ValueError("AFT action mode mismatch")
    if manifest.get("train_episodes") != list(data.episodes) or manifest.get("validation_episodes") != list(
        data.validation_episodes
    ):
        raise ValueError("AFT train/validation split mismatch")
    for filename, key in [("norm_stats.json", "norm_stats_sha256"), ("adjacency.json", "adjacency_sha256")]:
        if file_hash(directory / filename) != manifest.get(key):
            raise ValueError(f"AFT {filename} hash mismatch")
    if check_source:
        source = pathlib.Path(data.root) / "meta/tabero_conversion.json"
        if file_hash(source) != manifest.get("conversion_sha256"):
            raise ValueError("AFT dataset conversion hash mismatch")
        mapping_hashes = manifest.get("source_mapping_sha256", {})
        conversion = json.loads(source.read_text())
        if (conversion.get("version", 0) >= 4 or conversion.get("frame_exclusion")) and not mapping_hashes:
            raise ValueError("AFT filtered source mapping hashes missing")
        for episode in (*data.episodes, *data.validation_episodes):
            if mapping_hashes:
                mapping = pathlib.Path(data.root) / "meta/source_mapping" / f"episode_{episode:06d}.json"
                if not mapping.exists() or file_hash(mapping) != mapping_hashes.get(str(episode)):
                    raise ValueError(f"AFT source mapping hash mismatch for episode {episode}")
    return manifest
