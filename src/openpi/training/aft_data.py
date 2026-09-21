"""Explicit, episode-local physical target windows for the A/F/T slow model.

Compact timestamps are not evidence that the original trajectory was continuous.
Callers must supply audited adjacency flags for original sample intervals.
"""

import numpy as np

from openpi.models.aft_types import AFTTargets


def build_episode_windows(rows, anchor, horizon=50, force_history=8, sensor_offset=1,
                          contiguous_edges=None, action_state_step_offset=0):
    if horizon < 1 or force_history < 1 or sensor_offset != 1:
        raise ValueError('Positive horizon/history and next-observation sensor_offset=1 required')
    state = np.asarray(rows['state'])
    n = len(state)
    if not 0 <= anchor < n:
        raise ValueError('anchor outside episode')
    if contiguous_edges is None:
        raise ValueError('Audited contiguous_edges required; compact timestamps are insufficient')
    edges = np.asarray(contiguous_edges)
    if edges.shape != (n - 1,) or edges.dtype != bool:
        raise ValueError('contiguous_edges must be boolean [episode_length-1]')
    if action_state_step_offset not in (0, 1):
        raise ValueError('action_state_step_offset must be 0 or 1')
    episodes = np.asarray(rows['episode_index'])
    if episodes.shape != (n,) or np.unique(episodes).size != 1:
        raise ValueError('Exactly one episode required')
    arrays = {key: np.asarray(rows[key]) for key in ('state', 'actions', 'wrist_wrench', 'tactile_marker_motion')}
    shapes = {'state': (n, 7), 'actions': (n, 7), 'wrist_wrench': (n, 6),
              'tactile_marker_motion': (n, 9, 198, 2)}
    for key, value in arrays.items():
        if value.shape != shapes[key] or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f'{key} must be floating {shapes[key]}, got {value.shape}')
        if not np.isfinite(value).all():
            raise ValueError(f'{key} must be finite')
    past = anchor + np.arange(1 - force_history, 1)
    # Do not pool force measurements from before a known temporal discontinuity.
    gaps_before = np.flatnonzero(~edges[:anchor])
    segment_start = int(gaps_before[-1] + 1) if gaps_before.size else 0
    past_valid = past >= segment_start
    past = np.clip(past, segment_start, anchor)
    action_indices = anchor + np.arange(horizon)
    future = action_indices + sensor_offset

    def valid_through(indices):
        result = (indices >= anchor) & (indices < n)
        for i, end in enumerate(indices):
            result[i] &= bool(edges[anchor:min(int(end), n - 1)].all())
        return result

    sensor_mask = valid_through(future)
    action_mask = valid_through(action_indices + action_state_step_offset)
    future_clipped = np.minimum(future, n - 1)
    motion = arrays['tactile_marker_motion'][future_clipped]
    return dict(
        force_history=arrays['wrist_wrench'][past].astype(np.float32),
        force_history_mask=past_valid,
        marker_history=arrays['tactile_marker_motion'][anchor].astype(np.float32),
        targets=AFTTargets(
            actions=arrays['actions'][np.minimum(action_indices, n - 1)].astype(np.float32),
            shear=(motion[:, -1] - motion[:, 0]).reshape(horizon, 396).astype(np.float32),
            wrench=arrays['wrist_wrench'][future_clipped].astype(np.float32),
            action_mask=action_mask, sensor_mask=sensor_mask,
        ),
    )
