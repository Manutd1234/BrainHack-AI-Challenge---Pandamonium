from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

N_CHANNELS = 25
AGENT_VC_H, AGENT_VC_W = 7, 5
BASE_VC_R = 8
BASE_VC_S = 2 * BASE_VC_R + 1

NUM_ACTIONS = 6
MAX_HEALTH = 60.0
MAX_BASE_HEALTH = 60.0
MAX_RESOURCES = 10.0
MAX_TEAM_BOMBS = 10
MAX_FROZEN = 3
NUM_ITERS = 200
GRID_SIZE = 16

SCALAR_DIM = 14


def _as_array(x, shape, dtype=np.float32, default=0.0) -> np.ndarray:
    if x is None:
        return np.full(shape, default, dtype=dtype)
    arr = np.asarray(x, dtype=dtype)
    if arr.shape != tuple(shape):
        out = np.full(shape, default, dtype=dtype)
        slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
        try:
            out[slices] = arr[slices]
        except Exception:
            pass
        return out
    return arr


def _scalar(x, default: float) -> float:
    if x is None:
        return float(default)
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def flatten_obs(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    agent_vc_raw = obs.get("agent_viewcone", obs.get("viewcone"))
    agent_vc = _as_array(agent_vc_raw, (AGENT_VC_H, AGENT_VC_W, N_CHANNELS))
    agent_vc = np.transpose(agent_vc, (2, 0, 1))

    base_vc_raw = obs.get("base_viewcone")
    base_vc = _as_array(base_vc_raw, (BASE_VC_S, BASE_VC_S, N_CHANNELS))
    base_vc = np.transpose(base_vc, (2, 0, 1))

    direction = int(_scalar(obs.get("direction", 0), 0.0)) % 4
    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[direction] = 1.0

    loc = _as_array(obs.get("location", [0, 0]), (2,)) / max(GRID_SIZE - 1, 1)
    base_loc = _as_array(obs.get("base_location", [0, 0]), (2,)) / max(GRID_SIZE - 1, 1)

    health = np.array([_scalar(obs.get("health", MAX_HEALTH), MAX_HEALTH) / MAX_HEALTH], dtype=np.float32)
    frozen = np.array([_scalar(obs.get("frozen_ticks", 0), 0.0) / MAX_FROZEN], dtype=np.float32)
    base_h = np.array([_scalar(obs.get("base_health", MAX_BASE_HEALTH), MAX_BASE_HEALTH) / MAX_BASE_HEALTH], dtype=np.float32)
    resources = np.array([_scalar(obs.get("team_resources", 0.0), 0.0) / MAX_RESOURCES], dtype=np.float32)
    bombs = np.array([_scalar(obs.get("team_bombs", 0), 0.0) / MAX_TEAM_BOMBS], dtype=np.float32)
    step = np.array([_scalar(obs.get("step", 0), 0.0) / NUM_ITERS], dtype=np.float32)

    scalars = np.concatenate([
        dir_oh, loc.astype(np.float32), base_loc.astype(np.float32),
        health, frozen, base_h, resources, bombs, step
    ])

    mask_raw = obs.get("action_mask", [1] * NUM_ACTIONS)
    action_mask = _as_array(mask_raw, (NUM_ACTIONS,), dtype=np.uint8, default=1)
    if action_mask.sum() == 0:
        action_mask[4] = 1

    return agent_vc, base_vc, scalars, action_mask


def pack_for_sb3(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    agent_vc, base_vc, scalars, action_mask = flatten_obs(obs)
    return {
        "agent_vc": agent_vc.astype(np.float32),
        "base_vc": base_vc.astype(np.float32),
        "scalars": scalars.astype(np.float32),
        "action_mask": action_mask.astype(np.uint8),
    }
