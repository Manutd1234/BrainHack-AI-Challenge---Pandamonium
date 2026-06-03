"""
Observation flattening — single source of truth for the obs -> tensor conversion.

The til_environment Bomberman obs is a dict with mixed shapes. Both training
(via the env wrapper) and inference (via ae_manager) call into this module so
the formats can never drift apart.

Channels we keep / shapes we produce:
  - agent_viewcone : (25, 7, 5)   float32      -> "ego" CNN input
  - base_viewcone  : (25, S, S)   float32      -> "base" CNN input (S = 2*r+1, r=8 default)
  - scalars        : (D,)         float32      -> direction(one-hot 4) + location(2)
                                                  + base_location(2) + health(1)
                                                  + frozen_ticks(1) + base_health(1)
                                                  + team_resources(1) + team_bombs(1)
                                                  + step_norm(1)               => D=14
  - action_mask    : (6,)         uint8        -> legal actions

If a key is missing (the spec-page example is a stale subset), sensible defaults
fill in so inference still produces a legal move.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# --- Constants -------------------------------------------------------------- #
N_CHANNELS = 25          # ViewChannel enum width
AGENT_VC_H, AGENT_VC_W = 7, 5
BASE_VC_R = 8            # default base vision_radius; base_viewcone is (2r+1)
BASE_VC_S = 2 * BASE_VC_R + 1    # 17

NUM_ACTIONS = 6          # FORWARD, BACKWARD, LEFT, RIGHT, STAY, PLACE_BOMB
MAX_HEALTH = 60.0
MAX_BASE_HEALTH = 60.0   # adjust if your config differs
MAX_RESOURCES = 10.0
MAX_TEAM_BOMBS = 10
MAX_FROZEN = 3
NUM_ITERS = 200
GRID_SIZE = 16

SCALAR_DIM = 4 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1   # = 14
# direction(4) + loc(2) + base_loc(2) + health(1) + frozen(1)
# + base_health(1) + team_resources(1) + team_bombs(1) + step(1)


def _as_array(x, shape, dtype=np.float32, default=0.0) -> np.ndarray:
    """Coerce arbitrary input (list/np/scalar/None) into a fixed-shape array."""
    if x is None:
        return np.full(shape, default, dtype=dtype)
    arr = np.asarray(x, dtype=dtype)
    if arr.shape != tuple(shape):
        out = np.full(shape, default, dtype=dtype)
        # best-effort copy of overlapping region
        slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
        try:
            out[slices] = arr[slices]
        except Exception:
            pass
        return out
    return arr


def flatten_obs(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (agent_vc, base_vc, scalars, action_mask) ready for a torch model.

    Shapes:
      agent_vc:   (25, 7, 5)
      base_vc:    (25, 17, 17)
      scalars:    (14,)
      action_mask:(6,) uint8
    """
    # --- viewcones --- #
    # til_environment gives (7,5,25) -> we transpose to (25,7,5) for conv2d
    agent_vc_raw = obs.get("agent_viewcone", obs.get("viewcone"))
    agent_vc = _as_array(agent_vc_raw, (AGENT_VC_H, AGENT_VC_W, N_CHANNELS))
    agent_vc = np.transpose(agent_vc, (2, 0, 1))   # (25, 7, 5)

    base_vc_raw = obs.get("base_viewcone")
    base_vc = _as_array(base_vc_raw, (BASE_VC_S, BASE_VC_S, N_CHANNELS))
    base_vc = np.transpose(base_vc, (2, 0, 1))    # (25, 17, 17)

    # --- scalars --- #
    direction = int(obs.get("direction", 0)) % 4
    dir_oh = np.zeros(4, dtype=np.float32)
    dir_oh[direction] = 1.0

    loc = _as_array(obs.get("location", [0, 0]), (2,)) / max(GRID_SIZE - 1, 1)
    base_loc = _as_array(obs.get("base_location", [0, 0]), (2,)) / max(GRID_SIZE - 1, 1)

    health = np.array([float(obs.get("health", MAX_HEALTH)) / MAX_HEALTH], dtype=np.float32)
    frozen = np.array([float(obs.get("frozen_ticks", 0)) / MAX_FROZEN], dtype=np.float32)
    base_h = np.array([float(obs.get("base_health", MAX_BASE_HEALTH)) / MAX_BASE_HEALTH], dtype=np.float32)
    resources = np.array([float(obs.get("team_resources", 0.0)) / MAX_RESOURCES], dtype=np.float32)
    bombs = np.array([float(obs.get("team_bombs", 0)) / MAX_TEAM_BOMBS], dtype=np.float32)
    step = np.array([float(obs.get("step", 0)) / NUM_ITERS], dtype=np.float32)

    scalars = np.concatenate(
        [dir_oh, loc.astype(np.float32), base_loc.astype(np.float32),
         health, frozen, base_h, resources, bombs, step]
    )   # (14,)

    # --- action mask --- #
    mask_raw = obs.get("action_mask", [1] * NUM_ACTIONS)
    action_mask = _as_array(mask_raw, (NUM_ACTIONS,), dtype=np.uint8, default=1)
    # If everything is masked (shouldn't happen but guard), allow STAY
    if action_mask.sum() == 0:
        action_mask[4] = 1

    return agent_vc, base_vc, scalars, action_mask


def pack_for_sb3(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Pack into the Dict-Box format the SB3 policy expects (training + inference)."""
    agent_vc, base_vc, scalars, action_mask = flatten_obs(obs)
    return {
        "agent_vc": agent_vc.astype(np.float32),
        "base_vc": base_vc.astype(np.float32),
        "scalars": scalars.astype(np.float32),
        "action_mask": action_mask.astype(np.uint8),
    }
