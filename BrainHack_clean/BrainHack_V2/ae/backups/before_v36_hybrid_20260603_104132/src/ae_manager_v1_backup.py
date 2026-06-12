from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from obs_utils import pack_for_sb3

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get(
    "AE_MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "phase1_final.zip"),
)
DEVICE = os.environ.get("AE_DEVICE", "cpu")


class AEManager:
    def __init__(self, model_path: str = MODEL_PATH, device: str = DEVICE):
        logger.info("Loading AE policy from %s on %s", model_path, device)
        self.model = MaskablePPO.load(model_path, device=device)
        self.model.policy.set_training_mode(False)

        dummy = pack_for_sb3({})
        with torch.no_grad():
            self._predict_packed(dummy)

    def ae(self, observation: Dict[str, Any]) -> int:
        packed = pack_for_sb3(observation)
        return self._predict_packed(packed)

    def rl(self, observation: Dict[str, Any]) -> int:
        return self.ae(observation)

    def reset(self) -> None:
        return None

    def _predict_packed(self, packed: Dict[str, np.ndarray]) -> int:
        action_mask = packed["action_mask"].astype(bool)
        obs_for_model = {
            "agent_vc": packed["agent_vc"][None, ...],
            "base_vc": packed["base_vc"][None, ...],
            "scalars": packed["scalars"][None, ...],
        }

        action, _ = self.model.predict(
            obs_for_model,
            deterministic=True,
            action_masks=action_mask[None, ...],
        )
        a = int(np.asarray(action).reshape(-1)[0])

        if a < 0 or a >= len(action_mask) or not action_mask[a]:
            legal = np.where(action_mask)[0]
            a = int(legal[0]) if legal.size > 0 else 4

        return a
