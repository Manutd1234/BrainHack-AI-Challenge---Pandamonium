"""
Feature extractor for the Bomberman observation.

Architecture:
    agent_vc (25,7,5)   -> Conv(32,3,p1) -> Conv(64,3,p1) -> GAP -> 64-d  ┐
    base_vc  (25,17,17) -> Conv(32,3,p1) -> Conv(64,3,p1) -> Conv(64,3,p1)│
                          -> GAP -> 64-d                                  ├─> concat -> Linear(256) -> ReLU
    scalars  (14,)      -> Linear(64) -> ReLU -> 64-d                     ┘

256-d feature vector feeds the MaskablePPO actor + critic heads.

Why this shape:
- The viewcone is sparse binary channels. 3x3 convs with padding=1 are cheap
  and let each cell aggregate immediate-neighbour context (one bomb cell next
  to an enemy cell, for example).
- Global Average Pool collapses spatial dim while keeping channel semantics —
  more robust than flattening and ~10x fewer params.
- Two towers (ego + base) because the ego-viewcone is what's immediately
  around the agent while base_viewcone gives strategic context near your base;
  mixing them in a single conv would force the net to learn the offset.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BombermanFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        n_ch = observation_space["agent_vc"].shape[0]   # 25

        # --- agent viewcone tower (7x5) --- #
        self.agent_cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                # -> (B, 64)
        )

        # --- base viewcone tower (17x17) --- #
        self.base_cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                # -> (B, 64)
        )

        scalar_dim = observation_space["scalars"].shape[0]
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Linear(64 + 64 + 64, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs):
        a = self.agent_cnn(obs["agent_vc"])
        b = self.base_cnn(obs["base_vc"])
        s = self.scalar_mlp(obs["scalars"])
        return self.fuse(torch.cat([a, b, s], dim=1))
