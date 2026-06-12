from __future__ import annotations
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BombermanFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        n_ch = observation_space["agent_vc"].shape[0]

        self.agent_cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.base_cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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
