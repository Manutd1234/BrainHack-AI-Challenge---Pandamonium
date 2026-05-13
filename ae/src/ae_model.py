"""Shared ResNet-bottleneck policy modules for AE inference and SAC training."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

NUM_ACTIONS = 6
AGENT_VIEW_SHAPE = (7, 5, 25)
BASE_VIEW_SHAPE = (5, 5, 25)
SCALAR_SIZE = 21


class BottleneckBlock(nn.Module):
    """Small ResNet bottleneck block for compact grid observations."""

    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        out_channels = bottleneck_channels * 4
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(1, bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(1, bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.norm3 = nn.GroupNorm(1, out_channels)
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = F.silu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        return F.silu(x + residual)


class ViewconeEncoder(nn.Module):
    def __init__(self, in_channels: int = 25, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.SiLU(),
            BottleneckBlock(64, 16),
            BottleneckBlock(64, 32),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SACPolicyNetwork(nn.Module):
    """Discrete-action policy used by the AE manager."""

    def __init__(self, scalar_size: int = SCALAR_SIZE, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.agent_encoder = ViewconeEncoder(out_dim=128)
        self.base_encoder = ViewconeEncoder(out_dim=96)
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_size, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128 + 96 + 96, 192),
            nn.SiLU(),
            nn.Linear(192, num_actions),
        )

    def forward(
        self,
        agent_view: torch.Tensor,
        base_view: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.agent_encoder(agent_view),
                self.base_encoder(base_view),
                self.scalar_net(scalars),
            ],
            dim=-1,
        )
        return self.policy_head(features)


class SACCriticNetwork(nn.Module):
    """Q-network for discrete SAC training."""

    def __init__(self, scalar_size: int = SCALAR_SIZE, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.agent_encoder = ViewconeEncoder(out_dim=128)
        self.base_encoder = ViewconeEncoder(out_dim=96)
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_size, 96),
            nn.SiLU(),
            nn.Linear(96, 96),
            nn.SiLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(128 + 96 + 96, 192),
            nn.SiLU(),
            nn.Linear(192, num_actions),
        )

    def forward(
        self,
        agent_view: torch.Tensor,
        base_view: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.agent_encoder(agent_view),
                self.base_encoder(base_view),
                self.scalar_net(scalars),
            ],
            dim=-1,
        )
        return self.q_head(features)


def _array_with_shape(value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value if value is not None else np.zeros(shape), dtype=np.float32)
    if array.shape != shape:
        try:
            array = array.reshape(shape)
        except ValueError:
            array = np.zeros(shape, dtype=np.float32)
    return np.nan_to_num(array, copy=False)


def _one_hot(index: int, size: int) -> list[float]:
    result = [0.0] * size
    if 0 <= index < size:
        result[index] = 1.0
    return result


def _first(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return default
        return float(value[0])
    if value is None:
        return default
    return float(value)


def action_mask_from_observation(observation: dict[str, Any]) -> np.ndarray:
    mask = np.asarray(
        observation.get("action_mask", [1, 1, 1, 1, 1, 1]),
        dtype=np.float32,
    )
    if mask.shape != (NUM_ACTIONS,):
        fixed = np.zeros(NUM_ACTIONS, dtype=np.float32)
        fixed[: min(NUM_ACTIONS, mask.size)] = mask.ravel()[:NUM_ACTIONS]
        mask = fixed
    if mask.max(initial=0) <= 0:
        mask[-1] = 1.0
    return mask


def encode_observation_np(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    agent_view = _array_with_shape(
        observation.get("agent_viewcone"),
        AGENT_VIEW_SHAPE,
    ).transpose(2, 0, 1)
    base_view = _array_with_shape(
        observation.get("base_viewcone"),
        BASE_VIEW_SHAPE,
    ).transpose(2, 0, 1)

    location = observation.get("location", [0, 0])
    base_location = observation.get("base_location", [0, 0])
    direction = int(observation.get("direction", 0))
    mask = action_mask_from_observation(observation)

    scalars = np.asarray(
        [
            float(location[0]) / 15.0,
            float(location[1]) / 15.0,
            float(base_location[0]) / 15.0,
            float(base_location[1]) / 15.0,
            *_one_hot(direction, 4),
            _first(observation.get("health"), 60.0) / 100.0,
            float(observation.get("frozen_ticks", 0)) / 20.0,
            _first(observation.get("base_health"), 100.0) / 100.0,
            _first(observation.get("team_resources"), 0.0) / 100.0,
            float(observation.get("team_bombs", 0)) / 10.0,
            float(observation.get("step", 0)) / 200.0,
            *mask.tolist(),
        ],
        dtype=np.float32,
    )
    return agent_view, base_view, scalars, mask


def batch_encode_observations(
    observations: Iterable[dict[str, Any]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [encode_observation_np(obs) for obs in observations]
    agent, base, scalars, masks = zip(*encoded)
    return (
        torch.from_numpy(np.stack(agent)).to(device),
        torch.from_numpy(np.stack(base)).to(device),
        torch.from_numpy(np.stack(scalars)).to(device),
        torch.from_numpy(np.stack(masks)).to(device),
    )


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(action_mask <= 0, -1.0e9)
