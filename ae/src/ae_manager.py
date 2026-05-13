"""Manages the AE model.

The manager uses a trained discrete-SAC ResNet policy when weights are present
and falls back to a tactical exploration heuristic otherwise. This keeps the
submission functional before long-running AE training has completed.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AEManager:
    def __init__(self):
        self.visited: set[tuple[int, int]] = set()
        self.visit_counts: dict[tuple[int, int], int] = {}
        self.last_action = None
        self.stuck_counter = 0
        self.last_location: tuple[int, int] | None = None
        self.step_count = 0
        self.policy = None
        self.device = None
        self._load_policy()

    def _load_policy(self) -> None:
        candidates = [
            os.getenv("AE_POLICY_PATH"),
            "models/ae/sac_resnet_policy.pt",
            "models/sac_resnet_policy.pt",
            "sac_resnet_policy.pt",
        ]
        candidates = [path for path in candidates if path]

        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                import torch
                from ae_model import SACPolicyNetwork

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(path, map_location=self.device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.policy = SACPolicyNetwork()
                self.policy.load_state_dict(state_dict)
                self.policy.to(self.device)
                self.policy.eval()
                logger.info("Loaded AE SAC policy from %s on %s", path, self.device)
                return
            except Exception as exc:
                logger.warning("Could not load AE policy %s: %s", path, exc)

        logger.info("No AE SAC policy found; using tactical heuristic.")

    def _policy_action(self, observation: dict[str, Any], action_mask: list[int]) -> int | None:
        if self.policy is None or self.device is None:
            return None

        try:
            import torch
            from ae_model import batch_encode_observations, masked_logits

            with torch.no_grad():
                agent_view, base_view, scalars, masks = batch_encode_observations(
                    [observation],
                    self.device,
                )
                logits = self.policy(agent_view, base_view, scalars)
                logits = masked_logits(logits, masks)
                action = int(torch.argmax(logits, dim=-1).item())
            if 0 <= action < len(action_mask) and action_mask[action] == 1:
                return action
        except Exception as exc:
            logger.warning("AE policy inference failed; falling back: %s", exc)
        return None

    def ae(self, observation: dict[str, Any]) -> int:
        """Gets the next action for the agent."""
        action_mask = observation.get("action_mask", [1, 1, 1, 1, 1, 1])
        location = tuple(observation.get("location", [0, 0]))
        direction = int(observation.get("direction", 0))
        step = int(observation.get("step", 0))
        agent_viewcone = observation.get("agent_viewcone", None)
        frozen_ticks = int(observation.get("frozen_ticks", 0))
        team_resources = self._first(observation.get("team_resources"), 0.0)
        team_bombs = int(observation.get("team_bombs", 0))

        self.step_count = step

        if frozen_ticks > 0:
            return 5

        if self.last_location == location:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_location = location
        self.visited.add(location)
        self.visit_counts[location] = self.visit_counts.get(location, 0) + 1

        if action_mask[3] == 1:
            return 3

        policy_action = self._policy_action(observation, action_mask)
        if policy_action is not None:
            return policy_action

        if self.stuck_counter > 6 and action_mask[4] == 1 and team_bombs > 0:
            self.stuck_counter = 0
            return 4

        if self.stuck_counter > 3:
            self.stuck_counter = 0
            turn_actions = [a for a in (1, 2) if action_mask[a] == 1]
            if turn_actions:
                return random.choice(turn_actions)
            return self._random_valid(action_mask)

        return self._heuristic_action(
            action_mask=action_mask,
            location=location,
            direction=direction,
            agent_viewcone=agent_viewcone,
            team_resources=team_resources,
            team_bombs=team_bombs,
        )

    def _heuristic_action(
        self,
        action_mask: list[int],
        location: tuple[int, int],
        direction: int,
        agent_viewcone: Any,
        team_resources: float,
        team_bombs: int,
    ) -> int:
        dx = [0, 1, 0, -1]
        dy = [-1, 0, 1, 0]

        directional_locations = {
            0: (location[0] + dx[direction], location[1] + dy[direction]),
            1: (
                location[0] + dx[(direction - 1) % 4],
                location[1] + dy[(direction - 1) % 4],
            ),
            2: (
                location[0] + dx[(direction + 1) % 4],
                location[1] + dy[(direction + 1) % 4],
            ),
        }

        blocked = {0: action_mask[0] == 0, 1: False, 2: False}
        frontier_bonus = {0: 0.0, 1: 0.0, 2: 0.0}

        if agent_viewcone is not None:
            try:
                vc = np.asarray(agent_viewcone, dtype=np.float32)
                if vc.shape[0] > 0 and vc.shape[1] > 3:
                    blocked[1] = bool(vc[0, 1, 0] > 0.5)
                    blocked[2] = bool(vc[0, 3, 0] > 0.5)
                    frontier_bonus = self._viewcone_frontier_bonus(vc)
            except Exception:
                pass

        choices = []
        for action in (0, 1, 2):
            if action_mask[action] == 0 or blocked[action]:
                continue
            visit_penalty = self.visit_counts.get(directional_locations[action], 0)
            turn_penalty = 0.15 if action in (1, 2) else 0.0
            score = visit_penalty + turn_penalty - frontier_bonus[action]
            choices.append((action, score))

        if choices:
            choices.sort(key=lambda item: item[1])
            best_score = choices[0][1]
            ties = [action for action, score in choices if abs(score - best_score) < 0.05]
            return random.choice(ties)

        if action_mask[4] == 1 and team_bombs > 0 and team_resources >= 1:
            return 4
        return self._random_valid(action_mask)

    @staticmethod
    def _viewcone_frontier_bonus(viewcone: np.ndarray) -> dict[int, float]:
        # Channel semantics can evolve; using occupancy-style low values keeps
        # this bonus conservative while still steering toward open unknown space.
        open_cells = (viewcone[:, :, 0] < 0.5).astype(np.float32)
        return {
            0: float(open_cells[:4, 1:4].mean()),
            1: float(open_cells[:3, :2].mean()),
            2: float(open_cells[:3, 3:].mean()),
        }

    @staticmethod
    def _first(value: Any, default: float) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return default
            return float(value[0])
        if value is None:
            return default
        return float(value)

    @staticmethod
    def _random_valid(action_mask: list[int]) -> int:
        valid = [i for i, m in enumerate(action_mask) if m == 1]
        if valid:
            return random.choice(valid)
        return 5
