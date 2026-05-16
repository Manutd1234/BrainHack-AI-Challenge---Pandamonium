"""PPO-backed AE manager with a high-scoring rule fallback."""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)

GRID_SIZE = int(os.getenv("AE_GRID_SIZE", "16"))
CHECKPOINT_PATH = os.getenv("AE_CHECKPOINT_PATH", "/app/model/policy.zip")

FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
PLACE_BOMB = 5

DIR_DELTA = {
    0: (1, 0),   # RIGHT
    1: (0, 1),   # DOWN
    2: (-1, 0),  # LEFT
    3: (0, -1),  # UP
}

VISIBLE = 0
WALL_RIGHT = 1
WALL_DOWN = 2
WALL_LEFT = 3
WALL_UP = 4
TILE_RECON = 6
TILE_MISSION = 7
TILE_RESOURCE = 8
ENEMY_AGENT = 10
ENEMY_BASE = 12
DESTR_WALL_RIGHT = 13
DESTR_WALL_DOWN = 14
DESTR_WALL_LEFT = 15
DESTR_WALL_UP = 16
ALLY_BOMB = 17
ENEMY_BOMB = 18
ALLY_BOMB_TIMER = 19
ENEMY_BOMB_TIMER = 20

WALL_CHANNELS = {
    0: WALL_RIGHT,
    1: WALL_DOWN,
    2: WALL_LEFT,
    3: WALL_UP,
}
DESTRUCTIBLE_CHANNELS = {
    0: DESTR_WALL_RIGHT,
    1: DESTR_WALL_DOWN,
    2: DESTR_WALL_LEFT,
    3: DESTR_WALL_UP,
}
COLLECTIBLE_CHANNELS = {
    TILE_MISSION: 5.0,
    TILE_RESOURCE: 2.0,
    TILE_RECON: 1.0,
}


class AEManager:
    """Returns one legal action for each AE observation."""

    def __init__(self) -> None:
        self.policy = None
        self.policy_kind = ""
        self._try_load_policy()
        self.reset()

    def reset(self) -> None:
        """Reset per-round rule-agent memory."""
        self.visited: set[tuple[int, int]] = set()
        self.visit_counts: dict[tuple[int, int], int] = {}
        self.known_open: set[tuple[int, int]] = set()
        self.known_walls: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.destructible_walls: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self.collectibles: dict[tuple[int, int], float] = {}
        self.enemy_bases: set[tuple[int, int]] = set()
        self.enemy_agents: set[tuple[int, int]] = set()
        self.base_threats: set[tuple[int, int]] = set()
        self.base_location: tuple[int, int] | None = None
        self.base_health = 100.0
        self.ally_bombs: dict[tuple[int, int], float] = {}
        self.enemy_bombs: dict[tuple[int, int], float] = {}
        self.last_location: tuple[int, int] | None = None
        self.last_action: int | None = None
        self.stuck_count = 0
        self.last_step = -1
        LOGGER.info("AEManager rule memory reset")

    def ae(self, observation: dict[str, Any]) -> int:
        """Return an action integer for a TIL AE observation."""
        step = int(observation.get("step", 0))
        if step == 0 or step < self.last_step:
            self.reset()
        self.last_step = step

        action_mask = self._action_mask(observation)
        self._update_memory(observation)

        if self._is_frozen(observation):
            return self._finish_action(STAY, observation)

        if self.policy is not None:
            action = self._policy_act(observation, action_mask)
            if self._is_legal(action, action_mask):
                return self._finish_action(action, observation)

        action = self._rule_act(observation, action_mask)
        return self._finish_action(action, observation)

    def act(self, observation: dict[str, Any]) -> int:
        """Compatibility alias for the standalone draft server."""
        return self.ae(observation)

    def _try_load_policy(self) -> None:
        if not os.path.exists(CHECKPOINT_PATH):
            LOGGER.warning("No PPO checkpoint at %s; using rule fallback", CHECKPOINT_PATH)
            return

        try:
            from sb3_contrib import MaskablePPO

            self.policy = MaskablePPO.load(CHECKPOINT_PATH, device="cpu")
            self.policy_kind = "maskable_ppo"
            LOGGER.info("Loaded MaskablePPO policy from %s", CHECKPOINT_PATH)
            return
        except Exception as exc:
            LOGGER.warning("MaskablePPO load failed: %s", exc)

        try:
            from stable_baselines3 import PPO

            self.policy = PPO.load(CHECKPOINT_PATH, device="cpu")
            self.policy_kind = "ppo"
            LOGGER.info("Loaded PPO policy from %s", CHECKPOINT_PATH)
        except Exception as exc:
            LOGGER.warning("PPO load failed: %s; using rule fallback", exc)
            self.policy = None

    def _policy_act(self, observation: dict[str, Any], action_mask: list[int]) -> int:
        try:
            obs = self._numpy_observation(observation)
            if self.policy_kind == "maskable_ppo":
                action, _ = self.policy.predict(
                    obs,
                    deterministic=True,
                    action_masks=np.asarray(action_mask, dtype=np.int8),
                )
            else:
                action, _ = self.policy.predict(obs, deterministic=True)
            return int(np.asarray(action).item())
        except Exception as exc:
            LOGGER.warning("Policy inference failed: %s", exc)
            return -1

    def _rule_act(self, observation: dict[str, Any], action_mask: list[int]) -> int:
        location = self._location(observation)
        direction = int(observation.get("direction", 0)) % 4

        escape = self._escape_danger(location, direction, action_mask)
        if escape is not None:
            return escape

        if self._is_legal(PLACE_BOMB, action_mask) and self._should_bomb_now(location):
            return PLACE_BOMB

        target_path = self._choose_target_path(location)
        if target_path:
            action = self._step_toward(target_path[0], location, direction, action_mask)
            if self._is_legal(action, action_mask):
                return action

        if (
            self.stuck_count >= 2
            and self._is_legal(PLACE_BOMB, action_mask)
            and self._adjacent_destructible_wall(location)
        ):
            return PLACE_BOMB

        return self._fallback_action(location, direction, action_mask)

    def _update_memory(self, observation: dict[str, Any]) -> None:
        location = self._location(observation)

        if self.last_location == location and self.last_action in (FORWARD, BACKWARD):
            self.stuck_count += 1
        elif self.last_location != location:
            self.stuck_count = 0

        self.visited.add(location)
        self.visit_counts[location] = self.visit_counts.get(location, 0) + 1
        self.known_open.add(location)
        self.collectibles.pop(location, None)
        self.enemy_agents.clear()
        self.base_threats.clear()
        self.base_location = self._optional_location(observation.get("base_location"))
        self.base_health = self._scalar(observation.get("base_health"), 100.0)
        self.ally_bombs = {
            cell: timer - 1 for cell, timer in self.ally_bombs.items() if timer > 1
        }
        self.enemy_bombs = {
            cell: timer - 1 for cell, timer in self.enemy_bombs.items() if timer > 1
        }

        direction = int(observation.get("direction", 0)) % 4
        view = observation.get("agent_viewcone", observation.get("viewcone", []))
        view_array = np.asarray(view, dtype=np.float32)

        if view_array.ndim == 3 and view_array.shape[-1] >= 21:
            self._parse_channel_viewcone(view_array, direction, location)
        elif view_array.ndim >= 2:
            self._parse_legacy_viewcone(view_array, direction, location)

        base_view = np.asarray(observation.get("base_viewcone", []), dtype=np.float32)
        if (
            self.base_location is not None
            and base_view.ndim == 3
            and base_view.shape[-1] >= 21
        ):
            self._parse_base_viewcone(base_view, self.base_location)

    def _parse_channel_viewcone(
        self,
        view: np.ndarray,
        direction: int,
        location: tuple[int, int],
    ) -> None:
        rows, cols = view.shape[:2]
        self_row = min(2, rows // 2)
        self_col = cols // 2

        for row in range(rows):
            for col in range(cols):
                cell = view[row, col]
                if cell[VISIBLE] <= 0 and cell[ENEMY_BASE] <= 0 and cell[ENEMY_AGENT] <= 0:
                    continue

                world = self._view_to_world(
                    row - self_row,
                    col - self_col,
                    direction,
                    location,
                )
                if not self._in_bounds(world):
                    continue

                self.known_open.add(world)

                for wall_dir, channel in WALL_CHANNELS.items():
                    edge = self._edge(world, wall_dir)
                    if edge is None:
                        continue
                    if cell[channel] > 0:
                        self.known_walls.add(edge)
                    elif edge in self.known_walls and cell[VISIBLE] > 0:
                        self.known_walls.discard(edge)

                for wall_dir, channel in DESTRUCTIBLE_CHANNELS.items():
                    edge = self._edge(world, wall_dir)
                    if edge is not None and cell[channel] > 0:
                        self.destructible_walls.add(edge)

                collectible_score = 0.0
                for channel, score in COLLECTIBLE_CHANNELS.items():
                    if cell[channel] > 0:
                        collectible_score = max(collectible_score, score)
                if collectible_score > 0:
                    self.collectibles[world] = collectible_score
                elif world in self.collectibles and cell[VISIBLE] > 0:
                    self.collectibles.pop(world, None)

                if cell[ENEMY_BASE] > 0:
                    self.enemy_bases.add(world)
                if cell[ENEMY_AGENT] > 0:
                    self.enemy_agents.add(world)

                if cell.shape[0] > ALLY_BOMB and cell[ALLY_BOMB] > 0:
                    timer = float(cell[ALLY_BOMB_TIMER]) if cell.shape[0] > ALLY_BOMB_TIMER else 4.0
                    self.ally_bombs[world] = timer
                else:
                    self.ally_bombs.pop(world, None)

                if cell[ENEMY_BOMB] > 0:
                    timer = float(cell[ENEMY_BOMB_TIMER]) if cell.shape[0] > ENEMY_BOMB_TIMER else 1.0
                    self.enemy_bombs[world] = timer
                else:
                    self.enemy_bombs.pop(world, None)

    def _parse_base_viewcone(
        self,
        view: np.ndarray,
        base_location: tuple[int, int],
    ) -> None:
        rows, cols = view.shape[:2]
        center_row = rows // 2
        center_col = cols // 2

        for row in range(rows):
            for col in range(cols):
                cell = view[row, col]
                if cell[VISIBLE] <= 0 and cell[ENEMY_AGENT] <= 0:
                    continue

                world = (
                    base_location[0] + row - center_row,
                    base_location[1] + col - center_col,
                )
                if not self._in_bounds(world):
                    continue

                self.known_open.add(world)

                for wall_dir, channel in WALL_CHANNELS.items():
                    edge = self._edge(world, wall_dir)
                    if edge is None:
                        continue
                    if cell[channel] > 0:
                        self.known_walls.add(edge)
                    elif edge in self.known_walls and cell[VISIBLE] > 0:
                        self.known_walls.discard(edge)

                for wall_dir, channel in DESTRUCTIBLE_CHANNELS.items():
                    edge = self._edge(world, wall_dir)
                    if edge is not None and cell[channel] > 0:
                        self.destructible_walls.add(edge)

                if cell[ENEMY_AGENT] > 0:
                    self.enemy_agents.add(world)
                    self.base_threats.add(world)

                if cell.shape[0] > ENEMY_BOMB and cell[ENEMY_BOMB] > 0:
                    timer = float(cell[ENEMY_BOMB_TIMER]) if cell.shape[0] > ENEMY_BOMB_TIMER else 1.0
                    self.enemy_bombs[world] = timer

    def _parse_legacy_viewcone(
        self,
        view: np.ndarray,
        direction: int,
        location: tuple[int, int],
    ) -> None:
        rows, cols = view.shape[:2]
        self_row = 0
        self_col = cols // 2
        for row in range(rows):
            for col in range(cols):
                world = self._view_to_world(
                    row - self_row,
                    col - self_col,
                    direction,
                    location,
                )
                if not self._in_bounds(world):
                    continue
                if float(np.ravel(view[row, col])[0]) == 0:
                    self.known_open.add(world)

    def _choose_target_path(self, location: tuple[int, int]) -> list[tuple[int, int]]:
        scored_targets: list[tuple[float, tuple[int, int]]] = []

        for base in self.enemy_bases:
            if self._in_bounds(base):
                for target in self._bombing_positions(base):
                    scored_targets.append((140.0, target))

        for threat in self.base_threats:
            if not self._in_bounds(threat):
                continue
            urgency = 90.0
            if self.base_location is not None:
                urgency += max(0.0, 5.0 - self._manhattan(threat, self.base_location)) * 14.0
            urgency += max(0.0, 100.0 - self.base_health) * 0.6
            for target in self._bombing_positions(threat):
                scored_targets.append((urgency, target))

        for cell, value in self.collectibles.items():
            if self._in_bounds(cell):
                scored_targets.append((value * 18.0, cell))

        for agent in self.enemy_agents:
            if self._in_bounds(agent):
                for target in self._bombing_positions(agent):
                    scored_targets.append((40.0, target))

        if scored_targets:
            best_path = self._best_scored_path(location, scored_targets)
            if best_path:
                return best_path

        frontier_targets = [
            cell
            for x in range(GRID_SIZE)
            for y in range(GRID_SIZE)
            for cell in [(x, y)]
            if cell not in self.visited and not self._dangerous(cell)
        ]
        return self._nearest_path(location, frontier_targets)

    def _best_scored_path(
        self,
        location: tuple[int, int],
        scored_targets: list[tuple[float, tuple[int, int]]],
    ) -> list[tuple[int, int]]:
        best_path: list[tuple[int, int]] = []
        best_value = float("-inf")

        deduped = {}
        for score, target in scored_targets:
            if self._in_bounds(target):
                deduped[target] = max(score, deduped.get(target, 0.0))

        for target, score in deduped.items():
            path = self._path_to_any(location, {target})
            if not path:
                continue
            value = score - 1.15 * len(path) - 0.35 * self.visit_counts.get(target, 0)
            if value > best_value:
                best_value = value
                best_path = path

        return best_path

    def _nearest_path(
        self,
        location: tuple[int, int],
        targets: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        return self._path_to_any(location, set(targets))

    def _path_to_any(
        self,
        start: tuple[int, int],
        goals: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if start in goals:
            return []

        queue = deque([(start, [])])
        seen = {start}

        while queue:
            cell, path = queue.popleft()
            neighbors = sorted(
                self._neighbors(cell),
                key=lambda item: (
                    self.visit_counts.get(item, 0),
                    min((self._manhattan(item, goal) for goal in goals), default=0),
                ),
            )
            for neighbor in neighbors:
                if neighbor in seen or self._dangerous(neighbor):
                    continue
                seen.add(neighbor)
                next_path = path + [neighbor]
                if neighbor in goals:
                    return next_path
                queue.append((neighbor, next_path))

        return []

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        result = []
        for direction, delta in DIR_DELTA.items():
            neighbor = (cell[0] + delta[0], cell[1] + delta[1])
            if self._in_bounds(neighbor) and not self._has_wall(cell, neighbor, direction):
                result.append(neighbor)
        return result

    def _step_toward(
        self,
        target: tuple[int, int],
        location: tuple[int, int],
        direction: int,
        action_mask: list[int],
    ) -> int:
        dx = target[0] - location[0]
        dy = target[1] - location[1]
        target_dir = self._direction_from_delta(dx, dy)
        if target_dir is None:
            return self._fallback_action(location, direction, action_mask)

        if target_dir == direction and self._is_legal(FORWARD, action_mask):
            return FORWARD
        if target_dir == (direction + 2) % 4 and self._is_legal(BACKWARD, action_mask):
            return BACKWARD
        if target_dir == (direction + 3) % 4 and self._is_legal(LEFT, action_mask):
            return LEFT
        if target_dir == (direction + 1) % 4 and self._is_legal(RIGHT, action_mask):
            return RIGHT

        return self._fallback_action(location, direction, action_mask)

    def _fallback_action(
        self,
        location: tuple[int, int],
        direction: int,
        action_mask: list[int],
    ) -> int:
        forward = self._destination(location, direction, FORWARD)
        backward = self._destination(location, direction, BACKWARD)

        if (
            self._is_legal(FORWARD, action_mask)
            and not self._dangerous(forward)
            and self.visit_counts.get(forward, 0) <= self.visit_counts.get(location, 0)
        ):
            return FORWARD
        for action in (RIGHT, LEFT):
            if self._is_legal(action, action_mask):
                return action
        if self._is_legal(BACKWARD, action_mask) and not self._dangerous(backward):
            return BACKWARD
        return self._first_legal(action_mask)

    def _escape_danger(
        self,
        location: tuple[int, int],
        direction: int,
        action_mask: list[int],
    ) -> int | None:
        if not self._dangerous(location):
            return None

        for action in (FORWARD, BACKWARD):
            destination = self._destination(location, direction, action)
            if self._is_legal(action, action_mask) and not self._dangerous(destination):
                return action
        for turn in (LEFT, RIGHT):
            if not self._is_legal(turn, action_mask):
                continue
            new_direction = (direction + (3 if turn == LEFT else 1)) % 4
            forward = self._destination(location, new_direction, FORWARD)
            if not self._dangerous(forward):
                return turn
        if self._is_legal(STAY, action_mask) and not self._dangerous(location):
            return STAY
        return self._first_legal(action_mask)

    def _should_bomb_now(self, location: tuple[int, int]) -> bool:
        if not self._can_escape_after_bomb(location):
            return False

        high_value_targets = self.enemy_bases | self.enemy_agents
        for target in high_value_targets:
            if self._blast_reaches(location, target):
                return True
        return False

    def _bombing_positions(self, target: tuple[int, int]) -> list[tuple[int, int]]:
        positions = []
        for x in range(max(0, target[0] - 2), min(GRID_SIZE, target[0] + 3)):
            for y in range(max(0, target[1] - 2), min(GRID_SIZE, target[1] + 3)):
                cell = (x, y)
                if (
                    self._blast_reaches(cell, target)
                    and not self._dangerous(cell)
                    and self._can_escape_after_bomb(cell)
                ):
                    positions.append(cell)
        return positions

    def _blast_reaches(self, origin: tuple[int, int], target: tuple[int, int]) -> bool:
        if max(abs(origin[0] - target[0]), abs(origin[1] - target[1])) > 2:
            return False
        return True

    def _adjacent_destructible_wall(self, location: tuple[int, int]) -> bool:
        for direction in DIR_DELTA:
            edge = self._edge(location, direction)
            if edge in self.destructible_walls:
                return True
        return False

    def _dangerous(self, cell: tuple[int, int]) -> bool:
        for bomb, timer in self.ally_bombs.items():
            if timer <= 4.0 and self._blast_reaches(bomb, cell):
                return True
        for bomb, timer in self.enemy_bombs.items():
            if timer <= 2.0 and self._blast_reaches(bomb, cell):
                return True
        return False

    def _can_escape_after_bomb(self, location: tuple[int, int]) -> bool:
        """Return whether there is a nearby tile outside a newly placed bomb."""
        queue = deque([(location, 0)])
        seen = {location}

        while queue:
            cell, depth = queue.popleft()
            if depth > 0 and not self._blast_reaches(location, cell) and not self._dangerous(cell):
                return True
            if depth >= 4:
                continue
            for neighbor in self._neighbors(cell):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, depth + 1))

        return False

    def _view_to_world(
        self,
        row_offset: int,
        col_offset: int,
        direction: int,
        location: tuple[int, int],
    ) -> tuple[int, int]:
        x, y = location
        if direction == 0:
            return x + row_offset, y + col_offset
        if direction == 1:
            return x - col_offset, y + row_offset
        if direction == 2:
            return x - row_offset, y - col_offset
        return x + col_offset, y - row_offset

    def _destination(
        self,
        location: tuple[int, int],
        direction: int,
        action: int,
    ) -> tuple[int, int]:
        move_dir = direction if action == FORWARD else (direction + 2) % 4
        delta = DIR_DELTA[move_dir]
        return location[0] + delta[0], location[1] + delta[1]

    def _direction_from_delta(self, dx: int, dy: int) -> int | None:
        for direction, delta in DIR_DELTA.items():
            if (dx, dy) == delta:
                return direction
        return None

    def _edge(
        self,
        cell: tuple[int, int],
        direction: int,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        delta = DIR_DELTA[direction]
        neighbor = (cell[0] + delta[0], cell[1] + delta[1])
        if not self._in_bounds(cell) or not self._in_bounds(neighbor):
            return None
        return tuple(sorted((cell, neighbor)))

    def _has_wall(
        self,
        cell: tuple[int, int],
        neighbor: tuple[int, int],
        direction: int,
    ) -> bool:
        edge = self._edge(cell, direction)
        return edge in self.known_walls

    def _location(self, observation: dict[str, Any]) -> tuple[int, int]:
        location = observation.get("location", [0, 0])
        return int(location[0]), int(location[1])

    def _optional_location(self, value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        arr = np.asarray(value).reshape(-1)
        if arr.size < 2:
            return None
        location = (int(arr[0]), int(arr[1]))
        return location if self._in_bounds(location) else None

    def _scalar(self, value: Any, default: float) -> float:
        if value is None:
            return default
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return default
        return float(arr[0])

    def _is_frozen(self, observation: dict[str, Any]) -> bool:
        return int(observation.get("frozen_ticks", 0)) > 0

    def _action_mask(self, observation: dict[str, Any]) -> list[int]:
        mask = observation.get("action_mask")
        if mask is None:
            bombs_value = observation.get("team_bombs", 0)
            bombs = int(np.asarray(bombs_value).reshape(-1)[0])
            return [1, 1, 1, 1, 1, int(bombs > 0)]
        return [int(v) for v in list(mask)]

    def _is_legal(self, action: int, action_mask: list[int]) -> bool:
        return 0 <= action < len(action_mask) and bool(action_mask[action])

    def _first_legal(self, action_mask: list[int]) -> int:
        for action in (FORWARD, BACKWARD, LEFT, RIGHT, PLACE_BOMB, STAY):
            if self._is_legal(action, action_mask):
                return action
        return STAY

    def _finish_action(self, action: int, observation: dict[str, Any]) -> int:
        action_mask = self._action_mask(observation)
        if not self._is_legal(action, action_mask):
            action = self._first_legal(action_mask)
        self.last_action = int(action)
        self.last_location = self._location(observation)
        if action == PLACE_BOMB:
            self.ally_bombs[self.last_location] = 4.0
        return int(action)

    def _numpy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in observation.items():
            if key in {"direction", "frozen_ticks", "team_bombs", "step"}:
                result[key] = int(value)
            else:
                result[key] = np.asarray(value)
        return result

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < GRID_SIZE and 0 <= cell[1] < GRID_SIZE

    def _manhattan(self, left: tuple[int, int], right: tuple[int, int]) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])
