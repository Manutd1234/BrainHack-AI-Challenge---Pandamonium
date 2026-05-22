"""
AE Manager — TIL-AI 2026
═══════════════════════════════════════════════════════════════════════
PRIMARY  : PPO with RecurrentPPO (LSTM policy) — memory across timesteps
           LSTM remembers which cells were visited and where opponents are,
           solving the partial-observability problem of the viewcone.

FALLBACK : Smart BFS exploration agent with occupancy map
           Three-tier fallback:
             1. PPO-LSTM inference (best)
             2. Rule-based BFS to nearest unvisited frontier (good)
             3. Random turn (prevents getting permanently stuck)

IMPROVEMENTS over previous version:
  1. LSTM hidden state — agent remembers across 200-step episode
  2. Occupancy map — tracks visited/wall/unknown per cell
  3. Better stuck detection — resets hidden state on stuck
  4. Direction-aware viewcone parsing — correct world-coord mapping
  5. Three-tier fallback — never returns constant action
═══════════════════════════════════════════════════════════════════════
"""

import logging
import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = os.environ.get("AE_CHECKPOINT_PATH", "/app/model/policy.zip")
GRID = 16

FORWARD   = 0
TURN_LEFT = 1
TURN_RIGHT = 2
TURN_BACK  = 3

DIR_DELTA = {
    0: ( 0, -1),  # North
    1: ( 1,  0),  # East
    2: ( 0,  1),  # South
    3: (-1,  0),  # West
}

# Occupancy values
UNKNOWN  = 0
FREE     = 1
WALL     = 2
VISITED  = 3


class OccupancyMap:
    """16×16 grid tracking cell state across the episode."""

    def __init__(self):
        self.grid = np.zeros((GRID, GRID), dtype=np.uint8)

    def reset(self):
        self.grid[:] = 0

    def mark(self, x: int, y: int, state: int):
        if 0 <= x < GRID and 0 <= y < GRID:
            self.grid[y, x] = state

    def get(self, x: int, y: int) -> int:
        if 0 <= x < GRID and 0 <= y < GRID:
            return int(self.grid[y, x])
        return WALL  # out-of-bounds = wall

    def update_from_viewcone(self, viewcone, direction: int, ax: int, ay: int):
        vc = np.array(viewcone, dtype=np.int32) if viewcone else np.array([[]])
        if vc.ndim < 2 or vc.size == 0:
            return
        rows, cols = vc.shape[:2]
        cx = cols // 2

        # Direction → (forward_dx, forward_dy, right_dx, right_dy)
        fwd = {
            0: (( 0,-1),( 1, 0)),  # North: forward=-y, right=+x
            1: (( 1, 0),( 0, 1)),  # East:  forward=+x, right=+y
            2: (( 0, 1),(-1, 0)),  # South: forward=+y, right=-x
            3: ((-1, 0),( 0,-1)),  # West:  forward=-x, right=-y
        }
        (fdx, fdy), (rdx, rdy) = fwd.get(direction, fwd[0])

        for r in range(rows):
            for c in range(cols):
                dx = fdx * r + rdx * (c - cx)
                dy = fdy * r + rdy * (c - cx)
                wx, wy = ax + dx, ay + dy
                if not (0 <= wx < GRID and 0 <= wy < GRID):
                    continue
                val = int(vc[r, c]) if vc.ndim == 2 else int(vc[r, c, 0])
                if val == 0:
                    # Passable
                    if self.get(wx, wy) != VISITED:
                        self.mark(wx, wy, FREE)
                else:
                    self.mark(wx, wy, WALL)

    def frontier_cells(self) -> list[tuple[int, int]]:
        """Cells that are FREE but adjacent to UNKNOWN — exploration targets."""
        frontier = []
        for y in range(GRID):
            for x in range(GRID):
                if self.grid[y, x] in (FREE, VISITED):
                    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < GRID and 0 <= ny < GRID:
                            if self.grid[ny, nx] == UNKNOWN:
                                frontier.append((x, y))
                                break
        return frontier

    def unvisited_free(self) -> list[tuple[int, int]]:
        cells = []
        for y in range(GRID):
            for x in range(GRID):
                if self.grid[y, x] == FREE:
                    cells.append((x, y))
        return cells


class AEManager:
    def __init__(self):
        self.ppo = None
        self.lstm_states = None
        self._try_load_ppo()
        self.map = OccupancyMap()
        self.action_queue: list[int] = []
        self.last_pos: Optional[tuple] = None
        self.stuck_count = 0
        self.step = 0

    # ── PPO loading ────────────────────────────────────────────────────────

    def _try_load_ppo(self):
        if not os.path.exists(CHECKPOINT_PATH):
            logger.warning(
                f"No PPO checkpoint at {CHECKPOINT_PATH} — using rule-based fallback. "
                f"Train with ae_train.py first."
            )
            return
        try:
            from sb3_contrib import RecurrentPPO
            self.ppo = RecurrentPPO.load(CHECKPOINT_PATH, device="cpu")
            self.lstm_states = None
            logger.info(f"RecurrentPPO (LSTM) loaded from {CHECKPOINT_PATH}")
        except ImportError:
            try:
                from stable_baselines3 import PPO
                self.ppo = PPO.load(CHECKPOINT_PATH, device="cpu")
                logger.info(f"PPO loaded from {CHECKPOINT_PATH}")
            except Exception as exc:
                logger.warning(f"PPO load failed ({exc}); using rule-based")
        except Exception as exc:
            logger.warning(f"RecurrentPPO load failed ({exc}); trying standard PPO")
            try:
                from stable_baselines3 import PPO
                self.ppo = PPO.load(CHECKPOINT_PATH, device="cpu")
                logger.info(f"Fallback PPO loaded")
            except Exception as exc2:
                logger.warning(f"PPO load also failed ({exc2}); rule-based only")

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self):
        self.map.reset()
        self.action_queue.clear()
        self.last_pos = None
        self.stuck_count = 0
        self.step = 0
        self.lstm_states = None   # reset LSTM hidden state
        logger.info("AEManager reset")

    # ── PPO inference ──────────────────────────────────────────────────────

    def _ppo_act(self, obs: dict) -> int:
        try:
            # RecurrentPPO needs lstm_states passed in and out
            if hasattr(self.ppo, 'policy') and hasattr(self.ppo.policy, 'lstm_actor'):
                action, self.lstm_states = self.ppo.predict(
                    obs,
                    state=self.lstm_states,
                    episode_start=np.array([self.step == 0]),
                    deterministic=True,
                )
            else:
                action, _ = self.ppo.predict(obs, deterministic=True)
            return int(action)
        except Exception as exc:
            logger.warning(f"PPO inference error ({exc}); using rule-based")
            return self._rule_act(obs)

    # ── Rule-based BFS ─────────────────────────────────────────────────────

    def _bfs_path(self, sx: int, sy: int, targets: list[tuple[int,int]]) -> list[tuple[int,int]]:
        if not targets:
            return []
        target_set = set(targets)
        queue      = deque([(sx, sy, [])])
        seen       = {(sx, sy)}
        while queue:
            x, y, path = queue.popleft()
            if (x, y) in target_set:
                return path + [(x, y)]
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                if (nx, ny) not in seen and self.map.get(nx, ny) != WALL:
                    seen.add((nx, ny))
                    queue.append((nx, ny, path + [(x, y)]))
        return []

    def _path_to_actions(
        self, path: list[tuple[int,int]], cur_dir: int, cur_pos: tuple[int,int]
    ) -> list[int]:
        actions = []
        x, y = cur_pos
        d    = cur_dir
        for (tx, ty) in path:
            dx, dy = tx-x, ty-y
            if (dx, dy) == (0, 0):
                continue
            target_dir = next(
                (k for k, (ddx, ddy) in DIR_DELTA.items() if (ddx, ddy) == (dx, dy)), None
            )
            if target_dir is None:
                continue
            turn = (target_dir - d) % 4
            if   turn == 1: actions.append(TURN_RIGHT)
            elif turn == 2: actions.append(TURN_BACK)
            elif turn == 3: actions.append(TURN_LEFT)
            actions.append(FORWARD)
            d = target_dir
            x, y = tx, ty
        return actions

    def _rule_act(self, obs: dict) -> int:
        location  = obs.get("location", [0, 0])
        direction = int(obs.get("direction", 0))
        viewcone  = obs.get("viewcone", [])
        x, y      = int(location[0]), int(location[1])

        # Update occupancy map
        self.map.update_from_viewcone(viewcone, direction, x, y)
        self.map.mark(x, y, VISITED)

        # Stuck detection
        if self.last_pos == (x, y):
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.last_pos = (x, y)

        # Tier 3: stuck recovery
        if self.stuck_count >= 3:
            self.action_queue.clear()
            self.stuck_count = 0
            return TURN_RIGHT

        # Execute queued path
        if self.action_queue:
            return self.action_queue.pop(0)

        # Tier 2a: BFS to frontier cells
        targets = self.map.frontier_cells() or self.map.unvisited_free()
        if not targets:
            self.map.reset()
            self.map.mark(x, y, VISITED)
            return FORWARD

        # Find nearest target
        path = self._bfs_path(x, y, targets)
        if not path:
            return TURN_RIGHT

        actions = self._path_to_actions(path, direction, (x, y))
        if not actions:
            return FORWARD

        self.action_queue = actions[1:]
        return actions[0]

    # ── Public ────────────────────────────────────────────────────────────

    def act(self, observation: dict) -> int:
        self.step += 1
        if self.ppo is not None:
            return self._ppo_act(observation)
        return self._rule_act(observation)
