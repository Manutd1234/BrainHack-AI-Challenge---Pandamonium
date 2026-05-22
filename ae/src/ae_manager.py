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
  6. Hybrid Safety-First dodging and Strategic Bombing filters
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

# Correct Environment Actions (from actions.py)
FORWARD     = 0
BACKWARD    = 1
LEFT        = 2
RIGHT       = 3
STAY        = 4
PLACE_BOMB  = 5

# Authorized direction delta (matches Direction IntEnum RIGHT=0, DOWN=1, LEFT=2, UP=3)
DIR_DELTA = {
    0: ( 1,  0),  # RIGHT (East)
    1: ( 0,  1),  # DOWN (South)
    2: (-1,  0),  # LEFT (West)
    3: ( 0, -1),  # UP (North)
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
        vc = np.array(viewcone, dtype=np.float32) if viewcone is not None and len(viewcone) > 0 else np.array([[]])
        if vc.ndim < 3 or vc.size == 0:
            return
        rows, cols = vc.shape[:2]
        cx = cols // 2

        # direction (0=RIGHT, 1=DOWN, 2=LEFT, 3=UP)
        fwd = {
            0: (( 1,  0), ( 0,  1)),  # RIGHT: forward=+x, right=+y
            1: (( 0,  1), (-1,  0)),  # DOWN:  forward=+y, right=-x
            2: ((-1,  0), ( 0, -1)),  # LEFT:  forward=-x, right=-y
            3: (( 0, -1), ( 1,  0)),  # UP:    forward=-y, right=+x
        }
        (fdx, fdy), (rdx, rdy) = fwd.get(direction, fwd[0])

        for r in range(rows):
            for c in range(cols):
                if vc[r, c, 0] != 1.0:  # Only parse visible cells
                    continue
                dx = fdx * (r - 2) + rdx * (c - cx)
                dy = fdy * (r - 2) + rdy * (c - cx)
                wx, wy = ax + dx, ay + dy
                if not (0 <= wx < GRID and 0 <= wy < GRID):
                    continue
                # If there are any wall edges on the cell, mark it blocked
                # WALL_RIGHT=1, WALL_DOWN=2, WALL_LEFT=3, WALL_UP=4
                has_wall_edge = (
                    vc[r, c, 1] == 1.0 or
                    vc[r, c, 2] == 1.0 or
                    vc[r, c, 3] == 1.0 or
                    vc[r, c, 4] == 1.0
                )
                if has_wall_edge:
                    self.mark(wx, wy, WALL)
                else:
                    if self.get(wx, wy) != VISITED:
                        self.mark(wx, wy, FREE)

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
        self.ppo_loaded = False
        self.map = OccupancyMap()
        self.action_queue: list[int] = []
        self.last_pos: Optional[tuple] = None
        self.stuck_count = 0
        self.step = 0

    def _ensure_ppo_loaded(self):
        if not self.ppo_loaded:
            self._try_load_ppo()
            self.ppo_loaded = True

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
            if turn == 1:
                actions.append(RIGHT)
                d = (d + 1) % 4
            elif turn == 2:
                actions.append(BACKWARD)
                x, y = tx, ty
                continue
            elif turn == 3:
                actions.append(LEFT)
                d = (d - 1) % 4
            actions.append(FORWARD)
            x, y = tx, ty
        return actions

    def _rule_act(self, obs: dict) -> int:
        location  = obs.get("location", [0, 0])
        direction = int(obs.get("direction", 0))
        viewcone  = obs.get("agent_viewcone", obs.get("viewcone", []))
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
            return RIGHT

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
            return RIGHT

        actions = self._path_to_actions(path, direction, (x, y))
        if not actions:
            return FORWARD

        self.action_queue = actions[1:]
        return actions[0]

    # ── Safety-Dodging & Bombing Helpers ────────────────────────────────────

    def _los_to_cell(self, bx: int, by: int, tx: int, ty: int) -> bool:
        if bx == tx and by == ty:
            return True
        x0, y0 = bx, by
        x1, y1 = tx, ty
        dx = x1 - x0
        dy = y1 - y0
        nx = abs(dx)
        ny = abs(dy)
        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0
        px, py = x0, y0
        ix = iy = 0
        while ix < nx or iy < ny:
            if (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
                px += sign_x
                py += sign_y
                ix += 1
                iy += 1
            elif (1 + 2 * ix) * ny < (1 + 2 * iy) * nx:
                px += sign_x
                ix += 1
            else:
                py += sign_y
                iy += 1
            if (px, py) != (tx, ty):
                if self.map.get(px, py) == WALL:
                    return False
        return True

    def _get_threatened_cells(self, bombs: list[tuple[int, int]]) -> set[tuple[int, int]]:
        threatened = set()
        for bx, by in bombs:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    tx, ty = bx + dx, by + dy
                    if 0 <= tx < GRID and 0 <= ty < GRID:
                        if self._los_to_cell(bx, by, tx, ty):
                            threatened.add((tx, ty))
        return threatened


    def _find_escape_path(self, sx: int, sy: int, threatened: set[tuple[int, int]]) -> list[tuple[int, int]]:
        queue = deque([(sx, sy, [])])
        seen = {(sx, sy)}
        while queue:
            x, y, path = queue.popleft()
            if (x, y) not in threatened:
                return path + [(x, y)]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID and 0 <= ny < GRID:
                    if (nx, ny) not in seen and self.map.get(nx, ny) != WALL:
                        seen.add((nx, ny))
                        queue.append((nx, ny, path + [(x, y)]))
        return []

    def _can_safely_place_bomb(self, ax: int, ay: int, current_bombs: list[tuple[int, int]]) -> bool:
        all_bombs = current_bombs + [(ax, ay)]
        threatened = self._get_threatened_cells(all_bombs)
        escape_path = self._find_escape_path(ax, ay, threatened)
        return len(escape_path) > 0

    def _fallback_act(self, observation: dict) -> int:
        if self.ppo is not None:
            return self._ppo_act(observation)
        return self._rule_act(observation)

    # ── Public ────────────────────────────────────────────────────────────

    def act(self, observation: dict) -> int:
        self.step += 1
        self._ensure_ppo_loaded()

        # 1. Parse observation and get key state variables
        location  = observation.get("location", [0, 0])
        direction = int(observation.get("direction", 0))
        viewcone  = observation.get("agent_viewcone", observation.get("viewcone", []))
        action_mask = observation.get("action_mask", [1, 1, 1, 1, 1, 1])
        x, y      = int(location[0]), int(location[1])

        # 2. Update occupancy map
        self.map.update_from_viewcone(viewcone, direction, x, y)
        self.map.mark(x, y, VISITED)

        # 3. Analyze viewcone
        vc = np.array(viewcone, dtype=np.float32) if viewcone is not None and len(viewcone) > 0 else np.array([[]])
        
        # If viewcone is empty/invalid, use fallback directly
        if vc.ndim < 3 or vc.size == 0:
            return self._fallback_act(observation)

        rows, cols = vc.shape[:2]
        cx = cols // 2

        (fdx, fdy), (rdx, rdy) = DIR_DELTA.get(direction, DIR_DELTA[0])

        # Collect active bombs in world coords
        visible_bombs = []
        for r in range(rows):
            for c in range(cols):
                if vc[r, c, 17] == 1.0 or vc[r, c, 18] == 1.0:
                    dx = fdx * (r - 2) + rdx * (c - cx)
                    dy = fdy * (r - 2) + rdy * (c - cx)
                    bx, by = x + dx, y + dy
                    if 0 <= bx < GRID and 0 <= by < GRID:
                        visible_bombs.append((bx, by))

        # Get threatened cells
        threatened = self._get_threatened_cells(visible_bombs)

        # 4. SAFETY Dodging check
        if (x, y) in threatened:
            escape_path = self._find_escape_path(x, y, threatened)
            if len(escape_path) >= 2:
                tx, ty = escape_path[1]
                dx, dy = tx - x, ty - y
                target_dir = next((k for k, (ddx, ddy) in DIR_DELTA.items() if (ddx, ddy) == (dx, dy)), None)
                if target_dir is not None:
                    turn = (target_dir - direction) % 4
                    if turn == 2 and action_mask[BACKWARD] == 1:
                        self.action_queue.clear()
                        return BACKWARD
                    elif turn == 1 and action_mask[RIGHT] == 1:
                        self.action_queue.clear()
                        return RIGHT
                    elif turn == 3 and action_mask[LEFT] == 1:
                        self.action_queue.clear()
                        return LEFT
                    elif turn == 0 and action_mask[FORWARD] == 1:
                        self.action_queue.clear()
                        return FORWARD

        # 5. STRATEGIC BOMBING check
        if action_mask[PLACE_BOMB] == 1:
            is_adj_enemy_or_breakable = False
            # Destructible walls adjacent to agent cell (2, 2)
            if (vc[2, 2, 13] == 1.0 or
                vc[2, 2, 14] == 1.0 or
                vc[2, 2, 15] == 1.0 or
                vc[2, 2, 16] == 1.0):
                is_adj_enemy_or_breakable = True
                
            # Adjacent cells (ahead, behind, left, right)
            adj_coords = [(3, 2), (1, 2), (2, 1), (2, 3)]
            for ar, ac in adj_coords:
                if ar < rows and ac < cols:
                    if vc[ar, ac, 10] == 1.0 or vc[ar, ac, 12] == 1.0:
                        is_adj_enemy_or_breakable = True
                        break

            if is_adj_enemy_or_breakable:
                if self._can_safely_place_bomb(x, y, visible_bombs):
                    self.action_queue.clear()
                    return PLACE_BOMB

        # 6. BASE ACTION selection (PPO or BFS Explorer)
        base_act = self._fallback_act(observation)

        # 7. PRO-ACTIVE SAFETY BLOCK: avoid walking into danger
        if base_act == FORWARD and action_mask[FORWARD] == 1:
            next_pos = (x + fdx, y + fdy)
            if next_pos in threatened:
                self.action_queue.clear()
                if action_mask[STAY] == 1:
                    return STAY
                elif action_mask[LEFT] == 1:
                    return LEFT
                elif action_mask[RIGHT] == 1:
                    return RIGHT

        if base_act == BACKWARD and action_mask[BACKWARD] == 1:
            next_pos = (x - fdx, y - fdy)
            if next_pos in threatened:
                self.action_queue.clear()
                if action_mask[STAY] == 1:
                    return STAY
                elif action_mask[LEFT] == 1:
                    return LEFT
                elif action_mask[RIGHT] == 1:
                    return RIGHT

        return base_act
