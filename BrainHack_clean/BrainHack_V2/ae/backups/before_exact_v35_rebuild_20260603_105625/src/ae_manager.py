from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Any

from ae_manager_core import AEManager as CoreAEManager

FORWARD, BACKWARD, LEFT, RIGHT, STAY, PLACE_BOMB = 0, 1, 2, 3, 4, 5
GRID = 16
CENTER = (7.5, 7.5)
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DELTA_TO_DIR = {delta: i for i, delta in enumerate(DIRS)}
OPPOSITE = {0: 2, 1: 3, 2: 0, 3: 1}

VISIBLE = 0
WALL_RIGHT, WALL_DOWN, WALL_LEFT, WALL_UP = 1, 2, 3, 4
TILE_RECON, TILE_MISSION, TILE_RESOURCE = 6, 7, 8
ENEMY_AGENT, ENEMY_BASE = 10, 12
DESTR_WALL_RIGHT, DESTR_WALL_DOWN, DESTR_WALL_LEFT, DESTR_WALL_UP = 13, 14, 15, 16
ALLY_BOMB, ENEMY_BOMB, ALLY_BOMB_TIMER, ENEMY_BOMB_TIMER = 17, 18, 19, 20

WALL_CHANNELS = [WALL_RIGHT, WALL_DOWN, WALL_LEFT, WALL_UP]
DESTR_CHANNELS = [DESTR_WALL_RIGHT, DESTR_WALL_DOWN, DESTR_WALL_LEFT, DESTR_WALL_UP]
COLLECTIBLE_CHANNELS = {TILE_MISSION: "mission", TILE_RESOURCE: "resource", TILE_RECON: "recon"}
COLLECTIBLE_VALUE = {"mission": 12.0, "resource": 5.5, "recon": 1.5}


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", "off"}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _ch(channels: Any, idx: int) -> float:
    try:
        return float(channels[idx])
    except Exception:
        return 0.0


def _mask(obs: dict[str, Any]) -> list[int]:
    raw = obs.get("action_mask") or [0, 0, 0, 0, 1, 0]
    out = [0, 0, 0, 0, 0, 0]
    for i in range(min(6, len(raw))):
        out[i] = 1 if raw[i] else 0
    return out


def _fallback(obs: dict[str, Any]) -> int:
    mask = _mask(obs)
    for action in (FORWARD, BACKWARD, RIGHT, LEFT, STAY, PLACE_BOMB):
        if mask[action]:
            return action
    return STAY


def _cell_from(value: Any, default: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    try:
        return (_as_int(value[0]), _as_int(value[1]))
    except Exception:
        return default


def _in_bounds(cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < GRID and 0 <= cell[1] < GRID


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _center_bonus(cell: tuple[int, int]) -> float:
    return 1.0 - (abs(cell[0] - CENTER[0]) + abs(cell[1] - CENTER[1])) / 15.0


def _agent_view_world(loc, direction, row, col, height, width):
    behind = 2 if height >= 7 else height // 2
    left = 2 if width >= 5 else width // 2
    front = row - behind
    side = col - left
    x, y = loc
    if direction == 0:
        return (x + front, y + side)
    if direction == 1:
        return (x - side, y + front)
    if direction == 2:
        return (x - front, y - side)
    return (x + side, y - front)


def _base_view_world(base, row, col, height, width):
    return (base[0] + row - height // 2, base[1] + col - width // 2)


@dataclass
class GuardState:
    last_step: int | None = None
    loc: tuple[int, int] = (0, 0)
    prev_loc: tuple[int, int] | None = None
    direction: int = 0
    base_loc: tuple[int, int] = (0, 0)
    obs_step: int = 0
    blocked_edges: set = field(default_factory=set)
    destructible_edges: set = field(default_factory=set)
    collectibles: dict = field(default_factory=dict)
    visible_collectibles: set = field(default_factory=set)
    visible_enemy_bases: set = field(default_factory=set)
    visible_enemy_agents: set = field(default_factory=set)
    bombs: dict = field(default_factory=dict)
    visits: dict = field(default_factory=dict)
    last_action: int = STAY
    stuck: int = 0
    bomb_cooldown: int = 0

    def reset(self):
        self.__dict__.update(GuardState().__dict__)

    def update(self, obs: dict[str, Any]) -> None:
        step = _as_int(obs.get("step"), 0)
        delta = max(0, step - self.obs_step)
        self.obs_step = step
        self.bomb_cooldown = max(0, self.bomb_cooldown - delta)

        aged = {}
        for pos, (team, timer) in self.bombs.items():
            timer -= delta
            if timer > 0:
                aged[pos] = (team, timer)
        self.bombs = aged

        self.prev_loc = self.loc
        self.loc = _cell_from(obs.get("location"), self.loc)
        self.direction = _as_int(obs.get("direction"), self.direction) % 4
        self.base_loc = _cell_from(obs.get("base_location"), self.base_loc)

        if self.prev_loc == self.loc and self.last_action in (FORWARD, BACKWARD):
            self.stuck += 1
        elif self.prev_loc != self.loc:
            self.stuck = 0

        self.visits[self.loc] = self.visits.get(self.loc, 0) + 1
        self.collectibles.pop(self.loc, None)

        self.visible_collectibles = set()
        self.visible_enemy_bases = set()
        self.visible_enemy_agents = set()

        mask_now = _mask(obs)
        self._set_edge(self.loc, self.direction, not bool(mask_now[FORWARD]))
        self._set_edge(self.loc, OPPOSITE[self.direction], not bool(mask_now[BACKWARD]))

        self._decode_agent_view(obs.get("agent_viewcone") or [])
        self._decode_base_view(obs.get("base_viewcone") or [])

    def _set_edge(self, cell, direction, blocked):
        if not _in_bounds(cell):
            return
        key = (cell[0], cell[1], direction)
        if blocked:
            self.blocked_edges.add(key)
        else:
            self.blocked_edges.discard(key)
        dx, dy = DIRS[direction]
        other = (cell[0] + dx, cell[1] + dy)
        if _in_bounds(other):
            rkey = (other[0], other[1], OPPOSITE[direction])
            if blocked:
                self.blocked_edges.add(rkey)
            else:
                self.blocked_edges.discard(rkey)

    def _set_destructible(self, cell, direction, destructible):
        key = (cell[0], cell[1], direction)
        if destructible:
            self.destructible_edges.add(key)
        else:
            self.destructible_edges.discard(key)
        dx, dy = DIRS[direction]
        other = (cell[0] + dx, cell[1] + dy)
        if _in_bounds(other):
            rkey = (other[0], other[1], OPPOSITE[direction])
            if destructible:
                self.destructible_edges.add(rkey)
            else:
                self.destructible_edges.discard(rkey)

    def _consume_cell(self, cell, channels, from_agent_view):
        if not _in_bounds(cell):
            return
        visible = _ch(channels, VISIBLE) > 0.4 or any(_ch(channels, i) > 0.4 for i in range(min(25, len(channels))))
        if not visible:
            return

        for d, wall_ch in enumerate(WALL_CHANNELS):
            self._set_edge(cell, d, _ch(channels, wall_ch) > 0.4)
            self._set_destructible(cell, d, _ch(channels, DESTR_CHANNELS[d]) > 0.4)

        kind = None
        for channel, candidate in COLLECTIBLE_CHANNELS.items():
            if _ch(channels, channel) > 0.4:
                kind = candidate
                break

        if kind:
            self.collectibles[cell] = kind
            self.visible_collectibles.add(cell)
        else:
            self.collectibles.pop(cell, None)

        if from_agent_view and _ch(channels, ENEMY_BASE) > 0.4:
            self.visible_enemy_bases.add(cell)
        if from_agent_view and _ch(channels, ENEMY_AGENT) > 0.4:
            self.visible_enemy_agents.add(cell)

        if _ch(channels, ALLY_BOMB) > 0.4:
            self.bombs[cell] = ("ally", max(1, round(_ch(channels, ALLY_BOMB_TIMER))))
        if _ch(channels, ENEMY_BOMB) > 0.4:
            self.bombs[cell] = ("enemy", max(1, round(_ch(channels, ENEMY_BOMB_TIMER))))

    def _decode_agent_view(self, view):
        h = len(view)
        if not h:
            return
        w = len(view[0]) if isinstance(view[0], list) else 0
        for r, row in enumerate(view):
            if not isinstance(row, list):
                continue
            for c in range(min(w, len(row))):
                self._consume_cell(_agent_view_world(self.loc, self.direction, r, c, h, w), row[c], True)

    def _decode_base_view(self, view):
        h = len(view)
        if not h:
            return
        w = len(view[0]) if isinstance(view[0], list) else 0
        for r, row in enumerate(view):
            if not isinstance(row, list):
                continue
            for c in range(min(w, len(row))):
                self._consume_cell(_base_view_world(self.base_loc, r, c, h, w), row[c], False)

    def _edge_blocked(self, cell, direction):
        return (cell[0], cell[1], direction) in self.blocked_edges

    def _enemy_bomb_danger(self, cell):
        for pos, (team, timer) in self.bombs.items():
            if team == "enemy" and timer <= 3 and _cheb(pos, cell) <= 2:
                return True
        return False

    def _line_clear(self, a, b):
        x, y = a
        tx, ty = b
        while (x, y) != (tx, ty):
            dx = 0 if tx == x else (1 if tx > x else -1)
            dy = 0 if ty == y else (1 if ty > y else -1)
            if dx != 0 and dy != 0:
                hd = 0 if dx > 0 else 2
                vd = 1 if dy > 0 else 3
                if self._edge_blocked((x, y), hd) and self._edge_blocked((x, y), vd):
                    return False
                x += dx
                y += dy
            else:
                d = DELTA_TO_DIR[(dx, dy)]
                if self._edge_blocked((x, y), d):
                    return False
                x += dx
                y += dy
        return True

    def _can_blast(self, target):
        return _cheb(self.loc, target) <= 2 and self._line_clear(self.loc, target)

    def confirmed_bomb(self, obs):
        mask = _mask(obs)
        bombs = _as_int(obs.get("team_bombs"), 0)
        if not mask[PLACE_BOMB] or bombs <= 0 or self.bomb_cooldown > 0:
            return False
        for target in list(self.visible_enemy_bases) + list(self.visible_enemy_agents):
            if self._can_blast(target):
                return True
        return False

    def useful_bomb_context(self):
        if self.visible_enemy_bases or self.visible_enemy_agents:
            return True
        return any((self.loc[0], self.loc[1], d) in self.destructible_edges for d in range(4))

    def next_cell_for_action(self, action):
        if action == FORWARD:
            d = self.direction
        elif action == BACKWARD:
            d = OPPOSITE[self.direction]
        else:
            return self.loc
        dx, dy = DIRS[d]
        return (self.loc[0] + dx, self.loc[1] + dy)

    def action_for_dir(self, target_dir, obs):
        mask = _mask(obs)
        if target_dir == self.direction and mask[FORWARD]:
            return FORWARD
        if target_dir == OPPOSITE[self.direction] and mask[BACKWARD]:
            return BACKWARD
        if target_dir == (self.direction + 3) % 4 and mask[LEFT]:
            return LEFT
        if target_dir == (self.direction + 1) % 4 and mask[RIGHT]:
            return RIGHT
        if mask[RIGHT]:
            return RIGHT
        if mask[LEFT]:
            return LEFT
        return _fallback(obs)

    def adjacent_pickup_action(self, obs):
        mask = _mask(obs)
        best = None

        for action, direction in ((FORWARD, self.direction), (BACKWARD, OPPOSITE[self.direction])):
            if not mask[action]:
                continue
            dx, dy = DIRS[direction]
            nxt = (self.loc[0] + dx, self.loc[1] + dy)
            if not _in_bounds(nxt) or self._enemy_bomb_danger(nxt):
                continue
            kind = self.collectibles.get(nxt)
            if not kind:
                continue
            score = COLLECTIBLE_VALUE[kind] + 0.8 * _center_bonus(nxt) - 0.3 * self.visits.get(nxt, 0)
            if best is None or score > best[0]:
                best = (score, action)

        if best and best[0] >= 1.4:
            return best[1]

        side_best = None
        for direction in ((self.direction + 3) % 4, (self.direction + 1) % 4):
            dx, dy = DIRS[direction]
            nxt = (self.loc[0] + dx, self.loc[1] + dy)
            if not _in_bounds(nxt):
                continue
            kind = self.collectibles.get(nxt)
            if kind not in {"mission", "resource"}:
                continue
            action = self.action_for_dir(direction, obs)
            if action not in (LEFT, RIGHT):
                continue
            score = COLLECTIBLE_VALUE[kind] - 0.9
            if side_best is None or score > side_best[0]:
                side_best = (score, action)

        if side_best and side_best[0] >= 4.0:
            return side_best[1]

        return None

    def escape_action(self, obs):
        mask = _mask(obs)
        choices = []
        for action, direction in ((FORWARD, self.direction), (BACKWARD, OPPOSITE[self.direction])):
            if not mask[action]:
                continue
            dx, dy = DIRS[direction]
            nxt = (self.loc[0] + dx, self.loc[1] + dy)
            if not _in_bounds(nxt) or self._enemy_bomb_danger(nxt):
                continue
            score = -self.visits.get(nxt, 0) + _center_bonus(nxt)
            if nxt in self.collectibles:
                score += COLLECTIBLE_VALUE[self.collectibles[nxt]]
            choices.append((score, action))
        if choices:
            return max(choices)[1]
        for action in (RIGHT, LEFT, STAY):
            if mask[action]:
                return action
        return _fallback(obs)

    def stuck_action(self, obs):
        if self.stuck < 2:
            return None
        return self.escape_action(obs)


class AEManager:
    def __init__(self):
        self.core = CoreAEManager()
        self.guard = GuardState()

    def reset(self):
        self.guard.reset()
        if hasattr(self.core, "reset"):
            self.core.reset()

    def ae(self, observation):
        if not observation:
            self.reset()
            return STAY

        obs = observation
        step = _as_int(obs.get("step"), 0)
        if step == 0 and self.guard.last_step not in (None, 0):
            self.reset()
        elif self.guard.last_step is not None and step < self.guard.last_step:
            self.reset()
        self.guard.last_step = step
        self.guard.update(obs)

        mask = _mask(obs)
        if not any(mask):
            return STAY
        if _as_int(obs.get("frozen_ticks"), 0) > 0:
            return STAY if mask[STAY] else _fallback(obs)

        try:
            action = int(self.core.ae(obs))
        except Exception:
            action = _fallback(obs)

        if action < 0 or action >= 6 or not mask[action]:
            action = _fallback(obs)

        if _enabled("AE_BOMB_OVERRIDE") and self.guard.confirmed_bomb(obs):
            action = PLACE_BOMB
            self.guard.bomb_cooldown = 2

        elif action == PLACE_BOMB and _enabled("AE_BLOCK_SPEC_BOMBS", "0") and not self.guard.useful_bomb_context():
            action = _fallback(obs)

        elif _enabled("AE_ESCAPE_OVERRIDE"):
            nxt = self.guard.next_cell_for_action(action)
            if self.guard._enemy_bomb_danger(self.guard.loc) or self.guard._enemy_bomb_danger(nxt):
                action = self.guard.escape_action(obs)

        if action != PLACE_BOMB and _enabled("AE_PICKUP_OVERRIDE"):
            pickup = self.guard.adjacent_pickup_action(obs)
            if pickup is not None and mask[pickup]:
                action = pickup

        if action in (FORWARD, BACKWARD) and _enabled("AE_STUCK_OVERRIDE"):
            recovery = self.guard.stuck_action(obs)
            if recovery is not None and mask[recovery]:
                action = recovery

        if not mask[action]:
            action = _fallback(obs)

        self.guard.last_action = int(action)
        return int(action)
