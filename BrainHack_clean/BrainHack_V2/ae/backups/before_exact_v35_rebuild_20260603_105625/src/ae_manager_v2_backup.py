from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Any

FORWARD, BACKWARD, LEFT, RIGHT, STAY, PLACE_BOMB = 0, 1, 2, 3, 4, 5
GRID = 16
CENTER = (7.5, 7.5)
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DELTA_TO_DIR = {delta: i for i, delta in enumerate(DIRS)}
OPPOSITE = {0: 2, 1: 3, 2: 0, 3: 1}

VISIBLE = 0
WALL_RIGHT, WALL_DOWN, WALL_LEFT, WALL_UP = 1, 2, 3, 4
TILE_EMPTY, TILE_RECON, TILE_MISSION, TILE_RESOURCE = 5, 6, 7, 8
ENEMY_AGENT, ENEMY_BASE = 10, 12
DESTR_WALL_RIGHT, DESTR_WALL_DOWN, DESTR_WALL_LEFT, DESTR_WALL_UP = 13, 14, 15, 16
ALLY_BOMB, ENEMY_BOMB, ALLY_BOMB_TIMER, ENEMY_BOMB_TIMER = 17, 18, 19, 20
ENEMY_AGENT_HEALTH, ENEMY_BASE_HEALTH = 22, 24

WALL_CHANNELS = [WALL_RIGHT, WALL_DOWN, WALL_LEFT, WALL_UP]
DESTR_CHANNELS = [DESTR_WALL_RIGHT, DESTR_WALL_DOWN, DESTR_WALL_LEFT, DESTR_WALL_UP]
COLLECTIBLE_CHANNELS = {TILE_MISSION: "mission", TILE_RESOURCE: "resource", TILE_RECON: "recon"}
COLLECTIBLE_VALUE = {"mission": 9.0, "resource": 4.2, "recon": 1.35}


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


def _fallback(obs: dict[str, Any] | None) -> int:
    if not obs:
        return STAY
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


def _clamp_cell(x: float, y: float) -> tuple[int, int]:
    return (max(0, min(GRID - 1, round(x))), max(0, min(GRID - 1, round(y))))


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
class PlannerState:
    last_step: int | None = None
    obs_step: int = 0
    loc: tuple[int, int] = (0, 0)
    direction: int = 0
    base_loc: tuple[int, int] = (0, 0)
    blocked_edges: set = field(default_factory=set)
    destructible_edges: set = field(default_factory=set)
    seen_cells: set = field(default_factory=set)
    collectibles: dict = field(default_factory=dict)
    enemy_bases: dict = field(default_factory=dict)
    enemy_base_health: dict = field(default_factory=dict)
    enemy_agents: dict = field(default_factory=dict)
    enemy_agent_health: dict = field(default_factory=dict)
    predicted_bases: set = field(default_factory=set)
    bombs: dict = field(default_factory=dict)
    visits: dict = field(default_factory=dict)
    last_action: int = STAY
    prev_loc: tuple[int, int] | None = None
    stuck: int = 0
    bomb_cooldown: int = 0

    def update(self, obs):
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
        self._refresh_predicted_bases()

        if self.prev_loc == self.loc and self.last_action in (FORWARD, BACKWARD):
            self.stuck += 1
        elif self.prev_loc != self.loc:
            self.stuck = 0

        self.visits[self.loc] = self.visits.get(self.loc, 0) + 1
        self.seen_cells.add(self.loc)
        self.collectibles.pop(self.loc, None)

        self.visible_enemy_bases = set()
        self.visible_enemy_agents = set()
        self.visible_collectibles = set()

        self._decode_agent_view(obs.get("agent_viewcone") or [])
        self._decode_base_view(obs.get("base_viewcone") or [])

        cutoff = self.obs_step - 12
        self.enemy_agents = {p: s for p, s in self.enemy_agents.items() if s >= cutoff}

    def _refresh_predicted_bases(self):
        bx, by = self.base_loc
        vx = bx - CENTER[0]
        vy = by - CENTER[1]
        if abs(vx) + abs(vy) < 2.0:
            return
        predicted = set()
        for k in range(1, 6):
            angle = k * pi / 3.0
            rx = vx * cos(angle) - vy * sin(angle)
            ry = vx * sin(angle) + vy * cos(angle)
            cx, cy = _clamp_cell(CENTER[0] + rx, CENTER[1] + ry)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    cell = (cx + dx, cy + dy)
                    if _in_bounds(cell):
                        predicted.add(cell)
        self.predicted_bases = predicted

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
        try:
            visible = _ch(channels, VISIBLE) > 0.4 or any(float(x) for x in channels[:25])
        except Exception:
            visible = False
        if not visible:
            return

        self.seen_cells.add(cell)
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
        elif visible or _ch(channels, TILE_EMPTY) > 0.4:
            self.collectibles.pop(cell, None)

        if _ch(channels, ENEMY_BASE) > 0.4:
            self.enemy_bases[cell] = self.obs_step
            self.enemy_base_health[cell] = max(0.0, _ch(channels, ENEMY_BASE_HEALTH))
            if from_agent_view:
                self.visible_enemy_bases.add(cell)

        if _ch(channels, ENEMY_AGENT) > 0.4:
            self.enemy_agents[cell] = self.obs_step
            self.enemy_agent_health[cell] = max(0.0, _ch(channels, ENEMY_AGENT_HEALTH))
            if from_agent_view:
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

    def _neighbors(self, cell, avoid_bombs=True):
        out = []
        for d, (dx, dy) in enumerate(DIRS):
            nxt = (cell[0] + dx, cell[1] + dy)
            if not _in_bounds(nxt):
                continue
            if self._edge_blocked(cell, d):
                continue
            if avoid_bombs and self._enemy_bomb_danger(nxt):
                continue
            out.append(nxt)
        return out

    def _bfs(self, start, avoid_bombs=True):
        dist = {start: 0}
        prev = {}
        q = deque([start])
        while q:
            cell = q.popleft()
            for nxt in self._neighbors(cell, avoid_bombs):
                if nxt in dist:
                    continue
                dist[nxt] = dist[cell] + 1
                prev[nxt] = cell
                q.append(nxt)
        return dist, prev

    def _path(self, target, prev):
        if target == self.loc:
            return [self.loc]
        if target not in prev:
            return []
        path = [target]
        while path[-1] != self.loc:
            path.append(prev[path[-1]])
        path.reverse()
        return path

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

    def _should_bomb(self, obs):
        mask = _mask(obs)
        bombs = _as_int(obs.get("team_bombs"), 0)
        if not mask[PLACE_BOMB] or bombs <= 0:
            return False
        if self.bomb_cooldown > 0 and not self.visible_enemy_bases:
            return False

        for base in self.visible_enemy_bases:
            if self._can_blast(base):
                return True
        for enemy in self.visible_enemy_agents:
            if self._can_blast(enemy):
                return True

        if bombs >= 2 or self.obs_step > 90:
            for cell in self.predicted_bases:
                if _cheb(self.loc, cell) <= 1:
                    return True

        if self.stuck >= 2:
            for d in range(4):
                if (self.loc[0], self.loc[1], d) in self.destructible_edges:
                    return True
        return False

    def _action_for_dir(self, target_dir, obs):
        mask = _mask(obs)
        if target_dir == self.direction and mask[FORWARD]:
            return FORWARD
        if target_dir == OPPOSITE[self.direction] and mask[BACKWARD]:
            return BACKWARD
        if target_dir == (self.direction + 3) % 4 and mask[LEFT]:
            return LEFT
        if target_dir == (self.direction + 1) % 4 and mask[RIGHT]:
            return RIGHT
        left_distance = (self.direction - target_dir) % 4
        right_distance = (target_dir - self.direction) % 4
        if left_distance <= right_distance and mask[LEFT]:
            return LEFT
        if mask[RIGHT]:
            return RIGHT
        return _fallback(obs)

    def _action_toward(self, nxt, obs):
        dx = nxt[0] - self.loc[0]
        dy = nxt[1] - self.loc[1]
        target_dir = DELTA_TO_DIR.get((dx, dy))
        if target_dir is None:
            return _fallback(obs)
        return self._action_for_dir(target_dir, obs)

    def _greedy_visible_action(self, obs):
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
            score = COLLECTIBLE_VALUE.get(kind, 0.0)
            score += 0.7 * _center_bonus(nxt)
            score += max(0.0, 1.0 - 0.35 * self.visits.get(nxt, 0))
            if kind or self.visits.get(nxt, 0) == 0:
                if best is None or score > best[0]:
                    best = (score, action)
        if best and best[0] >= 1.0:
            return best[1]

        side_best = None
        for direction in ((self.direction + 3) % 4, (self.direction + 1) % 4):
            dx, dy = DIRS[direction]
            nxt = (self.loc[0] + dx, self.loc[1] + dy)
            if not _in_bounds(nxt):
                continue
            kind = self.collectibles.get(nxt)
            if not kind:
                continue
            action = self._action_for_dir(direction, obs)
            if action not in (LEFT, RIGHT):
                continue
            score = COLLECTIBLE_VALUE.get(kind, 0.0) - 0.8
            if side_best is None or score > side_best[0]:
                side_best = (score, action)
        if side_best and side_best[0] >= 2.5:
            return side_best[1]
        return None

    def _attack_cells_for_base(self, base):
        cells = []
        for x in range(max(0, base[0] - 2), min(GRID, base[0] + 3)):
            for y in range(max(0, base[1] - 2), min(GRID, base[1] + 3)):
                cell = (x, y)
                if cell != base and self._line_clear(cell, base):
                    cells.append(cell)
        return cells

    def _best_path(self):
        dist, prev = self._bfs(self.loc, avoid_bombs=True)
        if len(dist) < 5:
            dist, prev = self._bfs(self.loc, avoid_bombs=False)

        candidates = set(self.collectibles)
        candidates.update(self.predicted_bases)
        for base in self.enemy_bases:
            candidates.update(self._attack_cells_for_base(base))
        for cell in self.seen_cells:
            for dx, dy in DIRS:
                nxt = (cell[0] + dx, cell[1] + dy)
                if _in_bounds(nxt) and nxt not in self.seen_cells:
                    candidates.add(nxt)
        if not candidates:
            candidates.update((x, y) for x in range(GRID) for y in range(GRID))

        best_score = -10**9
        best_target = None
        for target in candidates:
            if target == self.loc or target not in dist:
                continue
            d = dist[target]
            if d <= 0:
                continue
            score = 0.0
            kind = self.collectibles.get(target)
            if kind:
                score += COLLECTIBLE_VALUE.get(kind, 1.0) * 4.0
            if target not in self.seen_cells:
                score += 5.0
            if target in self.predicted_bases:
                score += 9.0 + min(9.0, self.obs_step / 16.0)
            for base, seen_step in self.enemy_bases.items():
                if _cheb(target, base) <= 2:
                    score += 34.0
                    if self.enemy_base_health.get(base, 1.0) <= 0.45:
                        score += 10.0
                    score += max(0.0, 10.0 - 0.2 * (self.obs_step - seen_step))
            for enemy, seen_step in self.enemy_agents.items():
                if _cheb(target, enemy) <= 2:
                    score += max(0.0, 12.0 - 1.2 * (self.obs_step - seen_step))
            score += max(0.0, 3.2 - 0.65 * self.visits.get(target, 0))
            score += 2.5 * _center_bonus(target)
            score -= 0.95 * d
            if self._enemy_bomb_danger(target):
                score -= 20.0
            if score > best_score:
                best_score = score
                best_target = target

        return self._path(best_target, prev) if best_target is not None else []

    def _least_visited_action(self, obs):
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

    def choose_action(self, obs):
        mask = _mask(obs)
        if not any(mask):
            return STAY
        if _as_int(obs.get("frozen_ticks"), 0) > 0:
            return STAY if mask[STAY] else _fallback(obs)

        if self._should_bomb(obs):
            self.bomb_cooldown = 2
            return PLACE_BOMB

        greedy = self._greedy_visible_action(obs)
        if greedy is not None and mask[greedy]:
            return greedy

        if self._enemy_bomb_danger(self.loc):
            dist, prev = self._bfs(self.loc, avoid_bombs=True)
            safe = [cell for cell in dist if not self._enemy_bomb_danger(cell)]
            if safe:
                target = min(safe, key=lambda c: (dist[c], self.visits.get(c, 0)))
                path = self._path(target, prev)
                if len(path) >= 2:
                    return self._action_toward(path[1], obs)

        path = self._best_path()
        if len(path) >= 2:
            action = self._action_toward(path[1], obs)
            if mask[action]:
                return action

        return self._least_visited_action(obs)


class AEManager:
    def __init__(self):
        self.state = PlannerState()

    def reset(self):
        self.state = PlannerState()

    def ae(self, observation):
        if not observation:
            self.reset()
            return STAY
        try:
            step = _as_int(observation.get("step"), 0)
            if step == 0 and self.state.last_step not in (None, 0):
                self.reset()
            elif self.state.last_step is not None and step < self.state.last_step:
                self.reset()
            self.state.last_step = step
            self.state.update(observation)
            action = self.state.choose_action(observation)
            if not _mask(observation)[action]:
                action = _fallback(observation)
            self.state.last_action = int(action)
            return int(action)
        except Exception:
            return int(_fallback(observation))
