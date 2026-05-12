"""Manages the AE model."""

import random
import numpy as np


class AEManager:

    def __init__(self):
        # Action space:
        # 0: move forward
        # 1: turn left
        # 2: turn right
        # 3: interact / activate challenge
        # 4: place bomb
        # 5: do nothing / stay

        # Exploration state
        self.visited = set()
        self.last_action = None
        self.stuck_counter = 0
        self.last_location = None
        self.explore_dir = 0  # 0=forward, 1=left, 2=right
        self.step_count = 0

    def ae(self, observation: dict) -> int:
        """Gets the next action for the agent, based on the observation.

        Implements a smarter exploration strategy using viewcone data
        and location tracking to navigate the maze efficiently and
        activate challenges.

        Args:
            observation: The observation from the environment.

        Returns:
            An integer representing the action to take (0-5).
        """

        action_mask = observation.get("action_mask", [1, 1, 1, 1, 1, 1])
        location = tuple(observation.get("location", [0, 0]))
        direction = observation.get("direction", 0)
        health = observation.get("health", [60.0])[0]
        step = observation.get("step", 0)
        agent_viewcone = observation.get("agent_viewcone", None)
        team_resources = observation.get("team_resources", [0.0])[0]
        frozen_ticks = observation.get("frozen_ticks", 0)
        base_health = observation.get("base_health", [100.0])[0]

        self.step_count = step

        # If frozen, do nothing
        if frozen_ticks > 0:
            return 5

        # Detect if stuck (same location for too long)
        if self.last_location == location:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_location = location
        self.visited.add(location)

        # --- Priority 1: Interact with challenges if available ---
        if action_mask[3] == 1:
            return 3

        # --- Priority 2: If stuck, try to escape ---
        if self.stuck_counter > 3:
            self.stuck_counter = 0
            # Try turning to a new direction
            turn_actions = [a for a in [1, 2] if action_mask[a] == 1]
            if turn_actions:
                return random.choice(turn_actions)
            # Try placing a bomb if really stuck
            if action_mask[4] == 1 and self.stuck_counter > 6:
                return 4
            return self._random_valid(action_mask)

        # --- Priority 3: Use viewcone to navigate ---
        if agent_viewcone is not None:
            try:
                vc = np.array(agent_viewcone)
                # The viewcone is 7x5x25 - agent is at row 0, center col
                # Check if path ahead is clear (look at forward cells)
                # Channel interpretation varies, but we can check for walls

                # Simple heuristic: check if the cells ahead have obstacles
                # by looking at the sum of relevant channels
                forward_clear = True
                if vc.shape[0] > 1 and vc.shape[1] > 2:
                    # Check the cell directly ahead (row 1, center col 2)
                    ahead_cell = vc[1, 2, :]
                    # If wall channel has high value, path is blocked
                    # Channel 0 is typically walls/boundaries
                    if ahead_cell[0] > 0.5:
                        forward_clear = False

                if not forward_clear and action_mask[0] == 1:
                    # Path seems blocked, try turning
                    turn_actions = [a for a in [1, 2] if action_mask[a] == 1]
                    if turn_actions:
                        return random.choice(turn_actions)

            except Exception:
                pass  # Fall through to default logic

        # --- Priority 4: Explore unvisited areas ---
        # Move forward if possible (main exploration action)
        if action_mask[0] == 1:
            # Check adjacent cells for unvisited locations
            dx = [0, 1, 0, -1]  # direction 0=north, 1=east, 2=south, 3=west
            dy = [-1, 0, 1, 0]
            next_loc = (location[0] + dx[direction], location[1] + dy[direction])

            if next_loc not in self.visited:
                # Prefer moving to unvisited locations
                return 0
            elif random.random() < 0.7:
                # Still move forward most of the time even if visited
                return 0

        # --- Priority 5: Turn to explore new directions ---
        # Check which turns lead to unvisited areas
        turn_actions = []
        if action_mask[1] == 1:
            turn_actions.append(1)
        if action_mask[2] == 1:
            turn_actions.append(2)

        if turn_actions:
            # Alternate turning direction to avoid circles
            self.explore_dir = (self.explore_dir + 1) % len(turn_actions)
            return turn_actions[self.explore_dir % len(turn_actions)]

        # --- Priority 6: Place bomb if nothing else works ---
        if action_mask[4] == 1 and random.random() < 0.1:
            return 4

        # Fallback: random valid action
        return self._random_valid(action_mask)

    def _random_valid(self, action_mask):
        """Pick a random valid action."""
        valid = [i for i, m in enumerate(action_mask) if m == 1]
        return random.choice(valid) if valid else 0
