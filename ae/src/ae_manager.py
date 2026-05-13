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
        self.visit_counts = {} # Track how many times each tile was stepped on
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
        self.visit_counts[location] = self.visit_counts.get(location, 0) + 1

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

        # --- Priority 3: Greedy Local Visited-Count Navigation ---
        # Evaluates immediate neighbor tiles and chooses the direction 
        # pointing to the least-visited cell to maximize maze coverage.
        
        # Coordinates delta: 0=North, 1=East, 2=South, 3=West
        dx = [0, 1, 0, -1]
        dy = [-1, 0, 1, 0]
        
        # 1. Compute coordinate mapping for all adjacent directions
        loc_fwd = (location[0] + dx[direction], location[1] + dy[direction])
        
        dir_left = (direction - 1) % 4
        loc_left = (location[0] + dx[dir_left], location[1] + dy[dir_left])
        
        dir_right = (direction + 1) % 4
        loc_right = (location[0] + dx[dir_right], location[1] + dy[dir_right])
        
        dir_back = (direction + 2) % 4
        loc_back = (location[0] + dx[dir_back], location[1] + dy[dir_back])
        
        # 2. Detect walls to prune blocked directions
        front_blocked = (action_mask[0] == 0)
        left_blocked = False
        right_blocked = False
        
        if agent_viewcone is not None:
            try:
                vc = np.array(agent_viewcone)
                # The agent is at row 0, col 2. 
                # Left is row 0, col 1. Right is row 0, col 3.
                if vc.shape[0] > 0 and vc.shape[1] > 3:
                    # Channel 0 has walls (value > 0.5)
                    if vc[0, 1, 0] > 0.5:
                        left_blocked = True
                    if vc[0, 3, 0] > 0.5:
                        right_blocked = True
            except Exception:
                pass
                
        # 3. Evaluate and score each valid movement choice
        choices = []
        
        # Choice A: Move Forward (Action 0)
        if not front_blocked:
            v_fwd = self.visit_counts.get(loc_fwd, 0)
            choices.append((0, v_fwd))
            
        # Choice B: Turn Left (Action 1)
        if not left_blocked and action_mask[1] == 1:
            # Small penalty added to turning to break ties and encourage going straight
            v_left = self.visit_counts.get(loc_left, 0) + 0.1
            choices.append((1, v_left))
            
        # Choice C: Turn Right (Action 2)
        if not right_blocked and action_mask[2] == 1:
            v_right = self.visit_counts.get(loc_right, 0) + 0.1
            choices.append((2, v_right))
            
        # Choice D: Backtrack / Turn Around (Action 1 or 2)
        # High penalty - only used as a complete dead-end escape
        v_back = self.visit_counts.get(loc_back, 0) + 5.0
        back_act = 1 if action_mask[1] == 1 else (2 if action_mask[2] == 1 else 5)
        choices.append((back_act, v_back))
        
        # 4. Sort choices to select direction of least visited tile
        choices.sort(key=lambda x: x[1])
        
        if choices:
            # Tie breaker
            best_val = choices[0][1]
            ties = [c for c in choices if abs(c[1] - best_val) < 0.05]
            return random.choice(ties)[0]
            
        # --- Priority 4: Random Bomb Placement ---
        if action_mask[4] == 1 and random.random() < 0.05:
            return 4
        
        return self._random_valid(action_mask)

    def _random_valid(self, action_mask):
        """Pick a random valid action."""
        valid = [i for i, m in enumerate(action_mask) if m == 1]
        if valid:
            return random.choice(valid)
        # If no valid actions (e.g. completely frozen), return 5 (stay)
        return 5
