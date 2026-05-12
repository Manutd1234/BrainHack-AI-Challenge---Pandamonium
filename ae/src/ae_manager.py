"""Manages the AE model."""

import random


class AEManager:

    def __init__(self):
        # Action space:
        # 0: move forward
        # 1: turn left
        # 2: turn right
        # 3: interact / activate challenge
        # 4: place bomb
        # 5: do nothing / stay
        pass

    def ae(self, observation: dict) -> int:
        """Gets the next action for the agent, based on the observation.

        Uses the action_mask to only pick from valid actions, and implements
        a simple exploration heuristic: prefer moving forward when possible,
        otherwise turn randomly.

        Args:
            observation: The observation from the environment containing:
                - agent_viewcone: 7x5x25 float32 array
                - base_viewcone: 5x5x25 float32 array
                - direction: int (0-3)
                - location: [x, y]
                - base_location: [x, y]
                - health: [float]
                - frozen_ticks: int
                - base_health: [float]
                - team_resources: [float]
                - team_bombs: int
                - step: int
                - action_mask: [int, int, int, int, int, int]

        Returns:
            An integer representing the action to take (0-5).
        """

        action_mask = observation.get("action_mask", [1, 1, 1, 1, 1, 1])

        # Simple heuristic: prefer moving forward, then interact, then turn
        preferred_actions = [0, 3, 1, 2, 4, 5]

        for action in preferred_actions:
            if action < len(action_mask) and action_mask[action] == 1:
                # Add some randomness to avoid getting stuck in loops
                if action == 0 and random.random() < 0.8:
                    return 0  # move forward most of the time
                elif action in [1, 2] and random.random() < 0.3:
                    return action  # occasionally turn

        # Fallback: pick a random valid action
        valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
        if valid_actions:
            return random.choice(valid_actions)

        return 0  # default: move forward
