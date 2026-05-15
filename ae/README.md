# AE

Your AE challenge is to direct your agent through the Bomberman map while
collecting rewards, placing bombs, and avoiding frozen downtime.

## Input

The TIL endpoint is `POST /ae` on port `5005`:

```JSON
{
  "instances": [
    {
      "observation": {
        "agent_viewcone": [[[0, "..."]]],
        "base_viewcone": [[[0, "..."]]],
        "direction": 0,
        "location": [0, 0],
        "base_location": [0, 0],
        "health": [60.0],
        "frozen_ticks": 0,
        "base_health": [100.0],
        "team_resources": [0.0],
        "team_bombs": 3,
        "step": 0,
        "action_mask": [1, 1, 1, 1, 1, 1]
      }
    }
  ]
}
```

The response is:

```Python
{
    "predictions": [{"action": 0}]
}
```

Actions are `0=FORWARD`, `1=BACKWARD`, `2=LEFT`, `3=RIGHT`, `4=STAY`, and
`5=PLACE_BOMB`.

## Implementation

The manager first tries to load `model/policy.zip` as a MaskablePPO/PPO policy.
If no checkpoint exists, it uses a deterministic rule fallback that:

- Resets memory automatically when `step == 0`.
- Parses the 25-channel viewcone into a local map.
- Prioritizes mission, resource, and recon tiles.
- Places bombs when enemy bases or agents are inside blast range.
- Uses BFS frontier exploration when no reward target is known.
- Respects `action_mask` before returning an action.

Build immediately with the rule fallback:

```bash
docker build -t pandamonium-ae:v2 .
docker run -p 5005:5005 pandamonium-ae:v2
```

Optional PPO training:

```bash
pip install -e /path/to/til-26-ae
pip install -r requirements.txt
python ae_train.py
```

That writes `model/policy.zip`; rebuild the Docker image to include it.
