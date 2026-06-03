# AE: Autonomous Exploration Agent

This module serves the Autonomous Exploration challenge on `POST /ae` at port `5005` and resets round state through `GET /reset`.

AE is the highest-weighted challenge in the TIL-AI scoring mix. The Novice map is fixed, which makes a hybrid of reinforcement learning and classical path planning attractive.

## Input and Output

Input:

```json
{
  "instances": [
    {
      "observation": {
        "agent_viewcone": [[[0]]],
        "base_viewcone": [[[0]]],
        "direction": 0,
        "location": [0, 0],
        "base_location": [0, 0],
        "health": [60.0],
        "frozen_ticks": 0,
        "base_health": [100.0],
        "team_resources": [0.0],
        "team_bombs": 0,
        "step": 0,
        "action_mask": [1, 1, 1, 1, 1, 0]
      }
    }
  ]
}
```

Output:

```json
{
  "predictions": [
    {
      "action": 0
    }
  ]
}
```

The action is one of the legal discrete environment actions. The manager always respects `action_mask`; if the learned policy proposes an illegal action, a legal fallback is selected.

## Architecture

The AE design combines:

- MaskablePPO from `sb3-contrib`.
- CNN/MLP feature extraction for `agent_viewcone`, `base_viewcone`, and scalar state.
- Reward shaping for faster early learning.
- Opponent snapshots for league-style self-play.
- BFS/A* style fallback behavior when the policy is uncertain or invalid.

## Why PPO plus BFS/A*

PPO learns tactical behavior from interaction, but the Novice map has strong path-planning structure. BFS/A* style logic is useful for:

- reaching mission/challenge tiles,
- escaping local loops,
- returning toward base or safe zones,
- selecting a valid movement when the policy emits an invalid action,
- keeping behavior deterministic during bad observations.

The final inference stack favors the trained policy but keeps rule-based navigation as a safety net.

## Training Files

```text
ae/
├── train.py                   # phased PPO training entrypoint
├── requirements-train.txt     # training dependencies
├── requirements.txt           # inference dependencies
├── src/
│   ├── ae_manager.py          # policy loading and action selection
│   ├── ae_server.py           # FastAPI service
│   ├── features.py            # neural feature extractor
│   ├── obs_utils.py           # observation packing
│   └── wrappers.py            # environment wrapper, masks, shaping
└── model/                     # inference policy location
```

## Training Recipe

Create an isolated venv and install training dependencies:

```bash
cd /home/jupyter/BrainHack_clean/BrainHack_V2/ae
python -m venv .venv_ae
source .venv_ae/bin/activate
pip install -r requirements-train.txt
```

If `til_environment` is not installed in the venv, point Python to the local reference environment:

```bash
export PYTHONPATH=/home/jupyter/reference-til26/til-26-ae:$PYTHONPATH
```

Train in phases:

```bash
python train.py --phase 1 --total-envs 16 --device cuda --out-dir models
python train.py --phase 2 --total-envs 8 --device cuda --resume models/phase1_final.zip --out-dir models
python train.py --phase 3 --total-envs 8 --device cuda --resume models/phase2_final.zip --out-dir models
```

If phase 2 or phase 3 is unstable at 16 environments, reduce to 8 environments. This reduces throughput but often improves stability.

Prepare the inference checkpoint:

```bash
cp -f models/phase3_final.zip models/ae_policy.zip
```

If only phase 1 completed and a quick inference build is required:

```bash
cp -f models/phase1_final.zip models/ae_policy.zip
```

## TensorBoard

```bash
cd /home/jupyter/BrainHack_clean/BrainHack_V2/ae
source .venv_ae/bin/activate
python -m pip install "setuptools<81" tensorboard
tensorboard --logdir models/tb --port 6006 --host 0.0.0.0
```

Open the forwarded Workbench/Jupyter port for `6006`.

## Build and Submit

```bash
export TIL_FOLDER=/home/jupyter/BrainHack_clean/BrainHack_V2
cd "$TIL_FOLDER/ae"

ls -lh models/ae_policy.zip
til build ae v10
til test ae v10
til submit ae v10
```

## Debug Checklist

- If `ae_policy.zip.zip` appears in logs, the manager path appended `.zip` twice.
- If `numpy._core.numeric` is missing, align inference `numpy` and SB3 versions with the training environment.
- If `ModuleNotFoundError: src` appears while loading the checkpoint, the training wrapper module paths changed. Keep `src/` package structure consistent.
- If local test imports `pettingzoo.utils.AgentSelector` incorrectly, restore the environment's pinned `gymnasium` and `pettingzoo` versions.
