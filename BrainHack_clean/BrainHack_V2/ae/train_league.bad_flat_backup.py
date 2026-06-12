from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback

from til_environment import bomberman_env
from til_environment.config import default_config

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

ACTIONS = 6
LEARN_AGENT = "agent_0"

DEFAULT_LEAGUE = [
    ("random", "random", 0.60),
    ("bestv5", "/home/jupyter/ae/model/best/best_model_v5.zip", 0.20),
    ("v30freeze", "/home/jupyter/ae/model/policy.v30.freeze.zip", 0.10),
    ("policyfinal", "/home/jupyter/ae/model/v14_official_reward/policy_final.zip", 0.05),
    ("lowvar", "/home/jupyter/ae/model/checkpoints-Copy1/maskable_ppo_til26_ae_v13_lowvar_objective_11499264_steps.zip", 0.05),
]


def jsonable(obs: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in obs.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def obs_to_vec(obs: dict[str, Any]) -> np.ndarray:
    parts = []

    for key in ("agent_viewcone", "base_viewcone"):
        parts.append(np.asarray(obs.get(key), dtype=np.float32).ravel())

    def scalar(key: str, denom: float = 1.0) -> float:
        v = obs.get(key, 0.0)
        arr = np.asarray(v, dtype=np.float32).ravel()
        return float(arr[0]) / denom if arr.size else 0.0

    scalars = np.array(
        [
            scalar("direction", 3.0),
            scalar("location", 15.0),
            float(np.asarray(obs.get("location", [0, 0])).ravel()[1]) / 15.0,
            scalar("base_location", 15.0),
            float(np.asarray(obs.get("base_location", [0, 0])).ravel()[1]) / 15.0,
            scalar("health", 60.0),
            scalar("frozen_ticks", 3.0),
            scalar("base_health", 100.0),
            min(scalar("team_resources", 10.0), 10.0),
            min(scalar("team_bombs", 10.0), 10.0),
            scalar("step", 200.0),
        ],
        dtype=np.float32,
    )
    parts.append(scalars)

    mask = np.asarray(obs.get("action_mask", [0, 0, 0, 0, 1, 0]), dtype=np.float32).ravel()
    if mask.size != ACTIONS:
        mask = np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
    parts.append(mask)

    return np.concatenate(parts).astype(np.float32)


def legal_random(mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask > 0)
    if legal.size:
        return int(np.random.choice(legal))
    return 4


class ManagerOpponent:
    def __init__(self, name: str, policy_path: str, slot: str):
        self.name = name
        self.policy_path = policy_path
        sys.path.insert(0, str(SRC))
        old = os.environ.get("AE_CHECKPOINT_PATH")
        os.environ["AE_CHECKPOINT_PATH"] = policy_path
        module_name = f"league_mgr_{name}_{slot}".replace("-", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(module_name, SRC / "ae_manager.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load ae_manager.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.manager = mod.AEManager()
        if old is None:
            os.environ.pop("AE_CHECKPOINT_PATH", None)
        else:
            os.environ["AE_CHECKPOINT_PATH"] = old

    def reset(self):
        if hasattr(self.manager, "reset"):
            self.manager.reset()

    def act(self, obs: dict[str, Any]) -> int:
        try:
            return int(self.manager.ae(jsonable(obs)))
        except Exception:
            return legal_random(np.asarray(obs.get("action_mask", [0, 0, 0, 0, 1, 0])))


class LeagueEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, novice: bool = False, seed_base: int = 0):
        super().__init__()
        self.seed_base = seed_base
        self.episode = 0

        cfg = default_config()
        cfg.env.novice = novice
        self.env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
        self.learn_agent = LEARN_AGENT

        self.league = [(n, p, w) for n, p, w in DEFAULT_LEAGUE if p == "random" or Path(p).exists()]
        self.names = [x[0] for x in self.league]
        weights = np.asarray([x[2] for x in self.league], dtype=np.float64)
        self.weights = weights / weights.sum()

        self.manager_cache: dict[tuple[str, str], ManagerOpponent] = {}
        self.opponents: dict[str, Any] = {}

        self.env.reset(seed=seed_base)
        obs = self.env.observe(self.learn_agent)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=obs_to_vec(obs).shape, dtype=np.float32)
        self.action_space = spaces.Discrete(ACTIONS)
        self.current_obs = obs

    def _sample_opponent(self, agent: str):
        idx = int(np.random.choice(len(self.league), p=self.weights))
        name, path, _ = self.league[idx]
        if path == "random":
            return None
        key = (path, agent)
        if key not in self.manager_cache:
            self.manager_cache[key] = ManagerOpponent(name, path, agent)
        self.manager_cache[key].reset()
        return self.manager_cache[key]

    def action_masks(self):
        return np.asarray(self.current_obs.get("action_mask", [0, 0, 0, 0, 1, 0]), dtype=bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        actual_seed = self.seed_base + self.episode if seed is None else seed

        self.opponents = {}
        for agent in self.env.possible_agents:
            if agent != self.learn_agent:
                self.opponents[agent] = self._sample_opponent(agent)

        self.env.reset(seed=actual_seed)

        while self.env.agent_selection != self.learn_agent:
            agent = self.env.agent_selection
            obs, _, term, trunc, _ = self.env.last()
            action = None if term or trunc else self._opponent_action(agent, obs)
            self.env.step(action)

        self.current_obs = self.env.observe(self.learn_agent)
        return obs_to_vec(self.current_obs), {}

    def _opponent_action(self, agent: str, obs: dict[str, Any]) -> int:
        mask = np.asarray(obs.get("action_mask", [0, 0, 0, 0, 1, 0]), dtype=np.int8)
        opponent = self.opponents.get(agent)
        if opponent is None:
            return legal_random(mask)
        action = opponent.act(obs)
        if action < 0 or action >= ACTIONS or not mask[action]:
            return legal_random(mask)
        return int(action)

    def step(self, action: int):
        mask = self.action_masks()
        if action < 0 or action >= ACTIONS or not mask[action]:
            action = legal_random(mask)

        self.env.step(int(action))

        while self.env.agent_selection != self.learn_agent:
            agent = self.env.agent_selection
            obs, _, term, trunc, _ = self.env.last()
            opp_action = None if term or trunc else self._opponent_action(agent, obs)
            self.env.step(opp_action)
            if all(self.env.truncations.values()) or all(self.env.terminations.values()):
                break

        obs, reward, term, trunc, info = self.env.last()
        self.current_obs = obs
        return obs_to_vec(obs), float(reward), bool(term), bool(trunc), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--out", default="/home/jupyter/ae/model/league_v1")
    ap.add_argument("--seed", type=int, default=2606)
    ap.add_argument("--novice", action="store_true")
    ap.add_argument("--init", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    env = LeagueEnv(novice=args.novice, seed_base=args.seed)

    if args.init:
        try:
            model = MaskablePPO.load(args.init, env=env, device="cpu")
            print(f"Loaded init model: {args.init}")
        except Exception as e:
            print(f"Init model load failed, starting fresh: {e}")
            model = None
    else:
        model = None

    if model is None:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=6,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.015,
            clip_range=0.15,
            verbose=1,
            tensorboard_log=str(out / "tb"),
            device="cpu",
        )

    cb = CheckpointCallback(
        save_freq=500_000,
        save_path=str(out / "checkpoints"),
        name_prefix="league_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=True)
    model.save(str(out / "policy_final.zip"))
    print(f"saved {out / 'policy_final.zip'}")


if __name__ == "__main__":
    main()
