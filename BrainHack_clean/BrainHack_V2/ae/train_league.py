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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from til_environment import bomberman_env
from til_environment.config import default_config

ACTIONS = 6
LEARN_AGENT = "agent_0"
SRC = Path(__file__).resolve().parent / "src"

LEAGUE = [
    ("random", "random", 0.35),
    ("v30freeze", "/home/jupyter/ae/model/policy.v30.freeze.zip", 0.35),
    ("bestv5", "/home/jupyter/ae/model/best/best_model_v5.zip", 0.30),
]


def fix_obs(obs: dict[str, Any] | None) -> dict[str, Any]:
    if obs is None:
        obs = {}
    obs = dict(obs)
    if "action_mask" not in obs or obs["action_mask"] is None:
        obs["action_mask"] = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
    else:
        obs["action_mask"] = np.asarray(obs["action_mask"], dtype=np.uint8)
    return obs


def jsonable(obs: dict[str, Any]) -> dict[str, Any]:
    obs = fix_obs(obs)
    out = {}
    for k, v in obs.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def legal_random(mask) -> int:
    mask = np.asarray(mask, dtype=np.int8)
    legal = np.flatnonzero(mask > 0)
    if legal.size:
        return int(np.random.choice(legal))
    return 4


class ManagerOpponent:
    def __init__(self, name: str, policy_path: str, slot: str):
        self.name = name
        self.policy_path = policy_path
        self.slot = slot
        self.fail_count = 0

        sys.path.insert(0, str(SRC))
        old = os.environ.get("AE_CHECKPOINT_PATH")
        os.environ["AE_CHECKPOINT_PATH"] = policy_path

        module_name = f"league_mgr_{name}_{slot}_{abs(hash(policy_path))}".replace("-", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(module_name, SRC / "ae_manager.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not import ae_manager.py")

        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.manager = mod.AEManager()

        if old is None:
            os.environ.pop("AE_CHECKPOINT_PATH", None)
        else:
            os.environ["AE_CHECKPOINT_PATH"] = old

        print(f"Loaded manager opponent {name}: {policy_path}", flush=True)

    def reset(self):
        if hasattr(self.manager, "reset"):
            try:
                self.manager.reset()
            except Exception:
                pass

    def act(self, obs: dict[str, Any]) -> int:
        obs = fix_obs(obs)
        try:
            action = int(self.manager.ae(jsonable(obs)))
        except Exception as e:
            self.fail_count += 1
            if self.fail_count <= 3:
                print(f"{self.name} manager inference failed, fallback random: {e}", flush=True)
            return legal_random(obs["action_mask"])

        mask = np.asarray(obs["action_mask"], dtype=np.int8)
        if action < 0 or action >= ACTIONS or not mask[action]:
            return legal_random(mask)
        return int(action)


class LeagueEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, novice: bool = False, seed_base: int = 2606):
        super().__init__()
        self.seed_base = int(seed_base)
        self.episode = 0

        cfg = default_config()
        cfg.env.novice = bool(novice)
        self.env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)

        self.observation_space = self.env.observation_space(LEARN_AGENT)
        self.action_space = self.env.action_space(LEARN_AGENT)

        self.league = [(n, p, w) for n, p, w in LEAGUE if p == "random" or Path(p).exists()]
        weights = np.asarray([x[2] for x in self.league], dtype=np.float64)
        self.weights = weights / weights.sum()

        self.cache: dict[tuple[str, str], ManagerOpponent] = {}
        self.opponents: dict[str, ManagerOpponent | None] = {}
        self.current_obs: dict[str, Any] | None = None

    def _sample_opponent(self, agent: str):
        idx = int(np.random.choice(len(self.league), p=self.weights))
        name, path, _ = self.league[idx]
        if path == "random":
            return None
        key = (path, agent)
        if key not in self.cache:
            self.cache[key] = ManagerOpponent(name, path, agent)
        self.cache[key].reset()
        return self.cache[key]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        actual_seed = self.seed_base + self.episode if seed is None else int(seed)

        self.opponents = {}
        for agent in self.env.possible_agents:
            if agent != LEARN_AGENT:
                self.opponents[agent] = self._sample_opponent(agent)

        self.env.reset(seed=actual_seed)

        while self.env.agent_selection != LEARN_AGENT:
            agent = self.env.agent_selection
            obs, _, term, trunc, _ = self.env.last()
            action = None if term or trunc else self._opponent_action(agent, obs)
            self.env.step(action)

        self.current_obs = fix_obs(self.env.observe(LEARN_AGENT))
        return self.current_obs, {}

    def action_masks(self):
        if self.current_obs is None:
            return np.array([0, 0, 0, 0, 1, 0], dtype=bool)
        return np.asarray(self.current_obs.get("action_mask", [0, 0, 0, 0, 1, 0]), dtype=bool)

    def _opponent_action(self, agent: str, obs: dict[str, Any]) -> int:
        obs = fix_obs(obs)
        opponent = self.opponents.get(agent)
        if opponent is None:
            return legal_random(obs["action_mask"])
        return opponent.act(obs)

    def step(self, action: int):
        mask = self.action_masks()
        if action < 0 or action >= ACTIONS or not mask[action]:
            action = legal_random(mask)

        self.env.step(int(action))

        while self.env.agent_selection != LEARN_AGENT:
            agent = self.env.agent_selection
            obs, _, term, trunc, _ = self.env.last()
            action2 = None if term or trunc else self._opponent_action(agent, obs)
            self.env.step(action2)
            if all(self.env.terminations.values()) or all(self.env.truncations.values()):
                break

        obs, reward, term, trunc, info = self.env.last()
        self.current_obs = fix_obs(obs)
        return self.current_obs, float(reward), bool(term), bool(trunc), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--out", default="/home/jupyter/ae/model/league_manager_seeded_v1")
    ap.add_argument("--seed", type=int, default=2606)
    ap.add_argument("--novice", action="store_true")
    ap.add_argument("--init", default="/home/jupyter/ae/model/best/best_model_v5.zip")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    env = LeagueEnv(novice=args.novice, seed_base=args.seed)

    if args.init and Path(args.init).exists():
        model = PPO.load(args.init, env=env, device="cpu", custom_objects={"learning_rate": 1e-5, "lr_schedule": lambda _: 1e-5, "clip_range": lambda _: 0.08})
        print(f"Loaded PPO init model: {args.init}", flush=True)
        model.learning_rate = 2.5e-5
        model.lr_schedule = lambda _: 1e-5
        model.clip_range = lambda _: 0.08
        model.ent_coef = 0.001
        model.n_epochs = 2
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=2.5e-5,
            n_steps=2048,
            batch_size=512,
            n_epochs=2,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.002,
            clip_range=0.08,
            verbose=1,
            tensorboard_log=str(out / "tb"),
            device="cpu",
        )
        print("Started fresh PPO MultiInputPolicy", flush=True)

    cb = CheckpointCallback(
        save_freq=250_000,
        save_path=str(out / "checkpoints"),
        name_prefix="league_manager_ppo",
    )

    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    model.save(str(out / "policy_final.zip"))
    print(f"saved {out / 'policy_final.zip'}", flush=True)


if __name__ == "__main__":
    main()
