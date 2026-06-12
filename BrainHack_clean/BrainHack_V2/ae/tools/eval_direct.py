import argparse
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ae" / "src"))

from ae_manager import AEManager
from til_environment import bomberman_env
from til_environment.config import default_config

MAX_SCORE = 1000

def jsonable(obs):
    return {k: v if type(v) in (int, float) else v.tolist() for k, v in obs.items()}

def run(rounds, novice, seed):
    cfg = default_config()
    cfg.env.novice = novice
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    controlled = env.possible_agents[0]
    rewards = {agent: 0.0 for agent in env.possible_agents}
    manager = AEManager()
    t0 = time.time()

    for i in range(rounds):
        manager.reset()
        env.reset(seed=None if seed is None else seed + i)
        random.seed(None if seed is None else seed + i)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                action = None
            elif agent == controlled:
                action = int(manager.ae(jsonable(observation)))
            else:
                action = env.action_space(agent).sample()
            env.step(action)

    elapsed = time.time() - t0
    env.close()
    total = rewards[controlled]
    score = total / rounds / MAX_SCORE
    print(f"rounds={rounds} novice={novice} seed={seed}")
    print(f"total_rewards={total:.1f}")
    print(f"score={score:.6f}")
    print(f"elapsed={elapsed:.2f}s speed_score_est={1 - min(elapsed, 1800) / 1800:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--novice", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    run(args.rounds, args.novice, args.seed)
