from pathlib import Path
import time
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from til_environment import bomberman_env
from til_environment.config import default_config

CANDIDATES = [
    ("best_model_v5", "/home/jupyter/ae/model/best/best_model_v5.zip"),
    ("policy_v5", "/home/jupyter/ae/model/policy_v5.zip"),
    ("v30freeze", "/home/jupyter/ae/model/policy.v30.freeze.zip"),
    ("policy_final", "/home/jupyter/ae/model/v14_official_reward/policy_final.zip"),
    ("lowvarbest", "/home/jupyter/ae/model/checkpoints-Copy1/maskable_ppo_til26_ae_v13_lowvar_objective_11499264_steps.zip"),
    ("v7candidate", "/home/jupyter/ae/model/archives/policy_v7_candidate.zip"),
]

def legal_random(mask):
    legal = np.flatnonzero(np.asarray(mask) > 0)
    return int(np.random.choice(legal)) if len(legal) else 4

def load_model(path):
    try:
        return "maskable", MaskablePPO.load(path, device="cpu")
    except Exception as e1:
        try:
            return "ppo", PPO.load(path, device="cpu")
        except Exception as e2:
            return "fail", f"{e1} | {e2}"

def main():
    cfg = default_config()
    cfg.env.novice = False

    for name, path in CANDIDATES:
        print(f"\n===== {name}: {path} =====", flush=True)
        if not Path(path).exists():
            print("missing")
            continue

        kind, model = load_model(path)
        print("load:", kind, flush=True)
        if kind == "fail":
            print(model)
            continue

        env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
        failures = 0
        actions = 0
        total_reward = 0.0
        t0 = time.time()

        for seed in [2606, 2607]:
            env.reset(seed=seed)
            for agent in env.agent_iter():
                obs, reward, term, trunc, info = env.last()
                if agent == "agent_0":
                    total_reward += float(reward)
                if term or trunc:
                    action = None
                elif agent == "agent_0":
                    try:
                        if kind == "maskable":
                            action, _ = model.predict(
                                obs,
                                action_masks=np.asarray(obs["action_mask"], dtype=bool),
                                deterministic=True,
                            )
                        else:
                            action, _ = model.predict(obs, deterministic=True)
                        action = int(action)
                        mask = np.asarray(obs["action_mask"], dtype=np.int8)
                        if action < 0 or action >= 6 or not mask[action]:
                            failures += 1
                            action = legal_random(mask)
                    except Exception as e:
                        failures += 1
                        if failures <= 3:
                            print("predict_fail:", repr(e), flush=True)
                        action = legal_random(obs.get("action_mask", [0,0,0,0,1,0]))
                    actions += 1
                else:
                    action = legal_random(obs.get("action_mask", [0,0,0,0,1,0]))
                env.step(action)

        elapsed = time.time() - t0
        print(f"actions={actions} failures={failures} reward={total_reward:.1f} elapsed={elapsed:.1f}s", flush=True)

if __name__ == "__main__":
    main()
