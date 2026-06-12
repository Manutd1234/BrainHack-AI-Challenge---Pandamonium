import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from train_league import LeagueEnv

def run(model_path, seeds, episodes):
    scores = []
    model = MaskablePPO.load(model_path, device="cpu")
    for seed in seeds:
        for ep in range(episodes):
            env = LeagueEnv(novice=False, seed_base=seed + ep * 1000)
            obs, _ = env.reset(seed=seed + ep)
            total = 0.0
            done = False
            while not done:
                action, _ = model.predict(
                    obs,
                    action_masks=env.action_masks(),
                    deterministic=True,
                )
                obs, reward, term, trunc, _ = env.step(int(action))
                total += float(reward)
                done = bool(term or trunc)
            scores.append(total / 1000.0)
    print("scores:", scores)
    print("mean:", float(np.mean(scores)))
    print("median:", float(np.median(scores)))
    print("min:", float(np.min(scores)))
    print("max:", float(np.max(scores)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[11, 26, 42, 71, 101, 202, 303, 404])
    args = ap.parse_args()
    run(args.model, args.seeds, args.episodes)
