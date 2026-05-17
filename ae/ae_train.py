"""Train a shared PPO policy for the TIL-AI 2026 AE environment.

Run from this directory after installing the AE environment package:

    pip install -e /path/to/til-26-ae
    pip install -r requirements.txt
    python ae_train.py
"""

from __future__ import annotations

import os
from pathlib import Path

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor

from til_environment.bomberman_env import parallel_basic_env
from til_environment.config import default_config


TOTAL_STEPS = int(os.getenv("AE_TOTAL_STEPS", "10000000"))
N_ENVS = int(os.getenv("AE_N_ENVS", "8"))
N_CPUS = int(os.getenv("AE_N_CPUS", str(N_ENVS)))
SEED = int(os.getenv("AE_SEED", "88"))

PPO_CONFIG = {
    "policy": "MultiInputPolicy",
    "learning_rate": float(os.getenv("AE_LR", "0.0003")),
    "n_steps": int(os.getenv("AE_N_STEPS", "2048")),
    "batch_size": int(os.getenv("AE_BATCH_SIZE", "512")),
    "n_epochs": int(os.getenv("AE_N_EPOCHS", "10")),
    "gamma": float(os.getenv("AE_GAMMA", "0.99")),
    "gae_lambda": float(os.getenv("AE_GAE_LAMBDA", "0.95")),
    "clip_range": float(os.getenv("AE_CLIP_RANGE", "0.2")),
    "ent_coef": float(os.getenv("AE_ENT_COEF", "0.01")),
    "vf_coef": float(os.getenv("AE_VF_COEF", "0.5")),
    "max_grad_norm": float(os.getenv("AE_MAX_GRAD_NORM", "0.5")),
    "verbose": 1,
    "tensorboard_log": "./logs/tensorboard",
}


def _config():
    cfg = default_config()
    cfg.env.render_mode = None
    cfg.env.novice = os.getenv("AE_NOVICE", "true").lower() in {"1", "true", "yes"}
    cfg.rewards.stationary_penalty = -0.01
    cfg.rewards.invalid_action = -0.02
    cfg.rewards.agent_collide_wall = -0.01
    return cfg


def make_vec_env(num_envs: int, num_cpus: int):
    """Build a vectorized multi-agent self-play environment for SB3."""
    env = parallel_basic_env(env_wrappers=[], cfg=_config())
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=num_envs,
        num_cpus=num_cpus,
        base_class="stable_baselines3",
    )
    return _patch_seed_method(VecMonitor(env))


def _patch_seed_method(env):
    """Add missing seed methods on Supersuit vec-env wrappers for SB3."""
    current = env
    while current is not None:
        if not hasattr(current, "seed"):
            def seed(seed_value=None, _env=current):
                return [seed_value] * int(getattr(_env, "num_envs", 1))

            current.seed = seed
        current = getattr(current, "venv", None)
    return env


def train() -> None:
    Path("model/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("model/best").mkdir(parents=True, exist_ok=True)
    Path("logs/eval").mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(N_ENVS, N_CPUS)
    eval_env = make_vec_env(1, 1)
    model = PPO(env=train_env, seed=SEED, device=os.getenv("AE_DEVICE", "cuda"), **PPO_CONFIG)

    model.learn(
        total_timesteps=TOTAL_STEPS,
        progress_bar=True,
        callback=[
            CheckpointCallback(
                save_freq=max(1, 500_000 // max(1, N_ENVS)),
                save_path="model/checkpoints",
                name_prefix="ppo_til26_ae",
            ),
            EvalCallback(
                eval_env,
                eval_freq=max(1, 200_000 // max(1, N_ENVS)),
                n_eval_episodes=6,
                best_model_save_path="model/best",
                log_path="logs/eval",
                deterministic=True,
            ),
        ],
    )

    model.save("model/policy")
    print("Saved PPO checkpoint to model/policy.zip")


if __name__ == "__main__":
    train()
