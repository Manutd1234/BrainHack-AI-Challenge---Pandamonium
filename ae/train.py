"""
AE Training Script
Run on Workbench (with til_environment installed):
    conda activate env
    python train.py

Trains a PPO agent and saves the policy to models/ae/policy.zip
Then build the Docker image which bundles in the trained weights.
"""

import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import til_environment  # provided by the competition

os.makedirs("models/ae", exist_ok=True)

# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(rank: int, seed: int = 0):
    """Creates a single TIL environment instance."""
    def _init():
        env = til_environment.make(
            # Use "advanced" for variable map training (better generalisation)
            # Switch to "novice" if you are on the Novice track
            mode="advanced",
            seed=seed + rank,
        )
        return env
    return _init


N_ENVS = 8          # parallel environments — reduce if OOM
TOTAL_STEPS = 5_000_000
EVAL_FREQ = 50_000

# ---------------------------------------------------------------------------
# Vectorised envs
# ---------------------------------------------------------------------------
train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
train_env = VecMonitor(train_env)

eval_env = SubprocVecEnv([make_env(N_ENVS)])
eval_env = VecMonitor(eval_env)

# ---------------------------------------------------------------------------
# PPO model — tuned hyperparameters for gridworld
# ---------------------------------------------------------------------------
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.995,           # high gamma — rewards can be delayed in gridworld
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,         # encourages exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={
        "net_arch": [256, 256, 128],  # deeper net for complex gridworld
    },
    tensorboard_log="logs/ae_ppo",
    verbose=1,
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
checkpoint_cb = CheckpointCallback(
    save_freq=EVAL_FREQ,
    save_path="models/ae/checkpoints/",
    name_prefix="ppo_ae",
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="models/ae/",
    log_path="logs/ae_eval/",
    eval_freq=EVAL_FREQ // N_ENVS,
    n_eval_episodes=20,
    deterministic=True,
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print(f"Training PPO for {TOTAL_STEPS:,} steps across {N_ENVS} envs ...")
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[checkpoint_cb, eval_cb],
    progress_bar=True,
)

# Save final policy
model.save("models/ae/policy")
print("Training complete. Policy saved to models/ae/policy.zip")
print("Best policy (by eval reward) saved to models/ae/best_model.zip")
