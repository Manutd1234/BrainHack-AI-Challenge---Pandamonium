"""
AE Training — TIL-AI 2026
═══════════════════════════════════════════════════════════════════════
Phase 1 — RecurrentPPO with LSTM (10M steps)
  Uses sb3-contrib RecurrentPPO so the policy maintains hidden state
  across the 200-step episode. Solves partial observability (viewcone
  only shows what's directly ahead — LSTM remembers the rest).

Phase 2 — Self-play fine-tune (5M steps)
  Opponents drawn from rolling pool of past checkpoints.
  PoolUpdateCallback adds live policy to pool every 500K steps.

Curriculum:
  Shaped reward ramps from exploration-heavy (+0.1 new cell) in Phase 1
  to competition-realistic (+0.05 new cell) in Phase 2, forcing the
  agent to learn challenge activation and opponent interaction.

Output: model/policy.zip (RecurrentPPO LSTM checkpoint)
Install:
    pip install stable-baselines3[extra] sb3-contrib tensorboard
    pip install -e /path/to/til-26/
"""

import glob
import logging
import os
import random
import shutil

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from sb3_contrib import RecurrentPPO
    USE_RECURRENT = True
    logger.info("Using RecurrentPPO (LSTM policy)")
except ImportError:
    from stable_baselines3 import PPO as RecurrentPPO
    USE_RECURRENT = False
    logger.warning("sb3-contrib not found — falling back to standard PPO (no LSTM)")

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

try:
    from til_environment.env import TILEnvironment
except ImportError:
    raise ImportError("pip install -e /path/to/til-26/")

# ── Config ─────────────────────────────────────────────────────────────────
N_ENVS          = 8
PHASE1_STEPS    = 10_000_000
PHASE2_STEPS    = 5_000_000
POOL_SIZE       = 8
POOL_UPDATE_FREQ = 500_000

BASE_PATH      = "model/policy_base"
SELFPLAY_PATH  = "model/policy_selfplay"
POOL_DIR       = "model/pool"
LOG_DIR        = "logs"

PPO_CONFIG = dict(
    learning_rate  = 3e-4,
    n_steps        = 2048,
    batch_size     = 512,
    n_epochs       = 10,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.01,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,
    verbose        = 1,
    tensorboard_log= LOG_DIR,
)

LSTM_CONFIG = dict(
    **PPO_CONFIG,
    # RecurrentPPO extra params
    lstm_hidden_size = 256,
    n_lstm_layers    = 1,
    shared_lstm      = False,
    enable_critic_lstm = True,
)


# ── Environments ───────────────────────────────────────────────────────────

class ExplorationEnv(TILEnvironment):
    """Phase 1: heavy exploration shaping."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._visited: set = set()

    def reset(self, **kw):
        self._visited.clear()
        return super().reset(**kw)

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        loc = tuple(obs.get("location", [0, 0]))
        if loc not in self._visited:
            self._visited.add(loc)
            rew += 0.10   # strong exploration bonus in Phase 1
        return obs, rew, term, trunc, info


class SelfPlayEnv(TILEnvironment):
    """Phase 2: exploration shaping + pool opponent."""
    def __init__(self, pool_dir: str, *a, **kw):
        super().__init__(*a, **kw)
        self.pool_dir    = pool_dir
        self._opp        = None
        self._opp_path   = None
        self._visited: set = set()

    def reset(self, **kw):
        self._visited.clear()
        # Sample new opponent from pool
        files = glob.glob(os.path.join(self.pool_dir, "*.zip"))
        if files:
            path = random.choice(files)
            if path != self._opp_path:
                try:
                    self._opp = RecurrentPPO.load(path, device="cpu")
                    self._opp_path = path
                except Exception:
                    self._opp = None
        return super().reset(**kw)

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        loc = tuple(obs.get("location", [0, 0]))
        if loc not in self._visited:
            self._visited.add(loc)
            rew += 0.05   # lighter shaping in Phase 2
        return obs, rew, term, trunc, info


# ── Pool management ────────────────────────────────────────────────────────

class OpponentPool:
    def __init__(self, pool_dir: str, max_size: int = POOL_SIZE):
        self.pool_dir = pool_dir
        self.max_size = max_size
        os.makedirs(pool_dir, exist_ok=True)

    def add(self, model, step: int):
        path = os.path.join(self.pool_dir, f"opp_{step:010d}")
        model.save(path)
        logger.info(f"Pool: added opp at step {step:,}")
        files = sorted(glob.glob(os.path.join(self.pool_dir, "*.zip")))
        while len(files) > self.max_size:
            os.remove(files.pop(0))

    def __len__(self):
        return len(glob.glob(os.path.join(self.pool_dir, "*.zip")))


class PoolUpdateCallback(BaseCallback):
    def __init__(self, pool: OpponentPool, freq: int = POOL_UPDATE_FREQ):
        super().__init__()
        self.pool = pool
        self.freq = freq
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.freq:
            self.pool.add(self.model, self.num_timesteps)
            self._last = self.num_timesteps
        return True


# ── Training ───────────────────────────────────────────────────────────────

def make_phase1_env(seed=0):
    return lambda: ExplorationEnv(seed=seed)

def make_phase2_env(pool_dir, seed=0):
    return lambda: SelfPlayEnv(pool_dir=pool_dir, seed=seed)


def phase1():
    os.makedirs("model", exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    if os.path.exists(f"{BASE_PATH}.zip"):
        logger.info(f"Phase 1 checkpoint found — skipping Phase 1")
        return RecurrentPPO.load(BASE_PATH, device="cuda")

    logger.info("=" * 60)
    logger.info("PHASE 1 — RecurrentPPO LSTM (10M steps, exploration shaping)")
    logger.info("=" * 60)

    envs = SubprocVecEnv([make_phase1_env(i) for i in range(N_ENVS)])
    envs = VecMonitor(envs)
    eval_env = make_vec_env(TILEnvironment, n_envs=1)

    policy = "MlpLstmPolicy" if USE_RECURRENT else "MultiInputPolicy"
    config  = LSTM_CONFIG    if USE_RECURRENT else PPO_CONFIG

    model = RecurrentPPO(policy=policy, env=envs, device="cuda", **{
        k: v for k, v in config.items()
        if k not in ("policy",)
    })

    model.learn(
        total_timesteps=PHASE1_STEPS,
        progress_bar=True,
        callback=[
            CheckpointCallback(
                save_freq=500_000 // N_ENVS,
                save_path="model/ckpts_p1/",
                name_prefix="p1",
            ),
            EvalCallback(
                eval_env, eval_freq=200_000 // N_ENVS,
                n_eval_episodes=10, deterministic=True,
                best_model_save_path="model/best_p1/",
            ),
        ],
    )
    model.save(BASE_PATH)
    logger.info(f"Phase 1 done → {BASE_PATH}.zip")
    return model


def phase2(base_model):
    logger.info("=" * 60)
    logger.info("PHASE 2 — Self-play fine-tune (5M steps)")
    logger.info("=" * 60)

    pool = OpponentPool(POOL_DIR)
    pool.add(base_model, step=0)

    envs = SubprocVecEnv([make_phase2_env(POOL_DIR, i) for i in range(N_ENVS)])
    envs = VecMonitor(envs)
    eval_env = make_vec_env(TILEnvironment, n_envs=1)

    policy = "MlpLstmPolicy" if USE_RECURRENT else "MultiInputPolicy"
    config  = {**LSTM_CONFIG, "learning_rate": 1e-4, "ent_coef": 0.005} \
              if USE_RECURRENT else \
              {**PPO_CONFIG, "learning_rate": 1e-4, "ent_coef": 0.005}

    model = RecurrentPPO(policy=policy, env=envs, device="cuda", **{
        k: v for k, v in config.items()
        if k not in ("policy",)
    })
    model.set_parameters(base_model.get_parameters())

    model.learn(
        total_timesteps=PHASE2_STEPS,
        progress_bar=True,
        reset_num_timesteps=True,
        callback=[
            PoolUpdateCallback(pool),
            CheckpointCallback(
                save_freq=500_000 // N_ENVS,
                save_path="model/ckpts_p2/",
                name_prefix="p2",
            ),
            EvalCallback(
                eval_env, eval_freq=200_000 // N_ENVS,
                n_eval_episodes=10, deterministic=True,
                best_model_save_path="model/best_p2/",
            ),
        ],
    )
    model.save(SELFPLAY_PATH)
    shutil.copy(f"{SELFPLAY_PATH}.zip", "model/policy.zip")
    logger.info(f"Phase 2 done → model/policy.zip")
    return model


if __name__ == "__main__":
    base  = phase1()
    phase2(base)
    logger.info("Training complete → copy model/policy.zip to ae/model/policy.zip")
