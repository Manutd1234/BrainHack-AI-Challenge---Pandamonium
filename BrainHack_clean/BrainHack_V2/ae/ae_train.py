"""
AE Training — TIL-AI 2026
═══════════════════════════════════════════════════════════════════════
Phase 1 — RecurrentPPO with LSTM (10M steps)
  Uses sb3-contrib RecurrentPPO so the policy maintains hidden state
  across the 200-step episode. Solves partial observability (viewcone
  only shows what's directly ahead — LSTM remembers the rest).

Phase 2 — Self-play fine-tune (5M steps)
  Opponents drawn from rolling pool of past checkpoints.

Curriculum:
  Shaped reward ramps from exploration-heavy (+0.1 new cell) in Phase 1
  to competition-realistic (+0.05 new cell) in Phase 2, forcing the
  agent to learn challenge activation and opponent interaction.

Output: model/policy.zip (RecurrentPPO LSTM checkpoint)
Install:
    pip install stable-baselines3[extra] sb3-contrib supersuit tensorboard
    pip install -e /home/jupyter/til-26-ae
"""

import os
import glob
import random
import shutil
import logging
from pathlib import Path
import numpy as np
import supersuit as ss
from gymnasium import spaces

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
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecMonitor

try:
    from til_environment.bomberman_env import parallel_basic_env
    from til_environment.config import default_config
except ImportError:
    raise ImportError("Please run 'pip install -e /home/jupyter/til-26-ae' before training.")

# ── Config ─────────────────────────────────────────────────────────────────
N_ENVS           = 8
PHASE1_STEPS    = 10_000_000
PHASE2_STEPS    = 6_000_000
SEED            = 88

BASE_PATH      = "model/policy_base"
SELFPLAY_PATH  = "model/policy_selfplay"
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
    lstm_hidden_size = 256,
    n_lstm_layers    = 1,
    shared_lstm      = False,
    enable_critic_lstm = True,
)


# ── Reward Shaping Wrapper ──────────────────────────────────────────────────

from pettingzoo.utils.env import ParallelEnv

class RewardShapingParallelWrapper(ParallelEnv):
    """PettingZoo ParallelEnv wrapper for custom exploration reward shaping."""
    def __init__(self, env, bonus: float = 0.0):
        super().__init__()
        self.env = env
        self.bonus = bonus
        self._visited = {}
        # Propagate properties required by pettingzoo/supersuit
        self.possible_agents = env.possible_agents
        self.metadata = getattr(env, "metadata", {})

    @property
    def agents(self):
        return self.env.agents

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def reset(self, seed=None, options=None):
        self._visited = {agent: set() for agent in self.possible_agents}
        obs, infos = self.env.reset(seed=seed, options=options)
        for agent, agent_obs in obs.items():
            loc = tuple(agent_obs.get("location", [0, 0]))
            self._visited[agent].add(loc)
        return obs, infos

    def step(self, actions):
        obs, rews, terminations, truncations, infos = self.env.step(actions)
        if self.bonus > 0.0:
            for agent, agent_obs in obs.items():
                loc = tuple(agent_obs.get("location", [0, 0]))
                if loc not in self._visited[agent]:
                    self._visited[agent].add(loc)
                    rews[agent] += self.bonus
        return obs, rews, terminations, truncations, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


# ── Vectorized Environment Helpers ──────────────────────────────────────────

def _config():
    cfg = default_config()
    cfg.env.render_mode = None
    cfg.env.novice = True
    cfg.rewards.stationary_penalty = -0.01
    cfg.rewards.invalid_action = -0.02
    cfg.rewards.agent_collide_wall = -0.01
    return cfg

def _patch_observation_dtypes(env):
    """Cast env observations to the dtypes declared by their spaces."""
    original_reset = env.reset
    original_step = env.step

    def cast_many(observations):
        return {
            agent: _cast_to_space(observation, env.observation_space(agent))
            for agent, observation in observations.items()
        }

    def reset(*args, **kwargs):
        observations, infos = original_reset(*args, **kwargs)
        return cast_many(observations), infos

    def step(actions):
        observations, rewards, terminations, truncations, infos = original_step(actions)
        return cast_many(observations), rewards, terminations, truncations, infos

    env.reset = reset
    env.step = step
    return env

def _cast_to_space(value, space):
    if isinstance(space, spaces.Dict):
        return {
            key: _cast_to_space(value[key], subspace)
            for key, subspace in space.spaces.items()
        }
    if isinstance(space, spaces.Box):
        return np.asarray(value, dtype=space.dtype)
    if isinstance(space, spaces.Discrete):
        return np.asarray(value, dtype=space.dtype)
    return value

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

def make_vec_env(num_envs: int, num_cpus: int, bonus: float = 0.0):
    """Build a vectorized PettingZoo multi-agent environment with custom wrappers."""
    env = parallel_basic_env(env_wrappers=[], cfg=_config())
    env = RewardShapingParallelWrapper(env, bonus=bonus)
    env = _patch_observation_dtypes(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=num_envs,
        num_cpus=num_cpus,
        base_class="stable_baselines3",
    )
    return _patch_seed_method(VecMonitor(env))


# ── Training Phases ────────────────────────────────────────────────────────

def phase1():
    Path("model").mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    if os.path.exists(f"{BASE_PATH}.zip"):
        logger.info(f"Phase 1 checkpoint found — loading checkpoint")
        global USE_RECURRENT
        if USE_RECURRENT:
            try:
                model = RecurrentPPO.load(BASE_PATH, device="cuda")
                logger.info("Successfully loaded RecurrentPPO model.")
                return model
            except Exception as e:
                logger.warning(f"Failed to load as RecurrentPPO: {e}")
                logger.info("Falling back to standard PPO to load checkpoint.")
                from stable_baselines3 import PPO
                model = PPO.load(BASE_PATH, device="cuda")
                USE_RECURRENT = False
                logger.info("Successfully loaded standard PPO model. Disabling recurrent training.")
                return model
        else:
            from stable_baselines3 import PPO
            model = PPO.load(BASE_PATH, device="cuda")
            logger.info("Successfully loaded standard PPO model.")
            return model

    logger.info("=" * 60)
    logger.info("PHASE 1 — RecurrentPPO LSTM (10M steps, exploration shaping = +0.10)")
    logger.info("=" * 60)

    train_env = make_vec_env(N_ENVS, N_ENVS, bonus=0.10)
    eval_env = make_vec_env(1, 1, bonus=0.0)

    policy = "MlpLstmPolicy" if USE_RECURRENT else "MultiInputPolicy"
    config  = LSTM_CONFIG    if USE_RECURRENT else PPO_CONFIG

    model = RecurrentPPO(policy=policy, env=train_env, seed=SEED, device="cuda", **{
        k: v for k, v in config.items()
        if k not in ("policy", "tensorboard_log")
    })

    model.learn(
        total_timesteps=PHASE1_STEPS,
        progress_bar=True,
        callback=[
            CheckpointCallback(
                save_freq=max(1, 500_000 // N_ENVS),
                save_path="model/ckpts_p1/",
                name_prefix="p1",
            ),
            EvalCallback(
                eval_env, eval_freq=max(1, 200_000 // N_ENVS),
                n_eval_episodes=6, deterministic=True,
                best_model_save_path="model/best_p1/",
            ),
        ],
    )
    model.save(BASE_PATH)
    logger.info(f"Phase 1 done → {BASE_PATH}.zip")
    return model

def phase2(base_model):
    logger.info("=" * 60)
    logger.info("PHASE 2 — Fine-tuning (5M steps, exploration shaping = +0.05)")
    logger.info("=" * 60)

    train_env = make_vec_env(N_ENVS, N_ENVS, bonus=0.05)
    eval_env = make_vec_env(1, 1, bonus=0.0)

    policy = "MlpLstmPolicy" if USE_RECURRENT else "MultiInputPolicy"
    config  = {**LSTM_CONFIG, "learning_rate": 1e-4, "ent_coef": 0.005} \
              if USE_RECURRENT else \
              {**PPO_CONFIG, "learning_rate": 1e-4, "ent_coef": 0.005}

    if USE_RECURRENT:
        model = RecurrentPPO(policy=policy, env=train_env, seed=SEED, device="cuda", **{
            k: v for k, v in config.items()
            if k not in ("policy", "tensorboard_log")
        })
    else:
        from stable_baselines3 import PPO
        model = PPO(policy=policy, env=train_env, seed=SEED, device="cuda", **{
            k: v for k, v in config.items()
            if k not in ("policy", "tensorboard_log")
        })
    model.set_parameters(base_model.get_parameters())

    model.learn(
        total_timesteps=PHASE2_STEPS,
        progress_bar=True,
        reset_num_timesteps=True,
        callback=[
            CheckpointCallback(
                save_freq=max(1, 500_000 // N_ENVS),
                save_path="model/ckpts_p2/",
                name_prefix="p2",
            ),
            EvalCallback(
                eval_env, eval_freq=max(1, 200_000 // N_ENVS),
                n_eval_episodes=6, deterministic=True,
                best_model_save_path="model/best_p2/",
            ),
        ],
    )
    model.save(SELFPLAY_PATH)
    shutil.copy(f"{SELFPLAY_PATH}.zip", "model/policy.zip")
    logger.info(f"Phase 2 done → model/policy.zip")
    return model

if __name__ == "__main__":
    base = phase1()
    phase2(base)
    logger.info("Training complete → copy model/policy.zip to ae/model/policy.zip")
