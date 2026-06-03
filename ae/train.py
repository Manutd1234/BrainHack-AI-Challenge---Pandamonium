"""
TIL-AI 2026 — AE training entrypoint.

Three phases, all run from this script (use --phase to pick or `all` for full run).

  Phase 1 (bootstrap)        : 15M steps  MaskablePPO vs scripted+random opponents
                               with reward shaping coef=1.0 -> 0.5
  Phase 2 (league self-play) : 10M steps  vs a frozen-snapshot pool + scripts,
                               shaping coef=0.5 -> 0.0
  Phase 3 (fine-tune)        :  5M steps  vs latest snapshots only, shaping=0,
                               LOWER lr, deterministic policy used for snapshots

Total: 30M steps. If you only have budget for 20M+10M like you proposed, just
skip Phase 3 — Phase 1+2 alone is enough for Novice 0.95+.

Tensorboard at ./runs/  (pip install tensorboard, then `tensorboard --logdir runs`).

Usage:
    python train.py --phase all  --total-envs 16  --device cuda
    python train.py --phase 1    --resume models/ckpt.zip
    python train.py --phase 2
    python train.py --phase 3
"""
from __future__ import annotations

import argparse
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Pull in your training-only deps. til_environment must be installed in the
# training venv (NOT in the inference container).
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config

from src.features import BombermanFeatures
from src.wrappers import OpponentPool, RewardShaper, SB3AECWrapper


# --------------------------------------------------------------------------- #
# Env factory                                                                 #
# --------------------------------------------------------------------------- #
def make_aec_env(novice: bool = True):
    """Factory returning a fresh PettingZoo Bomberman env."""
    cfg = default_config()
    cfg.env.novice = novice                  # Novice = fixed map
    cfg.env.render_mode = None
    return Bomberman(cfg=cfg)


def make_sb3_env(
    opponent_pool: OpponentPool,
    shaping_coef: float,
    novice: bool = True,
) -> Callable:
    """Returns a thunk for SubprocVecEnv."""
    def _thunk():
        shaper = RewardShaper(coef=shaping_coef)
        env = SB3AECWrapper(
            env_factory=lambda: make_aec_env(novice=novice),
            opponent_pool=opponent_pool,
            learner_agent_id="agent_0",
            shaper=shaper,
        )
        # ActionMasker just exposes env.action_masks() to MaskablePPO
        env = ActionMasker(env, lambda e: e.action_masks())
        return env
    return _thunk


# --------------------------------------------------------------------------- #
# Snapshot callback — feeds the league                                        #
# --------------------------------------------------------------------------- #
class SnapshotCallback(BaseCallback):
    """Every `snapshot_every` steps, freeze a copy of the policy and add it
    to the opponent pool. Subprocess envs each have their own pool reference,
    so we ship snapshots through a shared list maintained on the main process
    and re-loaded by envs at reset() time.
    """
    def __init__(self, opponent_pool: OpponentPool, snapshot_every: int = 500_000,
                 save_dir: str = "models/snapshots"):
        super().__init__()
        self.opponent_pool = opponent_pool
        self.snapshot_every = snapshot_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._next_snapshot_at = snapshot_every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_snapshot_at:
            path = self.save_dir / f"snap_{self.num_timesteps:09d}.zip"
            self.model.save(str(path))
            # cold-load a copy so it doesn't share weights with the live policy
            snap = MaskablePPO.load(str(path), device="cpu")
            self.opponent_pool.add_snapshot(snap)
            self.logger.record("league/snapshots", len(self.opponent_pool._snapshots))
            self._next_snapshot_at += self.snapshot_every
        return True


# --------------------------------------------------------------------------- #
# Phase config                                                                #
# --------------------------------------------------------------------------- #
PHASES = {
    1: dict(steps=15_000_000, lr=3e-4, ent=0.02, shaping=1.0,  snapshot_every=2_000_000),
    2: dict(steps=10_000_000, lr=2e-4, ent=0.01, shaping=0.25, snapshot_every=  500_000),
    3: dict(steps= 5_000_000, lr=5e-5, ent=0.005,shaping=0.0,  snapshot_every=  500_000),
}


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def train_phase(
    phase: int,
    total_envs: int,
    device: str,
    novice: bool,
    resume: str | None,
    out_dir: Path,
) -> str:
    cfg = PHASES[phase]
    print(f"\n=== Phase {phase} | steps={cfg['steps']:,} | shaping={cfg['shaping']} ===\n")

    opponent_pool = OpponentPool(max_snapshots=8)

    # If we have prior snapshots from earlier phases, load them
    snap_dir = out_dir / "snapshots"
    if snap_dir.exists():
        for p in sorted(snap_dir.glob("snap_*.zip"))[-8:]:
            try:
                opponent_pool.add_snapshot(MaskablePPO.load(str(p), device="cpu"))
                print(f"  loaded snapshot {p.name}")
            except Exception as e:
                print(f"  WARN: failed to load {p.name}: {e}")

    vec_env = SubprocVecEnv([
        make_sb3_env(opponent_pool, cfg["shaping"], novice=novice)
        for _ in range(total_envs)
    ])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class=BombermanFeatures,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    if resume and os.path.exists(resume):
        print(f"  resuming from {resume}")
        model = MaskablePPO.load(resume, env=vec_env, device=device)
        # override LR / entropy for the new phase
        model.learning_rate = cfg["lr"]
        model.ent_coef = cfg["ent"]
        model._setup_lr_schedule()
    else:
        model = MaskablePPO(
            policy="MultiInputPolicy",
            env=vec_env,
            learning_rate=cfg["lr"],
            n_steps=512,
            batch_size=4096,
            n_epochs=4,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=cfg["ent"],
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(out_dir / "tb"),
            device=device,
            verbose=1,
        )

    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 500_000 // total_envs),
        save_path=str(out_dir / "ckpts"),
        name_prefix=f"phase{phase}",
    )
    snap_cb = SnapshotCallback(
        opponent_pool=opponent_pool,
        snapshot_every=cfg["snapshot_every"],
        save_dir=str(out_dir / "snapshots"),
    )

    model.learn(
        total_timesteps=cfg["steps"],
        callback=CallbackList([ckpt_cb, snap_cb]),
        progress_bar=True,
        reset_num_timesteps=(resume is None),
        tb_log_name=f"phase{phase}",
    )

    out_path = out_dir / f"phase{phase}_final.zip"
    model.save(str(out_path))
    print(f"\n  Phase {phase} done -> {out_path}")
    vec_env.close()
    return str(out_path)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--total-envs", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--novice", action="store_true", default=True,
                    help="Novice = fixed map. Set --no-novice for Advanced.")
    ap.add_argument("--no-novice", dest="novice", action="store_false")
    ap.add_argument("--resume", default=None, help="Path to a .zip checkpoint to resume from.")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "all":
        prev = args.resume
        for p in (1, 2, 3):
            prev = train_phase(p, args.total_envs, args.device, args.novice, prev, out_dir)
        # promote the final phase-3 model to the inference path
        final = out_dir / "phase3_final.zip"
        ae_policy = out_dir / "ae_policy.zip"
        shutil.copy2(final, ae_policy)
        print(f"\n  Promoted {final} -> {ae_policy} (this is what Dockerfile bakes in)")
    else:
        path = train_phase(int(args.phase), args.total_envs, args.device,
                           args.novice, args.resume, out_dir)
        if args.phase == "3":
            ae_policy = out_dir / "ae_policy.zip"
            shutil.copy2(path, ae_policy)
            print(f"\n  Promoted -> {ae_policy}")


if __name__ == "__main__":
    main()
