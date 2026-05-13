"""Train the AE agent with discrete Soft Actor-Critic.

Stable-Baselines3 SAC only supports continuous action spaces, while TIL-26 AE
uses six discrete actions. This script implements the discrete SAC update
directly and reuses the ResNet-bottleneck policy shipped in ``ae/src``.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from ae_model import (  # noqa: E402
    NUM_ACTIONS,
    SACCriticNetwork,
    SACPolicyNetwork,
    batch_encode_observations,
    masked_logits,
)


@dataclass
class Transition:
    obs: dict[str, Any]
    action: int
    reward: float
    next_obs: dict[str, Any]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def make_env(mode: str, seed: int):
    import til_environment

    if hasattr(til_environment, "make"):
        return til_environment.make(mode=mode, seed=seed)

    from til_environment import bomberman_env
    from til_environment.config import default_config

    config = default_config()
    config.env.novice = mode == "novice"
    return bomberman_env.basic_env(env_wrappers=[], cfg=config)


def as_plain_obs(observation: dict[str, Any]) -> dict[str, Any]:
    if observation is None:
        return {}
    plain = {}
    for key, value in observation.items():
        if hasattr(value, "tolist"):
            plain[key] = value.tolist()
        else:
            plain[key] = value
    return plain


@torch.no_grad()
def select_action(
    actor: SACPolicyNetwork,
    observation: dict[str, Any],
    epsilon: float,
    device: torch.device,
) -> int:
    mask = observation.get("action_mask", [1] * NUM_ACTIONS)
    valid = [idx for idx, allowed in enumerate(mask) if allowed == 1]
    if not valid:
        return NUM_ACTIONS - 1
    if random.random() < epsilon:
        return random.choice(valid)

    agent_view, base_view, scalars, action_mask = batch_encode_observations(
        [observation],
        device,
    )
    logits = actor(agent_view, base_view, scalars)
    logits = masked_logits(logits, action_mask)
    probs = F.softmax(logits, dim=-1)
    return int(torch.distributions.Categorical(probs=probs).sample().item())


def update_sac(
    actor: SACPolicyNetwork,
    critic1: SACCriticNetwork,
    critic2: SACCriticNetwork,
    target1: SACCriticNetwork,
    target2: SACCriticNetwork,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    alpha: float,
    tau: float,
    device: torch.device,
) -> dict[str, float]:
    batch = replay.sample(batch_size)
    observations = [transition.obs for transition in batch]
    next_observations = [transition.next_obs for transition in batch]
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

    agent, base, scalars, masks = batch_encode_observations(observations, device)
    next_agent, next_base, next_scalars, next_masks = batch_encode_observations(
        next_observations,
        device,
    )

    with torch.no_grad():
        next_logits = masked_logits(
            actor(next_agent, next_base, next_scalars),
            next_masks,
        )
        next_log_probs = F.log_softmax(next_logits, dim=-1)
        next_probs = next_log_probs.exp()
        next_q = torch.min(
            target1(next_agent, next_base, next_scalars),
            target2(next_agent, next_base, next_scalars),
        )
        next_value = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)
        target_q = rewards + gamma * (1.0 - dones) * next_value

    current_q1 = critic1(agent, base, scalars).gather(1, actions[:, None]).squeeze(1)
    current_q2 = critic2(agent, base, scalars).gather(1, actions[:, None]).squeeze(1)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), 5.0)
    critic_opt.step()

    logits = masked_logits(actor(agent, base, scalars), masks)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    q_min = torch.min(critic1(agent, base, scalars), critic2(agent, base, scalars))
    actor_loss = (probs * (alpha * log_probs - q_min)).sum(dim=-1).mean()

    actor_opt.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
    actor_opt.step()

    with torch.no_grad():
        for target, source in ((target1, critic1), (target2, critic2)):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)

    return {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
    }


def train_on_transition(
    transition: Transition,
    replay: ReplayBuffer,
    state: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    replay.add(transition)
    state["steps"] += 1
    if len(replay) < max(args.batch_size, args.warmup_steps):
        return

    for _ in range(args.updates_per_step):
        metrics = update_sac(
            state["actor"],
            state["critic1"],
            state["critic2"],
            state["target1"],
            state["target2"],
            state["actor_opt"],
            state["critic_opt"],
            replay,
            args.batch_size,
            args.gamma,
            args.alpha,
            args.tau,
            device,
        )
        state["last_metrics"] = metrics


def rollout_gym_env(env, state, replay, args, device) -> float:
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    obs = as_plain_obs(obs)
    total_reward = 0.0

    for _ in range(args.max_episode_steps):
        epsilon = max(args.min_epsilon, args.epsilon * (1 - state["steps"] / args.total_steps))
        action = select_action(state["actor"], obs, epsilon, device)
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, _info = step_result
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, _info = step_result
        next_obs = as_plain_obs(next_obs)
        total_reward += float(reward)
        train_on_transition(
            Transition(obs, action, float(reward), next_obs, done),
            replay,
            state,
            args,
            device,
        )
        obs = next_obs
        if done or state["steps"] >= args.total_steps:
            break
    return total_reward


def rollout_aec_env(env, state, replay, args, device) -> float:
    env.reset()
    controlled_agent = env.possible_agents[0]
    last_obs = None
    last_action = None
    total_reward = 0.0

    for agent in env.agent_iter(args.max_episode_steps * len(env.possible_agents)):
        observation, reward, termination, truncation, _info = env.last()
        done = bool(termination or truncation)
        reward = float(reward)

        if agent == controlled_agent:
            obs = as_plain_obs(observation)
            total_reward += reward
            if last_obs is not None and last_action is not None:
                train_on_transition(
                    Transition(last_obs, last_action, reward, obs, done),
                    replay,
                    state,
                    args,
                    device,
                )
            if done:
                action = None
                last_obs = None
                last_action = None
            else:
                epsilon = max(
                    args.min_epsilon,
                    args.epsilon * (1 - state["steps"] / args.total_steps),
                )
                action = select_action(state["actor"], obs, epsilon, device)
                last_obs = obs
                last_action = action
        else:
            action = None if done else env.action_space(agent).sample()

        env.step(action)
        if state["steps"] >= args.total_steps:
            break
    return total_reward


def save_policy(actor: SACPolicyNetwork, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": actor.state_dict()}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["novice", "advanced"], default="advanced")
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--min-epsilon", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("models/ae/sac_resnet_policy.pt"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = SACPolicyNetwork().to(device)
    critic1 = SACCriticNetwork().to(device)
    critic2 = SACCriticNetwork().to(device)
    target1 = SACCriticNetwork().to(device)
    target2 = SACCriticNetwork().to(device)
    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic2.state_dict())

    state = {
        "actor": actor,
        "critic1": critic1,
        "critic2": critic2,
        "target1": target1,
        "target2": target2,
        "actor_opt": torch.optim.Adam(actor.parameters(), lr=args.lr),
        "critic_opt": torch.optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()),
            lr=args.lr,
        ),
        "steps": 0,
        "last_metrics": {},
    }
    replay = ReplayBuffer(args.buffer_size)
    envs = [make_env(args.mode, args.seed + idx) for idx in range(args.envs)]
    next_save_step = args.save_freq

    print(f"Training discrete SAC on {device} for {args.total_steps:,} steps")
    episode = 0
    while state["steps"] < args.total_steps:
        env = envs[episode % len(envs)]
        if hasattr(env, "agent_iter") and hasattr(env, "possible_agents"):
            reward = rollout_aec_env(env, state, replay, args, device)
        else:
            reward = rollout_gym_env(env, state, replay, args, device)
        episode += 1

        if state["steps"] >= next_save_step:
            save_policy(actor, args.save_path)
            next_save_step += args.save_freq

        print(
            f"episode={episode} steps={state['steps']} reward={reward:.2f} "
            f"buffer={len(replay)} metrics={state['last_metrics']}"
        )

    save_policy(actor, args.save_path)
    for env in envs:
        close = getattr(env, "close", None)
        if close is not None:
            close()
    print(f"Training complete. Policy saved to {args.save_path}")


if __name__ == "__main__":
    main()
