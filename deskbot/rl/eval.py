"""
Evaluate a saved PPO policy on DeskBot.

Usage:
    python3 -m deskbot.rl.eval --task easy
    python3 -m deskbot.rl.eval --task medium --episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task",     default="easy", choices=["easy","medium","hard"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--model",    type=str, default=None,
                   help="Path to .zip model (default: models/ppo_deskbot_{task}.zip)")
    return p.parse_args()


def eval_policy(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from deskbot.rl.env import DeskBotGymEnv

    ROOT = Path(__file__).parent.parent.parent
    model_path = args.model or str(ROOT / "models" / f"ppo_deskbot_{args.task}.zip")

    if not Path(model_path).exists():
        print(f"No model found at {model_path}")
        print("Train first:  python3 -m deskbot.rl.train --task", args.task)
        return

    print(f"Loading {model_path}")
    model = PPO.load(model_path)

    env = DeskBotGymEnv(task=args.task)
    ep_rewards, ep_lengths, ep_orders = [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_r, ep_len = 0.0, 0
        final_order = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_r   += r
            ep_len += 1
            final_order = info.get("order", 0.0)
            done = terminated or truncated

        ep_rewards.append(ep_r)
        ep_lengths.append(ep_len)
        ep_orders.append(final_order)
        print(f"  ep {ep+1:3d} | steps={ep_len:3d} | "
              f"reward={ep_r:.4f} | order={final_order:.3f}")

    print(f"\nSummary over {args.episodes} episodes:")
    print(f"  Reward : {np.mean(ep_rewards):.4f} ± {np.std(ep_rewards):.4f}")
    print(f"  Order  : {np.mean(ep_orders):.4f} ± {np.std(ep_orders):.4f}")
    print(f"  Length : {np.mean(ep_lengths):.1f} steps avg")
    env.close()


if __name__ == "__main__":
    eval_policy(parse_args())
