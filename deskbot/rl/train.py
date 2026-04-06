"""
DeskBot RL training script — PPO on DeskBotGymEnv.

Usage:
    python3 -m deskbot.rl.train                         # easy, 100k steps
    python3 -m deskbot.rl.train --task medium --steps 500000
    python3 -m deskbot.rl.train --task hard   --steps 1000000 --n-envs 4

Saves model to: models/ppo_deskbot_{task}.zip
TensorBoard:    tensorboard --logdir logs/ppo_deskbot
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on DeskBot")
    p.add_argument("--task",    default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--steps",   type=int, default=100_000)
    p.add_argument("--n-envs",  type=int, default=1,
                   help="Number of parallel envs (needs sufficient RAM)")
    p.add_argument("--lr",      type=float, default=3e-4)
    p.add_argument("--ent-coef",type=float, default=0.01,
                   help="Entropy coefficient (exploration bonus)")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--resume",  action="store_true",
                   help="Resume training from existing checkpoint")
    return p.parse_args()


def make_env(task: str, seed: int, rank: int):
    """Factory for vectorised environment creation."""
    def _init():
        from deskbot.rl.env import DeskBotGymEnv
        env = DeskBotGymEnv(task=task)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    from deskbot.rl.env import DeskBotGymEnv

    ROOT      = Path(__file__).parent.parent.parent
    MODEL_DIR = ROOT / "models"
    LOG_DIR   = ROOT / "logs" / "ppo_deskbot"
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"ppo_deskbot_{args.task}.zip"

    print(f"\n{'='*55}")
    print(f"  DeskBot PPO Training")
    print(f"  task     : {args.task}")
    print(f"  steps    : {args.steps:,}")
    print(f"  n_envs   : {args.n_envs}")
    print(f"  lr       : {args.lr}")
    print(f"  ent_coef : {args.ent_coef}")
    print(f"  seed     : {args.seed}")
    print(f"  model    : {model_path}")
    print(f"{'='*55}\n")

    # ── Build vectorised env ──────────────────────────────────────────────────
    if args.n_envs == 1:
        env = Monitor(DeskBotGymEnv(task=args.task))
        env.reset(seed=args.seed)
    else:
        env = make_vec_env(
            make_env(args.task, args.seed, 0),
            n_envs  = args.n_envs,
            seed    = args.seed,
        )

    # Separate eval env (no monitoring noise)
    eval_env = Monitor(DeskBotGymEnv(task=args.task))
    eval_env.reset(seed=args.seed + 999)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(MODEL_DIR),
        log_path             = str(LOG_DIR),
        eval_freq            = max(5_000 // args.n_envs, 1),
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq  = max(20_000 // args.n_envs, 1),
        save_path  = str(MODEL_DIR),
        name_prefix = f"ppo_deskbot_{args.task}",
        verbose    = 1,
    )
    callbacks = CallbackList([eval_cb, ckpt_cb])

    # TensorBoard is optional
    try:
        import tensorboard  # noqa: F401
        tb_log = str(LOG_DIR)
    except ImportError:
        tb_log = None
        print("TensorBoard not installed — skipping TB logging (pip install tensorboard)")

    # ── Policy network: MLP with two 256-unit hidden layers ───────────────────
    policy_kwargs = dict(net_arch=[256, 256])

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume and model_path.exists():
        print(f"Resuming from {model_path}")
        model = PPO.load(str(model_path), env=env)
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = env,
            learning_rate   = args.lr,
            n_steps         = 512,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = args.ent_coef,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            tensorboard_log = tb_log,
            policy_kwargs   = policy_kwargs,
            verbose         = 1,
            seed            = args.seed,
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps = args.steps,
        callback        = callbacks,
        progress_bar    = True,
        reset_num_timesteps = not args.resume,
    )
    elapsed = time.time() - t0

    model.save(str(model_path))
    print(f"\nTraining done in {elapsed:.0f}s — saved {model_path}")

    # ── Quick eval ────────────────────────────────────────────────────────────
    print("\nEvaluating trained model (10 episodes)...")
    ep_rewards = []
    for ep in range(10):
        obs, _ = eval_env.reset(seed=ep)
        ep_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = eval_env.step(action)
            ep_r += r
            done = terminated or truncated
        ep_rewards.append(ep_r)

    mean_r = float(np.mean(ep_rewards))
    std_r  = float(np.std(ep_rewards))
    print(f"  Mean reward: {mean_r:.4f} ± {std_r:.4f}")
    print(f"  Min: {min(ep_rewards):.4f}  Max: {max(ep_rewards):.4f}")

    eval_env.close()
    env.close()
    return mean_r


if __name__ == "__main__":
    args = parse_args()
    train(args)
