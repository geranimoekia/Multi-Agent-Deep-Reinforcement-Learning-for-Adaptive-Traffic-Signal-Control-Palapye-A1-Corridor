"""
Train a PPO agent to control traffic signals in the SUMO network.

Run:
    python train_ppo.py

Outputs:
    ppo_models/   — checkpoint files saved every 10 000 steps
    ppo_logs/     — TensorBoard logs  (tensorboard --logdir ppo_logs)

Evaluation:
    A separate env fixed to the "normal" scenario evaluates the agent
    every 20 000 steps and saves the best model as ppo_models/best_model.
"""

import os
import sys

# ── SUMO path guard ───────────────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    guesses = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
    ]
    for g in guesses:
        if os.path.isdir(g):
            os.environ["SUMO_HOME"] = g
            sys.path.append(os.path.join(g, "tools"))
            print(f"[train_ppo] SUMO_HOME set to {g}")
            break
    else:
        print("[train_ppo] WARNING: SUMO_HOME not set and not auto-detected.")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from sumo_env import SumoEnv

MODEL_DIR = "ppo_models"
LOG_DIR   = "ppo_logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

TOTAL_TIMESTEPS = 2_000_000


def make_train_env() -> Monitor:
    """Training env: random scenario each episode."""
    return Monitor(SumoEnv(use_gui=False, scenario_name=None))


def make_eval_env() -> Monitor:
    """Eval env: fixed 'normal' scenario for comparable metrics."""
    return Monitor(SumoEnv(use_gui=False, scenario_name="normal"))


if __name__ == "__main__":
    print("=" * 60)
    print("PPO Traffic Signal Training")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Models saved to : {MODEL_DIR}/")
    print(f"  Logs saved to   : {LOG_DIR}/")
    print("=" * 60)

    train_env = make_train_env()
    eval_env  = make_eval_env()

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = 3e-4,
        n_steps         = 2048,       # steps collected before each update
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,       # discount — long-horizon traffic optimisation
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,       # entropy bonus keeps the agent exploring
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = MODEL_DIR,
        name_prefix = "ppo_traffic",
        verbose     = 1,
    )

    eval_cb = EvalCallback(
        eval_env            = eval_env,
        best_model_save_path = MODEL_DIR,
        log_path            = LOG_DIR,
        eval_freq           = 20_000,
        n_eval_episodes     = 3,
        deterministic       = True,
        verbose             = 1,
    )

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_cb, eval_cb],
    )

    final_path = os.path.join(MODEL_DIR, "ppo_traffic_final")
    model.save(final_path)
    print(f"\nTraining complete. Final model: {final_path}.zip")

    train_env.close()
    eval_env.close()
