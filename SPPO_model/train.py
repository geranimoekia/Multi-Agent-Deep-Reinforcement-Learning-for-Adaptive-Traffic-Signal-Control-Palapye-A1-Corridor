import os
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from env import SumoEnv


def main():

    # ============================================================
    # ENVIRONMENT SETUP
    # ============================================================
    sumo_config = "map.sumocfg"

    def make_env():
        return Monitor(SumoEnv(sumo_cfg=sumo_config, use_gui=False))

    env = DummyVecEnv([make_env])

    # ============================================================
    # LOGGING + FOLDERS
    # ============================================================
    run_name = time.strftime("PPO_run_%Y%m%d_%H%M%S")
    log_dir = f"./tb_logs/{run_name}/"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ============================================================
    # DEVICE SELECTION
    # ============================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    # ============================================================
    # CHECKPOINT CALLBACK
    # ============================================================
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=checkpoint_dir,
        name_prefix="ppo_traffic"
    )

    # ============================================================
    # EVAL CALLBACK
    # ============================================================
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # ============================================================
    # PPO MODEL
    # ============================================================
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2
    )

    # ============================================================
    # TRAINING
    # ============================================================
    TOTAL_TIMESTEPS = 200000

    print(f"🚦 Training PPO Agent for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="PPO_traffic_signal"
    )

    # ============================================================
    # SAVE FINAL MODEL
    # ============================================================
    model.save("ppo_traffic_final")

    print("\n🎉 Training completed successfully!")
    print("Final model saved as: ppo_traffic_final.zip")
    print(f"TensorBoard logs saved in: {log_dir}")


if __name__ == "__main__":
    main()
