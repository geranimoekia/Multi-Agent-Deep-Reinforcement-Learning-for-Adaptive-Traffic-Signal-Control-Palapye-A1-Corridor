"""
Run the trained PPO model in SUMO with the GUI.

Usage:
    python run_model.py                          # best model, normal scenario
    python run_model.py --scenario rush_hour_am  # specific scenario
    python run_model.py --model ppo_models/ppo_traffic_final.zip
"""

import os
import sys
import argparse

# ── SUMO path guard ───────────────────────────────────────────────────────────
if "SUMO_HOME" not in os.environ:
    for guess in [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
    ]:
        if os.path.isdir(guess):
            os.environ["SUMO_HOME"] = guess
            sys.path.append(os.path.join(guess, "tools"))
            break

from stable_baselines3 import PPO
from sumo_env import SumoEnv

SCENARIOS = ["low", "normal", "rush_hour_am", "rush_hour_pm", "holiday", "incident"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="ppo_models/best_model.zip",
        help="Path to model zip file"
    )
    parser.add_argument(
        "--scenario",
        default="normal",
        choices=SCENARIOS + ["random"],
        help="Traffic scenario to run"
    )
    args = parser.parse_args()

    print(f"\nLoading model : {args.model}")
    print(f"Scenario      : {args.scenario}")
    print("Opening SUMO GUI — close the window to exit.\n")

    model = PPO.load(args.model)

    env = SumoEnv(
        use_gui=True,
        scenario_name=None if args.scenario == "random" else args.scenario,
    )

    obs, info = env.reset()
    print(f"Episode started | scenario: {info.get('scenario', args.scenario)}")

    total_reward = 0.0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % 50 == 0:
            print(f"  step {step:4d} | reward {reward:+.3f} | cumulative {total_reward:+.1f}")

        if terminated or truncated:
            break

    print(f"\nEpisode finished after {step} steps | total reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
