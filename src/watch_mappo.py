"""
Watch the trained MAPPO policy control the Palapye triple intersection in SUMO-GUI.
Change SCENARIO below to any of: low, normal, rush_hour_am, rush_hour_pm, holiday, incident
"""

import torch
import mappo_env
from mappo_env import MAPPOEnv, TL_IDS, LOCAL_OBS_DIM, N_ACTIONS
from mappo_networks import Actor

SCENARIO = "rush_hour_am"   # <-- change this to try different scenarios
MODEL    = "mappo_models/best_actor.pth"

# Watch a longer episode than the 500-step training default (this run only).
mappo_env.MAX_SIM_STEPS = 2000

actor = Actor(obs_dim=LOCAL_OBS_DIM, n_actions=N_ACTIONS)
actor.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=True))
actor.eval()

env = MAPPOEnv(use_gui=True, scenario_name=SCENARIO)
obs_dict, _ = env.reset()

print(f"\nRunning MAPPO on scenario: {SCENARIO}")
print(f"Episode limit: {mappo_env.MAX_SIM_STEPS} sim-seconds (~{mappo_env.MAX_SIM_STEPS // mappo_env.DELTA_T} RL steps)")
print("Watch the SUMO window. Close it or press Ctrl+C to stop.\n")

step = 0
done = False
total_reward = 0.0

while not done:
    actions = {}
    for tl in TL_IDS:
        obs_t = torch.tensor(obs_dict[tl], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = actor.get_action(obs_t, deterministic=True)
        actions[tl] = int(action.item())

    obs_dict, _, reward, done, _ = env.step(actions)
    total_reward += reward
    step += 1

    if step % 50 == 0:
        print(f"  step {step:4d} | reward this window: {total_reward/step:.3f}")

print(f"\nEpisode finished — {step} steps | mean reward: {total_reward/step:.3f}")
