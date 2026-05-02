#!/usr/bin/env python3
"""
inference.py - Load trained PPO agent and run on QoSNetworkEnv.
Usage: python inference.py [--episodes 10] [--dataset lena_dataset_cleaned.csv]
"""
import os, sys, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from stable_baselines3 import PPO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from QoSNetworkEnv import QoSNetworkEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--dataset',  type=str, default='lena_dataset_cleaned.csv')
    parser.add_argument('--model',    type=str, default='ppo_qos_agent_best')
    args = parser.parse_args()

    env   = QoSNetworkEnv(ns3_data_path=args.dataset, episode_len=200)
    agent = PPO.load(args.model, env=env)
    print(f"Loaded model: {args.model}")
    print(f"Running {args.episodes} episodes...")

    all_rewards, all_violations = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_r, ep_v = 0.0, 0
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_r += reward
            ep_v += info["sla_violation"]
            done = terminated or truncated
        all_rewards.append(ep_r)
        all_violations.append(ep_v)
        print(f"  Ep {ep+1:2d}: reward={ep_r:6.2f}  violations={ep_v}")

    print(f"\nMean reward    : {np.mean(all_rewards):.3f}")
    print(f"Mean violations: {np.mean(all_violations):.1f}")

if __name__ == "__main__":
    main()