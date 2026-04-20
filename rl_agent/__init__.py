"""
QoSBuddy M6 — RL Agent module (DSO3.1)
========================================
Integrates the PPO policy trained on the NS-3 LENA digital twin
(`rl_agent/models/ppo_qos_agent_best.zip`) into the M6 serving stack.

Exposes:
  • QoSNetworkEnv — Gymnasium env backed by NS-3 data
  • RLPolicy      — singleton wrapper around the trained PPO (load once, reuse)
  • ACTION_NAMES  — 0=reroute, 1=throttle, 2=prioritize, 3=no_action
"""
from .env import QoSNetworkEnv
from .policy import RLPolicy, get_policy, ACTION_NAMES, OBS_COLS

__all__ = ["QoSNetworkEnv", "RLPolicy", "get_policy", "ACTION_NAMES", "OBS_COLS"]
