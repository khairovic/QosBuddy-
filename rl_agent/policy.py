"""
rl_agent/policy.py — Serving-side wrapper around the trained PPO agent.

The FastAPI backend instantiates a single RLPolicy at startup (in `lifespan`)
and reuses it for every /rl/predict and /rl/simulate request.

The PPO model was trained on the QoSNetworkEnv digital twin (8-dim obs,
4 discrete actions). See rl_agent/config.json for training hyperparameters
and benchmark results.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .env import QoSNetworkEnv

log = logging.getLogger("qosbuddy.rl")

ACTION_NAMES: Dict[int, str] = {0: "reroute", 1: "throttle", 2: "prioritize", 3: "no_action"}
OBS_COLS: List[str] = [
    "sinr_dl_db", "throughput_mbps", "delay_ms", "jitter_ms",
    "packet_loss_ratio", "prb_utilization", "retransmissions", "sla_risk_score",
]

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL = _MODULE_DIR / "models" / "ppo_qos_agent_best.zip"
_DEFAULT_DATASET = Path(os.getenv("QOSBUDDY_DATA_DIR", "./data")) / "lena_dataset_cleaned.csv"
_DEFAULT_CONFIG = _MODULE_DIR / "config.json"


class RLPolicy:
    """Lazy-loaded PPO wrapper. Safe to instantiate without the model present —
    .ready is False until .load() succeeds."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path or _DEFAULT_MODEL)
        self.dataset_path = Path(dataset_path or _DEFAULT_DATASET)
        self.config_path = Path(config_path or _DEFAULT_CONFIG)
        self._model = None
        self._env: Optional[QoSNetworkEnv] = None
        self._config: Dict = {}
        self.ready: bool = False
        self.error: Optional[str] = None

    def load(self) -> bool:
        """Load PPO weights + twin env. Returns True on success; logs and
        returns False on any failure so the rest of the API can still serve."""
        try:
            from stable_baselines3 import PPO
        except ImportError as e:
            self.error = f"stable-baselines3 not installed: {e}"
            log.warning(self.error)
            return False

        if not self.model_path.exists():
            self.error = f"PPO model not found at {self.model_path}"
            log.warning(self.error)
            return False

        try:
            ds = str(self.dataset_path) if self.dataset_path.exists() else None
            self._env = QoSNetworkEnv(ns3_data_path=ds, episode_len=200)
            self._model = PPO.load(str(self.model_path), env=self._env)
            if self.config_path.exists():
                self._config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.ready = True
            log.info(
                f"RLPolicy ready — model={self.model_path.name}, "
                f"dataset={'yes' if ds else 'synthetic'}"
            )
            return True
        except Exception as e:
            self.error = f"PPO load failed: {e}"
            log.exception(self.error)
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, obs: List[float], deterministic: bool = True) -> Dict:
        """Predict a single action. `obs` must be 8 floats matching OBS_COLS."""
        if not self.ready:
            raise RuntimeError(f"RLPolicy not ready: {self.error or 'not loaded'}")
        if len(obs) != 8:
            raise ValueError(f"obs must have 8 elements (got {len(obs)}): {OBS_COLS}")

        arr = np.asarray(obs, dtype=np.float32)
        action, _ = self._model.predict(arr, deterministic=deterministic)
        action = int(action)

        probs = self._action_distribution(arr)
        return {
            "action":         action,
            "action_name":    ACTION_NAMES[action],
            "probabilities":  probs,
            "deterministic":  deterministic,
            "obs":            {k: float(v) for k, v in zip(OBS_COLS, obs)},
        }

    def _action_distribution(self, obs: np.ndarray) -> Dict[str, float]:
        """Pull softmax probabilities from the PPO policy network so the
        dashboard can show confidence, not just a single action."""
        import torch
        try:
            obs_tensor, _ = self._model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                dist = self._model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy().flatten()
            return {ACTION_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)}
        except Exception as e:
            log.debug(f"action distribution unavailable: {e}")
            return {ACTION_NAMES[i]: 0.0 for i in range(4)}

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate(self, episodes: int = 10, deterministic: bool = True) -> Dict:
        """Run `episodes` live episodes on the digital twin. Returns per-episode
        reward + SLA violation counts plus aggregate stats."""
        if not self.ready:
            raise RuntimeError(f"RLPolicy not ready: {self.error or 'not loaded'}")
        episodes = max(1, min(int(episodes), 100))

        per_ep = []
        actions_hist = [0, 0, 0, 0]
        for ep in range(episodes):
            obs, _ = self._env.reset()
            ep_reward, ep_violations = 0.0, 0
            done = False
            while not done:
                action, _ = self._model.predict(obs, deterministic=deterministic)
                action = int(action)
                actions_hist[action] += 1
                obs, reward, terminated, truncated, info = self._env.step(action)
                ep_reward += float(reward)
                ep_violations += int(info.get("sla_violation", 0))
                done = terminated or truncated
            per_ep.append({
                "episode":    ep + 1,
                "reward":     round(ep_reward, 3),
                "violations": ep_violations,
            })

        rewards = [e["reward"] for e in per_ep]
        vios = [e["violations"] for e in per_ep]
        total_actions = sum(actions_hist) or 1
        return {
            "episodes": per_ep,
            "summary": {
                "episodes":        episodes,
                "mean_reward":     round(float(np.mean(rewards)), 3),
                "std_reward":      round(float(np.std(rewards)), 3),
                "mean_violations": round(float(np.mean(vios)), 2),
                "total_violations": int(sum(vios)),
                "action_mix": {
                    ACTION_NAMES[i]: round(actions_hist[i] / total_actions, 3)
                    for i in range(4)
                },
            },
        }

    # ── Metadata ──────────────────────────────────────────────────────────────

    def benchmark(self) -> Dict:
        """Return training-time benchmark comparison (PPO vs Random vs Rule-Based)."""
        bench = self._config.get("benchmark", {})
        return {
            "benchmark":      bench,
            "algorithm":      self._config.get("algorithm", "PPO"),
            "library":        self._config.get("library", "stable-baselines3"),
            "total_timesteps": self._config.get("total_timesteps"),
            "environment":    self._config.get("environment", "QoSNetworkEnv"),
            "dataset":        self._config.get("dataset", self.dataset_path.name),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_policy_singleton: Optional[RLPolicy] = None


def get_policy() -> RLPolicy:
    """Returns the shared RLPolicy. Lazy-loads on first call."""
    global _policy_singleton
    if _policy_singleton is None:
        _policy_singleton = RLPolicy()
        _policy_singleton.load()
    return _policy_singleton
