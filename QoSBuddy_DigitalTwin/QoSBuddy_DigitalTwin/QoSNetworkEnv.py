import gymnasium as gym
import numpy as np
import pandas as pd
import os
from gymnasium import spaces


class QoSNetworkEnv(gym.Env):
    """
    Digital Twin Gymnasium environment backed by real NS-3 LENA data.

    Observation (8 floats):
        [sinr_dl_db, throughput_mbps, delay_ms, jitter_ms,
         packet_loss_ratio, prb_utilization, retransmissions, sla_risk_score]

    Actions (Discrete 4):
        0=reroute  1=throttle  2=prioritize  3=no_action

    Reward:
        -1.0 per SLA violation
        +0.3 * risk_reduction vs previous step
        +0.05 proactive bonus (action taken, no violation)
    """
    metadata = {"render_modes": ["human"]}
    ACTION_NAMES = {0: "reroute", 1: "throttle", 2: "prioritize", 3: "no_action"}

    SLA = {
        1: {"sinr": -25.0, "delay": 300.0, "loss": 0.80, "tput": 0.01},
        2: {"sinr": -18.0, "delay": 200.0, "loss": 0.60, "tput": 0.05},
        3: {"sinr": -10.0, "delay": 120.0, "loss": 0.30, "tput": 0.20},
    }

    ACTION_FX = {
        0: {"sinr": +4.0,  "delay": 0.75, "loss": 0.70, "tput": 1.10},
        1: {"sinr": +0.0,  "delay": 0.90, "loss": 0.50, "tput": 0.65},
        2: {"sinr": +6.0,  "delay": 0.65, "loss": 0.45, "tput": 1.30},
        3: {"sinr": -1.5,  "delay": 1.10, "loss": 1.05, "tput": 0.95},
    }

    OBS_LOW  = np.array([-50,  0,    0,   0,   0,   0,   0,  0], dtype=np.float32)
    OBS_HIGH = np.array([ 90, 100, 5000, 2000, 1, 100, 1000, 10], dtype=np.float32)
    OBS_COLS = ['sinr_dl_db','throughput_mbps','delay_ms','jitter_ms',
                'packet_loss_ratio','prb_utilization','retransmissions','sla_risk_score']

    def __init__(self, ns3_data_path=None, episode_len=50, render_mode=None):
        super().__init__()
        self.episode_len = episode_len
        self.render_mode = render_mode
        self._step_idx   = 0
        self._prev_risk  = 0.0
        self._data       = None
        self._data_idx   = 0
        self.observation_space = spaces.Box(
            low=self.OBS_LOW, high=self.OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        if ns3_data_path and os.path.exists(ns3_data_path):
            self._load_data(ns3_data_path)

    def _load_data(self, path):
        df = pd.read_csv(path)
        INT_COLS   = ['timestamp','ue_id','mcs_dl','prb_utilization',
                      'retransmissions','cqi','load_level','mobility_speed','shadowing_enabled']
        FLOAT_COLS = ['sinr_dl_db','throughput_mbps','delay_ms','jitter_ms','packet_loss_ratio']
        for c in INT_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        for c in FLOAT_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna()
        for c in INT_COLS:
            df[c] = df[c].astype(int)
        df = df[~((df['throughput_mbps'] == 0) & (df['delay_ms'] == 0))]
        df = df[(df['sinr_dl_db'].between(-50,90)) & (df['throughput_mbps'].between(0,100)) &
                (df['delay_ms'].between(0,5000)) & (df['packet_loss_ratio'].between(0,1)) &
                (df['prb_utilization'].between(0,100)) & (df['cqi'].between(1,15)) &
                (df['load_level'].between(1,3))]
        df = df.sort_values(['ue_id','timestamp']).reset_index(drop=True)
        self._data = df
        print(f'[QoSNetworkEnv] Loaded {len(df):,} NS-3 rows from {path}')

    def _sla_risk_score(self, row, load_level):
        thr   = self.SLA.get(int(load_level), self.SLA[2])
        score = 0.0
        if row['sinr_dl_db']        < thr['sinr']:  score += 2.5
        if row['delay_ms']          > thr['delay']: score += 2.5
        if row['packet_loss_ratio'] > thr['loss']:  score += 2.5
        if row['throughput_mbps']   < thr['tput']:  score += 2.5
        return min(score, 10.0)

    def _get_raw_row(self):
        if self._data is not None:
            idx = self._data_idx % len(self._data)
            row = self._data.iloc[idx]
            self._data_idx += 1
            return {
                'sinr_dl_db':        float(row['sinr_dl_db']),
                'throughput_mbps':   float(row['throughput_mbps']),
                'delay_ms':          float(row['delay_ms']),
                'jitter_ms':         float(row['jitter_ms']),
                'packet_loss_ratio': float(row['packet_loss_ratio']),
                'prb_utilization':   float(row['prb_utilization']),
                'retransmissions':   float(row['retransmissions']),
                'load_level':        int(row['load_level']),
            }
        return {
            'sinr_dl_db':        float(np.random.normal(-8, 15)),
            'throughput_mbps':   float(np.clip(np.random.exponential(2), 0, 100)),
            'delay_ms':          float(np.clip(np.random.exponential(80), 0, 5000)),
            'jitter_ms':         float(np.clip(np.random.exponential(30), 0, 2000)),
            'packet_loss_ratio': float(np.clip(np.random.beta(1, 8), 0, 1)),
            'prb_utilization':   float(np.random.randint(5, 80)),
            'retransmissions':   float(np.random.randint(0, 20)),
            'load_level':        int(np.random.randint(1, 4)),
        }

    def _apply_action(self, raw, action):
        fx = self.ACTION_FX[action]
        n  = lambda s: np.random.normal(0, s)
        return {
            'sinr_dl_db':        float(np.clip(raw['sinr_dl_db'] + fx['sinr'] + n(1.5), -50, 90)),
            'throughput_mbps':   float(np.clip(raw['throughput_mbps'] * fx['tput'] + n(0.1), 0, 100)),
            'delay_ms':          float(np.clip(raw['delay_ms'] * fx['delay'] + n(5), 0, 5000)),
            'jitter_ms':         float(np.clip(raw['jitter_ms'] + n(2), 0, 2000)),
            'packet_loss_ratio': float(np.clip(raw['packet_loss_ratio'] * fx['loss'], 0, 1)),
            'prb_utilization':   float(np.clip(raw['prb_utilization'] + n(3), 0, 100)),
            'retransmissions':   float(np.clip(raw['retransmissions'] * fx['loss'] + n(1), 0, 1000)),
            'load_level':        raw['load_level'],
        }

    def _row_to_obs(self, row, risk):
        return np.array([
            row['sinr_dl_db'], row['throughput_mbps'], row['delay_ms'],
            row['jitter_ms'], row['packet_loss_ratio'], row['prb_utilization'],
            row['retransmissions'], risk,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_idx = 0
        n = len(self._data) if self._data is not None else 1
        self._data_idx = int(np.random.randint(0, max(1, n)))
        raw  = self._get_raw_row()
        risk = self._sla_risk_score(raw, raw['load_level'])
        self._prev_risk = risk
        obs  = self._row_to_obs(raw, risk)
        return obs, {'sla_violation': 0, 'risk_score': risk, 'action_name': 'reset'}

    def step(self, action):
        self._step_idx += 1
        raw      = self._get_raw_row()
        modified = self._apply_action(raw, action)
        risk     = self._sla_risk_score(modified, modified['load_level'])
        thr      = self.SLA.get(int(modified['load_level']), self.SLA[2])
        violation = int(
            modified['sinr_dl_db']        < thr['sinr']  or
            modified['delay_ms']          > thr['delay'] or
            modified['packet_loss_ratio'] > thr['loss']  or
            modified['throughput_mbps']   < thr['tput']
        )
        risk_reduction = max(0, self._prev_risk - risk)
        proactive = 0.05 if (action != 3 and not violation) else 0.0
        reward    = -1.0 * violation + 0.3 * risk_reduction + proactive
        self._prev_risk = risk
        obs = self._row_to_obs(modified, risk)
        terminated = False
        truncated  = self._step_idx >= self.episode_len
        info = {
            'sla_violation':   violation,
            'risk_score':      risk,
            'risk_reduction':  risk_reduction,
            'action_name':     self.ACTION_NAMES[action],
            'load_level':      modified['load_level'],
            'sinr_dl_db':      modified['sinr_dl_db'],
            'delay_ms':        modified['delay_ms'],
            'throughput_mbps': modified['throughput_mbps'],
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
