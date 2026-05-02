# Module M1 — Reinforcement Learning Agent for 5G QoS Optimization
## QoSBuddy Project — PFE Report Section

**Team:** PingWin — ESPRIT  
**Module:** M1 — RL Agent (DSO3.1)  
**Algorithm:** Proximal Policy Optimization (PPO)  
**Environment:** QoSNetworkEnv v3 | **Dataset:** lena_dataset_cleaned.csv

---

## 1. Context and Objective

Dynamic 5G networks experience rapid fluctuations in traffic load, channel quality, and user mobility that render static network management strategies insufficient. Rule-based controllers fail to adapt to unseen conditions and require continuous manual re-tuning. Reinforcement Learning (RL) offers an alternative: an agent that **learns a decision policy through experience**, mapping observed network states to corrective actions that proactively minimize SLA violations.

**Objective:** Train a PPO-based RL agent on real NS-3 LENA simulation data to autonomously optimize 5G QoS by selecting network management actions (rerouting, throttling, prioritization) that reduce SLA violations and minimize network risk scores in real time.

**Architectural integration:**

```
Real 5G Network
       │ KPI stream
       ▼
Digital Twin (QoSNetworkEnv)  ◄──── safe simulation sandbox
       │ state (8 features)
       ▼
   PPO Agent (M1)
       │ action (4 discrete)
       ▼
Network Action Executor  ──► Dashboard (M4) / Self-Healing Controller (M6)
```

---

## 2. Environment Design

### 2.1 Dataset

| Property | Value |
|---|---|
| **Source** | NS-3 LENA 5G network simulator |
| **File** | `lena_dataset_cleaned.csv` |
| **Size** | 18,580 rows (after cleaning) |
| **Coverage** | 5 User Equipment (UEs) |
| **Columns** | 16 features (9 int, 7 float) |

**Data cleaning steps:**
```python
df = df.dropna()                                   # Remove nulls
df = df[~((df.throughput_mbps == 0) &
           (df.delay_ms == 0))]                    # Remove degenerate rows
# Range validation:
df = df[(df.sinr_dl_db.between(-50, 90)) &
        (df.throughput_mbps.between(0, 100))  &
        (df.delay_ms.between(0, 5000))         &
        (df.packet_loss_ratio.between(0, 1))   &
        (df.prb_utilization.between(0, 100))   &
        (df.cqi.between(1, 15))                &
        (df.load_level.between(1, 3))]
df = df.sort_values(['ue_id', 'timestamp']).reset_index(drop=True)
```

### 2.2 Observation Space

The agent observes an 8-dimensional vector at each timestep:

```
Box(low=[-50, 0, 0, 0, 0, 0, 0, 0],
    high=[90, 100, 5000, 2000, 1, 100, 1000, 10],
    shape=(8,), dtype=float32)
```

| Index | Feature | Unit | Range | Description |
|---|---|---|---|---|
| 0 | `sinr_dl_db` | dB | [−50, 90] | Downlink SINR |
| 1 | `throughput_mbps` | Mbps | [0, 100] | Downlink throughput |
| 2 | `delay_ms` | ms | [0, 5000] | End-to-end delay |
| 3 | `jitter_ms` | ms | [0, 2000] | Delay variation |
| 4 | `packet_loss_ratio` | — | [0, 1] | Fraction of lost packets |
| 5 | `prb_utilization` | % | [0, 100] | Physical Resource Block usage |
| 6 | `retransmissions` | count | [0, 1000] | L1 retransmission count |
| 7 | `sla_risk_score` | score | [0, 10] | Composite SLA risk (0=safe, 10=critical) |

**SLA Risk Score Computation:**
```python
def _sla_risk_score(row, load_level):
    thr = SLA.get(load_level, SLA[2])
    score = 0.0
    if row['sinr_dl_db']        < thr['sinr']:  score += 2.5
    if row['delay_ms']          > thr['delay']: score += 2.5
    if row['packet_loss_ratio'] > thr['loss']:  score += 2.5
    if row['throughput_mbps']   < thr['tput']:  score += 2.5
    return min(score, 10.0)
```

### 2.3 Action Space

The agent selects from **4 discrete network management actions**:

```
Discrete(4)
```

| ID | Action | SINR Δ | Delay ×| Loss ×| Tput × | Purpose |
|---|---|---|---|---|---|---|
| 0 | `reroute` | +4.0 dB | ×0.75 | ×0.70 | ×1.10 | Redirect traffic to better path/cell |
| 1 | `throttle` | — | ×0.90 | ×0.50 | ×0.65 | Reduce congestion on overloaded path |
| 2 | `prioritize` | +6.0 dB | ×0.65 | ×0.45 | ×1.30 | Mark as high-priority/expedited forwarding |
| 3 | `no_action` | −1.5 dB | ×1.10 | ×1.05 | ×0.95 | Baseline; passive monitoring |

Each action applies stochastic noise ~N(0, σ) to simulate real-world variability.

### 2.4 SLA Thresholds (Load-Dependent)

Thresholds vary by network load level to reflect realistic operator contracts:

| Load Level | SINR (dB) | Delay (ms) | Loss Ratio | Tput (Mbps) |
|---|---|---|---|---|
| 1 — Low | ≥ −25.0 | ≤ 300 | ≤ 0.80 | ≥ 0.01 |
| 2 — Medium | ≥ −18.0 | ≤ 200 | ≤ 0.60 | ≥ 0.05 |
| 3 — High | ≥ −10.0 | ≤ 120 | ≤ 0.30 | ≥ 0.20 |

An **SLA violation** is triggered whenever ANY of the four KPI constraints is breached for the current load level.

### 2.5 Reward Function

The reward function encodes three objectives:

```python
violation      = int(sinr < thr['sinr'] or delay > thr['delay']
                     or loss > thr['loss'] or tput < thr['tput'])
risk_reduction = max(0, prev_risk - current_risk)
proactive      = 0.05 if (action != 3 and not violation) else 0.0

reward = -1.0 * violation + 0.3 * risk_reduction + proactive
```

| Component | Weight | Purpose |
|---|---|---|
| SLA violation penalty | −1.0 | Hard penalty for breaching any threshold |
| Risk reduction bonus | +0.3 | Reward for proactively lowering risk score |
| Proactive action bonus | +0.05 | Encourage intervention before violations |

**Reward range:** approximately [−1.0, +1.0] per step; cumulative over 200 steps per episode.

### 2.6 Episode Structure

| Parameter | Value |
|---|---|
| Episode length | 200 steps |
| Termination condition | Truncated at step 200 |
| Data sampling | Sequential cycling through dataset (random start per episode) |
| Number of UEs | 5 (in dataset) |

---

## 3. Algorithm — Proximal Policy Optimization (PPO)

### 3.1 Motivation

PPO was selected for the following reasons:
- **On-policy stability:** Clip-ratio constraint prevents destructive policy updates.
- **Sample efficiency:** Suitable for moderate environment sizes (18,580 rows, 200-step episodes).
- **Industry standard:** Widely deployed for continuous/discrete control in telecom optimization literature.
- **Ease of integration:** Stable-Baselines3 provides tested, production-grade implementation.

### 3.2 Hyperparameters

| Parameter | Value | Purpose |
|---|---|---|
| `learning_rate` | **3e-4** | Conservative Adam step size |
| `n_steps` | **512** | Rollout buffer length per environment |
| `batch_size` | **128** | Mini-batch size for gradient updates |
| `n_epochs` | **10** | Gradient update passes per rollout |
| `gamma` (γ) | **0.99** | Discount factor — values long-term rewards |
| `gae_lambda` (λ) | **0.95** | GAE bias-variance trade-off |
| `clip_range` | **0.2** | PPO clipping coefficient |
| `ent_coef` | **0.01** | Entropy bonus — promotes exploration |
| `vf_coef` | **0.5** | Value function loss coefficient |
| `max_grad_norm` | **0.5** | Gradient clipping threshold |
| `seed` | **42** | Reproducibility |

### 3.3 Policy Network Architecture

The policy (actor) and value (critic) networks share a 2-layer MLP trunk:

```
Observation (8 features)
        │
   Dense(128, ReLU)     ← 8×128 + 128 = 1,152 params
        │
   Dense(128, ReLU)     ← 128×128 + 128 = 16,512 params
        │
   ┌────┴────────────────────┐
   │ Policy Head             │  Value Head
   │ Dense(4, Linear)        │  Dense(1, Linear)
   │ → action logits         │  → V(s)
   └─────────────────────────┘

Total trainable parameters: 35,973
```

```python
policy_kwargs = dict(net_arch=[128, 128])
```

### 3.4 Training Setup

```python
model = PPO(
    policy         = 'MlpPolicy',
    env            = vec_env,                    # 4 parallel envs (VecEnv)
    learning_rate  = 3e-4,
    n_steps        = 512,
    batch_size     = 128,
    n_epochs       = 10,
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.01,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,
    policy_kwargs  = dict(net_arch=[128, 128]),
    verbose        = 0,
    seed           = 42,
    device         = 'cpu',
)

model.learn(
    total_timesteps = 200_000,
    callback        = CallbackList([eval_cb, metrics_cb]),
    progress_bar    = False,
)
```

**Training infrastructure:**

| Setting | Value |
|---|---|
| Parallel environments | 4 (VecEnv) |
| Total timesteps | 200,000 |
| Gradient updates | ~390 (200,000 / (512 × 4) ≈ 97 batches × 4 envs) |
| Evaluation frequency | Every 5,000 timesteps (10 deterministic episodes) |
| Best model checkpoint | Saved on highest mean reward at evaluation |
| Framework | Stable-Baselines3 v2.7.1 |
| Device | CPU |

---

## 4. Baselines

Two baseline policies were implemented to contextualize the PPO agent's performance:

### 4.1 Random Policy
Samples uniformly from {0, 1, 2, 3} at each step. Represents an untrained agent with no signal.

```python
action = env.action_space.sample()
```

### 4.2 Rule-Based Heuristic
Hand-crafted decision tree based on observed KPI thresholds:

```python
def rule_based_action(obs):
    sinr, tput, delay, jitter, pkt_loss, prb, retx, risk = obs
    if risk >= 5.0 or pkt_loss > 0.3:
        return 2   # prioritize
    if delay > 100:
        return 0   # reroute
    if tput < 1.0:
        return 1   # throttle
    return 3       # no_action
```

---

## 5. Training Dynamics

### 5.1 Episode Reward Evolution (~250 training episodes)

| Training Phase | Episodes | Mean Reward | Trend |
|---|---|---|---|
| Exploration (random-like) | 1–20 | 2 – 6 | Flat / noisy |
| Learning phase | 21–150 | 6 – 8 | Steadily increasing |
| Convergence | 151–250 | 9 – 10 | Plateau (asymptotic) |

The smoothed reward curve (window=30) shows a clear monotonic increase from episode 1 to convergence at ~episode 150–180, confirming successful policy improvement.

### 5.2 SLA Violations Per Episode

| Training Phase | Violations/Episode | Violation Rate |
|---|---|---|
| Initial (~ep 1–20) | 9–12 | 4.5–6.0% |
| Mid (~ep 50–150) | 5–8 | 2.5–4.0% |
| Converged (~ep 180–250) | 3–5 | **1.5–2.5%** |

The agent learns to anticipate and prevent SLA breaches, reducing violations by approximately 50% compared to early training.

### 5.3 Mean Risk Score per Episode

| Training Phase | Mean Risk/Step |
|---|---|
| Initial | 0.10 – 0.16 |
| Mid | 0.07 – 0.10 |
| Converged | **0.04 – 0.06** |

Risk score reduction demonstrates **proactive** rather than reactive behavior — the agent acts before violations occur, not just after them.

---

## 6. Evaluation Results

Evaluation protocol: **20 deterministic episodes** (200 steps each) loaded from the best model checkpoint (`ppo_qos_agent_best.zip`).

### 6.1 Benchmark Summary

| Metric | Random Policy | Rule-Based | **PPO (Best)** |
|---|---|---|---|
| **Mean Reward / Episode** | 5.3675 ± 1.63 | −1.1125 ± 1.70 | **9.040 ± 1.35** |
| **SLA Violation Rate** | 4.1% (8.2/ep) | 5.6% (11.2/ep) | **2.0% (4.0/ep)** |
| **Mean Risk Score/Step** | 0.1106 | 0.1637 | **0.0556** |

### 6.2 Improvement Over Baselines

| Improvement | PPO vs Random | PPO vs Rule-Based |
|---|---|---|
| Reward increase | +3.67 (+68.4%) | +10.15 (absolute) |
| Violation reduction | −2.1 pp (4.1% → 2.0%) | −3.6 pp (5.6% → 2.0%) |
| Risk score reduction | −0.055 (−49.5%) | −0.108 (−65.9%) |

> **Key finding:** The rule-based policy performs *worse* than random due to over-conservative thresholds (`risk ≥ 5.0`, `delay > 100 ms`) that trigger suboptimal actions too late or too aggressively. PPO learns nuanced, context-dependent behavior that outperforms both baselines.

### 6.3 Learned Action Distribution (50 evaluation episodes, 10,000 steps)

| Action | Count | Proportion | Interpretation |
|---|---|---|---|
| `throttle` (1) | 6,559 | **65.6%** | Default congestion control strategy |
| `prioritize` (2) | 2,530 | **25.3%** | Emergency high-risk mitigation |
| `reroute` (0) | 911 | **9.1%** | Selective path optimization |
| `no_action` (3) | ~0 | ~0% | Agent rarely stays passive |

The agent has learned a meaningful specialization: **throttle** handles routine congestion, **prioritize** is reserved for high-risk states, and **reroute** addresses severe delay cases. The near-absence of `no_action` confirms the agent is consistently proactive.

### 6.4 Per-Load-Level Performance

| Load Level | PPO Violation Rate | Challenge |
|---|---|---|
| 1 — Low | ~1.5% | Low — agent rarely needed |
| 2 — Medium | ~2.0–2.5% | Moderate — standard conditions |
| 3 — High | ~2.5–3.5% | High — agent must be most aggressive |

The agent adapts its strategy by load level, using more `prioritize` actions under high load (25–30%) and more `throttle` under low load (70–75%).

---

## 7. Generated Artifacts

| File | Description |
|---|---|
| `ppo_qos_agent_best.zip` | Best PPO checkpoint (highest eval reward) |
| `ppo_qos_agent_v1.zip` | Final PPO checkpoint (end of training) |
| `QoSNetworkEnv.py` | Custom Gymnasium environment (QoSNetworkEnv v3) |
| `inference.py` | Standalone inference script (load agent, run episodes) |
| `rl_agent_config.json` | Training config + benchmark results |
| `ns3_environment_config.json` | Environment config (observation/action/reward specs) |
| `outputs/training_curves.png` | Reward, violations, risk score over training episodes |
| `outputs/agent_comparison.png` | PPO vs Random vs Rule-Based bar charts |
| `outputs/policy_analysis.png` | Action distribution and per-load analysis |
| `outputs/sla_heatmap.png` | SLA violation heatmap by load level and action |

---

## 8. Limitations and Future Work

### 8.1 Current Limitations

| Limitation | Description |
|---|---|
| **Simulation gap** | Action effects are simulation-derived (NS-3 LENA), not validated on live 5G hardware. Real networks may have non-Gaussian action variability. |
| **Discrete action space** | Only 4 coarse actions; fine-grained control (e.g., specific PRB allocation, beam steering angles) is not representable. |
| **Stateless observation** | Agent sees only the current timestep; no temporal memory (past 3–5 steps could improve anticipation). |
| **Fixed SLA thresholds** | Hard-coded per load level; real networks use dynamic, per-UE SLA contracts. |
| **Small-scale evaluation** | Evaluated on 5 UEs; scalability to 50+ UEs under dense urban scenarios is unvalidated. |
| **No safety guarantees** | No formal worst-case violation bound or fallback mechanism. |

### 8.2 Recommended Extensions

1. **Recurrent policy (PPO + LSTM):** Add temporal context to the observation, enabling the agent to detect trends and anticipate deterioration.
2. **Multi-agent coordination:** Extend to multiple co-located agents managing different network slices simultaneously.
3. **Hybrid guardrail architecture:** Keep the rule-based layer as a safety filter; use PPO for fine-grained optimization within the safe region.
4. **Real-network testbed validation:** Deploy on an OpenAirInterface or srsRAN testbed for end-to-end validation.
5. **Continuous action space:** Reformulate with SAC or TD3 to enable fine-grained control parameters.

---

## 9. Conclusion

The PPO-based RL agent successfully learns a QoS optimization policy on real NS-3 LENA data. Across 20 evaluation episodes, the agent achieves:

- **Mean reward of 9.040** (+68.4% over random baseline)
- **SLA violation rate of 2.0%** (vs 4.1% random, 5.6% rule-based)
- **Mean risk score of 0.056** (vs 0.111 random, 0.164 rule-based)

The learned policy is **proactive** (near-zero `no_action` selections), **adaptive** (load-level-aware action distribution), and **generalizable** (consistent performance across 20 test episodes with ±1.35 reward std).

This module represents the core decision engine of the QoSBuddy architecture, feeding optimized network actions to the Dashboard (Module M4) and Self-Healing Controller (Module M6) for closed-loop 5G QoS management.

---

*Section prepared for QoSBuddy PFE Final Report — PingWin Team, ESPRIT.*
