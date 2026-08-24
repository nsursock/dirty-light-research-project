import os
import sys
import csv
import math
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import mlx.core as mx
from scripts.agents import PPO, SAC

# -----------------------------------------------------------------------------
# Pure MLX Toy Environments (Gymnasium-v1 Specifications)
# -----------------------------------------------------------------------------
class MLXCartPole:
    """Gymnasium CartPole-v1 environment in pure MLX (max 500 steps, solve >= 475)."""
    def __init__(self, max_steps=500):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.theta_threshold_radians = 12.0 * 2.0 * math.pi / 360.0
        self.x_threshold = 2.4
        self.max_steps = max_steps
        self.step_cnt = 0
        self.state = None

    def reset(self):
        self.state = mx.random.uniform(-0.05, 0.05, (4,))
        self.step_cnt = 0
        return self.state

    def step(self, action):
        act_val = action[0] if hasattr(action, "__getitem__") else action
        force = mx.clip(act_val, -1.0, 1.0) * self.force_mag
        x, x_dot, theta, theta_dot = self.state[0], self.state[1], self.state[2], self.state[3]
        costheta, sintheta = mx.cos(theta), mx.sin(theta)

        temp = (force + self.polemass_length * (theta_dot ** 2) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * (costheta ** 2) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = mx.stack([x, x_dot, theta, theta_dot])
        self.step_cnt += 1

        terminated = bool((mx.abs(x) > self.x_threshold).item() or (mx.abs(theta) > self.theta_threshold_radians).item())
        truncated = self.step_cnt >= self.max_steps
        done = terminated or truncated
        reward = 1.0 if not terminated else 0.0
        return self.state, reward, terminated, truncated

class MLXPendulum:
    """Gymnasium Pendulum-v1 environment in pure MLX (max 200 steps, solve >= -200)."""
    def __init__(self, max_steps=200):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g, self.m, self.l = 10.0, 1.0, 1.0
        self.max_steps = max_steps
        self.step_cnt = 0
        self.state = None

    def reset(self):
        high = mx.array([math.pi, 1.0])
        self.state = mx.random.uniform(-1.0, 1.0, (2,)) * high
        self.step_cnt = 0
        return self._get_obs()

    def _get_obs(self):
        th, thdot = self.state[0], self.state[1]
        return mx.stack([mx.cos(th), mx.sin(th), thdot / self.max_speed])

    def step(self, u):
        u_val = u[0] if hasattr(u, "__getitem__") else u
        u_clamped = mx.clip(u_val, -1.0, 1.0) * self.max_torque
        th, thdot = self.state[0], self.state[1]
        costs = ((th + math.pi) % (2.0 * math.pi) - math.pi) ** 2 + 0.1 * (thdot ** 2) + 0.001 * (u_clamped ** 2)
        newthdot = thdot + (3.0 * self.g / (2.0 * self.l) * mx.sin(th) + 3.0 / (self.m * (self.l ** 2)) * u_clamped) * self.dt
        newthdot = mx.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        self.state = mx.stack([newth, newthdot])
        self.step_cnt += 1
        truncated = self.step_cnt >= self.max_steps
        return self._get_obs(), -float(costs.item()), False, truncated

# -----------------------------------------------------------------------------
# Metric Helpers: Noise & Trend Analysis
# -----------------------------------------------------------------------------
def compute_noise_and_trend(values):
    """Calculates standard deviation (noise) and OLS slope (trend)."""
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    mean_v = sum(values) / n
    variance = sum((v - mean_v) ** 2 for v in values) / (n - 1)
    noise_std = math.sqrt(variance)

    mean_x = (n - 1) / 2.0
    num = sum((i - mean_x) * (values[i] - mean_v) for i in range(n))
    den = sum((i - mean_x) ** 2 for i in range(n))
    trend_slope = num / (den + 1e-8)
    return noise_std, trend_slope

def evaluate_agent(agent, env, num_episodes=10):
    total_rews = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            act = agent.predict(obs, deterministic=True)
            obs, r, term, trunc = env.step(act)
            ep_rew += r
            done = term or trunc
        total_rews.append(ep_rew)
    return sum(total_rews) / len(total_rews)

# -----------------------------------------------------------------------------
# PyTest V1 Benchmark Tests
# -----------------------------------------------------------------------------
def test_ppo_cartpole_v1(tmp_path):
    """Verifies PPO solves CartPole-v1 criteria (eval score >= 475.0/500.0)."""
    mx.random.seed(42)
    csv_file = str(tmp_path / "ppo_cartpole.csv")
    env = MLXCartPole(max_steps=500)
    agent = PPO(
        obs_dim=4, act_dim=1, n_steps=256, batch_size=64,
        n_epochs=10, learning_rate=3e-3, csv_path=csv_file
    )

    obs = env.reset()
    ep_rewards = []
    cur_rew = 0.0
    o_b, a_b, r_b, d_b, v_b, lp_b = [], [], [], [], [], []

    for step in range(12000):
        act, lp, _, val = agent.policy(obs[None, :])
        next_obs, rew, term, trunc = env.step(act[0])
        done = term or trunc
        cur_rew += rew

        o_b.append(obs)
        a_b.append(act[0])
        r_b.append(mx.array(rew))
        d_b.append(mx.array(float(term)))
        v_b.append(val[0])
        lp_b.append(lp[0])

        if len(o_b) >= agent.n_steps:
            _, _, _, next_v = agent.policy(next_obs[None, :])
            agent.train_on_rollout(
                mx.stack(o_b), mx.stack(a_b), mx.stack(r_b), mx.stack(d_b),
                mx.stack(v_b), mx.stack(lp_b), next_v[0],
                ep_rew_mean=sum(ep_rewards[-5:]) / max(len(ep_rewards[-5:]), 1)
            )
            o_b, a_b, r_b, d_b, v_b, lp_b = [], [], [], [], [], []

        if done:
            ep_rewards.append(cur_rew)
            cur_rew = 0.0
            obs = env.reset()
        else:
            obs = next_obs

    noise, trend = compute_noise_and_trend(ep_rewards)
    eval_score = evaluate_agent(agent, env, num_episodes=10)

    print(f"\n[CartPole-v1 PPO] Episodes: {len(ep_rewards)}, Noise: {noise:.2f}, Trend: {trend:.2f}")
    print(f"[CartPole-v1 PPO] Final Evaluation (10 eps): {eval_score:.1f} / 500.0 (Gym V1 Criterion >= 475.0)")

    assert os.path.exists(csv_file), "PPO CSV log must exist"
    assert noise > 0.0, "Metrics must display stochastic exploration noise"
    assert trend > 0.0, "Episode returns must display an upward learning trend"
    assert eval_score >= 475.0, f"PPO failed CartPole-v1 solving criterion (got {eval_score:.1f} < 475.0)"

def test_sac_pendulum_v1(tmp_path):
    """Verifies SAC solves Pendulum-v1 criteria (eval score >= -200.0/200 steps)."""
    mx.random.seed(42)
    csv_file = str(tmp_path / "sac_pendulum.csv")
    env = MLXPendulum(max_steps=200)
    agent = SAC(
        obs_dim=3, act_dim=1, learning_starts=200, batch_size=128,
        learning_rate=3e-3, tau=0.01, csv_path=csv_file
    )

    obs = env.reset()
    critic_losses = []

    for step in range(5000):
        if step < agent.learning_starts:
            act = mx.random.uniform(-1.0, 1.0, (1,))
        else:
            act = agent.predict(obs)

        next_obs, rew, term, trunc = env.step(act)
        agent.store(obs, act, rew / 10.0, next_obs, float(term))
        metrics = agent.train(gradient_steps=1)
        if metrics and "train/critic_loss" in metrics:
            critic_losses.append(metrics["train/critic_loss"])

        if trunc:
            obs = env.reset()
        else:
            obs = next_obs

    noise, trend = compute_noise_and_trend(critic_losses)
    eval_score = evaluate_agent(agent, env, num_episodes=10)

    print(f"\n[Pendulum-v1 SAC] Steps: 5000, Critic Noise: {noise:.4f}, Trend: {trend:.6f}")
    print(f"[Pendulum-v1 SAC] Final Evaluation (10 eps): {eval_score:.1f} (Gym V1 Criterion >= -200.0)")

    assert os.path.exists(csv_file), "SAC CSV log must exist"
    assert noise > 0.0, "SAC critic losses must exhibit minibatch stochastic noise"
    assert eval_score >= -200.0, f"SAC failed Pendulum-v1 solving criterion (got {eval_score:.1f} < -200.0)"

def test_save_and_load(tmp_path):
    """Verifies parameter persistence and restoration."""
    ppo_path = str(tmp_path / "ppo_test.npz")
    sac_path = str(tmp_path / "sac_test")

    ppo = PPO(obs_dim=4, act_dim=2)
    obs = mx.ones((1, 4))
    act1 = ppo.predict(obs, deterministic=True)
    ppo.save(ppo_path)

    ppo2 = PPO(obs_dim=4, act_dim=2)
    ppo2.load(ppo_path)
    act2 = ppo2.predict(obs, deterministic=True)
    assert mx.allclose(act1, act2).item(), "PPO loaded weights must match saved weights"

    sac = SAC(obs_dim=3, act_dim=1)
    obs_s = mx.ones((1, 3))
    s_act1 = sac.predict(obs_s, deterministic=True)
    sac.save(sac_path)

    sac2 = SAC(obs_dim=3, act_dim=1)
    sac2.load(sac_path)
    s_act2 = sac2.predict(obs_s, deterministic=True)
    assert mx.allclose(s_act1, s_act2).item(), "SAC loaded weights must match saved weights"
