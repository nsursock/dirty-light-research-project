import os
import csv
import time
import math
import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scripts.config import cfg
except ModuleNotFoundError:
    from config import cfg


PPO_METRICS_HEADER = [
    "time/total_timesteps", "time/fps", "time/time_elapsed", "train/learning_rate",
    "train/policy_loss", "train/value_loss", "train/entropy_loss", "train/approx_kl",
    "train/clip_fraction", "train/explained_variance", "rollout/ep_rew_mean", "rollout/ep_len_mean",
]


class CSVLogger:
    def __init__(self, filename, fieldnames=None):
        self.filename = filename
        self.fieldnames = fieldnames
        if self.filename:
            os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)
            if self.fieldnames and (not os.path.exists(self.filename) or os.path.getsize(self.filename) == 0):
                with open(self.filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()

    def log(self, row: dict):
        if not self.filename:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)
        file_exists = os.path.exists(self.filename) and os.path.getsize(self.filename) > 0
        fields = self.fieldnames or list(row.keys())
        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = mx.zeros((act_dim,))

    def __call__(self, obs, action=None):
        mu = self.actor(obs)
        std = mx.exp(self.log_std)
        if action is None:
            action = mu + mx.random.normal(mu.shape) * std
        log_prob = -0.5 * (((action - mu) ** 2) / (std ** 2 + 1e-8) + 2.0 * self.log_std + math.log(2.0 * math.pi))
        log_prob = mx.sum(log_prob, axis=-1)
        entropy = mx.sum(0.5 + 0.5 * math.log(2.0 * math.pi) + self.log_std)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


class PPO:
    """SB3-style PPO in pure MLX for high-level macro portfolio allocation."""
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=None,
        learning_rate=None,
        n_steps=None,
        batch_size=None,
        n_epochs=None,
        gamma=None,
        gae_lambda=None,
        clip_range=None,
        ent_coef=None,
        vf_coef=None,
        max_grad_norm=None,
        csv_path=None
    ):
        p_cfg = cfg.agents.ppo
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.learning_rate = learning_rate if learning_rate is not None else p_cfg.learning_rate
        self.n_steps = n_steps if n_steps is not None else p_cfg.n_steps
        self.batch_size = batch_size if batch_size is not None else p_cfg.batch_size
        self.n_epochs = n_epochs if n_epochs is not None else p_cfg.n_epochs
        self.gamma = gamma if gamma is not None else p_cfg.gamma
        self.gae_lambda = gae_lambda if gae_lambda is not None else p_cfg.gae_lambda
        self.clip_range = clip_range if clip_range is not None else p_cfg.clip_range
        self.ent_coef = ent_coef if ent_coef is not None else p_cfg.ent_coef
        self.vf_coef = vf_coef if vf_coef is not None else p_cfg.vf_coef
        self.max_grad_norm = max_grad_norm if max_grad_norm is not None else p_cfg.max_grad_norm
        h_dim = hidden_dim if hidden_dim is not None else p_cfg.hidden_dim
        csv_p = csv_path if csv_path is not None else p_cfg.csv_path

        self.policy = PPOActorCritic(obs_dim, act_dim, hidden_dim=h_dim)
        self.optimizer = opt.Adam(learning_rate=self.learning_rate)
        self.logger = CSVLogger(csv_p, fieldnames=PPO_METRICS_HEADER)
        self.num_timesteps = 0
        self.start_time = time.time()

    def predict(self, obs, deterministic=False):
        if not isinstance(obs, mx.array):
            obs = mx.array(obs, dtype=mx.float32)
        is_single = obs.ndim == 1
        if is_single:
            obs = obs[None, :]
        mu = self.policy.actor(obs)
        if deterministic:
            return mu[0] if is_single else mu
        std = mx.exp(self.policy.log_std)
        act = mu + mx.random.normal(mu.shape) * std
        return act[0] if is_single else act

    def compute_gae(self, rewards, values, dones, next_val):
        if rewards.ndim == 1:
            rewards = rewards[:, None]
            values = values[:, None]
            dones = dones[:, None]
            val_next = next_val.reshape(1) if isinstance(next_val, mx.array) else mx.array([float(next_val)])
        else:
            val_next = next_val if isinstance(next_val, mx.array) else mx.full((rewards.shape[1],), float(next_val))
            if val_next.ndim == 0:
                val_next = mx.full((rewards.shape[1],), float(val_next.item()))
            elif val_next.ndim == 2:
                val_next = val_next.squeeze(-1)

        T, B = rewards.shape
        advs = [None] * T
        gae = mx.zeros((B,), dtype=mx.float32)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * val_next * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advs[t] = gae
            val_next = values[t]
        advs = mx.stack(advs, axis=0)
        returns = advs + values
        return advs, returns

    def train_on_rollout(self, obs, actions, rewards, dones, values, log_probs, next_val, ep_rew_mean=0.0, ep_len_mean=0.0):
        advs, returns = self.compute_gae(rewards, values, dones, next_val)
        flat_obs = obs.reshape(-1, self.obs_dim)
        flat_act = actions.reshape(-1, self.act_dim)
        flat_advs = advs.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_vals = values.reshape(-1)
        flat_lps = log_probs.reshape(-1)

        self.num_timesteps += flat_obs.shape[0]
        advs_norm = (flat_advs - mx.mean(flat_advs)) / (mx.std(flat_advs) + 1e-8)

        N = flat_obs.shape[0]
        p_losses, v_losses, e_losses, kls, clip_fracs = [], [], [], [], []

        def ppo_loss(model, b_obs, b_act, b_old_lp, b_adv, b_ret):
            _, new_lp, entropy, vals = model(b_obs, b_act)
            ratio = mx.exp(new_lp - b_old_lp)
            p_loss = -mx.mean(mx.minimum(ratio * b_adv, mx.clip(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * b_adv))
            v_loss = 0.5 * mx.mean((vals - b_ret) ** 2)
            tot_loss = p_loss + self.vf_coef * v_loss - self.ent_coef * entropy
            approx_kl = mx.mean(b_old_lp - new_lp)
            clip_frac = mx.mean((mx.abs(ratio - 1.0) > self.clip_range).astype(mx.float32))
            return tot_loss, (p_loss, v_loss, entropy, approx_kl, clip_frac)

        loss_and_grad_fn = nn.value_and_grad(self.policy, ppo_loss)
        bs = min(self.batch_size, N)

        for _ in range(self.n_epochs):
            indices = mx.random.permutation(N)
            for start in range(0, N, bs):
                idx = indices[start:start + bs]
                (_, (p_loss, v_loss, ent, kl, clip_frac)), grads = loss_and_grad_fn(
                    self.policy, flat_obs[idx], flat_act[idx], flat_lps[idx], advs_norm[idx], flat_returns[idx]
                )
                if self.max_grad_norm > 0:
                    grads, _ = opt.clip_grad_norm(grads, self.max_grad_norm)
                self.optimizer.update(self.policy, grads)
                mx.eval(self.policy.state, self.optimizer.state)

                p_losses.append(p_loss.item())
                v_losses.append(v_loss.item())
                e_losses.append(ent.item())
                kls.append(kl.item())
                clip_fracs.append(clip_frac.item())

        elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / max(elapsed, 1e-6))
        exp_var = float((1.0 - mx.var(flat_returns - flat_vals) / (mx.var(flat_returns) + 1e-8)).item())
        metrics = {
            "time/total_timesteps": self.num_timesteps,
            "time/fps": fps,
            "time/time_elapsed": elapsed,
            "train/learning_rate": self.learning_rate,
            "train/policy_loss": float(sum(p_losses) / len(p_losses)),
            "train/value_loss": float(sum(v_losses) / len(v_losses)),
            "train/entropy_loss": float(sum(e_losses) / len(e_losses)),
            "train/approx_kl": float(sum(kls) / len(kls)),
            "train/clip_fraction": float(sum(clip_fracs) / len(clip_fracs)),
            "train/explained_variance": exp_var,
            "rollout/ep_rew_mean": float(ep_rew_mean),
            "rollout/ep_len_mean": float(ep_len_mean),
        }
        self.logger.log(metrics)
        return metrics
        self.logger.log(metrics)
        return metrics

    def save(self, path):
        self.policy.save_weights(path)

    def load(self, path):
        self.policy.load_weights(path)
