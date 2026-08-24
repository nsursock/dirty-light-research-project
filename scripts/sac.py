import time
import math
import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_map

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scripts.config import cfg
    from scripts.ppo import CSVLogger
except ModuleNotFoundError:
    from config import cfg
    from ppo import CSVLogger


SAC_METRICS_HEADER = [
    "time/total_timesteps", "time/fps", "time/time_elapsed", "train/learning_rate",
    "train/actor_loss", "train/critic_loss", "train/ent_coef", "train/ent_coef_loss",
    "rollout/ep_rew_mean", "rollout/ep_len_mean",
]


class SACActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def __call__(self, obs):
        h = self.net(obs)
        return self.mu(h), mx.clip(self.log_std(h), -20.0, 2.0)

    def sample(self, obs, deterministic=False):
        mu, log_std = self(obs)
        if deterministic:
            return mx.tanh(mu), None
        std = mx.exp(log_std)
        u = mu + mx.random.normal(mu.shape) * std
        a = mx.tanh(u)
        lp = -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
        lp = mx.sum(lp, axis=-1, keepdims=True) - mx.sum(mx.log(mx.clip(1.0 - a ** 2, 1e-6, 1.0)), axis=-1, keepdims=True)
        return a, lp


class SACCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def __call__(self, obs, act):
        x = mx.concatenate([obs, act], axis=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, buffer_size, obs_dim, act_dim):
        self.buffer_size = int(buffer_size)
        self.obs = mx.zeros((self.buffer_size, obs_dim))
        self.act = mx.zeros((self.buffer_size, act_dim))
        self.rew = mx.zeros((self.buffer_size, 1))
        self.next_obs = mx.zeros((self.buffer_size, obs_dim))
        self.done = mx.zeros((self.buffer_size, 1))
        self.pos, self.size = 0, 0

    def add(self, obs, act, rew, next_obs, done):
        if not isinstance(obs, mx.array): obs = mx.array(obs, dtype=mx.float32)
        if not isinstance(act, mx.array): act = mx.array(act, dtype=mx.float32)
        if not isinstance(next_obs, mx.array): next_obs = mx.array(next_obs, dtype=mx.float32)
        if not isinstance(rew, mx.array): rew = mx.array(rew, dtype=mx.float32)
        if not isinstance(done, mx.array): done = mx.array(done, dtype=mx.float32)

        if obs.ndim == 1:
            obs, act, next_obs = obs[None, :], act[None, :], next_obs[None, :]
            rew, done = rew.reshape(1, 1), done.reshape(1, 1)
        else:
            rew = rew.reshape(-1, 1) if rew.ndim == 1 else rew
            done = done.reshape(-1, 1) if done.ndim == 1 else done

        B = obs.shape[0]
        idx = (mx.arange(B) + self.pos) % self.buffer_size
        self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx] = obs, act, rew, next_obs, done
        self.pos = (self.pos + B) % self.buffer_size
        self.size = min(self.size + B, self.buffer_size)

    def sample(self, batch_size):
        idx = mx.random.randint(0, self.size, (batch_size,))
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]


class SAC:
    """SB3-style SAC in pure MLX for low-level continuous trade execution."""
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dim=None,
        learning_rate=None,
        buffer_size=None,
        learning_starts=None,
        batch_size=None,
        tau=None,
        gamma=None,
        train_freq=None,
        gradient_steps=None,
        ent_coef=None,
        target_entropy=None,
        max_grad_norm=None,
        csv_path=None
    ):
        s_cfg = cfg.agents.sac
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.learning_rate = learning_rate if learning_rate is not None else s_cfg.learning_rate
        self.buffer_size = buffer_size if buffer_size is not None else s_cfg.buffer_size
        self.learning_starts = learning_starts if learning_starts is not None else s_cfg.learning_starts
        self.batch_size = batch_size if batch_size is not None else s_cfg.batch_size
        self.tau = tau if tau is not None else s_cfg.tau
        self.gamma = gamma if gamma is not None else s_cfg.gamma
        self.train_freq = train_freq if train_freq is not None else s_cfg.train_freq
        self.gradient_steps = gradient_steps if gradient_steps is not None else s_cfg.gradient_steps
        self.max_grad_norm = max_grad_norm if max_grad_norm is not None else s_cfg.max_grad_norm
        h_dim = hidden_dim if hidden_dim is not None else s_cfg.hidden_dim
        csv_p = csv_path if csv_path is not None else s_cfg.csv_path
        e_coef = ent_coef if ent_coef is not None else s_cfg.ent_coef
        tgt_ent = target_entropy if target_entropy is not None else s_cfg.target_entropy

        self.actor = SACActor(obs_dim, act_dim, hidden_dim=h_dim)
        self.critic = SACCritic(obs_dim, act_dim, hidden_dim=h_dim)
        self.target_critic = SACCritic(obs_dim, act_dim, hidden_dim=h_dim)
        self.target_critic.update(self.critic.parameters())

        self.actor_opt = opt.Adam(learning_rate=self.learning_rate)
        self.critic_opt = opt.Adam(learning_rate=self.learning_rate)

        self.target_entropy = -float(act_dim) if tgt_ent == "auto" else float(tgt_ent)
        self.auto_entropy = e_coef == "auto"
        self.log_ent_coef = mx.array([0.0]) if self.auto_entropy else mx.log(mx.array([float(e_coef)]))

        self.replay_buffer = ReplayBuffer(self.buffer_size, obs_dim, act_dim)
        self.logger = CSVLogger(csv_p, fieldnames=SAC_METRICS_HEADER)
        self.num_timesteps = 0
        self.start_time = time.time()

    @property
    def alpha(self):
        return mx.exp(self.log_ent_coef)

    def predict(self, obs, deterministic=False):
        if not isinstance(obs, mx.array):
            obs = mx.array(obs, dtype=mx.float32)
        is_single = obs.ndim == 1
        if is_single:
            obs = obs[None, :]
        a, _ = self.actor.sample(obs, deterministic=deterministic)
        return a[0] if is_single else a

    def store(self, obs, act, rew, next_obs, done):
        self.replay_buffer.add(obs, act, rew, next_obs, done)
        b = obs.shape[0] if obs.ndim > 1 else 1
        self.num_timesteps += b

    def train_step(self):
        b_obs, b_act, b_rew, b_next_obs, b_done = self.replay_buffer.sample(self.batch_size)

        # 1. Target Q
        next_act, next_lp = self.actor.sample(b_next_obs)
        t_q1, t_q2 = self.target_critic(b_next_obs, next_act)
        target_q = b_rew + self.gamma * (1.0 - b_done) * (mx.minimum(t_q1, t_q2) - self.alpha * next_lp)

        # 2. Critic Update
        def critic_loss(model, obs, act, tgt):
            q1, q2 = model(obs, act)
            return 0.5 * mx.mean((q1 - tgt) ** 2) + 0.5 * mx.mean((q2 - tgt) ** 2)

        c_loss, c_grads = nn.value_and_grad(self.critic, critic_loss)(self.critic, b_obs, b_act, target_q)
        if self.max_grad_norm > 0:
            c_grads, _ = opt.clip_grad_norm(c_grads, self.max_grad_norm)
        self.critic_opt.update(self.critic, c_grads)

        # 3. Actor Update
        def actor_loss(model, critic, obs, alpha):
            a, lp = model.sample(obs)
            q1, q2 = critic(obs, a)
            return mx.mean(alpha * lp - mx.minimum(q1, q2)), lp

        (a_loss, lp), a_grads = nn.value_and_grad(self.actor, actor_loss)(self.actor, self.critic, b_obs, self.alpha)
        if self.max_grad_norm > 0:
            a_grads, _ = opt.clip_grad_norm(a_grads, self.max_grad_norm)
        self.actor_opt.update(self.actor, a_grads)

        # 4. Entropy Temperature Update
        ent_loss_val = 0.0
        if self.auto_entropy:
            lp_stop = mx.stop_gradient(lp)
            alpha_grad = -mx.mean(lp_stop + self.target_entropy)
            self.log_ent_coef = mx.clip(self.log_ent_coef - self.learning_rate * alpha_grad, -5.0, 5.0)
            ent_loss_val = float((-self.log_ent_coef * (lp_stop + self.target_entropy)).mean().item())

        # 5. Target Network Polyak Update
        new_target = tree_map(
            lambda tp, p: (1.0 - self.tau) * tp + self.tau * p,
            self.target_critic.parameters(),
            self.critic.parameters()
        )
        self.target_critic.update(new_target)

        mx.eval(self.actor.state, self.critic.state, self.target_critic.state, self.log_ent_coef)
        return a_loss.item(), c_loss.item(), self.alpha.item(), ent_loss_val

    def train(self, gradient_steps=None, ep_rew_mean=0.0, ep_len_mean=0.0):
        if self.replay_buffer.size < self.learning_starts:
            return {}

        steps = gradient_steps or self.gradient_steps
        a_losses, c_losses, alphas, ent_losses = [], [], [], []

        for _ in range(steps):
            al, cl, alpha_v, el = self.train_step()
            a_losses.append(al)
            c_losses.append(cl)
            alphas.append(alpha_v)
            ent_losses.append(el)

        elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / max(elapsed, 1e-6))
        metrics = {
            "time/total_timesteps": self.num_timesteps,
            "time/fps": fps,
            "time/time_elapsed": elapsed,
            "train/learning_rate": self.learning_rate,
            "train/actor_loss": float(sum(a_losses) / len(a_losses)),
            "train/critic_loss": float(sum(c_losses) / len(c_losses)),
            "train/ent_coef": float(sum(alphas) / len(alphas)),
            "train/ent_coef_loss": float(sum(ent_losses) / len(ent_losses)),
            "rollout/ep_rew_mean": float(ep_rew_mean),
            "rollout/ep_len_mean": float(ep_len_mean),
        }
        self.logger.log(metrics)
        return metrics

    def save(self, path):
        self.actor.save_weights(f"{path}_actor.npz")
        self.critic.save_weights(f"{path}_critic.npz")

    def load(self, path):
        self.actor.load_weights(f"{path}_actor.npz")
        self.critic.load_weights(f"{path}_critic.npz")
