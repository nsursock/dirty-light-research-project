import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scripts.config import cfg
    from scripts.ppo import PPO, PPOActorCritic, CSVLogger
    from scripts.sac import SAC, SACActor, SACCritic, ReplayBuffer
    from scripts.data import generate_multi_tf_data, get_timeframe_ratio, FEATURE_NAMES, BINANCE_TIMEFRAMES
    from scripts.report import create_run_dir, TradeHistoryLogger
except ModuleNotFoundError:
    from config import cfg
    from ppo import PPO, PPOActorCritic, CSVLogger
    from sac import SAC, SACActor, SACCritic, ReplayBuffer
    from data import generate_multi_tf_data, get_timeframe_ratio, FEATURE_NAMES, BINANCE_TIMEFRAMES
    from report import create_run_dir, TradeHistoryLogger

__all__ = ["PPO", "SAC", "PPOActorCritic", "SACActor", "SACCritic", "ReplayBuffer", "CSVLogger", "run_hrl_simulation"]


def run_hrl_simulation(config=None, log_dir=None, high_tf: str = "1h", low_tf: str = "5m"):
    """Runs a hierarchical reinforcement learning crypto bot simulation (PPO Manager at high TF + SAC Worker at low TF)."""
    c = config or cfg
    sim_cfg = c.simulation
    high_tf = high_tf or getattr(sim_cfg, "high_tf", "1h")
    low_tf = low_tf or getattr(sim_cfg, "low_tf", "5m")
    print(f"Diet research please... Initializing HRL Crypto Bot [PPO Manager ({high_tf}) -> SAC Worker ({low_tf})]")

    num_candles, num_symbols = sim_cfg.num_candles, sim_cfg.num_symbols
    macro_period = get_timeframe_ratio(high_tf, low_tf)
    done_interval = sim_cfg.done_interval

    run_dir = Path(log_dir) if log_dir else create_run_dir(base_dir="logs")
    ppo_csv = str(run_dir / "ppo_manager.csv")
    sac_csv = str(run_dir / "sac_worker.csv")
    TradeHistoryLogger(run_dir / "trade_history.csv")

    low_data, high_data, ratio = generate_multi_tf_data(
        num_candles=num_candles, num_symbols=num_symbols, high_tf=high_tf, low_tf=low_tf, config=c
    )
    mx.eval(low_data, high_data)

    obs_dim = num_symbols * len(FEATURE_NAMES)
    goal_dim = num_symbols
    worker_obs_dim = obs_dim + goal_dim
    worker_act_dim = num_symbols

    ppo_manager = PPO(obs_dim=obs_dim, act_dim=goal_dim, n_steps=20, batch_size=10, n_epochs=4, csv_path=ppo_csv)
    sac_worker = SAC(obs_dim=worker_obs_dim, act_dim=worker_act_dim, learning_starts=32, batch_size=32, csv_path=sac_csv)

    t_start = time.time()
    total_steps = 0
    m_obs, m_acts, m_rews, m_dones, m_vals, m_lps = [], [], [], [], [], []
    macro_reward = 0.0
    ep_rews, cur_ep_rew = [], 0.0

    curr_low_obs = low_data[0].reshape(-1)
    goal = mx.zeros((1, goal_dim))

    with tqdm(total=num_candles - 1, desc="HRL Simulation", unit="step") as pbar:
        for t in range(num_candles - 1):
            curr_low_norm = (curr_low_obs - mx.mean(curr_low_obs)) / (mx.std(curr_low_obs) + 1e-6)

            # 1. Macro Step: PPO Manager operates at high timeframe (e.g. 1h, 4h, 1d)
            if t % macro_period == 0:
                t_high = min(t // macro_period, high_data.shape[0] - 1)
                high_obs = high_data[t_high].reshape(-1)
                high_obs_norm = (high_obs - mx.mean(high_obs)) / (mx.std(high_obs) + 1e-6)

                if len(m_obs) >= ppo_manager.n_steps:
                    _, _, _, next_v = ppo_manager.policy(high_obs_norm[None, :])
                    ppo_manager.train_on_rollout(
                        mx.stack(m_obs, axis=0), mx.stack(m_acts, axis=0), mx.stack(m_rews, axis=0),
                        mx.stack(m_dones, axis=0), mx.stack(m_vals, axis=0), mx.stack(m_lps, axis=0),
                        next_v[0], ep_rew_mean=sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1)
                    )
                    m_obs, m_acts, m_rews, m_dones, m_vals, m_lps = [], [], [], [], [], []

                goal, lp, _, val = ppo_manager.policy(high_obs_norm[None, :])
                m_obs.append(high_obs_norm)
                m_acts.append(goal[0])
                m_vals.append(val[0])
                m_lps.append(lp[0])
                macro_reward = 0.0

            # 2. Micro Step: SAC Worker executes at low timeframe (e.g. 1m, 5m, 15m)
            worker_obs = mx.concatenate([curr_low_norm, goal[0]], axis=-1)
            worker_act = sac_worker.predict(worker_obs)

            next_low_obs = low_data[t + 1].reshape(-1)
            next_low_norm = (next_low_obs - mx.mean(next_low_obs)) / (mx.std(next_low_obs) + 1e-6)
            price_ret = low_data[t + 1, :, 7]  # log_ret
            # Worker reward: profit + goal alignment incentive
            goal_alignment = -float(mx.mean((worker_act - goal[0]) ** 2).item())
            worker_rew = float(mx.sum(worker_act * price_ret).item()) + 0.1 * goal_alignment
            macro_reward += worker_rew
            cur_ep_rew += worker_rew

            done = 1.0 if (t + 1) % done_interval == 0 else 0.0
            if done:
                ep_rews.append(cur_ep_rew)
                cur_ep_rew = 0.0

            next_worker_obs = mx.concatenate([next_low_norm, goal[0]], axis=-1)
            sac_worker.store(worker_obs, worker_act, worker_rew, next_worker_obs, done)
            sac_worker.train(gradient_steps=1, ep_rew_mean=sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1))

            if (t + 1) % macro_period == 0 or t == num_candles - 2:
                m_rews.append(mx.array(macro_reward))
                m_dones.append(mx.array(done))

            curr_low_obs = next_low_obs
            total_steps += 1
            pbar.update(1)
            if (t + 1) % 10 == 0 or t == num_candles - 2:
                elapsed = max(time.time() - t_start, 1e-6)
                fps = total_steps / elapsed
                mean_ep_rew = sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1)
                pbar.set_postfix(fps=f"{fps:.1f}", rew=f"{worker_rew:.3f}", ep_rew=f"{mean_ep_rew:.2f}")

    if len(m_obs) > 0:
        t_high = min((num_candles - 1) // macro_period, high_data.shape[0] - 1)
        high_obs = high_data[t_high].reshape(-1)
        high_obs_norm = (high_obs - mx.mean(high_obs)) / (mx.std(high_obs) + 1e-6)
        _, _, _, next_v = ppo_manager.policy(high_obs_norm[None, :])
        ppo_manager.train_on_rollout(
            mx.stack(m_obs, axis=0), mx.stack(m_acts, axis=0), mx.stack(m_rews, axis=0),
            mx.stack(m_dones, axis=0), mx.stack(m_vals, axis=0), mx.stack(m_lps, axis=0),
            next_v[0], ep_rew_mean=sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1)
        )

    total_time = time.time() - t_start
    print(f"HRL Crypto Bot finished: {total_steps} micro-steps in {total_time:.3f}s ({total_steps / total_time:.1f} FPS)")
    return ppo_manager, sac_worker


if __name__ == "__main__":
    run_hrl_simulation()
