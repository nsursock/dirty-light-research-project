import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scripts.config import cfg
    from scripts.env import MultiCryptoDexPerpEnv
    from scripts.ppo import PPO
    from scripts.sac import SAC
    from scripts.data import get_timeframe_ratio, BINANCE_TIMEFRAMES
    from scripts.report import create_run_dir, TradeHistoryLogger, generate_breakdown_report, DEFAULT_SYMBOLS, get_symbol_name
except ModuleNotFoundError:
    from config import cfg
    from env import MultiCryptoDexPerpEnv
    from ppo import PPO
    from sac import SAC
    from data import get_timeframe_ratio, BINANCE_TIMEFRAMES
    from report import create_run_dir, TradeHistoryLogger, generate_breakdown_report, DEFAULT_SYMBOLS, get_symbol_name


def create_agents(obs_dim: int, num_symbols: int, ppo_csv: str | None = None, sac_csv: str | None = None, sac_train_freq: int | None = None):
    """Initializes PPO Manager and SAC Worker."""
    goal_dim = num_symbols
    worker_obs_dim = obs_dim + goal_dim
    worker_act_dim = num_symbols
    ppo_manager = PPO(obs_dim=obs_dim, act_dim=goal_dim, n_steps=32, batch_size=16, n_epochs=4, csv_path=ppo_csv)
    sac_worker = SAC(obs_dim=worker_obs_dim, act_dim=worker_act_dim, learning_starts=32, batch_size=32, train_freq=sac_train_freq, csv_path=sac_csv)
    return ppo_manager, sac_worker


def train_hrl(
    total_timesteps: int = 3000,
    num_envs: int = 2,
    train_scheme: str = "joint",
    margin_mode: str = "isolated",
    high_tf: str = "1h",
    low_tf: str = "5m",
    macro_period: int | None = None,
    num_symbols: int = 4,
    num_candles: int = 300,
    alt_interval: int = 200,
    train_freq: int | None = None,
    log_dir: str | Path | None = None,
):
    """Trains PPO Manager (high TF) + SAC Worker (low TF) with vectorized pure MLX env."""
    period = macro_period or get_timeframe_ratio(high_tf, low_tf)
    print(f"Diet research please... Starting Training [{train_scheme.upper()} | {num_envs} env(s) | {margin_mode} margin | High TF: {high_tf} -> Low TF: {low_tf} (ratio {period})]")
    env = MultiCryptoDexPerpEnv(num_envs=num_envs, num_symbols=num_symbols, num_candles=num_candles, margin_mode=margin_mode, high_tf=high_tf, low_tf=low_tf, record_trades=False)
    obs_dim = env.obs_dim
    ppo_csv = str(Path(log_dir) / "ppo_manager.csv") if log_dir else None
    sac_csv = str(Path(log_dir) / "sac_worker.csv") if log_dir else None
    sac_tf = train_freq if train_freq is not None else max(1, num_envs // 2)
    ppo_manager, sac_worker = create_agents(obs_dim, num_symbols, ppo_csv=ppo_csv, sac_csv=sac_csv, sac_train_freq=sac_tf)

    obs, _ = env.reset(seed=100)
    goals = mx.zeros((num_envs, num_symbols)) if num_envs > 1 else mx.zeros((num_symbols,))
    macro_rews = mx.zeros((num_envs,)) if num_envs > 1 else mx.zeros((1,))
    m_buf = {"obs": [], "act": [], "rew": [], "done": [], "val": [], "lp": []}
    ep_rews, cur_ep_rew = [], 0.0

    timesteps = 0
    t_start = time.time()

    with tqdm(total=total_timesteps, desc="Training HRL", unit="step") as pbar:
        while timesteps < total_timesteps:
            worker_train = True if train_scheme == "joint" else ((timesteps // alt_interval) % 2 == 0)
            manager_train = True if train_scheme == "joint" else ((timesteps // alt_interval) % 2 == 1)

            curr_obs = obs[None, :] if num_envs == 1 else obs
            curr_obs_norm = (curr_obs - mx.mean(curr_obs, axis=-1, keepdims=True)) / (mx.std(curr_obs, axis=-1, keepdims=True) + 1e-6)

            # Macro Step (PPO Manager at High TF)
            if env.t % period == 0:
                macro_obs = env.get_macro_obs()
                macro_obs = macro_obs[None, :] if num_envs == 1 else macro_obs
                macro_obs_norm = (macro_obs - mx.mean(macro_obs, axis=-1, keepdims=True)) / (mx.std(macro_obs, axis=-1, keepdims=True) + 1e-6)
                if len(m_buf["obs"]) >= ppo_manager.n_steps:
                    if manager_train:
                        _, _, _, next_v = ppo_manager.policy(macro_obs_norm)
                        ppo_manager.train_on_rollout(
                            mx.concatenate(m_buf["obs"], axis=0), mx.concatenate(m_buf["act"], axis=0),
                            mx.concatenate(m_buf["rew"], axis=0), mx.concatenate(m_buf["done"], axis=0),
                            mx.concatenate(m_buf["val"], axis=0), mx.concatenate(m_buf["lp"], axis=0),
                            next_v[0] if num_envs == 1 else next_v.mean(),
                            ep_rew_mean=sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1)
                        )
                    for k in m_buf:
                        m_buf[k].clear()

                goal, lp, _, val = ppo_manager.policy(macro_obs_norm)
                goals = goal[0] if num_envs == 1 else goal
                m_buf["obs"].append(macro_obs_norm)
                m_buf["act"].append(goal)
                m_buf["val"].append(val)
                m_buf["lp"].append(lp)
                macro_rews = mx.zeros_like(macro_rews)

            # Micro Step (SAC Worker at Low TF)
            g_in = goals[None, :] if num_envs == 1 else goals
            worker_obs = mx.concatenate([curr_obs_norm, g_in], axis=-1)
            worker_act = sac_worker.predict(worker_obs, deterministic=False)
            next_obs, rew, done, _, info = env.step(worker_act)

            macro_rews = macro_rews + rew
            step_rew = float(rew) if num_envs == 1 else float(mx.mean(rew).item())
            cur_ep_rew += step_rew

            next_o = next_obs[None, :] if num_envs == 1 else next_obs
            next_obs_norm = (next_o - mx.mean(next_o, axis=-1, keepdims=True)) / (mx.std(next_o, axis=-1, keepdims=True) + 1e-6)
            next_worker_obs = mx.concatenate([next_obs_norm, g_in], axis=-1)

            sac_worker.store(worker_obs, worker_act, rew, next_worker_obs, float(done) if num_envs == 1 else mx.full((num_envs,), float(done)))
            if worker_train and (timesteps % sac_worker.train_freq == 0):
                sac_worker.train(gradient_steps=sac_worker.gradient_steps, ep_rew_mean=sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1))

            if (env.t % period == 0) or done:
                m_buf["rew"].append(macro_rews[None, :] if num_envs == 1 else macro_rews)
                m_buf["done"].append(mx.full((1,) if num_envs == 1 else (num_envs,), float(done)))

            inc = num_envs
            timesteps += inc
            pbar.update(inc)
            if timesteps % max(10, num_envs) == 0 or timesteps >= total_timesteps:
                fps = timesteps / max(time.time() - t_start, 1e-6)
                mean_ep_rew = sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1)
                mode_str = "joint" if train_scheme == "joint" else ("worker" if worker_train else "manager")
                pbar.set_postfix(fps=f"{fps:.1f}", ep_rew=f"{mean_ep_rew:.2f}", mode=mode_str, eps=len(ep_rews))

            if done:
                ep_rews.append(cur_ep_rew)
                cur_ep_rew = 0.0
                obs, _ = env.reset(seed=timesteps + 17)
                goals = mx.zeros((num_envs, num_symbols)) if num_envs > 1 else mx.zeros((num_symbols,))
            else:
                obs = next_obs

    total_time = time.time() - t_start
    print(f"Training finished: {timesteps} steps in {total_time:.2f}s ({timesteps / total_time:.1f} FPS)")
    return ppo_manager, sac_worker


def evaluate_hrl(
    ppo_manager,
    sac_worker,
    num_episodes: int = 5,
    margin_mode: str = "isolated",
    high_tf: str = "1h",
    low_tf: str = "5m",
    macro_period: int | None = None,
    num_symbols: int = 4,
    num_candles: int = 300,
    log_dir: str | Path | None = None,
):
    """Evaluates trained agents over test episodes and reports Martin ratio & trading metrics."""
    period = macro_period or get_timeframe_ratio(high_tf, low_tf)
    print(f"\nDiet research please... Starting Evaluation ({num_episodes} eps | {margin_mode} | {high_tf}->{low_tf})")
    env = MultiCryptoDexPerpEnv(num_symbols=num_symbols, num_candles=num_candles, margin_mode=margin_mode, high_tf=high_tf, low_tf=low_tf)
    trade_logger = TradeHistoryLogger(Path(log_dir) / "trade_history.csv") if log_dir else None
    results = []
    eval_start = time.time()
    total_eval_steps = 0

    with tqdm(total=num_episodes, desc="Evaluating HRL", unit="ep") as pbar:
        for ep in range(num_episodes):
            obs, info = env.reset(seed=42 + ep * 1000)
            done = False
            goal = mx.zeros((num_symbols,))
            total_reward = 0.0

            while not done:
                total_eval_steps += 1
                if env.t % period == 0:
                    macro_obs = env.get_macro_obs()
                    macro_obs_norm = (macro_obs - mx.mean(macro_obs)) / (mx.std(macro_obs) + 1e-6)
                    goal = ppo_manager.predict(macro_obs_norm, deterministic=True)
                obs_norm = (obs - mx.mean(obs)) / (mx.std(obs) + 1e-6)
                worker_obs = mx.concatenate([obs_norm, goal], axis=-1)
                action = sac_worker.predict(worker_obs, deterministic=True)
                obs, rew, done, _, info = env.step(action)
                total_reward += rew

                if trade_logger and "trades" in info:
                    for tr in info["trades"]:
                        tr_dict = dict(tr)
                        s_idx = tr_dict.pop("symbol_idx", 0)
                        tr_dict["symbol"] = get_symbol_name(s_idx)
                        tr_dict["strategy_id"] = f"Episode {ep + 1}"
                        trade_logger.log_trade(tr_dict)

            ret_pct = ((info["equity"] - env.initial_capital) / env.initial_capital) * 100.0
            results.append({
                "episode": ep + 1, "equity": info["equity"], "return_pct": ret_pct,
                "drawdown_pct": info["drawdown"] * 100.0, "ulcer_index": info["ulcer_index"],
                "martin_ratio": info["martin_ratio"], "liquidations": info["total_liquidations"], "trades": info["total_trades"],
            })
            eval_elapsed = max(time.time() - eval_start, 1e-6)
            eval_fps = total_eval_steps / eval_elapsed
            pbar.update(1)
            pbar.set_postfix(
                fps=f"{eval_fps:.1f}", ret=f"{ret_pct:+.2f}%", eq=f"{info['equity']:.2f}",
                martin=f"{info['martin_ratio']:.2f}", max_dd=f"{info['drawdown']*100:.1f}%"
            )
            pbar.write(f"  Ep {ep + 1:02d} | Return: {ret_pct:+6.2f}% | Final Eq: {info['equity']:8.2f} | MaxDD: {info['drawdown']*100:5.2f}% | Martin: {info['martin_ratio']:6.2f} | Liqs: {info['total_liquidations']}")

    if log_dir:
        generate_breakdown_report(Path(log_dir) / "trade_history.csv", Path(log_dir) / "breakdown.txt", initial_capital=env.initial_capital)

    mean_ret = sum(r["return_pct"] for r in results) / len(results)
    mean_martin = sum(r["martin_ratio"] for r in results) / len(results)
    mean_dd = sum(r["drawdown_pct"] for r in results) / len(results)
    print("-" * 75)
    print(f"Summary: Mean Return: {mean_ret:+.2f}% | Mean Martin: {mean_martin:.2f} | Mean DD: {mean_dd:.2f}%")
    print("-" * 75)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Multi Crypto Pure MLX Trading Bot")
    parser.add_argument("--mode", type=str, choices=["train", "test", "full"], default="full", help="Execution mode")
    parser.add_argument("--timesteps", "-t", type=int, default=2000, help="Training timesteps override")
    parser.add_argument("--episodes", "-e", type=int, default=5, help="Testing episodes override")
    parser.add_argument("--num_envs", "-n", type=int, default=2, help="Number of parallel training environments")
    parser.add_argument("--train_scheme", type=str, choices=["joint", "alternating"], default="joint", help="Training scheme")
    parser.add_argument("--margin_mode", type=str, choices=["isolated", "cross"], default="isolated", help="Margin mode")
    parser.add_argument("--high_tf", type=str, default="1h", help="Binance high timeframe for PPO Manager (e.g. 1h, 4h, 1d)")
    parser.add_argument("--low_tf", type=str, default="5m", help="Binance low timeframe for SAC Worker (e.g. 1m, 5m, 15m)")
    parser.add_argument("--num_symbols", type=int, default=4, help="Number of crypto assets")
    parser.add_argument("--num_candles", type=int, default=300, help="Candles per episode")
    parser.add_argument("--macro_period", type=int, default=None, help="Macro period override (default: auto from timeframes)")
    parser.add_argument("--train_freq", type=int, default=None, help="SAC train frequency")
    parser.add_argument("--log_dir", type=str, default=None, help="Custom logging output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else create_run_dir(base_dir="logs")
    print(f"Diet research please... Simulation output run directory: {log_dir}")
    ppo_manager, sac_worker = None, None

    if args.mode in ("train", "full"):
        ppo_manager, sac_worker = train_hrl(
            total_timesteps=args.timesteps, num_envs=args.num_envs, train_scheme=args.train_scheme,
            margin_mode=args.margin_mode, high_tf=args.high_tf, low_tf=args.low_tf,
            macro_period=args.macro_period, num_symbols=args.num_symbols, num_candles=args.num_candles,
            train_freq=args.train_freq, log_dir=log_dir,
        )

    if args.mode in ("test", "full"):
        if ppo_manager is None or sac_worker is None:
            sample_env = MultiCryptoDexPerpEnv(num_symbols=args.num_symbols, num_candles=args.num_candles, high_tf=args.high_tf, low_tf=args.low_tf)
            ppo_manager, sac_worker = create_agents(sample_env.obs_dim, args.num_symbols)

        evaluate_hrl(
            ppo_manager, sac_worker, num_episodes=args.episodes, margin_mode=args.margin_mode,
            high_tf=args.high_tf, low_tf=args.low_tf, macro_period=args.macro_period,
            num_symbols=args.num_symbols, num_candles=args.num_candles, log_dir=log_dir,
        )

    TradeHistoryLogger(log_dir / "trade_history.csv")
    for fname in ["ppo_manager.csv", "sac_worker.csv"]:
        fpath = log_dir / fname
        if not fpath.exists():
            fpath.touch()
    generate_breakdown_report(log_dir / "trade_history.csv", log_dir / "breakdown.txt")


if __name__ == "__main__":
    main()
