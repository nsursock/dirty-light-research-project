import sys
import time
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scripts.config import cfg, load_config
    from scripts.env import MultiCryptoDexPerpEnv
    from scripts.ppo import PPO
    from scripts.sac import SAC
    from scripts.agents import compute_goal_alignment
    from scripts.data import get_timeframe_ratio, generate_multi_tf_data
    from scripts.report import create_run_dir, TradeHistoryLogger, generate_breakdown_report, get_symbol_name, compute_financial_metrics
    from scripts.visualize import generate_trade_figures
except ModuleNotFoundError:
    from config import cfg, load_config
    from env import MultiCryptoDexPerpEnv
    from ppo import PPO
    from sac import SAC
    from agents import compute_goal_alignment
    from data import get_timeframe_ratio, generate_multi_tf_data
    from report import create_run_dir, TradeHistoryLogger, generate_breakdown_report, get_symbol_name, compute_financial_metrics
    from visualize import generate_trade_figures

VIZ_THEMES = ("synthwave", "ghibli")


def resolve_theme(theme: str | None = None) -> str:
    """Pick viz theme: explicit name, random, or config default."""
    t = (theme or cfg.visualization.default_theme or "synthwave").lower()
    if t == "random":
        return random.choice(VIZ_THEMES)
    return t if t in VIZ_THEMES else cfg.visualization.default_theme


def create_agents(obs_dim: int, num_symbols: int, ppo_csv: str | None = None, sac_csv: str | None = None, sac_train_freq: int | None = None):
    """Initializes PPO Manager and SAC Worker."""
    goal_dim = num_symbols * 5  # side, leverage, collateral, tp, sl
    worker_obs_dim = obs_dim + goal_dim
    worker_act_dim = num_symbols * 5
    ppo_manager = PPO(obs_dim=obs_dim, act_dim=goal_dim, n_steps=8, batch_size=4, n_epochs=4, csv_path=ppo_csv)
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
    goals = mx.zeros((num_envs, num_symbols * 5)) if num_envs > 1 else mx.zeros((num_symbols * 5,))
    macro_rews = mx.zeros((num_envs,)) if num_envs > 1 else mx.zeros((1,))
    m_buf = {"obs": [], "act": [], "rew": [], "done": [], "val": [], "lp": []}
    ep_rews, cur_ep_rew = [], 0.0
    macro_rollout_steps = max(1, ppo_manager.n_steps // max(num_envs, 1))

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
                if len(m_buf["obs"]) >= macro_rollout_steps and len(m_buf["rew"]) == len(m_buf["obs"]):
                    if manager_train:
                        _, _, _, next_v = ppo_manager.policy(macro_obs_norm)
                        ppo_manager.train_on_rollout(
                            mx.stack(m_buf["obs"], axis=0), mx.stack(m_buf["act"], axis=0),
                            mx.stack(m_buf["rew"], axis=0), mx.stack(m_buf["done"], axis=0),
                            mx.stack(m_buf["val"], axis=0), mx.stack(m_buf["lp"], axis=0),
                            next_v[0] if num_envs == 1 else next_v,
                            ep_rew_mean=sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1)
                        )
                    for k in m_buf:
                        m_buf[k].clear()

                goal, lp, _, val = ppo_manager.policy(macro_obs_norm)
                goals = goal[0] if num_envs == 1 else goal
                m_buf["obs"].append(macro_obs_norm[0] if num_envs == 1 else macro_obs_norm)
                m_buf["act"].append(goal[0] if num_envs == 1 else goal)
                m_buf["val"].append(val[0] if num_envs == 1 else val)
                m_buf["lp"].append(lp[0] if num_envs == 1 else lp)
                macro_rews = mx.zeros_like(macro_rews)

            # Micro Step (SAC Worker at Low TF)
            g_in = goals[None, :] if num_envs == 1 else goals
            worker_obs = mx.concatenate([curr_obs_norm, g_in], axis=-1)
            worker_act = sac_worker.predict(worker_obs, deterministic=False)
            next_obs, rew, done, _, info = env.step(worker_act)
            mx.eval(next_obs, rew)

            macro_rews = macro_rews + rew
            step_rew = float(rew) if num_envs == 1 else float(mx.mean(rew).item())
            cur_ep_rew += step_rew

            next_o = next_obs[None, :] if num_envs == 1 else next_obs
            next_obs_norm = (next_o - mx.mean(next_o, axis=-1, keepdims=True)) / (mx.std(next_o, axis=-1, keepdims=True) + 1e-6)
            next_worker_obs = mx.concatenate([next_obs_norm, g_in], axis=-1)

            align_rew = compute_goal_alignment(worker_act, g_in, num_symbols)
            worker_rew = rew + (0.1 * align_rew if num_envs > 1 else float(0.1 * align_rew.item()))
            sac_worker.store(worker_obs, worker_act, worker_rew, next_worker_obs, float(done) if num_envs == 1 else mx.full((num_envs,), float(done)))
            if worker_train and (timesteps % sac_worker.train_freq == 0):
                sac_worker.train(gradient_steps=sac_worker.gradient_steps, ep_rew_mean=sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1))

            if (env.t % period == 0) or done:
                if len(m_buf["rew"]) < len(m_buf["obs"]):
                    m_buf["rew"].append(macro_rews[0] if num_envs == 1 else macro_rews)
                    m_buf["done"].append(mx.array(float(done)) if num_envs == 1 else mx.full((num_envs,), float(done)))
                    macro_rews = mx.zeros_like(macro_rews)

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
                goals = mx.zeros((num_envs, num_symbols * 5)) if num_envs > 1 else mx.zeros((num_symbols * 5,))
            else:
                obs = next_obs

    if len(m_buf["obs"]) > 0 and len(m_buf["rew"]) == len(m_buf["obs"]) and manager_train:
        macro_obs = env.get_macro_obs()
        macro_obs = macro_obs[None, :] if num_envs == 1 else macro_obs
        macro_obs_norm = (macro_obs - mx.mean(macro_obs, axis=-1, keepdims=True)) / (mx.std(macro_obs, axis=-1, keepdims=True) + 1e-6)
        _, _, _, next_v = ppo_manager.policy(macro_obs_norm)
        ppo_manager.train_on_rollout(
            mx.stack(m_buf["obs"], axis=0), mx.stack(m_buf["act"], axis=0),
            mx.stack(m_buf["rew"], axis=0), mx.stack(m_buf["done"], axis=0),
            mx.stack(m_buf["val"], axis=0), mx.stack(m_buf["lp"], axis=0),
            next_v[0] if num_envs == 1 else next_v,
            ep_rew_mean=sum(ep_rews[-10:]) / max(len(ep_rews[-10:]), 1)
        )
        for k in m_buf:
            m_buf[k].clear()

    total_time = time.time() - t_start
    print(f"Training finished: {timesteps} steps in {total_time:.2f}s ({timesteps / total_time:.1f} FPS)")
    return ppo_manager, sac_worker


def evaluate_hrl(
    ppo_manager, sac_worker, num_episodes: int = 5, margin_mode: str = "isolated",
    high_tf: str = "1h", low_tf: str = "5m", macro_period: int | None = None,
    num_symbols: int = 4, num_candles: int = 300, eval_envs: int | None = None,
    log_dir: str | Path | None = None, theme: str | None = None,
):
    """Vectorized fast evaluation of trained agents across parallel test episodes."""
    import numpy as np
    period = macro_period or get_timeframe_ratio(high_tf, low_tf)
    theme = resolve_theme(theme)
    print(f"\nDiet research please... Starting Vectorized Evaluation ({num_episodes} eps | {margin_mode} | {high_tf}->{low_tf} | theme={theme})")
    trade_logger = TradeHistoryLogger(Path(log_dir) / "trade_history.csv") if log_dir else None
    results, eval_start, total_eval_steps = [], time.time(), 0
    max_batch = eval_envs or min(num_episodes, 128)

    with tqdm(total=num_episodes, desc="Evaluating HRL", unit="ep") as pbar:
        ep_done = 0
        while ep_done < num_episodes:
            b_size = min(max_batch, num_episodes - ep_done)
            rec = (trade_logger is not None)
            low_list, high_list = [], []
            for i in range(b_size):
                mx.random.seed(42 + (ep_done + i) * 1000)
                l, h, _ = generate_multi_tf_data(num_candles=num_candles, num_symbols=num_symbols, high_tf=high_tf, low_tf=low_tf)
                low_list.append(l)
                high_list.append(h)
            low_batch = mx.stack(low_list, axis=0) if b_size > 1 else low_list[0]
            high_batch = mx.stack(high_list, axis=0) if b_size > 1 else high_list[0]
            env = MultiCryptoDexPerpEnv(
                num_envs=b_size, data=low_batch, high_tf_data=high_batch, num_symbols=num_symbols,
                num_candles=num_candles, margin_mode=margin_mode, high_tf=high_tf, low_tf=low_tf, record_trades=rec
            )
            obs, info = env.reset()
            goals = mx.zeros((b_size, num_symbols * 5)) if b_size > 1 else mx.zeros((num_symbols * 5,))

            batch_trades = []
            for t in range(num_candles - 1):
                total_eval_steps += b_size
                curr_obs = obs[None, :] if b_size == 1 else obs
                c_norm = (curr_obs - mx.mean(curr_obs, axis=-1, keepdims=True)) / (mx.std(curr_obs, axis=-1, keepdims=True) + 1e-6)
                if t % period == 0:
                    m_obs = env.get_macro_obs()
                    m_obs = m_obs[None, :] if b_size == 1 else m_obs
                    m_norm = (m_obs - mx.mean(m_obs, axis=-1, keepdims=True)) / (mx.std(m_obs, axis=-1, keepdims=True) + 1e-6)
                    g_out = ppo_manager.predict(m_norm, deterministic=True)
                    goals = g_out[0] if b_size == 1 else g_out
                g_in = goals[None, :] if b_size == 1 else goals
                action = sac_worker.predict(mx.concatenate([c_norm, g_in], axis=-1), deterministic=True)
                obs, rew, done, _, step_info = env.step(action)
                if rec and "trades" in step_info:
                    for tr in step_info["trades"]:
                        tr_d = dict(tr)
                        e_idx = tr_d.pop("env_idx", 0)
                        tr_d["symbol"] = get_symbol_name(tr_d.pop("symbol_idx", 0))
                        tr_d["strategy_id"] = f"Episode {ep_done + e_idx + 1}"
                        batch_trades.append(tr_d)
                        if trade_logger:
                            trade_logger.log_trade(tr_d)
                if done: break

            eqs = np.array(env.equity[:, 0]) if b_size > 1 else np.array([float(env.equity[0, 0].item())])
            peaks = np.array(env.peak_equity[:, 0]) if b_size > 1 else np.array([float(env.peak_equity[0, 0].item())])
            ulcers = np.array(mx.sqrt(env.sum_dd_sq[:, 0] / max(env.step_cnt, 1))) if b_size > 1 else np.array([float(info["ulcer_index"])])

            for i in range(b_size):
                ep_num = ep_done + i + 1
                ep_id = f"Episode {ep_num}"
                sub_trades = [tr for tr in batch_trades if tr.get("strategy_id") == ep_id]
                if sub_trades:
                    m = compute_financial_metrics(sub_trades, initial_capital=cfg.env.initial_capital)
                    ret_v = float(m["return_pct"].replace("%", ""))
                    eq_v = max(0.0, cfg.env.initial_capital + float(m["net_profit"]))
                    dd_v = float(m["max_dd"].replace("%", ""))
                    martin_v = float(m["martin"]) if m["martin"] != "n/a" else 0.0
                    n_trades = int(m["num_trades"])
                else:
                    ret_v = max(-100.0, ((float(eqs[i]) - env.initial_capital) / env.initial_capital) * 100.0)
                    eq_v = max(0.0, float(eqs[i]))
                    dd_v = np.maximum(0.0, (float(peaks[i]) - float(eqs[i])) / (float(peaks[i]) + 1e-8)) * 100.0
                    martin_v = (ret_v / 100.0) / (float(ulcers[i]) + 1e-6)
                    n_trades = int(env.total_trades if i == 0 else 0)

                results.append({
                    "episode": ep_num, "equity": eq_v, "return_pct": ret_v,
                    "drawdown_pct": dd_v, "ulcer_index": float(ulcers[i]),
                    "martin_ratio": martin_v, "liquidations": int(env.total_liq_count.item() if i == 0 else 0),
                    "trades": n_trades,
                })
                pbar.write(f"  Ep {ep_num:02d} | Return: {ret_v:+6.2f}% | Final Eq: {eq_v:8.2f} | MaxDD: {dd_v:5.2f}% | Martin: {martin_v:6.2f}")

            ep_done += b_size
            eval_fps = total_eval_steps / max(time.time() - eval_start, 1e-6)
            pbar.update(b_size)
            pbar.set_postfix(fps=f"{eval_fps:.1f}", mean_ret=f"{sum(r['return_pct'] for r in results)/len(results):+.2f}%")

    if log_dir:
        th = Path(log_dir) / "trade_history.csv"
        generate_breakdown_report(th, Path(log_dir) / "breakdown.txt", initial_capital=cfg.env.initial_capital)
        generate_trade_figures(th, out_dir=log_dir, theme=theme, initial_capital=cfg.env.initial_capital)

    mean_ret = sum(r["return_pct"] for r in results) / len(results)
    mean_martin = sum(r["martin_ratio"] for r in results) / len(results)
    mean_dd = sum(r["drawdown_pct"] for r in results) / len(results)
    print("-" * 75)
    print(f"Summary: Mean Return: {mean_ret:+.2f}% | Mean Martin: {mean_martin:.2f} | Mean DD: {mean_dd:.2f}%")
    print("-" * 75)
    return results


def parse_args():
    s = cfg.simulation
    p = argparse.ArgumentParser(description="Multi Crypto Pure MLX Trading Bot")
    p.add_argument("--mode", choices=["train", "test", "full"], default="full")
    p.add_argument("--stage", default=None, help="S0/S1/S2/S3 or stage YAML path (budget only)")
    p.add_argument("--timesteps", "-t", type=int, default=None)
    p.add_argument("--episodes", "-e", type=int, default=None)
    p.add_argument("--num_envs", "-n", type=int, default=None)
    p.add_argument("--eval_envs", type=int, default=None, help="Parallel evaluation envs")
    p.add_argument("--train_scheme", choices=["joint", "alternating"], default="joint")
    p.add_argument("--margin_mode", choices=["isolated", "cross"], default=cfg.env.margin_mode)
    p.add_argument("--high_tf", default=s.high_tf)
    p.add_argument("--low_tf", default=s.low_tf)
    p.add_argument("--num_symbols", type=int, default=s.num_symbols)
    p.add_argument("--num_candles", type=int, default=s.num_candles)
    p.add_argument("--macro_period", type=int, default=s.get("macro_period"))
    p.add_argument("--train_freq", type=int, default=None)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--theme", choices=["synthwave", "ghibli", "random"], default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_cfg = load_config(stage=args.stage) if args.stage else cfg
    tr, ev = run_cfg.get("training") or {}, run_cfg.get("evaluation") or {}
    timesteps = args.timesteps if args.timesteps is not None else int(tr.get("total_timesteps", 2000))
    num_envs = args.num_envs if args.num_envs is not None else int(tr.get("n_envs", 2))
    episodes = args.episodes if args.episodes is not None else int(ev.get("episodes", 5))
    stage_name = (run_cfg.get("stage") or {}).get("name", args.stage or "base")
    theme = resolve_theme(args.theme)
    log_dir = Path(args.log_dir) if args.log_dir else create_run_dir(base_dir="logs")
    print(f"Diet research please... stage={stage_name} | t={timesteps} n_envs={num_envs} eval_eps={episodes} | log={log_dir} | theme={theme}")
    ppo_manager, sac_worker = None, None
    kw = dict(margin_mode=args.margin_mode, high_tf=args.high_tf, low_tf=args.low_tf, macro_period=args.macro_period, num_symbols=args.num_symbols, num_candles=args.num_candles)

    if args.mode in ("train", "full"):
        ppo_manager, sac_worker = train_hrl(total_timesteps=timesteps, num_envs=num_envs, train_scheme=args.train_scheme, train_freq=args.train_freq, log_dir=log_dir, **kw)
    if args.mode in ("test", "full"):
        if ppo_manager is None or sac_worker is None:
            env0 = MultiCryptoDexPerpEnv(num_symbols=args.num_symbols, num_candles=args.num_candles, high_tf=args.high_tf, low_tf=args.low_tf)
            ppo_manager, sac_worker = create_agents(env0.obs_dim, args.num_symbols)
        evaluate_hrl(ppo_manager, sac_worker, num_episodes=episodes, eval_envs=args.eval_envs, log_dir=log_dir, theme=theme, **kw)

    TradeHistoryLogger(log_dir / "trade_history.csv")
    for fname in ("ppo_manager.csv", "sac_worker.csv"):
        (log_dir / fname).touch(exist_ok=True)
    th = log_dir / "trade_history.csv"
    generate_breakdown_report(th, log_dir / "breakdown.txt")
    generate_trade_figures(th, out_dir=log_dir, theme=theme, initial_capital=cfg.env.initial_capital)


if __name__ == "__main__":
    main()
