import sys
import time
import argparse
import resource
import subprocess
import re
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scripts.config import cfg
    from scripts.env import MultiCryptoDexPerpEnv
    from scripts.main import create_agents
    from scripts.data import get_timeframe_ratio
except ModuleNotFoundError:
    from config import cfg
    from env import MultiCryptoDexPerpEnv
    from main import create_agents
    from data import get_timeframe_ratio


def get_system_swap_mb() -> float:
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], stderr=subprocess.DEVNULL).decode("utf-8")
            m = re.search(r"used\s*=\s*([\d\.]+)M", out)
            return float(m.group(1)) if m else 0.0
        except Exception:
            return 0.0
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                info = {l.split(":")[0].strip(): float(l.split(":")[1].strip().split()[0]) for l in f if ":" in l}
            return (info["SwapTotal"] - info["SwapFree"]) / 1024.0 if ("SwapTotal" in info and "SwapFree" in info) else 0.0
        except Exception:
            return 0.0
    return 0.0


def get_memory_stats(base_swap: float = 0.0) -> dict:
    swap, rss = get_system_swap_mb(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = (rss / (1024 * 1024)) if sys.platform == "darwin" else (rss / 1024.0)
    mlx_act = getattr(mx, "get_active_memory", lambda: 0)() / (1024 * 1024)
    mlx_peak = getattr(mx, "get_peak_memory", lambda: 0)() / (1024 * 1024)
    return {"rss_mb": rss_mb, "mlx_active_mb": mlx_act, "mlx_peak_mb": mlx_peak, "swap_used_mb": swap, "swap_delta_mb": max(0.0, swap - base_swap)}


def bench_env_fps(num_envs: int, steps: int = 200, steps_per_env: int | None = None, num_symbols: int = 4, num_candles: int = 300) -> float:
    n_steps = steps_per_env or steps
    env = MultiCryptoDexPerpEnv(num_envs=num_envs, num_symbols=num_symbols, num_candles=num_candles, record_trades=False)
    obs, _ = env.reset(seed=100)
    act = mx.random.uniform(-1.0, 1.0, (num_envs, num_symbols)) if num_envs > 1 else mx.random.uniform(-1.0, 1.0, (num_symbols,))
    for _ in range(5):
        obs, rew, done, _, _ = env.step(act)
        mx.eval(obs, rew)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        obs, rew, done, _, _ = env.step(act)
        mx.eval(obs, rew)
    return (num_envs * n_steps) / max(time.perf_counter() - t0, 1e-6)


def bench_train_fps(num_envs: int, steps: int = 100, steps_per_env: int | None = None, num_symbols: int = 4, num_candles: int = 300) -> float:
    n_steps = steps_per_env or steps
    env = MultiCryptoDexPerpEnv(num_envs=num_envs, num_symbols=num_symbols, num_candles=num_candles, record_trades=False)
    obs_dim, period, sac_tf = env.obs_dim, get_timeframe_ratio("1h", "5m"), max(1, num_envs // 2)
    ppo_manager, sac_worker = create_agents(obs_dim, num_symbols, sac_train_freq=sac_tf)
    obs, _ = env.reset(seed=100)
    goals = mx.zeros((num_envs, num_symbols)) if num_envs > 1 else mx.zeros((num_symbols,))
    macro_rews, m_buf = (mx.zeros((num_envs,)) if num_envs > 1 else mx.zeros((1,))), {"obs": [], "act": [], "rew": [], "done": [], "val": [], "lp": []}
    w_obs = mx.concatenate([obs[None, :] if num_envs == 1 else obs, goals[None, :] if num_envs == 1 else goals], axis=-1)
    mx.eval(sac_worker.predict(w_obs))

    timesteps, t0 = 0, time.perf_counter()
    for _ in range(n_steps):
        curr_obs = obs[None, :] if num_envs == 1 else obs
        curr_obs_norm = (curr_obs - mx.mean(curr_obs, axis=-1, keepdims=True)) / (mx.std(curr_obs, axis=-1, keepdims=True) + 1e-6)

        if env.t % period == 0:
            macro_obs = env.get_macro_obs()
            macro_obs = macro_obs[None, :] if num_envs == 1 else macro_obs
            macro_obs_norm = (macro_obs - mx.mean(macro_obs, axis=-1, keepdims=True)) / (mx.std(macro_obs, axis=-1, keepdims=True) + 1e-6)
            if len(m_buf["obs"]) >= ppo_manager.n_steps:
                _, _, _, next_v = ppo_manager.policy(macro_obs_norm)
                ppo_manager.train_on_rollout(mx.concatenate(m_buf["obs"], axis=0), mx.concatenate(m_buf["act"], axis=0), mx.concatenate(m_buf["rew"], axis=0), mx.concatenate(m_buf["done"], axis=0), mx.concatenate(m_buf["val"], axis=0), mx.concatenate(m_buf["lp"], axis=0), next_v[0] if num_envs == 1 else next_v.mean())
                for k in m_buf: m_buf[k].clear()
            goal, lp, _, val = ppo_manager.policy(macro_obs_norm)
            goals = goal[0] if num_envs == 1 else goal
            m_buf["obs"].append(macro_obs_norm); m_buf["act"].append(goal); m_buf["val"].append(val); m_buf["lp"].append(lp)
            macro_rews = mx.zeros_like(macro_rews)

        g_in = goals[None, :] if num_envs == 1 else goals
        worker_obs = mx.concatenate([curr_obs_norm, g_in], axis=-1)
        worker_act = sac_worker.predict(worker_obs)
        next_obs, rew, done, _, _ = env.step(worker_act)

        macro_rews = macro_rews + rew
        next_o = next_obs[None, :] if num_envs == 1 else next_obs
        next_obs_norm = (next_o - mx.mean(next_o, axis=-1, keepdims=True)) / (mx.std(next_o, axis=-1, keepdims=True) + 1e-6)
        next_worker_obs = mx.concatenate([next_obs_norm, g_in], axis=-1)

        sac_worker.store(worker_obs, worker_act, rew, next_worker_obs, float(done) if num_envs == 1 else mx.full((num_envs,), float(done)))
        if timesteps % sac_worker.train_freq == 0:
            sac_worker.train(gradient_steps=sac_worker.gradient_steps)

        if (env.t % period == 0) or done:
            m_buf["rew"].append(macro_rews[None, :] if num_envs == 1 else macro_rews)
            m_buf["done"].append(mx.full((1,) if num_envs == 1 else (num_envs,), float(done)))

        timesteps += num_envs
        if done:
            obs, _ = env.reset(seed=timesteps + 17)
            goals = mx.zeros((num_envs, num_symbols)) if num_envs > 1 else mx.zeros((num_symbols,))
        else:
            obs = next_obs
        mx.eval(obs, rew)

    return timesteps / max(time.perf_counter() - t0, 1e-6)


def profile_breakdown(num_envs: int = 256, steps: int = 100, num_symbols: int = 4, num_candles: int = 300) -> dict:
    print(f"Diet research please... Profiling % wall-time breakdown for n_envs={num_envs}...")
    env = MultiCryptoDexPerpEnv(num_envs=num_envs, num_symbols=num_symbols, num_candles=num_candles, record_trades=False)
    obs_dim, period = env.obs_dim, get_timeframe_ratio("1h", "5m")
    ppo_manager, sac_worker = create_agents(obs_dim, num_symbols, sac_train_freq=max(1, num_envs // 2))
    obs, _ = env.reset(seed=100)
    goals = mx.zeros((num_envs, num_symbols)) if num_envs > 1 else mx.zeros((num_symbols,))
    macro_rews = mx.zeros((num_envs,)) if num_envs > 1 else mx.zeros((1,))
    m_buf = {"obs": [], "act": [], "rew": [], "done": [], "val": [], "lp": []}
    times = {k: 0.0 for k in ["env_step", "manager_forward", "worker_forward", "sac_critic", "sac_actor", "ppo_update", "buffer_ops", "obs_construction", "reward_computation", "mlx_sync"]}

    timesteps, t_start = 0, time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        curr_obs = obs[None, :] if num_envs == 1 else obs
        curr_obs_norm = (curr_obs - mx.mean(curr_obs, axis=-1, keepdims=True)) / (mx.std(curr_obs, axis=-1, keepdims=True) + 1e-6)
        times["obs_construction"] += time.perf_counter() - t0

        if env.t % period == 0:
            macro_obs = env.get_macro_obs()
            macro_obs = macro_obs[None, :] if num_envs == 1 else macro_obs
            macro_obs_norm = (macro_obs - mx.mean(macro_obs, axis=-1, keepdims=True)) / (mx.std(macro_obs, axis=-1, keepdims=True) + 1e-6)
            if len(m_buf["obs"]) >= ppo_manager.n_steps:
                t0 = time.perf_counter()
                _, _, _, next_v = ppo_manager.policy(macro_obs_norm)
                ppo_manager.train_on_rollout(mx.concatenate(m_buf["obs"], axis=0), mx.concatenate(m_buf["act"], axis=0), mx.concatenate(m_buf["rew"], axis=0), mx.concatenate(m_buf["done"], axis=0), mx.concatenate(m_buf["val"], axis=0), mx.concatenate(m_buf["lp"], axis=0), next_v[0] if num_envs == 1 else next_v.mean())
                times["ppo_update"] += time.perf_counter() - t0
                for k in m_buf: m_buf[k].clear()

            t0 = time.perf_counter()
            goal, lp, _, val = ppo_manager.policy(macro_obs_norm)
            times["manager_forward"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            goals = goal[0] if num_envs == 1 else goal
            m_buf["obs"].append(macro_obs_norm); m_buf["act"].append(goal); m_buf["val"].append(val); m_buf["lp"].append(lp)
            times["buffer_ops"] += time.perf_counter() - t0
            macro_rews = mx.zeros_like(macro_rews)

        t0 = time.perf_counter()
        g_in = goals[None, :] if num_envs == 1 else goals
        worker_obs = mx.concatenate([curr_obs_norm, g_in], axis=-1)
        worker_act = sac_worker.predict(worker_obs)
        times["worker_forward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        next_obs, rew, done, _, _ = env.step(worker_act)
        times["env_step"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        macro_rews = macro_rews + rew
        next_o = next_obs[None, :] if num_envs == 1 else next_obs
        next_obs_norm = (next_o - mx.mean(next_o, axis=-1, keepdims=True)) / (mx.std(next_o, axis=-1, keepdims=True) + 1e-6)
        next_worker_obs = mx.concatenate([next_obs_norm, g_in], axis=-1)
        times["reward_computation"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        sac_worker.store(worker_obs, worker_act, rew, next_worker_obs, float(done) if num_envs == 1 else mx.full((num_envs,), float(done)))
        times["buffer_ops"] += time.perf_counter() - t0

        if timesteps % sac_worker.train_freq == 0 and sac_worker.replay_buffer.size >= sac_worker.learning_starts:
            b_obs, b_act, b_rew, b_next_obs, b_done = sac_worker.replay_buffer.sample(sac_worker.batch_size)
            t0 = time.perf_counter()
            next_act, next_lp = sac_worker.actor.sample(b_next_obs)
            t_q1, t_q2 = sac_worker.target_critic(b_next_obs, next_act)
            target_q = b_rew + sac_worker.gamma * (1.0 - b_done) * (mx.minimum(t_q1, t_q2) - sac_worker.alpha * next_lp)
            c_loss, c_grads = nn.value_and_grad(sac_worker.critic, lambda m, o, a, t: 0.5 * mx.mean((m(o, a)[0] - t) ** 2) + 0.5 * mx.mean((m(o, a)[1] - t) ** 2))(sac_worker.critic, b_obs, b_act, target_q)
            sac_worker.critic_opt.update(sac_worker.critic, c_grads)
            times["sac_critic"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            (a_loss, lp), a_grads = nn.value_and_grad(sac_worker.actor, lambda m, c, o, alpha: (mx.mean(alpha * m.sample(o)[1] - mx.minimum(*c(o, m.sample(o)[0]))), m.sample(o)[1]))(sac_worker.actor, sac_worker.critic, b_obs, sac_worker.alpha)
            sac_worker.actor_opt.update(sac_worker.actor, a_grads)
            times["sac_actor"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            mx.eval(sac_worker.actor.state, sac_worker.critic.state, sac_worker.target_critic.state)
            times["mlx_sync"] += time.perf_counter() - t0

        if (env.t % period == 0) or done:
            m_buf["rew"].append(macro_rews[None, :] if num_envs == 1 else macro_rews)
            m_buf["done"].append(mx.full((1,) if num_envs == 1 else (num_envs,), float(done)))

        timesteps += num_envs
        if done:
            obs, _ = env.reset(seed=timesteps + 17)
            goals = mx.zeros((num_envs, num_symbols)) if num_envs > 1 else mx.zeros((num_symbols,))
        else:
            obs = next_obs
        t0 = time.perf_counter()
        mx.eval(obs, rew)
        times["mlx_sync"] += time.perf_counter() - t0

    total_wall = max(time.perf_counter() - t_start, 1e-6)
    pcts = {k: (v / total_wall) * 100.0 for k, v in times.items()}
    print("\n" + "=" * 60)
    print(f"Profile Breakdown (n_envs={num_envs}, {timesteps} steps in {total_wall:.2f}s, {timesteps/total_wall:.1f} FPS)")
    print("=" * 60)
    for k, v in sorted(pcts.items(), key=lambda x: -x[1]):
        print(f"{k:<22} : {v:6.2f}% | {'#' * int(v / 2)}")
    print("=" * 60)
    return {"wall_time": total_wall, "fps": timesteps / total_wall, "breakdown_pct": pcts}


def run_benchmark(start_envs: int = 1, max_envs: int = 1024, steps: int = 100, steps_per_env: int | None = None, plateau_tol: float = 0.03, plateau_patience: int = 2, swap_thresh_mb: float = 50.0, num_symbols: int = 4, num_candles: int = 300, csv_out: str | None = None) -> list[dict]:
    n_steps = steps_per_env or steps
    print("Diet research please... Starting Benchmark: Doubling num_envs until FPS plateaus or swap is hit.")
    base_swap = get_system_swap_mb()
    results, best_train_fps, plateau_cnt, num_envs = [], 0.0, 0, start_envs
    hdr = f"{'Envs':>6} | {'Env FPS':>11} | {'Train FPS':>11} | {'MLX Peak(MB)':>13} | {'RSS(MB)':>10} | {'Swap(MB)':>10} | {'Status':<15}"
    print("-" * len(hdr) + "\n" + hdr + "\n" + "-" * len(hdr))
    stop_reason = ""

    while num_envs <= max_envs:
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        env_fps = bench_env_fps(num_envs=num_envs, steps=n_steps, num_symbols=num_symbols, num_candles=num_candles)
        train_fps = bench_train_fps(num_envs=num_envs, steps=n_steps, num_symbols=num_symbols, num_candles=num_candles)
        mem = get_memory_stats(base_swap=base_swap)

        if train_fps > best_train_fps * (1.0 + plateau_tol):
            best_train_fps, plateau_cnt, status = train_fps, 0, "scaling"
        else:
            plateau_cnt += 1
            status = f"plateau ({plateau_cnt}/{plateau_patience})"

        row = {"num_envs": num_envs, "env_fps": env_fps, "train_fps": train_fps, "mlx_active_mb": mem["mlx_active_mb"], "mlx_peak_mb": mem["mlx_peak_mb"], "rss_mb": mem["rss_mb"], "swap_used_mb": mem["swap_used_mb"], "swap_delta_mb": mem["swap_delta_mb"], "status": status}
        results.append(row)
        print(f"{num_envs:6d} | {env_fps:11.1f} | {train_fps:11.1f} | {mem['mlx_peak_mb']:13.2f} | {mem['rss_mb']:10.2f} | {mem['swap_used_mb']:10.2f} | {status:<15}")

        if mem["swap_delta_mb"] >= swap_thresh_mb:
            stop_reason = f"Swap limit exceeded (+{mem['swap_delta_mb']:.1f} MB >= {swap_thresh_mb:.1f} MB)"
            break
        if plateau_cnt >= plateau_patience:
            stop_reason = f"FPS plateaued across {plateau_patience} consecutive doublings"
            break
        num_envs *= 2

    if not stop_reason and num_envs > max_envs:
        stop_reason = f"Reached max_envs ceiling ({max_envs})"
    print("-" * len(hdr) + f"\nDiet research please... Benchmark complete: {stop_reason}")

    if csv_out and results:
        import csv
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    return results


def main():
    p = argparse.ArgumentParser(description="Benchmark Pure MLX Env & Training FPS and Memory scaling.")
    p.add_argument("--start_envs", type=int, default=1)
    p.add_argument("--max_envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--plateau_tol", type=float, default=0.03)
    p.add_argument("--plateau_patience", type=int, default=2)
    p.add_argument("--swap_thresh_mb", type=float, default=50.0)
    p.add_argument("--num_symbols", type=int, default=4)
    p.add_argument("--num_candles", type=int, default=300)
    p.add_argument("--csv_out", type=str, default=None)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile_envs", type=int, default=256)
    args = p.parse_args()

    if args.profile:
        profile_breakdown(num_envs=args.profile_envs, steps=args.steps, num_symbols=args.num_symbols, num_candles=args.num_candles)
    else:
        run_benchmark(start_envs=args.start_envs, max_envs=args.max_envs, steps=args.steps, plateau_tol=args.plateau_tol, plateau_patience=args.plateau_patience, swap_thresh_mb=args.swap_thresh_mb, num_symbols=args.num_symbols, num_candles=args.num_candles, csv_out=args.csv_out)


if __name__ == "__main__":
    main()
