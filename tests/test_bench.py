import sys
from pathlib import Path
import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.bench import bench_env_fps, bench_train_fps, get_memory_stats, get_system_swap_mb, run_benchmark


def test_memory_stats_and_swap():
    swap_mb = get_system_swap_mb()
    assert isinstance(swap_mb, float)
    assert swap_mb >= 0.0

    mem = get_memory_stats(base_swap=swap_mb)
    assert "rss_mb" in mem
    assert "mlx_active_mb" in mem
    assert "mlx_peak_mb" in mem
    assert "swap_used_mb" in mem
    assert "swap_delta_mb" in mem
    assert mem["rss_mb"] > 0.0


def test_bench_env_fps():
    fps = bench_env_fps(num_envs=2, steps_per_env=10, num_symbols=2, num_candles=20)
    assert isinstance(fps, float)
    assert fps > 0.0


def test_bench_train_fps():
    fps = bench_train_fps(num_envs=2, steps_per_env=10, num_symbols=2, num_candles=20)
    assert isinstance(fps, float)
    assert fps > 0.0


def test_run_benchmark(tmp_path):
    csv_file = str(tmp_path / "bench_results.csv")
    results = run_benchmark(
        start_envs=1,
        max_envs=4,
        steps_per_env=10,
        num_symbols=2,
        num_candles=20,
        plateau_patience=2,
        csv_out=csv_file,
    )
    assert len(results) >= 1
    assert Path(csv_file).exists()
    assert "env_fps" in results[0]
    assert "train_fps" in results[0]
    assert "mlx_peak_mb" in results[0]
    assert "rss_mb" in results[0]
