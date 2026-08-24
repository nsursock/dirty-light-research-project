# Dirty Light Research Project

Multi-crypto trading bot built with hierarchical reinforcement learning (HRL).
A **PPO Manager** trades on the high timeframe (e.g. `1h`) while **SAC Workers**
trade the low timeframe (e.g. `5m`), sharing goals through a vectorized,
leverage-aware DEX perpetuals environment. The hot path runs on **pure MLX**
on Apple Silicon.

## Features

- Hierarchical RL: PPO manager (high TF) + SAC workers (low TF)
- Vectorized multi-env, multi-symbol training with pure MLX
- Synthetic market data with regime-switching models (Bull, Bear, Range, Crash, Mania)
- Binance klines support for real market data
- Leverage-aware environment: isolated/cross margin, liquidation, funding fees, slippage
- Trade history logging and per-symbol breakdown reports
- Plotly visualizations and training benchmarks

## Requirements

- Python 3.11
- [MLX](https://github.com/ml-explore/mlx) (Apple Silicon)

```
pip install -r requirements.txt
```

## Usage

```bash
python scripts/main.py --mode train --timesteps 3000 --num_envs 2
```

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--mode` | `train`, `test`, or `full` | `full` |
| `--timesteps`, `-t` | Training timesteps | `2000` |
| `--episodes`, `-e` | Testing episodes | `5` |
| `--num_envs`, `-n` | Parallel environments | `2` |
| `--train_scheme` | `joint` or `alternating` | `joint` |
| `--margin_mode` | `isolated` or `cross` | `isolated` |
| `--high_tf` | High timeframe for PPO Manager | `1h` |
| `--low_tf` | Low timeframe for SAC Worker | `5m` |
| `--num_symbols` | Number of crypto assets | `4` |
| `--num_candles` | Candles per episode | `300` |

## Project Structure

```
scripts/    Training, agents, environment, data, reporting, visualization
utils/      Benchmarking utilities
tests/      Pytest suite
configs/    YAML configuration
docs/       Architecture and research notes
```

## Configuration

Base hyperparameters live in [`configs/config.yaml`](configs/config.yaml):
data generation, agent hyperparameters, simulation, environment (margin,
leverage, risk), and visualization settings.

**Training stages** (budget only; env/data stay fixed) live in `configs/stages/`:

| Stage | Purpose | Timesteps | Envs | Eval eps | Seeds |
|-------|---------|----------:|-----:|---------:|------:|
| S0 smoke | correctness | 250k | 64 | 10 | 1 |
| S1 baseline | learning sanity | 1M | 256 | 50 | 3 |
| S2 research | comparison | 5M | 256 | 100 | 3 |
| S3 final | validation | 10M | 256 | 500 | 5 |

```bash
python scripts/main.py --stage S0 --mode full
python scripts/main.py --stage S1 --mode train
python scripts/main.py --stage configs/stages/S2_research.yaml -t 100000  # CLI still overrides
```

## Benchmarks

Vectorized environment throughput and end-to-end training FPS scale efficiently with pure MLX on Apple Silicon:

```bash
python utils/bench.py
```

| Envs | Env FPS | Train FPS | MLX Peak (MB) | RSS (MB) | Swap (MB) | Status |
| ---: | ------: | --------: | ------------: | -------: | --------: | :----- |
| 1 | 192.2 | 146.3 | 91.73 | 117.31 | 0.00 | scaling |
| 2 | 2,933.8 | 764.6 | 181.34 | 120.64 | 0.00 | scaling |
| 4 | 6,249.9 | 1,385.3 | 270.26 | 123.30 | 0.00 | scaling |
| 8 | 11,462.1 | 2,631.1 | 359.06 | 126.22 | 0.00 | scaling |
| 16 | 23,057.5 | 4,710.4 | 355.09 | 126.84 | 0.00 | scaling |
| 32 | 46,755.6 | 8,112.6 | 181.48 | 127.72 | 0.00 | scaling |
| 64 | 102,333.8 | 11,926.4 | 270.62 | 129.06 | 0.00 | scaling |
| 128 | 200,262.8 | 18,818.1 | 359.98 | 138.47 | 0.00 | scaling |
| 256 | 365,683.8 | 23,701.9 | 448.70 | 153.95 | 0.00 | scaling |
| 512 | 719,534.4 | 26,965.2 | 183.26 | 179.33 | 0.00 | scaling |
| 1024 | 1,401,314.4 | 28,455.8 | 186.84 | 246.56 | 0.00 | scaling |
| 2048 | 2,634,987.3 | 27,732.7 | 281.98 | 380.84 | 0.00 | plateau (1/2) |
| 4096 | 4,138,539.0 | 27,675.5 | 206.15 | 643.06 | 0.00 | plateau (2/2) |

- **Peak Env Step Throughput:** `4.14M+ FPS` at `num_envs=4096`
- **Peak Training Throughput:** `28,455.8 FPS` at `num_envs=1024`
- **Memory Footprint:** Peak RSS < 650 MB, 0 MB swap utilized across 4096 parallel environments

## Tests

```bash
pytest
```

## Documentation

See [`docs/`](docs/) for architecture, backtesting validation, and research notes.

## License

TBD
