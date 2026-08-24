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

All hyperparameters live in [`configs/config.yaml`](configs/config.yaml):
data generation, agent hyperparameters, simulation, environment (margin,
leverage, risk), and visualization settings.

## Tests

```bash
pytest
```

## Documentation

See [`docs/`](docs/) for architecture, backtesting validation, and research notes.

## License

TBD
