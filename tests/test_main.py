import csv
import sys
from pathlib import Path
import pytest
import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.main import create_agents, train_hrl, evaluate_hrl, parse_args, main
from scripts.report import create_run_dir, TradeHistoryLogger, generate_breakdown_report, TRADE_HISTORY_COLUMNS
from scripts.visualize import generate_trade_figures


def test_create_agents():
    ppo_manager, sac_worker = create_agents(obs_dim=80, num_symbols=4)
    assert ppo_manager.obs_dim == 80
    assert ppo_manager.act_dim == 20  # 4 symbols * 5 (side, leverage, collateral, tp, sl)
    assert sac_worker.obs_dim == 100  # 80 + 20
    assert sac_worker.act_dim == 20   # 4 symbols * 5


def test_train_hrl_joint_and_alternating():
    for scheme in ["joint", "alternating"]:
        ppo_manager, sac_worker = train_hrl(
            total_timesteps=40,
            num_envs=2,
            train_scheme=scheme,
            margin_mode="isolated",
            num_symbols=3,
            num_candles=50,
            macro_period=5,
        )
        assert ppo_manager is not None
        assert sac_worker is not None


def test_evaluate_hrl():
    ppo_manager, sac_worker = create_agents(obs_dim=63, num_symbols=3)
    results = evaluate_hrl(
        ppo_manager=ppo_manager,
        sac_worker=sac_worker,
        num_episodes=2,
        margin_mode="cross",
        num_symbols=3,
        num_candles=50,
        macro_period=5,
    )
    assert len(results) == 2
    assert "martin_ratio" in results[0]
    assert "equity" in results[0]


def test_trade_history_columns_schema(tmp_path):
    log_path = tmp_path / "trade_history.csv"
    logger = TradeHistoryLogger(log_path, run_id="test_run", seed=42)
    logger.log_trade({
        "symbol": "BTC",
        "side": "buy",
        "position_effect": "open",
        "quantity": 0.5,
        "price": 50000.0,
        "notional_value": 25000.0,
        "leverage": 10.0,
        "net_pnl": 120.0,
    })

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        row = next(reader)

    assert headers == TRADE_HISTORY_COLUMNS
    # Check expected core columns from docs/reporting.md
    expected_sample_cols = [
        "trade_id", "timestamp", "timestamp_utc", "symbol", "exchange", "venue", "strategy_id", "run_id",
        "side", "position_effect", "quantity", "filled_quantity", "price", "notional_value", "leverage",
        "order_type", "liquidity_flag", "fee_amount", "fee_currency", "funding_fee", "spread_cost",
        "slippage_bps", "liquidation_fee", "liquidation_penalty", "realized_pnl", "unrealized_pnl",
        "gross_pnl", "net_pnl", "pnl_pct", "return_on_margin", "return_on_equity", "margin_type",
        "initial_margin", "maintenance_margin", "margin_used", "free_margin", "liquidation_price",
        "distance_to_liquidation_pct", "equity_before", "equity_after", "mark_price", "index_price",
        "oracle_price", "funding_rate", "funding_payment", "open_interest", "model_version",
        "policy_version", "config_hash", "git_commit", "seed", "data_version", "cost_basis_method"
    ]
    for col in expected_sample_cols:
        assert col in headers, f"Missing required column from reporting.md: {col}"


def test_simulation_run_dir_and_files(tmp_path, monkeypatch):
    run_dir = create_run_dir(base_dir=tmp_path)
    assert run_dir.exists()
    assert run_dir.is_dir()

    ppo_manager, sac_worker = train_hrl(
        total_timesteps=40,
        num_envs=2,
        train_scheme="joint",
        num_symbols=3,
        num_candles=40,
        macro_period=5,
        log_dir=run_dir,
    )

    evaluate_hrl(
        ppo_manager=ppo_manager,
        sac_worker=sac_worker,
        num_episodes=2,
        num_symbols=3,
        num_candles=40,
        macro_period=5,
        log_dir=run_dir,
    )

    ppo_file = run_dir / "ppo_manager.csv"
    sac_file = run_dir / "sac_worker.csv"
    trade_file = run_dir / "trade_history.csv"
    breakdown_file = run_dir / "breakdown.txt"
    perf_fig = run_dir / "trade_performance_synthwave.png"
    dist_fig = run_dir / "trade_distributions_synthwave.png"

    assert ppo_file.exists(), "ppo_manager.csv must exist in run dir"
    assert sac_file.exists(), "sac_worker.csv must exist in run dir"
    assert trade_file.exists(), "trade_history.csv must exist in run dir"
    assert breakdown_file.exists(), "breakdown.txt must exist in run dir"
    assert perf_fig.exists() and perf_fig.stat().st_size > 0, "trade performance 2x2 figure required after breakdown"
    assert dist_fig.exists() and dist_fig.stat().st_size > 0, "trade distributions 2x2 figure required after breakdown"

    with open(ppo_file, "r", encoding="utf-8") as f:
        ppo_rows = list(csv.reader(f))
        assert len(ppo_rows) > 1, "ppo_manager.csv should record training updates"

    with open(sac_file, "r", encoding="utf-8") as f:
        sac_rows = list(csv.reader(f))
        assert len(sac_rows) > 1, "sac_worker.csv should record training steps"

    with open(trade_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers == TRADE_HISTORY_COLUMNS
        rows = list(reader)
        assert len(rows) > 0, "trade_history.csv should have recorded trades during testing"

    with open(breakdown_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert "PER SYMBOL BREAKDOWN" in content
        assert "PER EPISODE BREAKDOWN" in content
        assert "Portfolio" in content
        for col_name in ["key (symbol, episode)", "num trades", "win rate %", "avg win", "avg loss", "return %", "net profit", "sharpe", "max dd", "risk reward", "sortino", "calmar", "profit factor", "martin"]:
            assert col_name in content


def test_breakdown_report_with_empty_and_custom_trades(tmp_path):
    empty_csv = tmp_path / "empty_trade_history.csv"
    TradeHistoryLogger(empty_csv)
    out_txt = tmp_path / "breakdown_empty.txt"
    report_empty = generate_breakdown_report(empty_csv, out_txt)
    assert out_txt.exists()
    assert "Portfolio" in report_empty
    assert "n/a" in report_empty

    # Custom trade with zero losses
    single_csv = tmp_path / "single_trade.csv"
    logger = TradeHistoryLogger(single_csv)
    logger.log_trade({"symbol": "BTC", "strategy_id": "Episode 1", "net_pnl": 150.0, "pnl_pct": 1.5})
    out_single = tmp_path / "breakdown_single.txt"
    report_single = generate_breakdown_report(single_csv, out_single)
    assert "BTC" in report_single
    assert "Episode 1" in report_single
    assert "Portfolio" in report_single
    assert "+150.00" in report_single


@pytest.mark.parametrize("theme", ["synthwave", "ghibli"])
def test_trade_figures_after_breakdown(tmp_path, theme):
    csv_path = tmp_path / "trade_history.csv"
    logger = TradeHistoryLogger(csv_path)
    logger.log_trade({"symbol": "BTC", "side": "long", "position_effect": "close", "exit_type": "take_profit",
                      "net_pnl": 100.0, "leverage": 10, "margin_used": 200})
    logger.log_trade({"symbol": "ETH", "side": "short", "position_effect": "close", "exit_type": "stop_loss",
                      "net_pnl": -50.0, "leverage": 25, "margin_used": 80})
    generate_breakdown_report(csv_path, tmp_path / "breakdown.txt")
    perf, dist = generate_trade_figures(csv_path, out_dir=tmp_path, theme=theme, initial_capital=10000.0)
    assert perf.exists() and perf.stat().st_size > 1000
    assert dist.exists() and dist.stat().st_size > 1000
    assert theme in perf.name and theme in dist.name


@pytest.mark.parametrize("high_tf,low_tf", [
    ("1h", "5m"),
    ("4h", "15m"),
    ("1d", "1h"),
])
def test_multi_timeframe_hrl_execution(high_tf, low_tf, tmp_path):
    """Verifies PPO Manager (high TF) + SAC Worker (low TF) across various Binance timeframes."""
    run_dir = tmp_path / f"run_{high_tf}_{low_tf}"
    ppo_mgr, sac_wkr = train_hrl(
        total_timesteps=20,
        num_envs=1,
        train_scheme="joint",
        high_tf=high_tf,
        low_tf=low_tf,
        num_symbols=2,
        num_candles=30,
        log_dir=run_dir,
    )
    assert ppo_mgr is not None
    assert sac_wkr is not None

    results = evaluate_hrl(
        ppo_mgr,
        sac_wkr,
        num_episodes=1,
        high_tf=high_tf,
        low_tf=low_tf,
        num_symbols=2,
        num_candles=30,
        log_dir=run_dir,
    )
    assert len(results) == 1
    assert "martin_ratio" in results[0]
    assert "equity" in results[0]


def test_statistically_significant_agent_csv_logging(tmp_path):
    """Verifies that PPO and SAC training outputs statistically rich CSV logs."""
    run_dir = tmp_path / "stat_run"
    train_hrl(
        total_timesteps=600,
        num_envs=2,
        train_scheme="joint",
        num_symbols=2,
        num_candles=60,
        macro_period=4,
        log_dir=run_dir,
    )
    ppo_file = run_dir / "ppo_manager.csv"
    sac_file = run_dir / "sac_worker.csv"

    assert ppo_file.exists()
    assert sac_file.exists()

    with open(ppo_file, "r", encoding="utf-8") as f:
        ppo_rows = list(csv.DictReader(f))
        assert len(ppo_rows) >= 5, f"PPO rows ({len(ppo_rows)}) should reflect frequent updates"
        assert all("train/policy_loss" in r for r in ppo_rows)

    with open(sac_file, "r", encoding="utf-8") as f:
        sac_rows = list(csv.DictReader(f))
        assert len(sac_rows) >= 50, f"SAC rows ({len(sac_rows)}) should reflect high frequency worker steps"
        assert all("train/critic_loss" in r for r in sac_rows)


def test_custom_config_cli(monkeypatch, tmp_path):
    """Verifies that --config correctly loads custom base configuration."""
    custom_cfg_path = tmp_path / "custom_config.yaml"
    custom_cfg_content = """
data:
  num_candles: 40
  num_symbols: 2
  features: ["open", "high", "low", "close", "volume", "funding", "regime", "log_ret", "sma20", "ema12", "ema26", "rsi14", "macd", "macd_sig", "macd_hist", "bb_upper", "bb_lower", "bb_pct", "atr14"]
  indicators:
    sma_window: 20
    ema_fast: 12
    ema_slow: 26
    macd_signal: 9
    rsi_window: 14
    bb_window: 20
    bb_k: 2.0
    atr_window: 14
  regimes:
    names: ["Bull", "Bear", "Range", "Crash", "Mania"]
    mu: [0.0005, -0.0005, 0.0, -0.003, 0.003]
    sigma: [0.015, 0.025, 0.008, 0.060, 0.045]
    funding_mu: [0.0003, -0.0002, 0.00005, -0.001, 0.0015]
    transition_matrix:
      - [0.97, 0.01, 0.01, 0.005, 0.005]
      - [0.01, 0.97, 0.01, 0.010, 0.000]
      - [0.02, 0.02, 0.95, 0.005, 0.005]
      - [0.05, 0.05, 0.05, 0.850, 0.000]
      - [0.02, 0.05, 0.03, 0.050, 0.850]
  ou:
    theta_f: 0.1
    sigma_f: 0.0005
simulation:
  high_tf: "1h"
  low_tf: "5m"
  macro_period: 12
  num_candles: 40
  num_symbols: 2
env:
  margin_mode: "cross"
  initial_capital: 5000.0
"""
    custom_cfg_path.write_text(custom_cfg_content, encoding="utf-8")
    run_dir = tmp_path / "custom_run"

    test_args = [
        "main.py",
        "--config", str(custom_cfg_path),
        "--timesteps", "20",
        "--episodes", "1",
        "--num_envs", "1",
        "--log_dir", str(run_dir),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()

    assert (run_dir / "trade_history.csv").exists()
    assert (run_dir / "breakdown.txt").exists()


