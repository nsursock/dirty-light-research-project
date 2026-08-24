import sys
from pathlib import Path
import pytest
import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.env import MultiCryptoDexPerpEnv
from scripts.data import generate_crypto_data, FEATURE_NAMES


def test_env_initialization_and_params():
    env = MultiCryptoDexPerpEnv(num_symbols=3, num_candles=100)
    obs, info = env.reset()
    assert obs.ndim == 1
    assert info["equity"] == 10000.0
    assert info["drawdown"] == 0.0
    assert info["margin_mode"] == "isolated"
    assert info["total_liquidations"] == 0
    assert env.min_leverage == 2.0
    assert env.max_leverage == 150.0
    assert env.min_risk == 0.01
    assert env.max_risk == 0.05
    assert env.min_collateral == 10.0
    assert env.max_collateral == 1000.0
    assert env.min_tp_pct == 0.01
    assert env.max_tp_pct == 0.15
    assert env.min_sl_pct == 0.005
    assert env.max_sl_pct == 0.05


def test_env_custom_param_sets():
    env = MultiCryptoDexPerpEnv(
        num_symbols=2, num_candles=50, min_leverage=5.0, max_leverage=50.0,
        min_risk_per_trade=0.02, max_risk_per_trade=0.04, min_collateral=20.0, max_collateral=500.0,
        min_take_profit_pct=0.02, max_take_profit_pct=0.20,
        min_stop_loss_pct=0.01, max_stop_loss_pct=0.08,
    )
    assert env.min_leverage == 5.0
    assert env.max_leverage == 50.0
    assert env.min_risk == 0.02
    assert env.max_risk == 0.04
    assert env.min_collateral == 20.0
    assert env.max_collateral == 500.0
    assert env.min_tp_pct == 0.02
    assert env.max_tp_pct == 0.20
    assert env.min_sl_pct == 0.01
    assert env.max_sl_pct == 0.08


def test_env_margin_modes():
    env_iso = MultiCryptoDexPerpEnv(num_symbols=3, num_candles=50, margin_mode="isolated")
    _, info_iso = env_iso.reset()
    assert info_iso["margin_mode"] == "isolated"

    env_cross = MultiCryptoDexPerpEnv(num_symbols=3, num_candles=50, margin_mode="cross")
    _, info_cross = env_cross.reset()
    assert info_cross["margin_mode"] == "cross"

    with pytest.raises(AssertionError):
        MultiCryptoDexPerpEnv(num_symbols=3, num_candles=50, margin_mode="invalid_mode")


def test_env_step_mechanics():
    for mode in ["isolated", "cross"]:
        env = MultiCryptoDexPerpEnv(num_symbols=4, num_candles=50, margin_mode=mode)
        obs, info = env.reset()
        action = mx.array([0.5, -0.5, 0.2, -0.1])
        next_obs, rew, done, truncated, info = env.step(action)
        assert next_obs.shape == obs.shape
        assert isinstance(rew, float)
        assert isinstance(done, bool)
        assert "martin_ratio" in info
        assert "ulcer_index" in info
        assert info["margin_mode"] == mode


def test_cost_model_and_fees():
    env = MultiCryptoDexPerpEnv(num_symbols=2, num_candles=50, fee_rate=0.001, slippage_coef=0.0)
    env.reset()
    act1 = mx.array([0.5, -0.5])
    _, _, _, _, info1 = env.step(act1)
    assert info1["costs"] > 0.0


def test_episode_done_on_candle_end():
    num_candles = 35
    env = MultiCryptoDexPerpEnv(num_symbols=2, num_candles=num_candles)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _, _ = env.step(mx.array([0.1, -0.1]))
        steps += 1
    assert steps == num_candles - 1
    assert done is True


def test_liquidation_and_bankruptcy():
    for mode in ["isolated", "cross"]:
        env = MultiCryptoDexPerpEnv(
            num_symbols=2, num_candles=100, initial_capital=100.0,
            min_leverage=10.0, max_leverage=100.0, maintenance_margin_rate=0.2, margin_mode=mode,
        )
        env.reset()
        done = False
        for _ in range(50):
            _, _, done, _, info = env.step(mx.array([1.0, -1.0]))
            if done:
                break
        assert env.equity <= env.initial_capital * 10.0


def test_pessimistic_fills_buy_and_sell():
    """Validates that fills are pessimistically priced against High/Low and slippage."""
    # Custom 2-step candle data: open=100, high=105, low=95, close=100, vol=1000, funding=0
    candle_0 = [100.0, 105.0, 95.0, 100.0, 1000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    candle_1 = [100.0, 105.0, 95.0, 100.0, 1000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    custom_data = mx.array([[candle_0, candle_0], [candle_1, candle_1]])

    env = MultiCryptoDexPerpEnv(data=custom_data, num_symbols=2, num_candles=2, slippage_coef=0.001, stop_loss_pct=0.0, take_profit_pct=0.0)
    env.reset()

    # Symbol 0 buys (+1.0), Symbol 1 sells (-1.0)
    _, _, _, _, info = env.step(mx.array([1.0, -1.0]))
    trades = info["trades"]
    buy_trade = [t for t in trades if t["side"] == "buy"][0]
    sell_trade = [t for t in trades if t["side"] == "sell"][0]

    # Buyer should pay higher than close (up to high)
    assert buy_trade["price"] >= 100.0
    assert buy_trade["price"] <= 105.0
    # Seller should receive lower than close (down to low)
    assert sell_trade["price"] <= 100.0
    assert sell_trade["price"] >= 95.0


def test_low_tf_liquidation_on_bar_extremes():
    """Validates that liquidation is triggered on intra-bar Low (long) or High (short) even if Close recovers."""
    # Long position opened at 100. Next bar drops to 80 (low) but closes at 99 (close).
    # With 10x leverage and MMR=0.05, liq_price is ~95. Low of 80 must trigger liquidation!
    c0 = [100.0, 102.0, 98.0, 100.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    c1 = [100.0, 101.0, 80.0, 99.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    custom_data = mx.array([[c0], [c1]])

    env = MultiCryptoDexPerpEnv(
        data=custom_data, num_symbols=1, num_candles=2,
        min_leverage=10.0, max_leverage=10.0, maintenance_margin_rate=0.05, stop_loss_pct=0.0,
    )
    env.reset()
    _, _, _, _, info = env.step(mx.array([1.0]))
    assert info["liquidated"] is True
    assert info["total_liquidations"] == 1


def test_take_profit_and_stop_loss_execution():
    """Validates take-profit and stop-loss intra-bar execution."""
    # Bar 0: price at 100
    # Bar 1: high reaches 110, low at 99 -> hits 5% TP (TP price 105)
    c0 = [100.0, 102.0, 98.0, 100.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    c1 = [100.0, 110.0, 99.0, 102.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    custom_data = mx.array([[c0], [c1]])

    env = MultiCryptoDexPerpEnv(
        data=custom_data, num_symbols=1, num_candles=2,
        min_leverage=1.0, max_leverage=1.0,
        take_profit_pct=0.05, stop_loss_pct=0.03,
    )
    env.reset()
    _, _, _, _, info = env.step(mx.array([1.0]))
    assert info["total_tp_hits"] == 1
    assert info["total_sl_hits"] == 0


def test_perp_pnl_tp_sl_scaling_with_leverage():
    """Validates that TP/SL are based on perp PnL % scaled by leverage (e.g. 50% TP at 10x is +5% underlying)."""
    # 1. Long at 10x leverage with 50% TP (requires +5% price move)
    c0 = [100.0, 100.0, 100.0, 100.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    c1 = [100.0, 105.5, 99.0, 104.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    custom_data = mx.array([[c0], [c1]])

    env_long = MultiCryptoDexPerpEnv(
        data=custom_data, num_symbols=1, num_candles=2,
        min_leverage=10.0, max_leverage=10.0,
        take_profit_pct=0.50, stop_loss_pct=0.20,
    )
    env_long.reset()
    _, _, _, _, info_long = env_long.step(mx.array([1.0]))
    assert info_long["total_tp_hits"] == 1
    assert info_long["total_sl_hits"] == 0

    # 2. Short at 10x leverage with 20% SL (requires +2% price move up)
    c1_short = [100.0, 102.5, 99.0, 101.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    custom_data_short = mx.array([[c0], [c1_short]])
    env_short = MultiCryptoDexPerpEnv(
        data=custom_data_short, num_symbols=1, num_candles=2,
        min_leverage=10.0, max_leverage=10.0,
        take_profit_pct=0.50, stop_loss_pct=0.20,
    )
    env_short.reset()
    _, _, _, _, info_short = env_short.step(mx.array([-1.0]))
    assert info_short["total_sl_hits"] == 1
    assert info_short["total_tp_hits"] == 0


def test_interpolated_liquidation_thresholds():
    """Validates 2x (-90% perp PnL) and 150x (-67% perp PnL) interpolated liquidation thresholds."""
    # 1. 2x leverage: 90% perp PnL loss -> 45% underlying price drop (entry 100 -> liq at 55.0)
    c0 = [100.0, 100.0, 100.0, 100.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    c1_safe_2x = [100.0, 100.0, 56.0, 60.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    c1_liq_2x = [100.0, 100.0, 54.0, 60.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)

    env_safe_2x = MultiCryptoDexPerpEnv(
        data=mx.array([[c0], [c1_safe_2x]]), num_symbols=1, num_candles=2,
        min_leverage=2.0, max_leverage=2.0, stop_loss_pct=0.0, take_profit_pct=0.0,
    )
    env_safe_2x.reset()
    _, _, _, _, info_safe_2x = env_safe_2x.step(mx.array([1.0]))
    assert info_safe_2x["total_liquidations"] == 0

    env_liq_2x = MultiCryptoDexPerpEnv(
        data=mx.array([[c0], [c1_liq_2x]]), num_symbols=1, num_candles=2,
        min_leverage=2.0, max_leverage=2.0, stop_loss_pct=0.0, take_profit_pct=0.0,
    )
    env_liq_2x.reset()
    _, _, _, _, info_liq_2x = env_liq_2x.step(mx.array([1.0]))
    assert info_liq_2x["total_liquidations"] == 1

    # 2. 150x leverage: 67% perp PnL loss -> 0.4467% underlying price drop (entry 100 -> liq at 99.5533)
    c1_safe_150x = [100.0, 100.0, 99.60, 99.70, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    c1_liq_150x = [100.0, 100.0, 99.50, 99.70, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)

    env_safe_150x = MultiCryptoDexPerpEnv(
        data=mx.array([[c0], [c1_safe_150x]]), num_symbols=1, num_candles=2,
        min_leverage=150.0, max_leverage=150.0, stop_loss_pct=0.0, take_profit_pct=0.0,
    )
    env_safe_150x.reset()
    _, _, _, _, info_safe_150x = env_safe_150x.step(mx.array([1.0]))
    assert info_safe_150x["total_liquidations"] == 0

    env_liq_150x = MultiCryptoDexPerpEnv(
        data=mx.array([[c0], [c1_liq_150x]]), num_symbols=1, num_candles=2,
        min_leverage=150.0, max_leverage=150.0, stop_loss_pct=0.0, take_profit_pct=0.0,
    )
    env_liq_150x.reset()
    _, _, _, _, info_liq_150x = env_liq_150x.step(mx.array([1.0]))
    assert info_liq_150x["total_liquidations"] == 1

    # 3. Custom configurable parameters (e.g. 2x liquidates at 80% loss -> liq_dist = 40% -> liq_price = 60.0)
    c1_safe_custom = [100.0, 100.0, 61.0, 70.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    c1_liq_custom = [100.0, 100.0, 59.0, 70.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)

    env_safe_custom = MultiCryptoDexPerpEnv(
        data=mx.array([[c0], [c1_safe_custom]]), num_symbols=1, num_candles=2,
        min_leverage=2.0, max_leverage=2.0, liq_threshold_low_pct=0.80, stop_loss_pct=0.0, take_profit_pct=0.0,
    )
    env_safe_custom.reset()
    _, _, _, _, info_safe_custom = env_safe_custom.step(mx.array([1.0]))
    assert info_safe_custom["total_liquidations"] == 0

    env_liq_custom = MultiCryptoDexPerpEnv(
        data=mx.array([[c0], [c1_liq_custom]]), num_symbols=1, num_candles=2,
        min_leverage=2.0, max_leverage=2.0, liq_threshold_low_pct=0.80, stop_loss_pct=0.0, take_profit_pct=0.0,
    )
    env_liq_custom.reset()
    _, _, _, _, info_liq_custom = env_liq_custom.step(mx.array([1.0]))
    assert info_liq_custom["total_liquidations"] == 1


def test_pessimistic_priority_sl_over_tp():
    """When both SL and TP price levels are within the Low-High range, pessimistic POV assumes SL is hit first."""
    # Bar 0: price at 100
    # Bar 1: high=110 (hits 5% TP @ 105), low=90 (hits 3% SL @ 97, above 1x liq price 50)
    c0 = [100.0, 102.0, 98.0, 100.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    c1 = [100.0, 110.0, 90.0, 100.0, 10000.0, 0.0] + [0.0] * (len(FEATURE_NAMES) - 6)
    custom_data = mx.array([[c0], [c1]])

    env = MultiCryptoDexPerpEnv(
        data=custom_data, num_symbols=1, num_candles=2,
        min_leverage=1.0, max_leverage=1.0,
        take_profit_pct=0.05, stop_loss_pct=0.03,
    )
    env.reset()
    _, _, _, _, info = env.step(mx.array([1.0]))
    # Pessimistic: SL must be triggered, not TP!
    assert info["total_sl_hits"] == 1
    assert info["total_tp_hits"] == 0


def test_env_5d_action_side_leverage_collateral_tp_sl():
    """Validates 5D action space per symbol: side (long/short/flat), leverage, collateral, tp, and sl."""
    env = MultiCryptoDexPerpEnv(
        num_symbols=3, num_candles=10,
        min_leverage=2.0, max_leverage=10.0,
        min_collateral=100.0, max_collateral=500.0,
        min_take_profit_pct=0.02, max_take_profit_pct=0.10,
        min_stop_loss_pct=0.01, max_stop_loss_pct=0.05,
    )
    obs, info = env.reset()

    # Symbol 0: Long (+1.0), leverage max (+1.0 -> 10x), col max (+1.0 -> 500), tp max (+1.0 -> 10%), sl min (-1.0 -> 1%)
    # Symbol 1: Short (-1.0), leverage min (-1.0 -> 2x), col min (-1.0 -> 100), tp min (-1.0 -> 2%), sl max (+1.0 -> 5%)
    # Symbol 2: Flat (0.0), any other parameters
    action_5d = mx.array([
        [1.0, 1.0, 1.0, 1.0, -1.0],      # Long
        [-1.0, -1.0, -1.0, -1.0, 1.0],   # Short
        [0.0, 0.0, 0.0, 0.0, 0.0],       # Flat
    ])
    next_obs, rew, done, _, info = env.step(action_5d)
    assert next_obs.shape == obs.shape
    assert isinstance(rew, float)

    # Symbol 0 should be long notional > 0, Symbol 1 short notional < 0, Symbol 2 flat == 0
    assert env.notionals[0, 0] > 0.0
    assert env.notionals[0, 1] < 0.0
    assert abs(float(env.notionals[0, 2].item())) < 1e-4


def test_env_falsy_param_handling():
    """Verifies that 0.0 / False values are explicitly honored and not replaced by config defaults."""
    env = MultiCryptoDexPerpEnv(
        num_symbols=2, num_candles=10,
        fee_rate=0.0, slippage_coef=0.0,
        take_profit_pct=0.0, stop_loss_pct=0.0,
    )
    assert env.fee_rate == 0.0
    assert env.slippage_coef == 0.0
    assert env.min_tp_pct == 0.0
    assert env.max_tp_pct == 0.0
    assert env.min_sl_pct == 0.0
    assert env.max_sl_pct == 0.0


def test_env_risk_fraction_collateral_bounds():
    """Verifies collateral = clamped(risk_fraction * balance, min_collateral, max_collateral)."""
    env = MultiCryptoDexPerpEnv(
        num_symbols=1, num_candles=10,
        initial_capital=10000.0,
        min_risk_per_trade=0.01, max_risk_per_trade=0.05,
        min_collateral=50.0, max_collateral=300.0,
        min_leverage=1.0, max_leverage=1.0,
    )
    env.reset()
    # 5D action: side=1.0, lev=-1.0 (1x), risk=1.0 (5% of 10000 = 500 -> clamped to 300)
    act = mx.array([[[1.0, -1.0, 1.0, 0.0, 0.0]]])
    env.step(act)
    # Notional should be clamped collateral (300) * leverage (1) = 300
    assert abs(float(env.notionals[0, 0].item()) - 300.0) < 1.0
