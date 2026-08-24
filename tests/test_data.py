import sys
from pathlib import Path
import pytest
import numpy as np
import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import cfg, load_config
from scripts.data import (
    generate_crypto_data,
    generate_multi_tf_data,
    resample_klines_mlx,
    get_timeframe_ratio,
    BINANCE_KLINE_ENDPOINT,
    BINANCE_FUTURES_KLINE_ENDPOINT,
    BINANCE_TIMEFRAMES,
    TIMEFRAME_MINUTES,
    FEATURE_NAMES,
    REGIME_NAMES,
    sma,
    ema,
    rsi,
    macd,
    bbands,
    atr,
)

MU_REG = np.array(cfg.data.regimes.mu)
SIG_REG = np.array(cfg.data.regimes.sigma)
F_MU_REG = np.array(cfg.data.regimes.funding_mu)


def _extract_feature_map(data_3d):
    arr = np.array(data_3d)
    return {name: arr[:, :, i] for i, name in enumerate(FEATURE_NAMES)}


# -----------------------------------------------------------------------------
# 1. Configuration System Contracts
# -----------------------------------------------------------------------------
def test_config_loading():
    """Validates that YAML config loads with correct hierarchical structure and types."""
    loaded_cfg = load_config()
    assert loaded_cfg.data.num_candles > 0
    assert loaded_cfg.data.num_symbols > 0
    assert len(loaded_cfg.data.regimes.names) == 5
    assert len(loaded_cfg.data.regimes.mu) == 5
    assert len(loaded_cfg.data.regimes.sigma) == 5
    assert len(loaded_cfg.data.regimes.funding_mu) == 5
    assert len(loaded_cfg.data.features) == 19
    assert loaded_cfg.agents.ppo.learning_rate > 0.0
    assert loaded_cfg.agents.sac.learning_rate > 0.0


# -----------------------------------------------------------------------------
# 2. GBM Price Mathematical Contracts
# -----------------------------------------------------------------------------
def test_gbm_price_positivity():
    """GBM price paths, opens, highs, and lows must be strictly positive."""
    mx.random.seed(42)
    data = generate_crypto_data(num_candles=500, num_symbols=5)
    fmap = _extract_feature_map(data)

    for key in ["open", "high", "low", "close"]:
        assert (fmap[key] > 0.0).all(), f"Prices in {key} must be strictly positive"


def test_gbm_log_return_moments_by_regime():
    """Log returns must match theoretical GBM mean and variance conditional on regime."""
    mx.random.seed(123)
    dt = 1.0 / 1440.0
    data = generate_crypto_data(num_candles=1000, num_symbols=25, dt=dt)
    fmap = _extract_feature_map(data)

    regimes = fmap["regime"].astype(int)
    log_ret = fmap["log_ret"]

    for r in range(5):
        mask = regimes == r
        n_samples = np.sum(mask)
        if n_samples < 500:
            continue

        r_samples = log_ret[mask]
        expected_mean = (MU_REG[r] - 0.5 * (SIG_REG[r] ** 2)) * dt
        expected_var = (SIG_REG[r] ** 2) * dt

        emp_mean = np.mean(r_samples)
        emp_var = np.var(r_samples, ddof=1)
        stderr_mean = np.sqrt(expected_var / n_samples)

        # Mean within 3.5 std errors
        z_score = abs(emp_mean - expected_mean) / stderr_mean
        assert z_score < 3.5, f"Regime {r} mean return z-score {z_score:.2f} too large"

        # Variance within 10% relative error for finite sample
        assert np.isclose(emp_var, expected_var, rtol=0.10), (
            f"Regime {r} variance mismatch: emp={emp_var:.8e}, exp={expected_var:.8e}"
        )


def test_gbm_normality_and_independence():
    """Within each regime, increments are Gaussian (skew~0, kurt~3) and uncorrelated."""
    mx.random.seed(42)
    data = generate_crypto_data(num_candles=1000, num_symbols=20)
    fmap = _extract_feature_map(data)

    regimes = fmap["regime"].astype(int)
    log_ret = fmap["log_ret"]

    for r in range(3):  # Check most persistent regimes
        samples = log_ret[regimes == r]
        if len(samples) < 1000:
            continue
        z = (samples - np.mean(samples)) / np.std(samples)
        skew = np.mean(z**3)
        kurt = np.mean(z**4)

        assert abs(skew) < 0.25, f"Regime {r} skewness {skew:.3f} exceeds tolerance"
        assert abs(kurt - 3.0) < 0.5, f"Regime {r} kurtosis {kurt:.3f} deviates from 3.0"

    # Lag-1 Autocorrelation of returns across paths should be ~0 (uncorrelated increments)
    acfs = []
    for col in range(log_ret.shape[1]):
        r_col = log_ret[:, col] - np.mean(log_ret[:, col])
        acf1 = np.sum(r_col[:-1] * r_col[1:]) / np.sum(r_col**2)
        acfs.append(acf1)
    assert abs(np.mean(acfs)) < 0.05, f"Average return autocorrelation {np.mean(acfs):.4f} not near 0"


def test_gbm_dt_scaling():
    """Log return variance must scale linearly with time step dt."""
    mx.random.seed(999)
    dt1 = 1.0 / 1440.0
    dt2 = 2.0 / 1440.0

    d1 = generate_crypto_data(num_candles=1000, num_symbols=20, dt=dt1)
    d2 = generate_crypto_data(num_candles=1000, num_symbols=20, dt=dt2)

    var1 = np.var(_extract_feature_map(d1)["log_ret"])
    var2 = np.var(_extract_feature_map(d2)["log_ret"])

    ratio = var2 / var1
    assert np.isclose(ratio, 2.0, rtol=0.15), f"dt scaling ratio {ratio:.3f} expected ~2.0"


# -----------------------------------------------------------------------------
# 3. OU Funding Mathematical Contracts
# -----------------------------------------------------------------------------
def test_ou_transition_kernel():
    """OU funding transition noise conditional on drift must match theoretical variance sigma_f^2 * dt."""
    mx.random.seed(777)
    dt = cfg.data.dt
    theta_f, sigma_f = cfg.data.ou.theta_f, cfg.data.ou.sigma_f
    data = generate_crypto_data(num_candles=1000, num_symbols=25, dt=dt)
    fmap = _extract_feature_map(data)

    funding = fmap["funding"]
    regimes = fmap["regime"].astype(int)
    mu_f = F_MU_REG[regimes]

    dx = funding[1:] - funding[:-1]
    expected_drift = theta_f * (mu_f[1:] - funding[:-1]) * dt
    noise = dx - expected_drift

    emp_noise_var = np.var(noise)
    th_noise_var = (sigma_f**2) * dt

    assert np.isclose(emp_noise_var, th_noise_var, rtol=0.08), (
        f"OU noise variance mismatch: emp={emp_noise_var:.10e}, exp={th_noise_var:.10e}"
    )
    # Zero-mean innovation check
    z_score = abs(np.mean(noise)) / np.sqrt(th_noise_var / noise.size)
    assert z_score < 3.0, f"OU noise mean z-score {z_score:.2f} is statistically non-zero"


def test_ou_stationarity_and_bounds():
    """Funding rates generated by OU process must stay within bounded realistic ranges."""
    mx.random.seed(42)
    data = generate_crypto_data(num_candles=1000, num_symbols=5)
    fmap = _extract_feature_map(data)
    funding = fmap["funding"]

    assert np.all(funding > -0.05) and np.all(funding < 0.05), "Funding rate out of realistic bounds"
    assert np.std(funding) > 0.0, "Funding rate must exhibit variance"


# -----------------------------------------------------------------------------
# 4. Joint Market Generator Invariants
# -----------------------------------------------------------------------------
def test_market_data_invariants():
    """Checks tensor dimensions, finitude (no NaNs), OHLC relationships, and volume positivity."""
    num_candles, num_symbols = 300, 4
    mx.random.seed(42)
    data = generate_crypto_data(num_candles=num_candles, num_symbols=num_symbols)
    mx.eval(data)

    assert data.shape == (num_candles, num_symbols, len(FEATURE_NAMES))
    fmap = _extract_feature_map(data)

    for name in FEATURE_NAMES:
        assert np.isfinite(fmap[name]).all(), f"Feature {name} contains non-finite values (NaN/Inf)"

    # OHLC Bar Invariants
    o, h, l, c = fmap["open"], fmap["high"], fmap["low"], fmap["close"]
    assert np.all(h >= o - 1e-6), "High must be >= Open"
    assert np.all(h >= c - 1e-6), "High must be >= Close"
    assert np.all(h >= l - 1e-6), "High must be >= Low"
    assert np.all(l <= o + 1e-6), "Low must be <= Open"
    assert np.all(l <= c + 1e-6), "Low must be <= Close"

    # Volume and Regime Invariants
    assert np.all(fmap["volume"] > 0.0), "Volume must be strictly positive"
    regimes = fmap["regime"]
    assert np.all(np.isin(regimes, [0, 1, 2, 3, 4])), "Regimes must be valid IDs in [0, 4]"


def test_indicator_contracts():
    """Verifies RSI bounded in [0, 100], BB upper >= lower, MACD consistency, and ATR >= 0."""
    mx.random.seed(42)
    data = generate_crypto_data(num_candles=200, num_symbols=2)
    fmap = _extract_feature_map(data)

    # RSI
    rsi_vals = fmap["rsi14"]
    assert np.all((rsi_vals >= 0.0) & (rsi_vals <= 100.0)), "RSI must be bounded between 0 and 100"

    # Bollinger Bands
    assert np.all(fmap["bb_upper"] >= fmap["bb_lower"] - 1e-6), "BB Upper must be >= BB Lower"

    # MACD Identity: hist = macd - signal
    macd_diff = fmap["macd"] - fmap["macd_sig"]
    assert np.allclose(macd_diff, fmap["macd_hist"], atol=1e-5), "MACD hist must equal macd - macd_sig"

    # ATR
    assert np.all(fmap["atr14"] >= 0.0), "ATR must be non-negative"


# -----------------------------------------------------------------------------
# 5. Reproducibility & Edge Cases
# -----------------------------------------------------------------------------
def test_reproducibility():
    """Same RNG seed must produce bitwise identical market tensors; different seeds must differ."""
    mx.random.seed(42)
    d1 = generate_crypto_data(num_candles=100, num_symbols=3)
    mx.random.seed(42)
    d2 = generate_crypto_data(num_candles=100, num_symbols=3)
    mx.random.seed(43)
    d3 = generate_crypto_data(num_candles=100, num_symbols=3)

    assert mx.all(d1 == d2).item(), "Same seed must yield identical tensors"
    assert not mx.all(d1 == d3).item(), "Different seeds must yield distinct paths"


@pytest.mark.parametrize("num_candles,num_symbols,dt,s0", [
    (30, 1, 1/1440, 100.0),
    (50, 2, 1/60, 50.0),
    (100, 10, 1/24, 1000.0),
])
def test_edge_cases_and_parameter_configurations(num_candles, num_symbols, dt, s0):
    mx.random.seed(1)
    data = generate_crypto_data(num_candles=num_candles, num_symbols=num_symbols, dt=dt, s0=s0)
    mx.eval(data)
    assert data.shape == (num_candles, num_symbols, len(FEATURE_NAMES))
    assert np.isfinite(np.array(data)).all()


def test_binance_timeframe_specifications():
    """Validates Binance Kline endpoints, standard intervals, and timeframe ratio conversions."""
    assert BINANCE_KLINE_ENDPOINT == "https://api.binance.com/api/v3/klines"
    assert BINANCE_FUTURES_KLINE_ENDPOINT == "https://fapi.binance.com/fapi/v1/klines"
    for tf in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]:
        assert tf in BINANCE_TIMEFRAMES
        assert tf in TIMEFRAME_MINUTES

    assert get_timeframe_ratio("1h", "1m") == 60
    assert get_timeframe_ratio("1h", "5m") == 12
    assert get_timeframe_ratio("4h", "15m") == 16
    assert get_timeframe_ratio("1d", "1h") == 24
    assert get_timeframe_ratio("1d", "15m") == 96


def test_multi_timeframe_generation_and_resampling():
    """Validates pure MLX multi-timeframe generation and low-to-high TF resampling."""
    low_data, high_data, ratio = generate_multi_tf_data(num_candles=120, num_symbols=3, high_tf="1h", low_tf="5m")
    mx.eval(low_data, high_data)
    assert ratio == 12
    assert low_data.shape == (120, 3, len(FEATURE_NAMES))
    assert high_data.shape == (10, 3, len(FEATURE_NAMES))

    # Low OHLC consistency check vs Resampled High OHLC
    # High Open must equal first sub-period Low Open
    assert np.isclose(np.array(high_data[0, 0, 0]), np.array(low_data[0, 0, 0]))
    # High Close must equal last sub-period Low Close
    assert np.isclose(np.array(high_data[0, 0, 3]), np.array(low_data[11, 0, 3]))
