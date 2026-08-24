import sys
from pathlib import Path
import mlx.core as mx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scripts.config import cfg
except ModuleNotFoundError:
    from config import cfg

# Binance Kline Endpoint Specification & Standard Timeframes
BINANCE_KLINE_ENDPOINT = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_KLINE_ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_TIMEFRAMES = [
    "1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"
]
TIMEFRAME_MINUTES = {
    "1s": 1.0 / 60.0, "1m": 1.0, "3m": 3.0, "5m": 5.0, "15m": 15.0, "30m": 30.0,
    "1h": 60.0, "2h": 120.0, "4h": 240.0, "6h": 360.0, "8h": 480.0, "12h": 720.0,
    "1d": 1440.0, "3d": 4320.0, "1w": 10080.0, "1M": 43200.0,
}


def get_timeframe_ratio(high_tf: str = "1h", low_tf: str = "5m") -> int:
    """Calculates integer ratio between high timeframe (PPO manager) and low timeframe (SAC worker)."""
    assert high_tf in TIMEFRAME_MINUTES, f"Unsupported high_tf '{high_tf}'. Valid: {list(TIMEFRAME_MINUTES.keys())}"
    assert low_tf in TIMEFRAME_MINUTES, f"Unsupported low_tf '{low_tf}'. Valid: {list(TIMEFRAME_MINUTES.keys())}"
    high_m = TIMEFRAME_MINUTES[high_tf]
    low_m = TIMEFRAME_MINUTES[low_tf]
    assert high_m >= low_m, f"high_tf ({high_tf}) must be >= low_tf ({low_tf})"
    return max(1, int(round(high_m / low_m)))


def sma(x, n):
    if x.shape[0] <= n:
        return mx.cumsum(x, axis=0) / mx.arange(1, x.shape[0] + 1)[:, None]
    cs = mx.pad(mx.cumsum(x, axis=0), [(1, 0), (0, 0)])
    return mx.pad((cs[n:] - cs[:-n]) / n, [(n - 1, 0), (0, 0)], mode="edge")


def ema(x, n):
    a = 2.0 / (n + 1.0)
    out = [x[0]]
    for t in range(1, x.shape[0]):
        out.append(out[-1] * (1.0 - a) + x[t] * a)
    res = mx.stack(out, axis=0)
    mx.eval(res)
    return res


def rsi(c, n=14):
    if c.shape[0] <= 1:
        return mx.full(c.shape, 50.0)
    d = c - mx.pad(c[:-1], [(1, 0), (0, 0)], mode="edge")
    rs = ema(mx.maximum(d, 0.0), n) / (ema(mx.maximum(-d, 0.0), n) + 1e-8)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(c, fast=12, slow=26, sig=9):
    m = ema(c, fast) - ema(c, slow)
    s = ema(m, sig)
    return m, s, m - s


def bbands(c, n=20, k=2.0):
    mid = sma(c, n)
    std = mx.sqrt(sma((c - mid) ** 2, n) + 1e-8)
    upper, lower = mid + k * std, mid - k * std
    return upper, lower, (c - lower) / (upper - lower + 1e-8)


def atr(h, l, c, n=14):
    if c.shape[0] <= 1:
        return h - l
    pc = mx.pad(c[:-1], [(1, 0), (0, 0)], mode="edge")
    tr = mx.maximum(h - l, mx.maximum(mx.abs(h - pc), mx.abs(l - pc)))
    return ema(tr, n)


REGIME_NAMES = cfg.data.regimes.names
FEATURE_NAMES = cfg.data.features


def compute_indicator_features(open_p, high, low, close, volume, funding, regimes, config=None):
    """Computes technical indicator stack in pure MLX for given OHLCV + funding + regimes."""
    c = config or cfg
    ind = c.data.indicators
    if close.shape[0] <= 1:
        log_ret = mx.zeros_like(close)
    else:
        log_ret = mx.pad(mx.log(mx.maximum(close[1:] / mx.maximum(close[:-1], 1e-8), 1e-8)), [(1, 0), (0, 0)], mode="edge")
    s20 = sma(close, ind.sma_window)
    e12, e26 = ema(close, ind.ema_fast), ema(close, ind.ema_slow)
    r14 = rsi(close, ind.rsi_window)
    m_line, m_sig, m_hist = macd(close, ind.ema_fast, ind.ema_slow, ind.macd_signal)
    bb_u, bb_l, bb_pct = bbands(close, ind.bb_window, ind.bb_k)
    a14 = atr(high, low, close, ind.atr_window)

    feats = [
        open_p, high, low, close, volume, funding, regimes.astype(mx.float32),
        log_ret, s20, e12, e26, r14,
        m_line, m_sig, m_hist, bb_u, bb_l, bb_pct, a14
    ]
    res = mx.stack(feats, axis=-1)
    mx.eval(res)
    return res


def generate_crypto_data(
    num_candles=None,
    num_symbols=None,
    dt=None,
    s0=None,
    config=None,
    timeframe: str = "5m",
):
    """Generates 3D tensor: (num_candles x num_symbols x num_features) at specified Binance timeframe."""
    c = config or cfg
    num_candles = num_candles if num_candles is not None else c.data.num_candles
    num_symbols = num_symbols if num_symbols is not None else c.data.num_symbols
    s0 = s0 if s0 is not None else c.data.s0
    if dt is None:
        tf_mins = TIMEFRAME_MINUTES.get(timeframe, 5.0)
        dt = (tf_mins / 1440.0) if hasattr(c.data, "dt") else c.data.dt

    mu_reg = mx.array(c.data.regimes.mu)
    sig_reg = mx.array(c.data.regimes.sigma)
    f_mu_reg = mx.array(c.data.regimes.funding_mu)
    P_cum = mx.cumsum(mx.array(c.data.regimes.transition_matrix), axis=-1)

    # 1. Markov Regime Simulation
    states = [mx.random.randint(0, len(c.data.regimes.names), (num_symbols,))]
    for _ in range(1, num_candles):
        r = mx.random.uniform(0.0, 1.0, (num_symbols, 1))
        states.append(mx.sum(r > P_cum[states[-1]], axis=-1))
    regimes = mx.stack(states, axis=0)

    mu = mu_reg[regimes]
    sigma = sig_reg[regimes]
    mu_f = f_mu_reg[regimes]

    # 2. Regime-dependent GBM OHLCV
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * (dt**0.5) * mx.random.normal((num_candles, num_symbols))
    close = s0 * mx.exp(mx.cumsum(log_ret, axis=0))
    open_p = mx.pad(close[:-1], [(1, 0), (0, 0)], mode="edge")
    ivol = sigma * (dt**0.5)
    high = mx.maximum(open_p, close) * (1.0 + mx.abs(mx.random.normal((num_candles, num_symbols))) * ivol)
    low = mx.minimum(open_p, close) * (1.0 - mx.abs(mx.random.normal((num_candles, num_symbols))) * ivol)
    volume = mx.exp(mx.random.normal((num_candles, num_symbols)) * c.data.volume_std + c.data.volume_mean)

    # 3. OU process for regime-aware funding rate
    theta_f = c.data.ou.theta_f
    sigma_f = c.data.ou.sigma_f
    f_noise = sigma_f * (dt**0.5) * mx.random.normal((num_candles, num_symbols))
    funding = [mx.zeros((num_symbols,))]
    for t in range(1, num_candles):
        funding.append(funding[-1] + theta_f * (mu_f[t] - funding[-1]) * dt + f_noise[t])
    funding = mx.stack(funding, axis=0)

    return compute_indicator_features(open_p, high, low, close, volume, funding, regimes, config=c)


def resample_klines_mlx(low_tf_feats: mx.array, ratio: int, config=None) -> mx.array:
    """Resamples low TF feature tensor to high TF feature tensor in pure MLX."""
    if ratio <= 1:
        return low_tf_feats
    T, N, _ = low_tf_feats.shape
    num_high = T // ratio
    if num_high == 0:
        return low_tf_feats
    trimmed = low_tf_feats[: num_high * ratio]
    reshaped = trimmed.reshape(num_high, ratio, N, -1)

    open_p = reshaped[:, 0, :, 0]
    high = mx.max(reshaped[:, :, :, 1], axis=1)
    low = mx.min(reshaped[:, :, :, 2], axis=1)
    close = reshaped[:, -1, :, 3]
    volume = mx.sum(reshaped[:, :, :, 4], axis=1)
    funding = reshaped[:, -1, :, 5]
    regimes = reshaped[:, -1, :, 6]

    return compute_indicator_features(open_p, high, low, close, volume, funding, regimes, config=config)


def generate_multi_tf_data(
    num_candles: int | None = None,
    num_symbols: int | None = None,
    high_tf: str = "1h",
    low_tf: str = "5m",
    config=None,
):
    """Generates synchronized low-TF and high-TF multi-crypto market tensors in pure MLX."""
    c = config or cfg
    num_candles = num_candles if num_candles is not None else getattr(c.simulation, "num_candles", 600)
    num_symbols = num_symbols if num_symbols is not None else getattr(c.simulation, "num_symbols", 4)
    ratio = get_timeframe_ratio(high_tf, low_tf)

    low_tf_data = generate_crypto_data(
        num_candles=num_candles,
        num_symbols=num_symbols,
        config=c,
        timeframe=low_tf,
    )
    high_tf_data = resample_klines_mlx(low_tf_data, ratio, config=c)
    return low_tf_data, high_tf_data, ratio


if __name__ == "__main__":
    low_data, high_data, r = generate_multi_tf_data(num_candles=600, num_symbols=4, high_tf="1h", low_tf="5m")
    mx.eval(low_data, high_data)
    print(f"Low TF (5m) shape: {low_data.shape} | High TF (1h, ratio {r}) shape: {high_data.shape}")

