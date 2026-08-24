# Technical Indicators

## Key Findings

**Strongest evidence:** Momentum/trend at longer horizons (12-month lookbacks). Short lookbacks (~1 day) are on the wrong side of momentum findings.

**Best-of-universe selection is noise:** ~all apparent technical analysis edge vanishes out-of-sample after costs.

**Raw OHLCV ≈ engineered TA in deep models:** Feeding large RSI/MACD stacks adds little once a neural network sees the raw price window.

**Wins come from:** More signal (cross-sectional, order-flow, learned latents) or better validation (deflated Sharpe, CPCV, PBO).

## Feature Categories

### Returns & Momentum
- Multi-horizon returns (1, 2, 4, 8, 12, 24, 48, 72, 168 bars)
- Return z-scores
- Distance to 52-week high (stronger than trailing return)

### Trend
- MA distance (close/EMA ratios rather than binary crossovers)
- Breakout signals
- Trend strength indicators

### Volatility
- Multi-scale realized volatility (HAR-RV approach)
- OHLC estimators (Parkinson, Garman-Klass, Yang-Zhang)
- Volatility dynamics (vol ratios, vol-of-vol)

### Volume
- Volume z-scores
- Volume surprise metrics
- Signed volume (return × volume)
- Volume-clock resampling (reduces noise)

### Price Geometry
- Range percentage (high-low/close)
- Body percentage
- Wick ratios
- Close location within range

### Regime Features
- Volatility regime classification
- Trend regime detection
- Market stress indicators

### Cross-Asset
- BTC returns/volatility as features for altcoins
- Cross-sectional return percentiles
- Asset beta to market

## Validation Requirements

When testing indicators:
- Demand t ≥ 3.0 statistical significance (not 1.96)
- Use deflated Sharpe ratio for multiple testing correction
- Estimate PBO (Probability of Backtest Overfitting)
- Prefer purged/embargoed holdouts (this repo: embargoed train/test by default; CPCV splits in `scripts/cpcv.py`)
- Include transaction costs in evaluation

## Practical Recommendations

**If staying with OHLCV+TA:**
- Use volume-clock resampling
- Add entropy/vol-regime features
- Extend lookback windows
- Use distance-to-high features
- Apply strict statistical validation

**Alternative approaches:**
- Raw OHLCV with learned representations (autoencoders)
- Order-flow features for short horizons
- Cross-sectional features for portfolio context
- Learned indicator computation (TINs)
