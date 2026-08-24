# Financial Signal Processing

State-of-the-art trading systems treat market data as a **low-SNR non-stationary signal**. Classical filters and modern decomposition methods sit *before* the predictor or policy so models see structure (trend, regime, multi-scale momentum) instead of raw tick noise.

## Key Findings

**Raw prices into deep RL are weak:** Low signal-to-noise makes policies chase microstructure noise → overtrading, fee bleed, and unstable value estimates. Signal-enhanced state (wavelets, VMD/EMD, Kalman/adaptive filters, multi-horizon patches) is a recurring SOTA pattern.

**Denoising ≠ free alpha:** Filters change *which* noise the model sees. They help when the failure mode is noise-driven churn; they do not invent predictability if gross signal is absent. Treat Kalman / wavelet gains as **hypotheses to falsify under costs**, not as transferable return multipliers (single-study anecdotes do not generalize).

**Prediction ≠ tradable edge:** LOB and forecasting papers repeatedly show high directional accuracy with **negative post-cost** returns. Always evaluate with fees, funding, spread, and liquidation where relevant.

**Transformers are not automatic winners:** PatchTST and frequency hybrids help on long sequences; DLinear/NLinear and Ridge/GBDT often match or beat heavy attention on non-stationary financial series. Keep simple baselines mandatory.

**Strongest short-horizon information is microstructure:** Limit-order-book, order-flow, and market-by-order data dominate candle OHLCV for 1m–5m price formation. Under an **OHLCV + TA + funding** constraint, prefer denser within-candle structure (multi-horizon returns, volatility estimators, volume-clock, funding/basis) over stacking more classical indicators.

**RL is the decision layer, not the signal layer:** Hierarchical RL remains a valid sequential-control architecture; literature still expects a **calibrated alpha / regime / risk representation** upstream. Volatility-scaled rewards and risk-aware objectives (Deep Hedging lineage, CVaR) matter as much as exploration schedules.

**Validation is part of SOTA:** Deflated Sharpe, PBO, CPCV, cost curves, and baseline ladders decide whether a filter “worked.” See `backtesting_validation.md`.

## Technique Taxonomy

### 1. Classical / adaptive filtering
- Kalman / adaptive Kalman (including regime-switching process noise)
- EWMA / exponential smoothing
- Fractional differentiation (stationarity while retaining memory — AFML)

**Role:** Estimate latent fair price or volatility; expose the **innovation** (residual) as a shock feature. Keep **execution prices raw**; filter observations only.

### 2. Spectral & mode decomposition
- Wavelets / Haar / wavelet packets (trend vs high-frequency bands)
- VMD, EMD / Hilbert–Huang, Empirical Wavelet Transform
- SSA, Fourier / multi-taper spectral density
- Dual-branch encoders: low-frequency trend path + high-frequency fluctuation path → policy (wavelet-enhanced DRL)

**Role:** Separate mixed-frequency entanglement before learning. Especially useful on noisy short-horizon bars when only candle data is available.

### 3. Multi-scale / patched representations
- Multi-horizon returns and hierarchical interpolation (N-HiTS-style)
- PatchTST-style patching (local context tokens instead of every bar)
- TimesNet / MICN multi-scale convolutions
- Volume-clock resampling (noise reduction vs calendar time)

**Role:** Align features with the decision horizon without inventing new data modalities.

### 4. Regime & latent state
- HMM / jump models / MS-GARCH on returns and volatility
- Soft regime posteriors as features or high-level gates
- Self-supervised encoders (e.g. TS2Vec, CoST seasonal–trend split) with heads for return, volatility, and regime

**Role:** Regime-specialized or gated policies often beat one shared-hyperparameter policy across heterogeneous market conditions.

### 5. Cross-asset & factor structure
- Shared latent factors (IPCA / autoencoder asset pricing)
- Cross-sectional returns, market-factor proxies, funding/basis across related instruments
- GNN / TFT–GNN hybrids for inter-asset relations

**Role:** Avoid treating each symbol’s one-bar return as an island.

### 6. Microstructure
- DeepLOB, market-by-order, Hawkes order-flow, imbalance / micro-price

**Role:** Highest evidence for short-horizon alpha when LOB or trade-tape data is available. Classical TA is not a substitute for the book.

### 7. Decision layer (uses processed signal)
- Hierarchical manager/worker architectures (e.g. EarnHFT, HRT, HRPM, feudal nets)
- Risk-sensitive / distributional RL; safety constraints on leverage and margin
- Volatility-scaled rewards; Deep Hedging-style frictions in the objective

**Role:** Size and execute *given* a cleaner state — does not replace signal work.

## Guidance When Limited to OHLCV + TA (+ Funding)

```text
OHLCV (+ volume) [+ funding]
        │
        ├─ classical TA (RSI, MA distance, ATR, …)
        ├─ multi-horizon returns / multi-scale volatility
        ├─ optional: Kalman / wavelet / VMD on observations
        ├─ funding / basis features (derivatives)
        └─ cross-asset / market-factor features
        │
        ▼
   multi-horizon / dual-timeframe state
        │
        ▼
   supervised & simple baselines  →  then sequential decision models
```

**Priority order under that constraint:**
1. Multi-horizon momentum + multi-scale volatility (HAR-style) + volume features  
2. Cross-asset / market factor + funding residual (if trading perps)  
3. Light denoising (Kalman or Haar dual-branch) on **observations only**, ablated vs raw  
4. Regime posterior as a feature or high-level gate  
5. Heavier VMD/EMD only if light denoising shows out-of-sample lift after costs  

**Do not:** replace fill or liquidation prices with filtered series; apply LOB-only methods without LOB data; treat timeframe or leverage sweeps as substitutes for signal enrichment.

## Horizon-Specific Emphasis

| Horizon class | Typical bars | Signal-processing emphasis |
| ------------- | ------------ | -------------------------- |
| Short / scalp | seconds–minutes | Highest noise → Kalman, wavelets, volume-clock; microstructure dominates when available |
| Intraday | minutes–hours | Multi-horizon features + regime; moderate denoise |
| Swing / position | hours–days | Less aggressive denoise; more trend, regime, funding accrual, and tail-risk features |

Validate across the horizons of interest; do not discard a horizon solely because tails are large — treat it as a risk-scale stress case.

## Practical Recommendations

1. **Filter observations, not fills.** Execution, mark-to-market, and liquidation use raw market prices.  
2. **Ablate, don’t assume.** Raw vs Kalman vs wavelet/VMD vs both — identical costs and purged/embargoed OOS protocol.  
3. **Baseline ladder first.** Buy-and-hold, random, momentum, mean-reversion, Ridge/DLinear under the same cost model (see `backtesting_validation.md`).  
4. **Cost and leverage sensitivity** on frozen policies before more RL training.  
5. **Report cost-component distributions** (fees, funding, liquidations, PnL) so “filter helped” is not confused with “model traded less.”  
6. **Demand DSR/PBO** when comparing filter variants (multiple testing).  
7. **Fix state SNR and costs before deep RL hyperparameter archaeology** — literature puts representation and frictions ahead of entropy/KL tweaks alone.

## Common Pitfalls

| Symptom | Likely misread | Better check |
| ------- | -------------- | ------------ |
| Exploration coefficient decays while reward falls | “Learning is succeeding” | Coefficient may track policy entropy vs a target, not performance |
| Large episode returns on long horizons | “Reward scale bug only” | Episode sums aggregate many steps and fatter tails; inspect per-step reward |
| Filter improves in-sample Sharpe | “Signal processing works” | Out-of-sample + costs + turnover; prediction ≠ PnL |
| Lower leverage but more liquidations | “Leverage band only” | Hold length × collateral × path; audit sizing and stops |

## Related Summaries

- `indicators_features.md` — OHLCV / TA feature design  
- `market_microstructure.md` — execution, LOB, volume-clock, denoising anecdotes  
- `trading_bot_architecture.md` — hierarchical and risk-aware RL  
- `backtesting_validation.md` — DSR, PBO, CPCV, baseline ladder  

Expanded paper lists and survey notes live in `docs/details.zip`.

## Recommended Reading Order

1. Gu, Kelly & Xiu — Empirical Asset Pricing via Machine Learning  
2. Huddleston et al. — Intraday market predictability  
3. DeepLOB; Briola et al. (prediction ≠ tradable)  
4. PatchTST; DLinear / NLinear (sequence baselines / anti-hype)  
5. Wavelet- / VMD-enhanced deep RL trading papers  
6. Adaptive Kalman / regime-switching filters  
7. Zhang, Zohren & Roberts — Deep RL for trading (volatility-scaled rewards)  
8. Deep Hedging — frictions and risk in the objective  
9. Bailey — PBO / Deflated Sharpe; CPCV comparisons  
