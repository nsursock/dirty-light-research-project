# Backtesting Validation & Statistical Rigor

State-of-the-art trading bot evaluation requires moving beyond simple profit/loss statements to rigorous statistical validation that addresses selection bias, overfitting, and non-stationarity.

## The Core Problem

**Reality Check:** Most reported trading bot performance is statistically misleading due to:
- Multiple testing bias (trying thousands of strategies)
- Selection bias (reporting only the best result)
- Look-ahead bias (using future information)
- Non-normal returns (fat tails, skewness)
- In-sample overfitting (curve-fitting to historical data)

## Key Statistical Tools

### 1. Deflated Sharpe Ratio (DSR)

**Purpose:** Corrects Sharpe ratio for multiple testing, selection bias, and non-normal returns.

**What it addresses:**
- Number of strategies tried
- Non-normal return distributions
- Selection bias from reporting the "best" result

**Reporting format:**
```
Raw Sharpe             3.10
Trials                 4,382
Skew                   ...
Excess kurtosis        ...
Expected max Sharpe    ...
Deflated Sharpe        ...
DSR probability        ...
```

**Key Paper:** Bailey & López de Prado (2014, Journal of Portfolio Management)

### 2. Probability of Backtest Overfitting (PBO)

**Purpose:** Estimates how likely a backtest-selected "winner" is actually overfit.

**Method:** Uses Combinatorially Symmetric Cross-Validation (CSCV) to simulate a wide variety of market situations.

**Key question:** Not "Was my backtest profitable?" but "How likely is the strategy I selected because it performed best IS actually an overfit?"

**Reporting format:**
```
PBO
CSCV partitions
IS/OOS rank correlation
OOS degradation
median OOS Sharpe
worst OOS Sharpe
```

**Key Paper:** Bailey, Borwein, López de Prado, Zhu (2017, Journal of Computational Finance)

### 3. Combinatorial Purged Cross-Validation (CPCV)

**Purpose:** The most rigorous out-of-sample validation method, superior to traditional walk-forward.

**Advantages:**
- Controls for leakage
- Simulates wider variety of market situations
- Lower PBO and better DSR statistics than conventional approaches

**Finding:** 2024 controlled comparison found CPCV superior to walk-forward validation in experiments with non-stationarity, autocorrelation, and regime shifts.

**Key Paper:** Arian, Norouzi Mobarekeh & Seco (2024, Expert Systems with Applications)

## In this repo

Two layers, not one:

**Engineering tests** (`pytest`, including `pytest -m p0`) check simulator invariants: causal fills, embargoed *slicing*, scalar-vs-MLX accounting, finite equity. A green suite means the code is internally consistent. It is **not** evidence of tradable edge.

**Statistical validation** default is a single embargoed train/test holdout (`data.cv: embargo` via `time_slice`). Combinatorial purged splits are implemented in `scripts/cpcv.py` (`cpcv_splits`) for path design; the default training run does not loop every combinatorial path. DSR and PBO are unimplemented. Passing pytest does not substitute for those.

### 4. White's Reality Check

**Purpose:** Bootstrap-based test for whether the best strategy found during specification search actually has predictive superiority over a benchmark.

**Core insight:** The reported Sharpe of the best strategy among 1,000 tested is not the same as the Sharpe of a strategy specified before seeing the data.

**Reporting requirement:** Record number of experiments, hyperparameter trials, models, features tested, and distribution of results.

**Key Paper:** White (2000, Econometrica)

### 5. Lo-Adjusted Sharpe Statistics

**Purpose:** Addresses statistical uncertainty in Sharpe ratios and corrects for serial correlation.

**Problem:** Naïve annualization can be wrong when returns are serially correlated, materially overstating annualized Sharpe.

**Reporting format:**
```
Sharpe                    2.13
Sharpe annualization      Lo-adjusted
return autocorrelation    ...
Sharpe standard error      ...
Sharpe confidence interval ...
```

**Key Paper:** Lo (2002, Financial Analysts Journal)

## Research-Grade Reporting Structure

A serious trading-bot report should include 8 layers:

1. **Data & Experiment Provenance**
   - Datasets, dates, assets, timeframe
   - Feature count, leakage controls

2. **Model**
   - Architecture, parameters, seed
   - Training procedure, reward

3. **Economic Performance**
   - CAGR, return, PnL, Sharpe, Sortino
   - Calmar, profit factor, win rate, turnover

4. **Risk**
   - Max DD, duration, VaR, CVaR
   - Tail loss, skew, kurtosis

5. **Execution**
   - Fees, spread, slippage, impact
   - Turnover, fill assumptions

6. **Generalization**
   - IS → validation → OOS
   - Walk-forward, CPCV
   - Regime × asset × timeframe

7. **Statistical Validity**
   - CI, bootstrap, PBO, DSR
   - Multiple testing, trials

8. **Robustness**
   - Cost stress, parameter sensitivity
   - Monte Carlo, perturbations
   - Seeds, datasets, baselines

## Critical Findings

### Research Design Sensitivity

**Finding:** Seemingly innocuous research-design choices (training window, data filters, portfolio construction) can cause very large differences in reported performance.

**Evidence:** 2026 study of 5,376 ML portfolios found nonstandard errors up to five times conventional standard errors. Only about one-third of portfolios remained significantly profitable after transaction costs.

**Implication:** Those choices need to be reported, not hidden in configuration files.

### The Transaction Cost Trap

**Finding:** Directional prediction can look excellent while post-cost returns are negative.

**Evidence:** 2025 study found SAC could show market-timing ability but didn't systematically beat 1/N, and high turnover turned negative under only 0.1% transaction costs.

**Implication:** Always test with realistic costs, not theoretical execution.

### Implementation Risk

**Finding:** The same strategy can produce different outputs in different backtesting engines.

**Evidence:** 2026 paper addresses reproducibility/engine-variance—essentially a critique that which number to trust given engine differences.

**Implication:** Standardize backtesting infrastructure and report all assumptions.

## Best Practices

1. **Strict untouched OOS**: Train → validation → development-OOS → FINAL LOCKED TEST
2. **Experiment ledger**: Record total trials, models, hyperparameter configs, features tested, backtests
3. **Cost sensitivity curve**: Test performance across different cost assumptions (0, 5, 10, 15, 20, 30, 50 bps)
4. **Regime decomposition**: Report performance by bull/bear/sideways/high-vol regimes
5. **Seed/model dispersion**: Report performance across multiple random seeds
6. **Baseline ladder**: Compare against buy-and-hold, coin flip, random policy, momentum, EMA, RSI/MACD, supervised models

## The Fundamental Shift

**Old question:** "How did this particular run perform?"

**New question:** "How much evidence do I have that this performance represents a real, generalizable edge rather than selection bias, overfitting, regime luck, or unrealistic execution assumptions?"

## Recommended Reading Order

1. PBO (Bailey et al.)
2. DSR (Bailey & López de Prado)
3. CPCV 2024 (Arian et al.)
4. Gu/Kelly/Xiu (Empirical Asset Pricing via ML)
5. Lalwani 2026 (Research Design Choices)
6. White Reality Check
7. Lo Sharpe Statistics
8. DRL Trading (Zhang, Zohren & Roberts)
