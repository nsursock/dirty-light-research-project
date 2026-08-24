# Synthetic Data Generation

State-of-the-art synthetic financial data generation has decisively shifted from simple models like Geometric Brownian Motion (GBM) to advanced generative architectures that capture realistic market properties.

## The Problem with GBM

**Key Flaw:** GBM cannot reproduce critical "stylized facts" of real financial markets:
- **Fat tails** (near-zero kurtosis vs. positive kurtosis in real data)
- **Volatility clustering** (negligible autocorrelation vs. significant positive autocorrelation)
- **Tail risk scenarios** essential for liquidation risk analysis

**Evidence:** CoFinDiff benchmark shows GBM exhibits kurtosis near zero while real data shows positive kurtosis; GBM shows negligible volatility autocorrelation while real data shows significant positive autocorrelation.

## Modern Generative Architectures

### 1. Diffusion Models (Current Frontier)

**Advantages:**
- Superior fidelity and stability compared to GANs/VAEs
- Better capture of stylized facts (fat tails, volatility clustering)
- Multimodal and conditional generation capabilities

**Key Implementations:**
- **FinDiff++**: Generates time-series, tabular data, and text in a single system
- **CoFinDiff**: Conditional diffusion model controllable by market regimes
- **Wavelet + DDPM**: Converts correlated time series to images, generates via diffusion, converts back

**Innovation:** Wavelet transformation allows capturing complex frequency-domain and time-domain patterns simultaneously.

### 2. GANs (Continued Evolution)

**Focus Areas:**
- **Enhanced control**: Conditional inputs (CausalGANs, MC-TE-GAN) for macroeconomic conditioning
- **Stability improvements**: WGAN-GP, learning rate balancers to prevent mode collapse
- **Hybrid approaches**: Combining with LSTMs (CGAN-LSTM) for crisis-period generation

**Specialized Use:**
- **Tail-GAN**: Purpose-built for tail-risk scenarios and VaR/CVaR estimation
- **CoMeTS-GAN**: Multivariate correlated generation with cross-asset correlation structure

### 3. VAEs (Domain-Specific)

**Strengths:**
- Controllable generation (e.g., Implied Volatility Surfaces with interpretable features)
- Data imputation (can reconstruct with only 5% of original data points)
- Interpretability and control for structured domains

**Use Case:** Illiquid markets or newly introduced derivatives where data is sparse.

## Market Microstructure Generation

**Next Frontier:** Generating order books and order flow, not just OHLCV bars.

**Key Papers:**
- **Limit Order Book Simulation with GANs**: Generates order flow/book dynamics
- **Painting the Market**: Diffusion models for LOB simulation
- **DiffLOB**: Adds counterfactual generation ("what would the book look like under different regime")

**Importance:** Order book generation determines spread, depth, and slippage-under-stress—critical for execution realism.

## Validation and Evaluation

**Critical Step:** Don't trust that data "looks" right. Implement rigorous validation:

**Methods:**
- **Model-based evaluation**: Compare forecasting model performance (ARIMA, LSTM, XGBoost) on real vs. synthetic data
- **Stylized facts checks**: Verify fat tails, volatility clustering, seasonality patterns
- **LOB-Bench**: Standardized realism metrics for LOB generators (ICML 2025)

**Key Insight:** If models perform similarly on both real and synthetic data, the synthetic data captures the underlying structure that matters for that specific task.

## Model Collapse Warning

**Finding:** Model collapse is mathematically inevitable—each training generation on synthetic data reduces variance and eliminates distribution tails containing rare but crucial patterns.

**Relevance:** GBM produces the symptom this literature warns about—thin-tailed stand-in for fat-tailed reality.

## Practical Recommendations

1. **Replace GBM** with diffusion- or GAN-based generators for any results you intend to trust, especially leverage/risk-sizing conclusions
2. **Keep GBM** only for pure wiring/smoke-tests (cheap, fast, fine for that purpose)
3. **Use conditional models** to generate data for specific market regimes (bull, bear, volatile)
4. **Implement rigorous validation** using model-based evaluation and stylized facts checks
5. **Consider LOB generation** for execution realism if market impact is a concern

## Key Takeaway

The SOTA is evolving rapidly. Using simple GBM is now a legacy approach. To achieve robust, generalizable performance:
- Consider diffusion models (current frontier)
- Implement conditioning for regime-specific generation
- Validate rigorously with quantitative fidelity checks
