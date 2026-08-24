# Trading Bot Architecture

State-of-the-art trading bot architecture has moved beyond simple rule-based systems to sophisticated multi-modal approaches combining deep reinforcement learning, large language models, and hierarchical decision-making.

## Key Architectural Paradigms

### 1. Hierarchical Reinforcement Learning (HRL)

**Problem:** Single RL agents struggle with the "curse of dimensionality" as the number of traded assets grows, leading to poor diversification.

**Solution:** Bi-level architectures separate concerns:
- **High-level controller** (e.g., PPO): Stock selection, position sizing, risk management
- **Low-level controller** (e.g., DDPG): Trade execution, timing, limit placement

**Examples:**
- **Hierarchical Reinforced Trader (HRT)**: Explicitly constrains trading behavior by turnover, drawdown, and execution feasibility
- **CVaR-PPO**: Embeds tail-risk awareness directly into policy objectives using trajectory sampling

### 2. LLM-Augmented Agents

**Shift:** LLMs are used as feature sources, not decision-makers. They process unstructured data (news, sentiment, filings) to generate signals that feed into RL policies.

**Key Frameworks:**
- **AlphaQuanter**: Agentic LLM-orchestrated RL trading framework with transparent reasoning
- **FinMem**: LLM agent with layered memory for balancing historical patterns with current dynamics
- **FinRL-DeepSeek**: Extends CVaR-PPO with LLM-generated risk assessment and trading recommendations

**Benefits:**
- Genuine alternative information source (news/sentiment) vs. redundant price-derived indicators
- Natural language justifications for auditability
- Ability to process vast amounts of unstructured data

### 3. Multi-Agent Systems

**Architecture:** Specialized agents with distinct roles:
- **Analyst Agent**: Market analysis and signal generation
- **Decision Agent**: Portfolio allocation and risk management
- **Risk Agent**: Position sizing and exposure limits
- **Execution Agent**: Order placement and timing
- **Explainability Agent**: Logs communication flow, generates natural language justifications

**Advantage:** Complete auditable decision-making pipeline with built-in transparency

### 4. Risk-Aware Objectives

**Problem:** Vanilla return-maximizing RL leads to excessive risk-taking or over-conservative behavior.

**Solution:** Incorporate risk metrics directly into reward functions:
- CVaR (Conditional Value at Risk) constraints
- Sortino ratio optimization
- Transaction cost penalties
- Volatility-targeting constraints
- Drawdown caps

**Finding:** Risk controls stabilize volatility but can cause policies to become overly conservative if penalties are too strong.

## Current Best Practices

1. **Hierarchical architectures** for complex decision spaces
2. **LLM-derived signals** as alternative information sources
3. **Risk-constrained objectives** instead of vanilla PPO/SAC
4. **Multi-agent systems** for transparency and auditability
5. **Continual learning** for regime adaptation (with strong safeguards)

## Key Insight

RL is not a universal winner. Literature shows mixed evidence that RL outperforms supervised learning or simple baselines. The strongest use cases are genuinely sequential problems: portfolio allocation, execution, and market making.

## Important Caveat

The field lacks strong evidence that RL agents work beyond backtests. Most models only show success on historical data, highlighting the need for rigorous out-of-sample validation before operational deployment.
