# Reporting & Accounting

## Trade History Schema

A professional-grade trade history CSV for perpetuals must capture the complete lifecycle of every position for P&L reconciliation, tax reporting, and risk analysis.

### Core Identification
- `trade_id` / `order_id` - Unique identifiers
- `timestamp` (UTC unix epoch milliseconds)
- `timestamp_utc` (ISO-8601 UTC, millisecond precision, e.g. `2020-01-01T00:00:02.123Z`)
- `symbol` (e.g., BTC)
- `exchange` / `venue`
- `strategy_id` / `run_id` - For model attribution

### Execution Details
- `side` (buy/sell)
- `position_effect` (open/close/increase/reduce)
- `quantity` / `filled_quantity`
- `price` / `notional_value`
- `leverage` at execution
- `order_type` (market/limit/stop)
- `liquidity_flag` (maker/taker)

### Costs & Fees
- `fee_amount` and `fee_currency`
- `funding_fee` (separate line item)
- `spread_cost`
- `slippage_bps`
- `liquidation_fee`
- `liquidation_penalty`

### P&L Attribution
- `realized_pnl`
- `unrealized_pnl`
- `gross_pnl`
- `net_pnl`
- `pnl_pct`
- `return_on_margin`
- `return_on_equity`

### Margin & Risk
- `margin_type` (cross/isolated)
- `initial_margin` / `maintenance_margin`
- `margin_used` / `free_margin`
- `liquidation_price`
- `distance_to_liquidation_pct`
- `equity_before` / `equity_after`

### Perpetuals-Specific
- `mark_price`
- `index_price`
- `oracle_price`
- `funding_rate`
- `funding_payment`
- `open_interest`

### Audit Trail
- `model_version` / `policy_version`
- `config_hash` / `git_commit`
- `seed`
- `data_version`
- `cost_basis_method` (FIFO/LIFO/average)

## Common Pitfalls

1. **Funding fees excluded** - Most exchange exports don't include funding; must be pulled separately
2. **Fees in different currency** - Need fiat value at time of payment, not just raw amount
3. **Wrong cost basis** - FIFO vs average-cost produces materially different realized P&L
4. **No position_effect field** - Can't correctly attribute P&L without open/increase/reduce/close distinction

## Ledger Architecture

Use four linked ledgers for clean separation:

```
TRADES (completed positions)
├── EXECUTIONS (individual fills)
├── FUNDING (funding events)
└── ACCOUNT_SNAPSHOTS (equity/margin state)

DECISIONS (model inputs/outputs)
```

**Key principle:** Make events the primitive; derive trade P&L from them. Never overwrite historical records—append corrections instead.

## Research Provenance

Track experiment metadata to interpret performance correctly:

- `experiment_id` / `run_id`
- `seed`
- `number_of_trials`
- `validation_scheme`
- `model_version` / `data_version`

Without this, you won't know how many experiments produced the "best" Sharpe.

## Regulatory Context

Growing demand for auditable, transparent trading systems:
- 2025 IRS requires Specific Identification documented contemporaneously
- Cost basis must be tracked per-wallet/exchange-account
- Book PnL (backtest) and tax PnL (compliance) can legitimately differ
- High-volume bots (2,000-20,000 trades/month) require individual transaction recording