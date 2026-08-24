import os
import csv
import math
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from tabulate import tabulate

DEFAULT_SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "AVAX", "DOGE", "LINK", "ADA",
    "DOT", "MATIC", "NEAR", "ATOM", "ARB", "OP", "APT", "SUI",
    "INJ", "TIA", "SEI", "FET", "RENDER", "RUNE", "AAVE", "UNI",
    "MKR", "SNX", "LDO", "CRV", "PENDLE", "DYDX", "GMX", "JUP",
]


def get_symbol_name(idx: int) -> str:
    """Returns unique crypto symbol for an index without modulo collisions."""
    return DEFAULT_SYMBOLS[idx] if idx < len(DEFAULT_SYMBOLS) else f"CRYPTO_{idx + 1}"

TRADE_HISTORY_COLUMNS = [
    # Core Identification
    "trade_id", "order_id", "timestamp", "timestamp_utc", "symbol", "exchange", "venue", "strategy_id", "run_id",
    # Execution Details
    "side", "position_effect", "exit_type", "quantity", "filled_quantity", "price", "notional_value", "leverage", "order_type", "liquidity_flag",
    # Costs & Fees
    "fee_amount", "fee_currency", "funding_fee", "spread_cost", "slippage_bps", "liquidation_fee", "liquidation_penalty",
    # P&L Attribution
    "realized_pnl", "unrealized_pnl", "gross_pnl", "net_pnl", "pnl_pct", "return_on_margin", "return_on_equity",
    # Margin & Risk
    "margin_type", "initial_margin", "maintenance_margin", "margin_used", "free_margin", "liquidation_price", "distance_to_liquidation_pct", "equity_before", "equity_after",
    # Perpetuals-Specific
    "mark_price", "index_price", "oracle_price", "funding_rate", "funding_payment", "open_interest",
    # Audit Trail
    "model_version", "policy_version", "config_hash", "git_commit", "seed", "data_version", "cost_basis_method",
]


def create_run_dir(base_dir: str | Path = "logs", prefix: str = "run_") -> Path:
    """Creates and returns a unique timestamped run directory in the logs folder."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / f"{prefix}{ts_str}"
    counter = 1
    orig_run_dir = run_dir
    while run_dir.exists():
        run_dir = base_path / f"{orig_run_dir.name}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class TradeHistoryLogger:
    """Logs individual perps trade executions following the docs/reporting.md schema."""

    def __init__(self, filename: str | Path, run_id: str | None = None, seed: int = 42):
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.filename.parent.name
        self.seed = seed
        self.trade_counter = 0
        self._init_file()

    def _init_file(self):
        if not self.filename.exists() or self.filename.stat().st_size == 0:
            with open(self.filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_HISTORY_COLUMNS)
                writer.writeheader()

    def log_trade(self, row: dict):
        """Appends a completed trade execution row to the trade history CSV."""
        self.trade_counter += 1
        now_utc = datetime.now(timezone.utc)
        ts_ms = int(now_utc.timestamp() * 1000)
        ts_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        full_row = {col: "" for col in TRADE_HISTORY_COLUMNS}
        full_row.update({
            "trade_id": f"T_{self.run_id}_{self.trade_counter:06d}",
            "order_id": f"ORD_{self.run_id}_{self.trade_counter:06d}",
            "timestamp": ts_ms,
            "timestamp_utc": ts_iso,
            "symbol": "BTC",
            "exchange": "DEX_MLX",
            "venue": "PureMLX_Perps",
            "strategy_id": "HRL_PPO_SAC",
            "run_id": self.run_id,
            "order_type": "market",
            "liquidity_flag": "taker",
            "fee_currency": "USD",
            "spread_cost": 0.0,
            "liquidation_fee": 0.0,
            "liquidation_penalty": 0.0,
            "model_version": "1.0.0",
            "policy_version": "PPO-SAC-HRL-v1",
            "config_hash": hashlib.sha256(self.run_id.encode()).hexdigest()[:12],
            "git_commit": "HEAD",
            "seed": self.seed,
            "data_version": "v1",
            "cost_basis_method": "FIFO",
        })
        full_row.update(row)

        formatted = {
            k: (f"{v:.6f}" if isinstance(v, float) else (str(v).lower() if isinstance(v, bool) else v))
            for k, v in full_row.items()
        }
        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_HISTORY_COLUMNS)
            writer.writerow(formatted)


def compute_financial_metrics(trades: list[dict], initial_capital: float = 10000.0) -> dict:
    """Computes key financial performance metrics for a collection of trade records."""
    num_trades = len(trades)
    if num_trades == 0:
        return {
            "num_trades": 0, "win_rate": "n/a", "avg_win": "n/a", "avg_loss": "n/a",
            "return_pct": "n/a", "net_profit": "n/a", "sharpe": "n/a", "max_dd": "n/a",
            "risk_reward": "n/a", "sortino": "n/a", "calmar": "n/a", "profit_factor": "n/a", "martin": "n/a",
        }

    pnls = [float(t.get("net_pnl", 0.0)) for t in trades]
    wins = [p for p in pnls if p > 1e-8]
    losses = [p for p in pnls if p < -1e-8]
    net_profit = sum(pnls)
    return_pct = (net_profit / initial_capital) * 100.0

    win_rate = f"{(len(wins) / num_trades) * 100.0:.2f}%"
    avg_win = f"{sum(wins) / len(wins):+.2f}" if wins else "n/a"
    avg_loss = f"{sum(losses) / len(losses):+.2f}" if losses else "n/a"
    rr = f"{abs(sum(wins) / len(wins)) / (abs(sum(losses) / len(losses)) + 1e-8):.2f}" if (wins and losses) else "n/a"
    pf = f"{sum(wins) / (abs(sum(losses)) + 1e-8):.2f}" if (losses and abs(sum(losses)) > 1e-8) else "n/a"

    eq = initial_capital
    peak = initial_capital
    dds, rets = [], []
    for p in pnls:
        rets.append(p / (eq + 1e-8))
        eq += p
        if eq > peak:
            peak = eq
        dds.append(max(0.0, (peak - eq) / (peak + 1e-8)))

    max_dd = max(dds) if dds else 0.0
    ulcer = (sum(d ** 2 for d in dds) / max(len(dds), 1)) ** 0.5

    mean_ret = sum(rets) / max(len(rets), 1)
    var_ret = sum((r - mean_ret) ** 2 for r in rets) / max(len(rets) - 1, 1) if len(rets) > 1 else 0.0
    std_ret = math.sqrt(var_ret)
    sharpe = f"{mean_ret / (std_ret + 1e-8):.2f}" if std_ret > 1e-8 else "n/a"

    downside_sq = [min(0.0, r) ** 2 for r in rets]
    downside_std = math.sqrt(sum(downside_sq) / max(len(downside_sq), 1))
    sortino = f"{mean_ret / (downside_std + 1e-8):.2f}" if downside_std > 1e-8 else "n/a"

    calmar = f"{return_pct / (max_dd * 100.0):.2f}" if max_dd > 1e-6 else "n/a"
    martin = f"{(net_profit / initial_capital) / (ulcer + 1e-8):.2f}" if ulcer > 1e-6 else "n/a"

    return {
        "num_trades": num_trades, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "return_pct": f"{return_pct:+.2f}%", "net_profit": f"{net_profit:+.2f}", "sharpe": sharpe,
        "max_dd": f"{max_dd * 100.0:.2f}%", "risk_reward": rr, "sortino": sortino,
        "calmar": calmar, "profit_factor": pf, "martin": martin,
    }


def leverage_bucket(v: float) -> str:
    """Maps a leverage value to its named bucket."""
    if v < 5:
        return "cruiser (1-5x)"
    if v < 10:
        return "charger (5-10x)"
    if v < 20:
        return "turbo (10-20x)"
    if v < 50:
        return "warp (20-50x)"
    if v < 100:
        return "hyper (50-100x)"
    return "singularity (100x+)"


LEVERAGE_BUCKET_ORDER = [
    "cruiser (1-5x)", "charger (5-10x)", "turbo (10-20x)",
    "warp (20-50x)", "hyper (50-100x)", "singularity (100x+)",
]


def collateral_bucket(v: float) -> str:
    """Maps a collateral (margin) value to its named bucket."""
    if v < 50:
        return "pocket (<$50)"
    if v < 200:
        return "small ($50-$200)"
    if v < 1000:
        return "standard ($200-$1k)"
    return "loaded (>=$1k)"


COLLATERAL_BUCKET_ORDER = [
    "pocket (<$50)", "small ($50-$200)", "standard ($200-$1k)", "loaded (>=$1k)",
]


def side_bucket(side: str) -> str:
    """Maps trade side ('long'/'buy' vs 'short'/'sell') to its named bucket."""
    s = str(side).lower()
    return "bull (long)" if s in ("long", "buy") else "bear (short)"


SIDE_BUCKET_ORDER = ["bull (long)", "bear (short)"]


def exit_type_bucket(t: dict) -> str:
    """Infers exit type from trade record attributes."""
    ext = str(t.get("exit_type", "")).lower()
    if ext in EXIT_TYPE_BUCKET_ORDER:
        return ext
    if float(t.get("liquidation_fee", 0.0) or 0.0) > 0:
        return "liquidation"
    pos_eff = str(t.get("position_effect", "")).lower()
    if pos_eff == "open":
        return "open"
    pnl = float(t.get("net_pnl", 0.0) or 0.0)
    if pos_eff in ("close", "reduce"):
        return "take_profit" if pnl > 1e-8 else ("stop_loss" if pnl < -1e-8 else "market_close")
    return "market_close"


EXIT_TYPE_BUCKET_ORDER = ["take_profit", "stop_loss", "market_close", "liquidation", "open"]


def generate_breakdown_report(trade_history_path: str | Path, output_path: str | Path | None = None, initial_capital: float = 10000.0) -> str:
    """Produces a formatted breakdown.txt summary per symbol and per episode with portfolio totals."""
    csv_file = Path(trade_history_path)
    trades = []
    if csv_file.exists() and csv_file.stat().st_size > 0:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            trades = list(reader)

    port_m = compute_financial_metrics(trades, initial_capital)

    def make_table_rows(keys, key_fn):
        rows = []
        for k in keys:
            sub = [t for t in trades if key_fn(t) == k]
            m = compute_financial_metrics(sub, initial_capital)
            rows.append([k, m["num_trades"], m["win_rate"], m["avg_win"], m["avg_loss"], m["return_pct"], m["net_profit"], m["sharpe"], m["max_dd"], m["risk_reward"], m["sortino"], m["calmar"], m["profit_factor"], m["martin"]])
        rows.append(["Portfolio", port_m["num_trades"], port_m["win_rate"], port_m["avg_win"], port_m["avg_loss"], port_m["return_pct"], port_m["net_profit"], port_m["sharpe"], port_m["max_dd"], port_m["risk_reward"], port_m["sortino"], port_m["calmar"], port_m["profit_factor"], port_m["martin"]])
        return rows

    # 1. Per Symbol Breakdown
    symbols = sorted(list(set(t.get("symbol", "UNKNOWN") for t in trades))) if trades else []
    sym_rows = make_table_rows(symbols, lambda t: t.get("symbol", "UNKNOWN"))

    # 2. Per Episode Breakdown
    episodes = sorted(list(set(t.get("strategy_id", "Episode 1") for t in trades))) if trades else []
    ep_rows = make_table_rows(episodes, lambda t: t.get("strategy_id", "Episode 1"))

    # 3. Per Leverage Breakdown
    lev_keys = sorted(set(leverage_bucket(float(t.get("leverage") or 0.0)) for t in trades), key=lambda b: LEVERAGE_BUCKET_ORDER.index(b)) if trades else []
    lev_rows = make_table_rows(lev_keys, lambda t: leverage_bucket(float(t.get("leverage") or 0.0)))

    # 4. Per Collateral Breakdown
    collat_keys = sorted(set(collateral_bucket(float(t.get("collateral") or t.get("margin_used") or t.get("initial_margin") or 0.0)) for t in trades), key=lambda b: COLLATERAL_BUCKET_ORDER.index(b)) if trades else []
    collat_rows = make_table_rows(collat_keys, lambda t: collateral_bucket(float(t.get("collateral") or t.get("margin_used") or t.get("initial_margin") or 0.0)))

    # 5. Per Side Breakdown
    side_keys = sorted(set(side_bucket(t.get("side", "long")) for t in trades), key=lambda b: SIDE_BUCKET_ORDER.index(b) if b in SIDE_BUCKET_ORDER else 99) if trades else []
    side_rows = make_table_rows(side_keys, lambda t: side_bucket(t.get("side", "long")))

    # 6. Per Exit Type Breakdown
    exit_keys = sorted(set(exit_type_bucket(t) for t in trades), key=lambda b: EXIT_TYPE_BUCKET_ORDER.index(b) if b in EXIT_TYPE_BUCKET_ORDER else 99) if trades else []
    exit_rows = make_table_rows(exit_keys, exit_type_bucket)

    cols = ["key (symbol, episode)", "num trades", "win rate %", "avg win", "avg loss", "return %", "net profit", "sharpe", "max dd", "risk reward", "sortino", "calmar", "profit factor", "martin"]
    lev_cols = ["leverage bucket"] + cols[1:]
    collat_cols = ["collateral bucket"] + cols[1:]
    side_cols = ["side"] + cols[1:]
    exit_cols = ["exit type"] + cols[1:]
    sym_table = tabulate(sym_rows, headers=cols, tablefmt="grid", disable_numparse=True)
    ep_table = tabulate(ep_rows, headers=cols, tablefmt="grid", disable_numparse=True)
    lev_table = tabulate(lev_rows, headers=lev_cols, tablefmt="grid", disable_numparse=True)
    collat_table = tabulate(collat_rows, headers=collat_cols, tablefmt="grid", disable_numparse=True)
    side_table = tabulate(side_rows, headers=side_cols, tablefmt="grid", disable_numparse=True)
    exit_table = tabulate(exit_rows, headers=exit_cols, tablefmt="grid", disable_numparse=True)

    report_lines = [
        "=" * 120,
        "                                  TRADE HISTORY PERFORMANCE BREAKDOWN",
        "=" * 120,
        "",
        "--- PER SYMBOL BREAKDOWN ---",
        sym_table,
        "",
        "--- PER EPISODE BREAKDOWN ---",
        ep_table,
        "",
        "--- PER LEVERAGE BREAKDOWN ---",
        lev_table,
        "",
        "--- PER COLLATERAL BREAKDOWN ---",
        collat_table,
        "",
        "--- PER SIDE BREAKDOWN ---",
        side_table,
        "",
        "--- PER EXIT TYPE BREAKDOWN ---",
        exit_table,
        "",
    ]
    report_text = "\n".join(report_lines)

    out_file = Path(output_path) if output_path else csv_file.parent / "breakdown.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text
