import csv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.data import generate_crypto_data, FEATURE_NAMES, REGIME_NAMES
    from scripts.config import cfg
    from scripts.report import (
        leverage_bucket, collateral_bucket, side_bucket, exit_type_bucket,
        LEVERAGE_BUCKET_ORDER, COLLATERAL_BUCKET_ORDER, SIDE_BUCKET_ORDER, EXIT_TYPE_BUCKET_ORDER,
    )
except ModuleNotFoundError:
    from data import generate_crypto_data, FEATURE_NAMES, REGIME_NAMES
    from config import cfg
    from report import (
        leverage_bucket, collateral_bucket, side_bucket, exit_type_bucket,
        LEVERAGE_BUCKET_ORDER, COLLATERAL_BUCKET_ORDER, SIDE_BUCKET_ORDER, EXIT_TYPE_BUCKET_ORDER,
    )

# synthwave = dark night · ghibli = light day
PALETTES = {
    "synthwave": {
        "mode": "dark",
        "base-100": "#1a103c", "base-200": "#241852", "base-300": "#3b2579",
        "primary": "#e779c1", "secondary": "#58c7f3", "accent": "#f3cc30",
        "neutral": "#f8f8f2", "info": "#58c7f3", "success": "#2dd4bf",
        "warning": "#f3cc30", "error": "#ff5370",
    },
    "ghibli": {
        "mode": "light",
        "base-100": "#f7f3e8", "base-200": "#efe8d8", "base-300": "#d4cbb8",
        "primary": "#3d7a6a", "secondary": "#b85c6e", "accent": "#c9a227",
        "neutral": "#2c3a3b", "info": "#4a8fa0", "success": "#3d7a6a",
        "warning": "#c9a227", "error": "#c45c48",
    },
}


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _theme_label(theme: str) -> str:
    return "SynthWave" if theme == "synthwave" else "Studio Ghibli"


def _palette(theme=None):
    theme = theme or cfg.visualization.default_theme
    return theme, PALETTES.get(theme, PALETTES["synthwave"])


def _apply_theme(fig, p, title, w=None, h=None, rangeslider=False):
    tmpl = "plotly_white" if p.get("mode") == "light" else "plotly_dark"
    fig.update_layout(
        title=dict(text=title, font=dict(color=p["neutral"], size=20)),
        template=tmpl, plot_bgcolor=p["base-100"], paper_bgcolor=p["base-200"],
        font=dict(color=p["neutral"], family="monospace"),
        width=w or cfg.visualization.width, height=h or 900,
        margin=dict(l=55, r=40, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=rangeslider,
    )
    fig.update_xaxes(gridcolor=p["base-300"], showgrid=True, zerolinecolor=p["base-300"])
    fig.update_yaxes(gridcolor=p["base-300"], showgrid=True, zerolinecolor=p["base-300"])


def _annot(fig, text, row, col, p):
    fig.add_annotation(
        text=text, xref="x domain", yref="y domain", x=0.02, y=0.98,
        xanchor="left", yanchor="top", showarrow=False, align="left",
        bgcolor=hex_to_rgba(p["base-200"], 0.88), bordercolor=p["primary"],
        borderwidth=1, borderpad=6, font=dict(size=11, color=p["neutral"], family="monospace"),
        row=row, col=col,
    )


def _save(fig, output_path):
    out = str(output_path)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    (fig.write_html if out.endswith(".html") else lambda p, **kw: fig.write_image(p, scale=2))(out)
    print(f"Saved figure to {out}")
    return fig


def plot_crypto_symbol(data_3d, symbol_idx=0, theme=None, title=None, output_path="crypto_inspection.png", window=None):
    """Multi-panel OHLC dashboard for one symbol (synthwave/ghibli)."""
    theme, p = _palette(theme)
    arr = np.array(data_3d[:, symbol_idx, :])
    if window is not None and window < arr.shape[0]:
        arr = arr[-window:]
    fm = {n: arr[:, i] for i, n in enumerate(FEATURE_NAMES)}
    T, t = arr.shape[0], np.arange(arr.shape[0])
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.44, 0.12, 0.16, 0.14, 0.14],
                        subplot_titles=[
                            f"[{_theme_label(theme)}] Price & Bands ({REGIME_NAMES[int(fm['regime'][-1])]})",
                            "Volume", "MACD (12,26,9)", "RSI (14)", "OU Funding"])
    fig.add_trace(go.Candlestick(x=t, open=fm["open"], high=fm["high"], low=fm["low"], close=fm["close"],
                                 increasing_line_color=p["success"], decreasing_line_color=p["error"], name="OHLC"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=fm["bb_upper"], line=dict(color=p["primary"], width=1.2, dash="dot"), name="BB U"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=fm["bb_lower"], fill="tonexty", fillcolor=hex_to_rgba(p["primary"], 0.10),
                             line=dict(color=p["primary"], width=1.2, dash="dot"), name="BB L"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=fm["ema12"], line=dict(color=p["accent"], width=1.8), name="EMA12"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=fm["ema26"], line=dict(color=p["secondary"], width=1.8), name="EMA26"), 1, 1)
    fig.add_trace(go.Bar(x=t, y=fm["volume"], marker_color=np.where(fm["close"] >= fm["open"], p["success"], p["error"]),
                         opacity=0.75, name="Vol"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=fm["macd"], line=dict(color=p["primary"], width=1.8), name="MACD"), 3, 1)
    fig.add_trace(go.Scatter(x=t, y=fm["macd_sig"], line=dict(color=p["secondary"], width=1.8), name="Sig"), 3, 1)
    fig.add_trace(go.Bar(x=t, y=fm["macd_hist"], marker_color=np.where(fm["macd_hist"] >= 0, p["success"], p["error"]),
                         opacity=0.65, name="Hist"), 3, 1)
    fig.add_trace(go.Scatter(x=t, y=fm["rsi14"], line=dict(color=p["accent"], width=1.8), name="RSI"), 4, 1)
    fig.add_hline(y=70, line=dict(color=p["error"], dash="dash", width=1), row=4, col=1)
    fig.add_hline(y=30, line=dict(color=p["success"], dash="dash", width=1), row=4, col=1)
    fig.add_trace(go.Scatter(x=t, y=fm["funding"], line=dict(color=p["primary"], width=1.8), name="Fund"), 5, 1)
    fig.add_hline(y=0, line=dict(color=p["base-300"], dash="dash", width=1), row=5, col=1)
    rc = [hex_to_rgba(p[k], a) for k, a in [("success", 0.12), ("error", 0.12), ("info", 0.08), ("error", 0.25), ("warning", 0.20)]]
    regimes = fm["regime"].astype(int)
    pts = np.where(np.diff(regimes, prepend=regimes[0]))[0]
    for seg in (np.split(np.arange(T), pts) if len(pts) else [np.arange(T)]):
        if len(seg):
            fig.add_vrect(x0=seg[0] - 0.5, x1=seg[-1] + 0.5, fillcolor=rc[regimes[seg[0]]],
                          layer="below", line_width=0, row=1, col=1)
    _apply_theme(fig, p, title or f"Multi-Regime Crypto (Symbol #{symbol_idx})",
                 h=cfg.visualization.height, rangeslider=False)
    return _save(fig, output_path)


def _load_trades(path, closed_only=True) -> list[dict]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not closed_only:
        return rows
    closed = [
        t for t in rows
        if str(t.get("position_effect", "")).lower() in ("close", "reduce")
        or str(t.get("exit_type", "")).lower() in ("take_profit", "stop_loss", "market_close", "liquidation")
    ]
    if not closed and rows:
        closed = [t for t in rows if str(t.get("position_effect", "")).lower() != "open" and str(t.get("exit_type", "")).lower() != "open"]
        closed = closed or rows
    return closed


def _f(row, key, default=0.0) -> float:
    try:
        v = row.get(key, "")
        return float(v) if v not in ("", None) else default
    except (TypeError, ValueError):
        return default


def _trade_series(trades, initial_capital):
    closed = [
        t for t in trades
        if str(t.get("position_effect", "")).lower() in ("close", "reduce")
        or str(t.get("exit_type", "")).lower() in ("take_profit", "stop_loss", "market_close", "liquidation")
    ]
    closed = closed or list(trades)
    closed.sort(key=lambda t: t.get("timestamp_utc") or t.get("timestamp") or "")
    dates, net_pnls, gross_pnls, net_eqc, gross_eqc, rets, dds = [], [], [], [], [], [], []
    net_eq = gross_eq = peak = initial_capital
    for t in closed:
        net_pnl = _f(t, "net_pnl")
        gross_pnl = _f(t, "gross_pnl", default=net_pnl)
        dates.append(t.get("timestamp_utc") or t.get("timestamp") or str(len(dates)))
        net_pnls.append(net_pnl)
        gross_pnls.append(gross_pnl)
        if t.get("pnl_pct") not in ("", None):
            ret = _f(t, "pnl_pct")
        elif t.get("return_on_margin") not in ("", None):
            ret = _f(t, "return_on_margin") * 100.0
        else:
            margin = _f(t, "margin_used") or _f(t, "initial_margin") or _f(t, "collateral") or (initial_capital * 0.02)
            ret = (net_pnl / (margin + 1e-8)) * 100.0
        ret = max(-100.0, ret)
        rets.append(ret)
        net_eq += net_pnl
        gross_eq += gross_pnl
        peak = max(peak, net_eq)
        net_eqc.append(net_eq)
        gross_eqc.append(gross_eq)
        dds.append(max(0.0, (peak - net_eq) / (peak + 1e-8)) * 100.0)
    return dates, np.asarray(net_pnls), np.asarray(gross_pnls), np.asarray(net_eqc), np.asarray(gross_eqc), np.asarray(rets), np.asarray(dds), net_eq, gross_eq, peak


def plot_trade_performance(trade_history_path, theme=None, initial_capital=10000.0, output_path="trade_performance.png"):
    """2x2: equity (gross/net) | trade returns | drawdown (downward) | returns distrib."""
    theme, p = _palette(theme)
    dates, net_pnls, gross_pnls, net_eq, gross_eq, rets, dds, final_net_eq, final_gross_eq, peak = _trade_series(
        _load_trades(trade_history_path), initial_capital
    )
    n = len(dates)
    if not n:
        dates, net_eq, gross_eq = ["n/a"], np.array([initial_capital]), np.array([initial_capital])
        net_pnls = gross_pnls = rets = dds = np.zeros(1)
        final_net_eq = final_gross_eq = peak = initial_capital
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.08,
                        subplot_titles=["Equity Curve", "Trade Returns", "Drawdown", "Returns Distribution"])
    fig.add_trace(go.Scatter(x=dates, y=gross_eq, mode="lines", line=dict(color=p["secondary"], width=1.8, dash="dot"),
                             name="Gross Eq"), 1, 1)
    fig.add_trace(go.Scatter(x=dates, y=net_eq, mode="lines", line=dict(color=p["primary"], width=2.2),
                             fill="tozeroy", fillcolor=hex_to_rgba(p["primary"], 0.12), name="Net Eq"), 1, 1)
    fig.add_hline(y=initial_capital, line=dict(color=p["base-300"], dash="dash", width=1), row=1, col=1)
    ret_pct = ((final_net_eq - initial_capital) / initial_capital) * 100.0 if n else 0.0
    _annot(fig, f"n={n}<br>net={final_net_eq:,.0f} ({ret_pct:+.1f}%)<br>gross={final_gross_eq:,.0f}<br>peak={peak:,.0f}", 1, 1, p)
    fig.add_trace(go.Scatter(
        x=dates, y=rets, mode="markers",
        marker=dict(color=[p["success"] if v >= 0 else p["error"] for v in rets], size=7, opacity=0.85,
                    line=dict(width=0.5, color=p["neutral"])),
        name="Ret%"), 1, 2)
    fig.add_hline(y=0, line=dict(color=p["base-300"], dash="dash", width=1), row=1, col=2)
    wins, losses = int(np.sum(net_pnls > 1e-8)) if n else 0, int(np.sum(net_pnls < -1e-8)) if n else 0
    avg_w = float(np.mean(rets[rets > 1e-8])) if wins else 0.0
    avg_l = float(np.mean(rets[rets < -1e-8])) if losses else 0.0
    _annot(fig, f"wins={wins} losses={losses}<br>avgW={avg_w:+.1f}%<br>avgL={avg_l:+.1f}%<br>netPnL={float(net_pnls.sum()):+.1f}", 1, 2, p)
    neg_dds = -np.asarray(dds)
    fig.add_trace(go.Scatter(x=dates, y=neg_dds, mode="lines", line=dict(color=p["error"], width=2),
                             fill="tozeroy", fillcolor=hex_to_rgba(p["error"], 0.18), name="DD%"), 2, 1)
    fig.add_hline(y=0, line=dict(color=p["base-300"], dash="dash", width=1), row=2, col=1)
    _annot(fig, f"maxDD=-{float(dds.max()):.2f}%<br>endDD=-{float(dds[-1]):.2f}%<br>meanDD=-{float(dds.mean()):.2f}%", 2, 1, p)
    fig.add_trace(go.Histogram(x=rets, nbinsx=min(40, max(10, n // 2 or 10)),
                               marker_color=p["secondary"], opacity=0.8, name="Ret% Dist"), 2, 2)
    mean_r, med_r, std_r = (float(rets.mean()), float(np.median(rets)), float(rets.std())) if n else (0.0, 0.0, 0.0)
    _annot(fig, f"mean={mean_r:+.2f}%<br>med={med_r:+.2f}%<br>std={std_r:.2f}%<br>n={n}", 2, 2, p)
    for r, c, xl, yl in [(1, 1, "close date", "equity ($)"), (1, 2, "close date", "trade return %"),
                         (2, 1, "close date", "drawdown %"), (2, 2, "return %", "count")]:
        fig.update_xaxes(title_text=xl, row=r, col=c)
        fig.update_yaxes(title_text=yl, row=r, col=c)
    _apply_theme(fig, p, f"[{_theme_label(theme)}] Trade Performance")
    return _save(fig, output_path)


def plot_trade_distributions(trade_history_path, theme=None, output_path="trade_distributions.png"):
    """2x2: leverage | collateral | side | exit-type distributions."""
    theme, p = _palette(theme)
    trades = _load_trades(trade_history_path)
    n = len(trades)

    def counts(order, fn):
        raw = {}
        for t in trades:
            k = fn(t)
            raw[k] = raw.get(k, 0) + 1
        keys = [k for k in order if k in raw] + [k for k in raw if k not in order]
        return keys or ["n/a"], ([raw[k] for k in keys] if keys else [0])

    specs = [
        (1, 1, *counts(LEVERAGE_BUCKET_ORDER, lambda t: leverage_bucket(_f(t, "leverage"))), p["primary"], "leverage"),
        (1, 2, *counts(COLLATERAL_BUCKET_ORDER, lambda t: collateral_bucket(
            _f(t, "collateral") or _f(t, "margin_used") or _f(t, "initial_margin"))), p["secondary"], "collateral"),
        (2, 1, *counts(SIDE_BUCKET_ORDER, lambda t: side_bucket(t.get("side", "long"))), p["accent"], "side"),
        (2, 2, *counts(EXIT_TYPE_BUCKET_ORDER, exit_type_bucket), p["info"], "exit"),
    ]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
                        subplot_titles=["Leverage Distribution", "Collateral Distribution",
                                        "Side Distribution", "Exit Type Distribution"])
    for row, col, keys, vals, color, kind in specs:
        fig.add_trace(go.Bar(x=keys, y=vals, marker_color=color, opacity=0.85, name=kind,
                             text=vals, textposition="outside"), row, col)
        total = sum(vals) or 1
        top = max(range(len(vals)), key=lambda i: vals[i])
        lines = [f"n={sum(vals)}"] + [
            f"{k.split('(')[0].strip()[:12]}={v} ({100 * v / total:.0f}%)"
            for k, v in sorted(zip(keys, vals), key=lambda kv: -kv[1])[:4]
        ] + [f"mode={keys[top][:18]}"]
        _annot(fig, "<br>".join(lines), row, col, p)
        fig.update_xaxes(title_text=kind, tickangle=-25, row=row, col=col)
        fig.update_yaxes(title_text="count", row=row, col=col)
    _apply_theme(fig, p, f"[{_theme_label(theme)}] Trade Style Distributions (n={n})")
    return _save(fig, output_path)


def generate_trade_figures(trade_history_path, out_dir=None, theme=None, initial_capital=10000.0):
    """After breakdown.txt: write performance + distribution 2x2 PNGs."""
    csv_path = Path(trade_history_path)
    dest = Path(out_dir) if out_dir else csv_path.parent
    dest.mkdir(parents=True, exist_ok=True)
    theme = theme or cfg.visualization.default_theme
    perf, dist = dest / f"trade_performance_{theme}.png", dest / f"trade_distributions_{theme}.png"
    plot_trade_performance(csv_path, theme=theme, initial_capital=initial_capital, output_path=perf)
    plot_trade_distributions(csv_path, theme=theme, output_path=dist)
    return perf, dist


if __name__ == "__main__":
    import argparse
    import mlx.core as mx
    ap = argparse.ArgumentParser(description="Crypto / trade history Plotly figures")
    ap.add_argument("--theme", choices=["synthwave", "ghibli"], default="synthwave")
    ap.add_argument("--symbol", type=int, default=0)
    ap.add_argument("--candles", type=int, default=600)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--symbols", type=int, default=3)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--trades", type=str, default=None, help="trade_history.csv → 2x2 figures")
    ap.add_argument("--capital", type=float, default=10000.0)
    args = ap.parse_args()
    if args.trades:
        generate_trade_figures(args.trades, theme=args.theme, initial_capital=args.capital)
    else:
        data = generate_crypto_data(num_candles=args.candles, num_symbols=args.symbols)
        mx.eval(data)
        plot_crypto_symbol(data, symbol_idx=args.symbol, theme=args.theme,
                           output_path=args.output or f"crypto_{args.theme}.png", window=args.window)
