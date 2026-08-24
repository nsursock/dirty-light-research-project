import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.data import generate_crypto_data, FEATURE_NAMES, REGIME_NAMES
    from scripts.config import cfg
except ModuleNotFoundError:
    from data import generate_crypto_data, FEATURE_NAMES, REGIME_NAMES
    from config import cfg

PALETTES = {
    "synthwave": {
        "base-100": "#1a103c",
        "base-200": "#241852",
        "base-300": "#3b2579",
        "primary": "#e779c1",
        "secondary": "#58c7f3",
        "accent": "#f3cc30",
        "neutral": "#f8f8f2",
        "info": "#58c7f3",
        "success": "#2dd4bf",
        "warning": "#f3cc30",
        "error": "#ff5370",
    },
    "ghibli": {
        "base-100": "#1e292a",
        "base-200": "#283739",
        "base-300": "#3a4f50",
        "primary": "#68b0ab",
        "secondary": "#c37b89",
        "accent": "#f4ca64",
        "neutral": "#ecebe4",
        "info": "#7db3c6",
        "success": "#68b0ab",
        "warning": "#f4ca64",
        "error": "#d66853",
    }
}

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"

def plot_crypto_symbol(data_3d, symbol_idx=0, theme=None, title=None, output_path="crypto_inspection.png", window=None):
    """
    Renders and saves a multi-panel dashboard for 1 crypto symbol as a high-res PNG (or HTML)
    using only standard semantic design tokens (base-100/200/300, primary, secondary, accent, neutral, info, success, warning, error).

    window: number of most-recent candles to display (None = all candles).
    """
    theme = theme or cfg.visualization.default_theme
    p = PALETTES.get(theme, PALETTES["synthwave"])
    arr = np.array(data_3d[:, symbol_idx, :]) # (T, F)
    if window is not None and window < arr.shape[0]:
        arr = arr[-window:]
    feat_map = {name: arr[:, i] for i, name in enumerate(FEATURE_NAMES)}
    T = arr.shape[0]
    time_idx = np.arange(T)

    theme_label = "SynthWave" if theme == "synthwave" else "Studio Ghibli"
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.44, 0.12, 0.16, 0.14, 0.14],
        subplot_titles=[
            f"[{theme_label}] Price Action & Bands (Current: {REGIME_NAMES[int(feat_map['regime'][-1])]} Regime)",
            "Volume",
            "MACD (12, 26, 9)",
            "RSI (14)",
            "OU Funding Rate"
        ]
    )

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=time_idx,
        open=feat_map["open"], high=feat_map["high"],
        low=feat_map["low"], close=feat_map["close"],
        increasing_line_color=p["success"],
        decreasing_line_color=p["error"],
        name="OHLC"
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=time_idx, y=feat_map["bb_upper"], line=dict(color=p["primary"], width=1.2, dash="dot"), name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_idx, y=feat_map["bb_lower"], fill="tonexty", fillcolor=hex_to_rgba(p["primary"], 0.10), line=dict(color=p["primary"], width=1.2, dash="dot"), name="BB Lower"), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=time_idx, y=feat_map["ema12"], line=dict(color=p["accent"], width=1.8), name="EMA 12"), row=1, col=1)
    fig.add_trace(go.Scatter(x=time_idx, y=feat_map["ema26"], line=dict(color=p["secondary"], width=1.8), name="EMA 26"), row=1, col=1)

    # 2. Volume
    vol_colors = np.where(feat_map["close"] >= feat_map["open"], p["success"], p["error"])
    fig.add_trace(go.Bar(x=time_idx, y=feat_map["volume"], marker_color=vol_colors, opacity=0.75, name="Volume"), row=2, col=1)

    # 3. MACD
    fig.add_trace(go.Scatter(x=time_idx, y=feat_map["macd"], line=dict(color=p["primary"], width=1.8), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=time_idx, y=feat_map["macd_sig"], line=dict(color=p["secondary"], width=1.8), name="Signal"), row=3, col=1)
    hist_colors = np.where(feat_map["macd_hist"] >= 0, p["success"], p["error"])
    fig.add_trace(go.Bar(x=time_idx, y=feat_map["macd_hist"], marker_color=hist_colors, opacity=0.65, name="Hist"), row=3, col=1)

    # 4. RSI
    fig.add_trace(go.Scatter(x=time_idx, y=feat_map["rsi14"], line=dict(color=p["accent"], width=1.8), name="RSI 14"), row=4, col=1)
    fig.add_hline(y=70, line=dict(color=p["error"], dash="dash", width=1), row=4, col=1)
    fig.add_hline(y=30, line=dict(color=p["success"], dash="dash", width=1), row=4, col=1)

    # 5. Funding Rate (OU process)
    fig.add_trace(go.Scatter(x=time_idx, y=feat_map["funding"], line=dict(color=p["primary"], width=1.8), name="Funding"), row=5, col=1)
    fig.add_hline(y=0, line=dict(color=p["base-300"], dash="dash", width=1), row=5, col=1)

    # 5 Market Regimes background shading mapped from semantic tokens:
    # 0 Bull -> success, 1 Bear -> error, 2 Range -> info, 3 Crash -> error (stronger), 4 Mania -> warning
    regime_colors = [
        hex_to_rgba(p["success"], 0.12),
        hex_to_rgba(p["error"], 0.12),
        hex_to_rgba(p["info"], 0.08),
        hex_to_rgba(p["error"], 0.25),
        hex_to_rgba(p["warning"], 0.20),
    ]

    regimes = feat_map["regime"].astype(int)
    change_pts = np.where(np.diff(regimes, prepend=regimes[0]))[0]
    segments = np.split(np.arange(T), change_pts) if len(change_pts) > 0 else [np.arange(T)]
    for seg in segments:
        if len(seg) == 0:
            continue
        reg_id = regimes[seg[0]]
        fig.add_vrect(
            x0=seg[0] - 0.5, x1=seg[-1] + 0.5,
            fillcolor=regime_colors[reg_id],
            layer="below", line_width=0,
            row=1, col=1
        )

    title_text = title or f"Multi-Regime Crypto Market Simulation (Symbol #{symbol_idx})"
    fig.update_layout(
        title=dict(text=title_text, font=dict(color=p["neutral"], size=22)),
        template="plotly_dark",
        plot_bgcolor=p["base-100"],
        paper_bgcolor=p["base-200"],
        font=dict(color=p["neutral"], family="monospace"),
        xaxis_rangeslider_visible=False,
        width=cfg.visualization.width,
        height=cfg.visualization.height,
        margin=dict(l=60, r=40, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    for i in range(1, 6):
        fig.update_xaxes(gridcolor=p["base-300"], showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor=p["base-300"], showgrid=True, row=i, col=1)

    if output_path.endswith(".html"):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path, scale=2)
    print(f"Saved figure to {output_path}")
    return fig

if __name__ == "__main__":
    import argparse
    import mlx.core as mx

    parser = argparse.ArgumentParser(description="Inspect simulated crypto market data with Plotly")
    parser.add_argument("--theme", choices=["synthwave", "ghibli"], default="synthwave", help="Color palette")
    parser.add_argument("--symbol", type=int, default=0, help="Symbol index to inspect")
    parser.add_argument("--candles", type=int, default=600, help="Total number of simulated candles generated")
    parser.add_argument("--window", type=int, default=None, help="Number of most-recent candles to display (None = all)")
    parser.add_argument("--symbols", type=int, default=3, help="Total number of simulated symbols")
    parser.add_argument("--output", type=str, default=None, help="Output PNG/HTML filepath")
    args = parser.parse_args()

    out_file = args.output or f"crypto_{args.theme}.png"
    data = generate_crypto_data(num_candles=args.candles, num_symbols=args.symbols)
    mx.eval(data)
    plot_crypto_symbol(data, symbol_idx=args.symbol, theme=args.theme, output_path=out_file, window=args.window)
