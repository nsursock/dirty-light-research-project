import sys
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from scripts.config import cfg
    from scripts.data import generate_multi_tf_data, get_timeframe_ratio, FEATURE_NAMES
except ModuleNotFoundError:
    from config import cfg
    from data import generate_multi_tf_data, get_timeframe_ratio, FEATURE_NAMES


class MultiCryptoDexPerpEnv:
    """Vectorized Pure MLX DEX Perps Trading Env with realistic fills, liqs, TP, and SL."""

    def __init__(
        self, num_envs: int = 1, data: mx.array | None = None, high_tf_data: mx.array | None = None,
        high_tf: str | None = None, low_tf: str | None = None, num_symbols: int | None = None,
        num_candles: int | None = None, margin_mode: str | None = None, max_open_pos_per_symbol: int = 1,
        initial_capital: float | None = None, min_leverage: float | None = None, max_leverage: float | None = None,
        min_risk_per_trade: float | None = None, max_risk_per_trade: float | None = None,
        min_collateral: float | None = None, max_collateral: float | None = None, fee_rate: float | None = None,
        slippage_coef: float | None = None, maintenance_margin_rate: float | None = None,
        liquidation_penalty: float | None = None, martin_penalty_weight: float | None = None,
        take_profit_pct: float | None = None, stop_loss_pct: float | None = None,
        record_trades: bool = False, config=None,
    ):
        c = config or cfg
        env_cfg, sim_cfg = getattr(c, "env", {}), getattr(c, "simulation", {})
        self.num_envs, self.num_symbols, self.num_candles = max(1, int(num_envs)), num_symbols or getattr(c.data, "num_symbols", 4), num_candles or getattr(c.data, "num_candles", 600)
        self.high_tf, self.low_tf = high_tf or getattr(sim_cfg, "high_tf", "1h"), low_tf or getattr(sim_cfg, "low_tf", "5m")
        self.macro_period = get_timeframe_ratio(self.high_tf, self.low_tf)
        self.margin_mode = (margin_mode or getattr(env_cfg, "margin_mode", "isolated")).lower()
        assert self.margin_mode in ("isolated", "cross"), f"Invalid margin_mode: {self.margin_mode}"
        self.max_open_pos, self.initial_capital = max_open_pos_per_symbol or getattr(env_cfg, "max_open_pos_per_symbol", 1), float(initial_capital or getattr(env_cfg, "initial_capital", 10000.0))
        self.min_leverage, self.max_leverage = float(min_leverage or getattr(env_cfg, "min_leverage", 2.0)), float(max_leverage or getattr(env_cfg, "max_leverage", 150.0))
        self.min_risk, self.max_risk = float(min_risk_per_trade or getattr(env_cfg, "min_risk_per_trade", 0.01)), float(max_risk_per_trade or getattr(env_cfg, "max_risk_per_trade", 0.05))
        self.min_collateral, self.max_collateral = float(min_collateral or getattr(env_cfg, "min_collateral", 10.0)), float(max_collateral or getattr(env_cfg, "max_collateral", 1000.0))
        self.fee_rate, self.slippage_coef = float(fee_rate or getattr(env_cfg, "fee_rate", 0.0006)), float(slippage_coef or getattr(env_cfg, "slippage_coef", 0.0001))
        self.fixed_mmr = float(maintenance_margin_rate) if maintenance_margin_rate is not None else getattr(env_cfg, "maintenance_margin_rate", None)
        self.liq_penalty, self.martin_weight = float(liquidation_penalty or getattr(env_cfg, "liquidation_penalty", 0.01)), float(martin_penalty_weight or getattr(env_cfg, "martin_penalty_weight", 5.0))
        self.bankrupt_thresh = float(getattr(env_cfg, "bankruptcy_threshold", 1e-4)) * self.initial_capital
        self.tp_pct, self.sl_pct = (float(take_profit_pct) if take_profit_pct is not None else getattr(env_cfg, "take_profit_pct", 0.04)), (float(stop_loss_pct) if stop_loss_pct is not None else getattr(env_cfg, "stop_loss_pct", 0.02))
        self.record_trades = record_trades
        self.data, self.high_tf_data = data, high_tf_data
        self.num_features = len(FEATURE_NAMES)
        self.obs_dim = self.num_symbols * self.num_features + self.num_symbols + 3
        self.reset(data=self.data, high_tf_data=self.high_tf_data)

    def reset(self, data: mx.array | None = None, high_tf_data: mx.array | None = None, seed: int | None = None):
        if seed is not None:
            mx.random.seed(seed)
            if data is None:
                self.data, self.high_tf_data, _ = generate_multi_tf_data(
                    num_candles=self.num_candles, num_symbols=self.num_symbols, high_tf=self.high_tf, low_tf=self.low_tf
                )
        elif data is not None:
            self.data, self.high_tf_data = data, high_tf_data
        elif self.data is None:
            self.data, self.high_tf_data, _ = generate_multi_tf_data(
                num_candles=self.num_candles, num_symbols=self.num_symbols, high_tf=self.high_tf, low_tf=self.low_tf
            )

        self.t = 0
        self.num_candles, self.num_symbols = self.data.shape[0], self.data.shape[1]
        self.equity, self.peak_equity = mx.full((self.num_envs, 1), self.initial_capital), mx.full((self.num_envs, 1), self.initial_capital)
        self.notionals, self.positions, self.entry_prices = mx.zeros((self.num_envs, self.num_symbols)), mx.zeros((self.num_envs, self.num_symbols)), mx.zeros((self.num_envs, self.num_symbols))
        self.sum_dd_sq, self.step_cnt = mx.zeros((self.num_envs, 1)), 0
        self.total_liq_count, self.total_tp_count, self.total_sl_count, self.total_trades = mx.zeros((), dtype=mx.int32), mx.zeros((), dtype=mx.int32), mx.zeros((), dtype=mx.int32), 0

        obs = self._get_obs()
        return (obs[0], self._get_info()) if self.num_envs == 1 else (obs, self._get_info())

    def _get_obs(self) -> mx.array:
        t_idx = min(self.t, self.num_candles - 1)
        market_feats = self.data[t_idx].reshape(-1)
        m_tile = mx.tile(market_feats[None, :], (self.num_envs, 1)) if self.num_envs > 1 else market_feats[None, :]
        norm_eq, norm_peak = self.equity / self.initial_capital, self.peak_equity / self.initial_capital
        dd = mx.maximum(mx.zeros_like(self.equity), (self.peak_equity - self.equity) / (self.peak_equity + 1e-8))
        return mx.concatenate([m_tile, self.positions, norm_eq, norm_peak, dd], axis=-1)

    def get_macro_obs(self) -> mx.array:
        if self.high_tf_data is None:
            return self._get_obs() if self.num_envs > 1 else self._get_obs()[0]
        t_high = min(self.t // self.macro_period, self.high_tf_data.shape[0] - 1)
        macro_feats = self.high_tf_data[t_high].reshape(-1)
        m_tile = mx.tile(macro_feats[None, :], (self.num_envs, 1)) if self.num_envs > 1 else macro_feats[None, :]
        norm_eq, norm_peak = self.equity / self.initial_capital, self.peak_equity / self.initial_capital
        dd = mx.maximum(mx.zeros_like(self.equity), (self.peak_equity - self.equity) / (self.peak_equity + 1e-8))
        obs = mx.concatenate([m_tile, self.positions, norm_eq, norm_peak, dd], axis=-1)
        return obs if self.num_envs > 1 else obs[0]

    def _get_info(self) -> dict:
        cur_dd = mx.maximum(mx.zeros_like(self.equity), (self.peak_equity - self.equity) / (self.peak_equity + 1e-8))
        ulcer = mx.sqrt(self.sum_dd_sq / max(self.step_cnt, 1))
        ret = (self.equity - self.initial_capital) / self.initial_capital
        martin = ret / (ulcer + 1e-6)
        return {
            "step": self.t, "equity": float(self.equity[0, 0].item()), "peak_equity": float(self.peak_equity[0, 0].item()),
            "drawdown": float(cur_dd[0, 0].item()), "ulcer_index": float(ulcer[0, 0].item()), "martin_ratio": float(martin[0, 0].item()),
            "margin_mode": self.margin_mode, "total_liquidations": int(self.total_liq_count.item()),
            "total_trades": int(self.total_trades), "total_tp_hits": int(self.total_tp_count.item()), "total_sl_hits": int(self.total_sl_count.item()),
        }

    def _extract_trades(self, delta_n, target_n, prev_n, is_liq, is_sl, is_tp, fill_p, lev, col, mmr, net_pnl, pnl, liq_pl, liq_ps, liq_pen, fund, slip, c_close, vol, sl_exit, tp_exit, prev_eq, new_eq, tot_col, liq_occ, liq_loss):
        d_not, t_not, p_not = np.array(delta_n[0]), np.array(target_n[0]), np.array(prev_n[0])
        liq_sym = np.array(is_liq[0]) if self.margin_mode == "isolated" else np.array([bool(liq_occ[0, 0].item())] * self.num_symbols)
        sl_sym, tp_sym = np.array(is_sl[0]), np.array(is_tp[0])
        traded = (np.abs(d_not) > 1.0) | liq_sym | sl_sym | tp_sym | ((np.abs(p_not) > 1.0) & (np.abs(t_not) <= 1.0))
        active = np.where(traded)[0]
        if len(active) == 0: return []
        fill_a, lev_a, col_a, mmr_a = np.array(fill_p[0]), np.array(lev[0]), np.array(col[0]), np.array(mmr[0])
        np_a, gp_a = np.array(net_pnl[0]), np.array(pnl[0])
        liq_la, liq_sa = np.array(liq_pl[0]), np.array(liq_ps[0])
        liq_pena = np.array(liq_pen[0]) if self.margin_mode == "isolated" and liq_pen is not None else None
        fund_a, slip_a, cc_a, vol_a = np.array(fund), np.array(slip[0]), np.array(c_close), np.array(vol)
        sl_ex_a, tp_ex_a = np.array(sl_exit[0]), np.array(tp_exit[0])
        peq_v, neq_v, tc_v = float(prev_eq[0, 0].item()), float(new_eq[0, 0].item()), float(tot_col[0, 0].item())
        records = []
        for s in active:
            dn, tn, pn = float(d_not[s]), float(t_not[s]), float(p_not[s])
            side = "buy" if (dn > 0 or (dn == 0 and pn > 0)) else "sell"
            is_l, is_s, is_t = bool(liq_sym[s]), bool(sl_sym[s]), bool(tp_sym[s])
            if is_l: et, pe = "liquidation", "close"
            elif is_s: et, pe = "stop_loss", "close"
            elif is_t: et, pe = "take_profit", "close"
            elif abs(tn) <= 1.0 and abs(pn) > 1.0: et, pe = "market_close", "close"
            elif abs(pn) <= 1.0 and abs(tn) > 1.0: et, pe = "open", "open"
            else: et, pe = "market_close", ("increase" if abs(tn) > abs(pn) else "reduce")
            s_lev, s_col = float(lev_a[s]), float(col_a[s])
            s_p = float(sl_ex_a[s]) if is_s else (float(tp_ex_a[s]) if is_t else float(fill_a[s]))
            s_not, s_mmr = (abs(dn) if abs(dn) > 1.0 else abs(pn)), float(mmr_a[s])
            s_np = -s_col if is_l else float(np_a[s])
            s_gp = -s_col if is_l else float(gp_a[s])
            lp = float(liq_la[s] if tn > 0 else (liq_sa[s] if tn < 0 else 0.0))
            dist_liq = max(0.0, (s_p - lp) / s_p * 100.0) if tn > 0 else (max(0.0, (lp - s_p) / s_p * 100.0) if tn < 0 else 100.0)
            lf = float(liq_pena[s]) if (self.margin_mode == "isolated" and is_l) else (float(liq_loss[0, 0].item()) if (self.margin_mode == "cross" and bool(liq_occ[0, 0].item())) else 0.0)
            records.append({
                "symbol_idx": int(s), "side": side, "position_effect": pe, "exit_type": et,
                "quantity": s_not / (s_p + 1e-8), "filled_quantity": s_not / (s_p + 1e-8), "price": s_p, "notional_value": s_not,
                "leverage": s_lev, "fee_amount": float(self.fee_rate * abs(dn)), "funding_fee": float(tn * fund_a[s]),
                "slippage_bps": (float(slip_a[s] * abs(dn)) / (abs(dn) + 1e-8)) * 10000.0 if abs(dn) > 1e-4 else 0.0,
                "liquidation_fee": lf, "liquidation_penalty": self.liq_penalty if is_l else 0.0,
                "realized_pnl": s_np if pe in ("close", "reduce") else 0.0, "unrealized_pnl": s_np if pe in ("open", "increase") else 0.0,
                "gross_pnl": s_gp, "net_pnl": s_np, "pnl_pct": (s_np / (s_col + 1e-8)) * 100.0,
                "return_on_margin": s_np / (s_col + 1e-8), "return_on_equity": s_np / (peq_v + 1e-8),
                "margin_type": self.margin_mode, "initial_margin": s_col, "maintenance_margin": abs(tn) * s_mmr, "margin_used": s_col, "free_margin": max(0.0, peq_v - tc_v),
                "liquidation_price": lp, "distance_to_liquidation_pct": dist_liq, "equity_before": peq_v, "equity_after": neq_v,
                "mark_price": float(cc_a[s]), "index_price": float(cc_a[s]), "oracle_price": float(cc_a[s]),
                "funding_rate": float(fund_a[s]), "funding_payment": float(tn * fund_a[s]), "open_interest": float(vol_a[s] * 10.0),
            })
        return records

    def step(self, action: mx.array | list):
        if not isinstance(action, mx.array):
            action = mx.array(action, dtype=mx.float32)
        if action.ndim == 1 and self.num_envs > 1:
            action = mx.tile(action[None, :], (self.num_envs, 1))
        elif action.ndim == 1 and self.num_envs == 1:
            action = action[None, :]

        action = mx.clip(action, -1.0, 1.0)
        abs_act, direction = mx.abs(action), mx.sign(action)
        is_active = abs_act > 1e-4

        # 1. Dynamic Collateral & Target Sizing
        risk = self.min_risk + abs_act * (self.max_risk - self.min_risk)
        clamped_col = mx.clip(risk * self.equity, self.min_collateral, self.max_collateral)
        tot_col = mx.sum(mx.where(is_active, clamped_col, 0.0), axis=-1, keepdims=True)
        scale_c = mx.minimum(mx.ones_like(self.equity), self.equity / (tot_col + 1e-8))
        collateral = mx.where(is_active, clamped_col * scale_c, 0.0)

        if self.max_leverage > self.min_leverage > 0:
            leverage = self.min_leverage * ((self.max_leverage / self.min_leverage) ** abs_act)
        else:
            leverage = mx.full(abs_act.shape, self.min_leverage)
        target_notional = mx.where(is_active, direction * collateral * leverage, 0.0)

        # 2. Pessimistic Low-TF Fills & Costs
        t_idx = min(self.t, self.num_candles - 1)
        c_open, c_high, c_low, c_close, vol, funding_rate = self.data[t_idx, :, 0], self.data[t_idx, :, 1], self.data[t_idx, :, 2], self.data[t_idx, :, 3], self.data[t_idx, :, 4], self.data[t_idx, :, 5]
        prev_notionals, delta_notional = self.notionals, target_notional - self.notionals
        abs_delta = mx.abs(delta_notional)
        pool_liq = mx.maximum(vol * c_close, 1e4)
        slippage_rate = self.slippage_coef * (abs_delta / pool_liq)
        slippage_cost = mx.sum(slippage_rate * abs_delta, axis=-1, keepdims=True)
        fee_cost = mx.sum(self.fee_rate * abs_delta, axis=-1, keepdims=True)
        total_costs = fee_cost + slippage_cost

        # Pessimistic fills: BUY at adverse high, SELL at adverse low
        pessimistic_fill = mx.where(
            delta_notional > 1e-4, mx.clip(c_close * (1.0 + slippage_rate), c_close, c_high),
            mx.where(delta_notional < -1e-4, mx.clip(c_close * (1.0 - slippage_rate), c_low, c_close), c_close)
        )
        is_new_pos = (mx.abs(prev_notionals) < 1e-4) | (mx.sign(target_notional) != mx.sign(prev_notionals))
        is_increase = (mx.sign(target_notional) == mx.sign(prev_notionals)) & (mx.abs(target_notional) > mx.abs(prev_notionals))
        weighted_entry = (mx.abs(prev_notionals) * self.entry_prices + abs_delta * pessimistic_fill) / (mx.abs(target_notional) + 1e-8)
        self.entry_prices = mx.where(is_active, mx.where(is_new_pos, pessimistic_fill, mx.where(is_increase, weighted_entry, self.entry_prices)), mx.zeros_like(self.entry_prices))

        # 3. Next Low-TF Candle & Intra-Bar Price Extremes
        next_t, done, liq_occurred = self.t + 1, False, False
        if next_t >= self.num_candles:
            done, n_open, n_high, n_low, n_close, n_funding = True, c_close, c_close, c_close, c_close, funding_rate
        else:
            n_candle = self.data[next_t]
            n_open, n_high, n_low, n_close, n_funding = n_candle[:, 0], n_candle[:, 1], n_candle[:, 2], n_candle[:, 3], n_candle[:, 5]

        # 4. Pessimistic Low-TF Liq, SL, TP Evaluation (Liq > SL > TP > Hold)
        mmr_sym = mx.full(leverage.shape, self.fixed_mmr) if self.fixed_mmr is not None else (1.0 / (2.0 * mx.maximum(leverage, 1.0)))
        is_long, is_short = target_notional > 1e-4, target_notional < -1e-4
        liq_p_long = self.entry_prices * (1.0 - (1.0 / mx.maximum(leverage, 1.0)) + mmr_sym)
        liq_p_short = self.entry_prices * (1.0 + (1.0 / mx.maximum(leverage, 1.0)) - mmr_sym)
        hit_liq = is_active & ((is_long & (n_low <= liq_p_long)) | (is_short & (n_high >= liq_p_short)))
        sl_val, sl_enabled = (float(self.sl_pct) if self.sl_pct is not None else -1.0), (self.sl_pct is not None and self.sl_pct > 0)
        tp_val, tp_enabled = (float(self.tp_pct) if self.tp_pct is not None else -1.0), (self.tp_pct is not None and self.tp_pct > 0)
        sl_dist = mx.minimum(mx.full(leverage.shape, sl_val), 0.75 / mx.maximum(leverage, 1.0)) if sl_enabled else mx.zeros_like(leverage)
        tp_dist = mx.minimum(mx.full(leverage.shape, tp_val), mx.maximum(1.5 * sl_dist, 1.5 / mx.maximum(leverage, 1.0))) if tp_enabled else mx.zeros_like(leverage)
        sl_p_long = self.entry_prices * (1.0 - sl_dist) if sl_enabled else mx.zeros_like(self.entry_prices)
        sl_p_short = self.entry_prices * (1.0 + sl_dist) if sl_enabled else mx.full(self.entry_prices.shape, 1e9)
        hit_sl = is_active & ((is_long & (n_low <= sl_p_long)) | (is_short & (n_high >= sl_p_short))) if sl_enabled else mx.zeros_like(is_active)
        tp_p_long = self.entry_prices * (1.0 + tp_dist) if tp_enabled else mx.full(self.entry_prices.shape, 1e9)
        tp_p_short = self.entry_prices * (1.0 - tp_dist) if tp_enabled else mx.zeros_like(self.entry_prices)
        hit_tp = is_active & ((is_long & (n_high >= tp_p_long)) | (is_short & (n_low <= tp_p_short))) if tp_enabled else mx.zeros_like(is_active)

        is_liq_sym, is_sl_sym = hit_liq, hit_sl & (~hit_liq)
        is_tp_sym = hit_tp & (~hit_liq) & (~is_sl_sym)
        is_hold_sym = is_active & (~hit_liq) & (~is_sl_sym) & (~is_tp_sym)
        sl_exit = mx.where(is_long, mx.minimum(sl_p_long, n_open), mx.maximum(sl_p_short, n_open))
        tp_exit = mx.where(is_long, tp_p_long, tp_p_short)
        exit_p = mx.where(is_sl_sym, sl_exit, mx.where(is_tp_sym, tp_exit, n_close))
        step_ret = mx.where(is_active, mx.where(is_long, (exit_p - c_close) / (c_close + 1e-8), (c_close - exit_p) / (c_close + 1e-8)), 0.0)
        pnl_sym = mx.abs(target_notional) * step_ret - (target_notional * n_funding)
        sum_act_notional = mx.sum(mx.abs(target_notional), axis=-1, keepdims=True)
        step_costs = (fee_cost + slippage_cost) * (mx.abs(target_notional) / (sum_act_notional + 1e-8))
        net_sym_pnl = pnl_sym - step_costs

        # 5. Margin Settlement & Liq Penalties
        if self.margin_mode == "cross":
            adverse_p = mx.where(is_long, n_low, mx.where(is_short, n_high, n_close))
            adverse_ret = mx.where(is_long, (adverse_p - c_close) / (c_close + 1e-8), (c_close - adverse_p) / (c_close + 1e-8))
            adverse_eq = self.equity + mx.sum(mx.abs(target_notional) * adverse_ret - (target_notional * n_funding), axis=-1, keepdims=True) - total_costs
            req_mmr = mx.sum(mx.abs(target_notional) * mmr_sym, axis=-1, keepdims=True)
            tot_open = mx.sum(mx.abs(target_notional), axis=-1, keepdims=True)
            cross_liq = (adverse_eq <= req_mmr) & (tot_open > 0)
            liq_loss = tot_open * self.liq_penalty
            liq_equity = mx.maximum(mx.zeros_like(self.equity), adverse_eq - liq_loss)
            norm_equity = mx.maximum(mx.zeros_like(self.equity), self.equity + mx.sum(net_sym_pnl, axis=-1, keepdims=True))
            new_equity = mx.where(cross_liq, liq_equity, norm_equity)
            liq_occurred = cross_liq
            self.total_liq_count = self.total_liq_count + mx.sum(cross_liq.astype(mx.int32))
            self.total_sl_count = self.total_sl_count + mx.sum(is_sl_sym.astype(mx.int32))
            self.total_tp_count = self.total_tp_count + mx.sum(is_tp_sym.astype(mx.int32))
            self.notionals = mx.where(cross_liq, mx.zeros_like(self.notionals), mx.where(is_hold_sym, target_notional, mx.zeros_like(target_notional)))
            self.entry_prices = mx.where(cross_liq, mx.zeros_like(self.entry_prices), mx.where(is_hold_sym, self.entry_prices, mx.zeros_like(self.entry_prices)))
        else:  # Isolated
            liq_penalties = mx.abs(target_notional) * self.liq_penalty
            eq_deltas = mx.where(is_liq_sym, -(collateral + liq_penalties), net_sym_pnl)
            new_equity = mx.maximum(mx.zeros_like(self.equity), self.equity + mx.sum(eq_deltas, axis=-1, keepdims=True))
            liq_occurred = mx.any(is_liq_sym, axis=-1, keepdims=True)
            self.total_liq_count = self.total_liq_count + mx.sum(is_liq_sym.astype(mx.int32))
            self.total_sl_count = self.total_sl_count + mx.sum(is_sl_sym.astype(mx.int32))
            self.total_tp_count = self.total_tp_count + mx.sum(is_tp_sym.astype(mx.int32))
            self.notionals = mx.where(is_hold_sym, target_notional, mx.zeros_like(target_notional))
            self.entry_prices = mx.where(is_hold_sym, self.entry_prices, mx.zeros_like(self.entry_prices))

        self.positions = self.notionals / (self.initial_capital * self.max_leverage + 1e-8)
        new_equity = mx.where(new_equity <= self.bankrupt_thresh, mx.zeros_like(new_equity), new_equity)
        self.t = next_t
        done = (self.t >= self.num_candles - 1)
        prev_equity, self.equity = self.equity, new_equity
        self.peak_equity = mx.maximum(self.peak_equity, self.equity)
        cur_dd = mx.maximum(mx.zeros_like(self.equity), (self.peak_equity - self.equity) / (self.peak_equity + 1e-8))
        dd_sq = cur_dd ** 2
        self.sum_dd_sq += dd_sq
        self.step_cnt += 1
        reward = (self.equity - prev_equity) / self.initial_capital - self.martin_weight * dd_sq - mx.where(liq_occurred, 1.0, 0.0) - mx.where(self.equity <= 0.0, 5.0, 0.0)

        info = {}
        if self.record_trades or self.num_envs == 1:
            trade_records = self._extract_trades(delta_notional, target_notional, prev_notionals, is_liq_sym, is_sl_sym, is_tp_sym, pessimistic_fill, leverage, collateral, mmr_sym, net_sym_pnl, pnl_sym, liq_p_long, liq_p_short, liq_penalties if self.margin_mode == "isolated" else None, funding_rate, slippage_rate, c_close, vol, sl_exit, tp_exit, prev_equity, new_equity, tot_col, liq_occurred, liq_loss if self.margin_mode == "cross" else None)
            self.total_trades += len(trade_records)
            info = self._get_info()
            info.update({"liquidated": bool(liq_occurred[0, 0].item()), "net_pnl": float((self.equity[0, 0] - prev_equity[0, 0]).item()), "costs": float(total_costs[0, 0].item()), "trades": trade_records, "total_trades": int(self.total_trades)})

        obs = self._get_obs()
        if self.num_envs == 1:
            return obs[0], float(reward[0, 0].item()), bool(done), False, info
        return obs, reward[:, 0], done, False, info


if __name__ == "__main__":
    print("Diet research please... Vectorized MultiCryptoDexPerpEnv")
    env = MultiCryptoDexPerpEnv(num_envs=64, num_symbols=4, num_candles=200, high_tf="1h", low_tf="5m")
    obs, info = env.reset()
    done = False
    while not done:
        obs, rew, done, _, info = env.step(mx.random.uniform(-1.0, 1.0, (64, 4)))
    print(f"Final Eq: {info['equity']:.2f}, DD: {info['drawdown']:.2%}, Martin: {info['martin_ratio']:.3f}")
