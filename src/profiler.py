# -*- coding: utf-8 -*-
"""
profiler — 이벤트 시점의 시장 상태 snapshot

기존 indicators + regime 외에 추가:
  - SR distance: HH20 / LL20 / EMA20/50/200 까지의 거리 (ATR 정규화)
  - 캔들 shape: body / upper_wick / lower_wick (ATR 정규화)
  - Pre-context movement: 5/15/30 봉 전 close 대비 net_disp (ATR 정규화)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 감 분석 대상 numeric features (Cohen's d 계산용)
PROFILE_NUMERIC = [
    # Indicators (이미 add_indicator_columns 결과에 있음)
    "atr_ratio", "rsi14", "mfi14", "stoch_k", "stoch_d",
    "bb_pctb", "bb_width_pct", "rvol", "vol_z",
    # Regime numeric
    "slope_pct", "above_pct",
    # Bar shape (add_sr_and_shape 에서 부여)
    "body_to_atr", "upper_wick_to_atr", "lower_wick_to_atr",
    # SR distance
    "dist_to_hh20_atr", "dist_to_ll20_atr",
    "dist_to_ema20_atr", "dist_to_ema50_atr", "dist_to_ema200_atr",
    # Pre-context movement
    "pre_5_disp_atr", "pre_15_disp_atr", "pre_30_disp_atr",
]

PROFILE_CATEGORICAL = ["regime", "ema_up", "ema_down", "bull_aligned",
                       "kst_h", "kst_m", "kst_hm", "dow_kst"]


def add_sr_and_shape(df: pd.DataFrame, *, atr_col: str = "atr14") -> pd.DataFrame:
    """SR 거리 + 캔들 shape feature 추가 (in-place 아님)."""
    out = df.copy()
    atr = out[atr_col].replace(0, np.nan)

    # Recent SR (rolling 20)
    out["hh20"] = out["high"].shift(1).rolling(20, min_periods=20).max()
    out["ll20"] = out["low"].shift(1).rolling(20, min_periods=20).min()
    out["dist_to_hh20_atr"] = (out["hh20"] - out["close"]) / atr
    out["dist_to_ll20_atr"] = (out["close"] - out["ll20"]) / atr
    out["dist_to_ema20_atr"]  = (out["close"] - out["ema20"])  / atr
    out["dist_to_ema50_atr"]  = (out["close"] - out["ema50"])  / atr
    out["dist_to_ema200_atr"] = (out["close"] - out["ema200"]) / atr

    # Bar shape
    body = (out["close"] - out["open"]).abs()
    upper_wick = out["high"] - np.maximum(out["close"], out["open"])
    lower_wick = np.minimum(out["close"], out["open"]) - out["low"]
    out["body_to_atr"]       = body       / atr
    out["upper_wick_to_atr"] = upper_wick / atr
    out["lower_wick_to_atr"] = lower_wick / atr

    # Pre-context movement (5/15/30 봉 net displacement, ATR 정규화)
    for n in (5, 15, 30):
        disp = out["close"] - out["close"].shift(n)
        out[f"pre_{n}_disp_atr"] = disp / atr

    return out


def attach_profile(events: pd.DataFrame, df_full: pd.DataFrame,
                   *, cols: list = None) -> pd.DataFrame:
    """events DataFrame 에 profile 컬럼 부여 (idx 기반 lookup)."""
    if events.empty:
        return events.copy()
    if "idx" not in events.columns:
        raise ValueError("events must have 'idx' column")
    if cols is None:
        cols = PROFILE_NUMERIC + PROFILE_CATEGORICAL

    out = events.copy()
    idxs = out["idx"].to_numpy()
    for col in cols:
        if col in df_full.columns:
            out[col] = df_full[col].to_numpy()[idxs]
    return out


if __name__ == "__main__":
    import os, time
    from indicators import add_indicator_columns
    from regime import attach_regime
    from events import zigzag_pivots, add_forward_leg, label_inflection, label_trend_leg_end

    DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"

    print("=== 5m profile smoke ===")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, "nq_5m.parquet"))
    df = add_indicator_columns(df, rvol_length=78)
    df = attach_regime(df, slope_window=288)
    df = add_sr_and_shape(df)
    print(f"  bars + full features: {len(df):,}  ({time.time()-t0:.1f}s)")

    zz = add_forward_leg(zigzag_pivots(df, threshold_atr_mult=1.5))
    inf = label_inflection(zz, min_back_atr=2.0, min_forward_atr=2.0)
    tre = label_trend_leg_end(zz, min_atr=3.0, min_bars=5, max_bars=200)
    print(f"  inflection (≥2.0×ATR both legs): {len(inf):,}")
    print(f"  trend_end  (≥3.0×ATR ≥5 bars):   {len(tre):,}")

    inf_p = attach_profile(inf, df)
    tre_p = attach_profile(tre, df)
    print(f"  inflection profile cols: {len([c for c in inf_p.columns if c in PROFILE_NUMERIC + PROFILE_CATEGORICAL])}")
    print("\n  inflection sample (5):")
    cols_show = ["kind", "leg_back_atr_ratio", "leg_forward_atr_ratio",
                 "rsi14", "stoch_k", "bb_pctb", "atr_ratio",
                 "dist_to_ema20_atr", "regime", "kst_h"]
    print(inf_p[cols_show].dropna().head(5).to_string())
    print("\n  trend_end sample (5):")
    cols_show2 = ["kind", "leg_atr_ratio", "leg_back_duration",
                  "rsi14", "bb_pctb", "atr_ratio", "rvol",
                  "dist_to_ema20_atr", "regime", "kst_h"]
    print(tre_p[cols_show2].dropna().head(5).to_string())
