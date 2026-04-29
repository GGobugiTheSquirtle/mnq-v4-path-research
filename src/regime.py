# -*- coding: utf-8 -*-
"""
regime — BULL / BEAR / MIXED 분류

5m_v4.pine 의 룰 그대로 (Pine 호환):
  EMA200 slope 측정창 = 288 bars (5m × 288 = 24h)
  Above ratio 측정창 = 50 bars (close > ema200 비율)

  BULL  : slope > +0.10% AND above_pct >= 70
  BEAR  : slope < -0.10% AND above_pct <= 30
  MIXED : 그 외

1m 차트의 경우 slope_window = 1440 (= 24h on 1m), above_window = 50 그대로.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def attach_regime(df: pd.DataFrame, *, slope_window: int = 288,
                  above_window: int = 50,
                  ema200_col: str = "ema200") -> pd.DataFrame:
    """df 에 regime 컬럼 부여 (BULL/BEAR/MIXED)."""
    out = df.copy()
    e200 = out[ema200_col]
    e200_lag = e200.shift(slope_window)
    slope_pct = (e200 - e200_lag) / e200_lag.replace(0, np.nan) * 100.0

    # above ratio: rolling sum of (close > ema200) / above_window
    above = (out["close"] > e200).astype(np.int8)
    above_pct = above.rolling(above_window, min_periods=above_window).sum() * (100.0 / above_window)

    out["slope_pct"] = slope_pct
    out["above_pct"] = above_pct

    bull = (slope_pct >  0.10) & (above_pct >= 70)
    bear = (slope_pct < -0.10) & (above_pct <= 30)
    out["regime"] = np.where(bull, "BULL", np.where(bear, "BEAR", "MIXED"))
    # NaN 처리: warmup 구간
    warmup = slope_pct.isna() | above_pct.isna()
    out.loc[warmup, "regime"] = "WARMUP"
    return out


def attach_regime_to_pivots(pivots: pd.DataFrame, df_with_regime: pd.DataFrame) -> pd.DataFrame:
    """pivot DataFrame 에 regime 컬럼 join (idx 기준)."""
    out = pivots.copy()
    if "idx" not in out.columns:
        raise ValueError("pivots must have 'idx' column")
    reg = df_with_regime["regime"].to_numpy()
    out["regime"] = reg[out["idx"].to_numpy()]
    return out


if __name__ == "__main__":
    import os, time
    from indicators import add_indicator_columns
    DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"

    for tf, rvol_n, slope_w in [("5m", 78, 288), ("1m", 390, 1440)]:
        print(f"\n=== {tf} regime ===")
        t0 = time.time()
        df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
        df = add_indicator_columns(df, rvol_length=rvol_n)
        df = attach_regime(df, slope_window=slope_w)
        print(f"  rows: {len(df):,}  ({time.time()-t0:.1f}s)")
        print(df["regime"].value_counts().to_string())
        # sample sample slope/above
        sample = df.dropna(subset=["slope_pct", "above_pct"]).iloc[::len(df)//5][:5]
        print("\n  sample (5 spread):")
        print(sample[["close", "ema200", "slope_pct", "above_pct", "regime"]].to_string())
