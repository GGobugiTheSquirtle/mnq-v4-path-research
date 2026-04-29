# -*- coding: utf-8 -*-
"""
pivots_ob — O (Order Block origin, BOS 룰)

Pine `MNQ_OBFVG_R4_v1.pine` R41A/R42A 룰 그대로:

  Bull OB R41A (5m 권장):
    swing_high_at_orig = highest(high[1], 10).shift(bos_confirm=3)
    bos_break_b   = close > swing_high_at_orig
    origin_was_bear = close[3] < open[3]      ← OB origin = 3봉 전 음봉
    raw_bos       = origin_was_bear AND bos_break_b AND in_user_hours
    bull_ob_confirmed = raw_bos AND NOT raw_bos[1]    (rising edge)

  Bear OB R42A (1m 권장):
    swing_low_at_orig = lowest(low[1], 10).shift(3)
    bos_break_s   = close < swing_low_at_orig
    origin_was_bull = close[3] > open[3]
    raw_bos       = origin_was_bull AND bos_break_s AND in_user_hours
    bear_ob_confirmed = raw_bos AND NOT raw_bos[1]

  in_user_hours : UTC 12:00 ~ 16:45  (KST 21:00 ~ 01:45)

Pivot index = BOS confirm 시점 (트레이더가 fire 인지하는 바).
origin_idx 컬럼 = 실제 origin bar (i - bos_confirm). forward path 측정 시 둘 다 사용 가능.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _in_user_hours(df: pd.DataFrame) -> pd.Series:
    utc_h = df.index.hour
    utc_m = df.index.minute
    utc_total = utc_h * 60 + utc_m
    return pd.Series((utc_total >= 720) & (utc_total <= 1005),
                     index=df.index, name="in_user_hours")


def find_ob_origins_bull(df: pd.DataFrame,
                         *, swing_lookback: int = 10, bos_confirm: int = 3) -> pd.DataFrame:
    swing_h = df["high"].shift(1).rolling(swing_lookback, min_periods=swing_lookback).max()
    swing_h_at_orig = swing_h.shift(bos_confirm)

    is_bear_at_orig = (df["close"].shift(bos_confirm) < df["open"].shift(bos_confirm))
    bos_break = df["close"] > swing_h_at_orig

    in_hours = _in_user_hours(df)
    raw_bos = (bos_break & is_bear_at_orig & in_hours).fillna(False)
    confirmed = raw_bos & ~raw_bos.shift(1).fillna(False)

    sel = df.loc[confirmed]
    if sel.empty:
        return pd.DataFrame(columns=["kind", "price", "idx", "origin_idx"]
                           ).rename_axis("ts_utc")

    pos = df.index.get_indexer(sel.index)
    origin_idx = pos - bos_confirm
    out = pd.DataFrame({
        "kind": "O_BULL",
        "price": sel["close"].to_numpy(),
        "idx":  pos,
        "origin_idx": origin_idx,
    }, index=sel.index)
    return out


def find_ob_origins_bear(df: pd.DataFrame,
                         *, swing_lookback: int = 10, bos_confirm: int = 3) -> pd.DataFrame:
    swing_l = df["low"].shift(1).rolling(swing_lookback, min_periods=swing_lookback).min()
    swing_l_at_orig = swing_l.shift(bos_confirm)

    is_bull_at_orig = (df["close"].shift(bos_confirm) > df["open"].shift(bos_confirm))
    bos_break = df["close"] < swing_l_at_orig

    in_hours = _in_user_hours(df)
    raw_bos = (bos_break & is_bull_at_orig & in_hours).fillna(False)
    confirmed = raw_bos & ~raw_bos.shift(1).fillna(False)

    sel = df.loc[confirmed]
    if sel.empty:
        return pd.DataFrame(columns=["kind", "price", "idx", "origin_idx"]
                           ).rename_axis("ts_utc")

    pos = df.index.get_indexer(sel.index)
    origin_idx = pos - bos_confirm
    out = pd.DataFrame({
        "kind": "O_BEAR",
        "price": sel["close"].to_numpy(),
        "idx":  pos,
        "origin_idx": origin_idx,
    }, index=sel.index)
    return out


def find_ob_origins(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    bull = find_ob_origins_bull(df, **kwargs)
    bear = find_ob_origins_bear(df, **kwargs)
    return pd.concat([bull, bear]).sort_index()


if __name__ == "__main__":
    import os, time
    DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"

    for tf in ("5m", "1m"):
        print(f"\n=== {tf} OB origins ===")
        t0 = time.time()
        df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
        print(f"  rows: {len(df):,}  ({time.time()-t0:.1f}s load)")
        t1 = time.time()
        piv = find_ob_origins(df)
        print(f"  pivots: {len(piv):,}  ({time.time()-t1:.1f}s)")
        print(piv["kind"].value_counts().to_string())
        print(f"  first: {piv.index[0]}")
        print(f"  last : {piv.index[-1]}")
        # in_user_hours 검증
        kst_hm_set = sorted(set((piv.index.tz_convert('Asia/Seoul').hour * 100 +
                                 piv.index.tz_convert('Asia/Seoul').minute).tolist()))
        print(f"  KST hm range: {kst_hm_set[0]} ~ {kst_hm_set[-1]} (n={len(kst_hm_set)})")
