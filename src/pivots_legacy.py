# -*- coding: utf-8 -*-
"""
pivots_legacy — L (Legacy V/W/SHORT/NOS 신호 발동점)

Pine 룰 그대로 포팅:
  5m (`MNQ_Signal_5m.pine`):
    L_W   : stoch_k < 30 + nearEMA20 (±1.0×ATR) + rvol >= 0.5  + !is_dead
    L_V   : stoch_k < 15 + nearEMA20 (±0.5×ATR) +              !is_dead
    L_S   : bb_pctb > 95  + !bull_aligned                      + !is_dead   (SHORT)
    L_NOS : stoch_k > 75  + mfi14  > 75 + bull_aligned                       (UP-bias)

  1m (`MNQ_Signal_1m_v2.pine`):
    L_V10  : stoch_k < 20 + nearEMA20 (±1.0×ATR) + rvol > 1.2                + !is_dead
    L_W15  : stoch_k < 15 + nearEMA20 (±1.5×ATR) + rvol > 1.2 + bull_aligned + !is_dead
    L_BB7  : bb_pctb < 10 + rvol > 0.8 + bull_aligned                        + !is_dead
    L_W15B : stoch_k < 20 + nearEMA20 (±0.7×ATR) +                             !is_dead   (basic W)
    L_NOS  : stoch_k > 75 + mfi14  > 75 + bull_aligned

is_dead = (rvol < 0.5)  OR  (KST hour ∈ [11, 16))   ← Pine 의 et_h ≥ 22 OR < 3 == KST 11:30~16
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _is_dead(df: pd.DataFrame) -> pd.Series:
    """is_dead = rvol<0.5 OR KST hour 11~15 (Pine ET 22-03)."""
    is_dead_vol  = df["rvol"] < 0.5
    is_dead_time = df["kst_h"].isin([11, 12, 13, 14, 15])
    return (is_dead_vol | is_dead_time).fillna(True)


def _near_ema20(df: pd.DataFrame, mult: float) -> pd.Series:
    return (df["close"] - df["ema20"]).abs() <= df["atr14"] * mult


def _to_pivots(df: pd.DataFrame, mask: pd.Series, kind: str) -> pd.DataFrame:
    """mask 가 True 인 row → pivot DataFrame. price = close (진입가 가정)."""
    sel = df.loc[mask]
    if sel.empty:
        return pd.DataFrame(columns=["kind", "price", "idx"]).rename_axis("ts_utc")
    pos = df.index.get_indexer(sel.index)
    out = pd.DataFrame({
        "kind": kind,
        "price": sel["close"].to_numpy(),
        "idx":  pos,
    }, index=sel.index)
    return out


def _dedupe_with_cooldown(df_pivots: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    """동일 kind 가 cooldown 내 중복이면 첫 occurrence 만."""
    if df_pivots.empty:
        return df_pivots
    out = df_pivots.sort_index().copy()
    out["_keep"] = True
    last_idx = {}
    keep = []
    for ts, row in out.iterrows():
        k = row["kind"]
        if k in last_idx and (row["idx"] - last_idx[k]) <= cooldown_bars:
            keep.append(False)
        else:
            keep.append(True)
            last_idx[k] = row["idx"]
    return out.loc[keep].drop(columns=["_keep"], errors="ignore")


# ─────────────────────────────────────────────────────────────────────────
# 5m  Legacy
# ─────────────────────────────────────────────────────────────────────────

def find_legacy_5m(df: pd.DataFrame, *, cooldown: int = 3) -> pd.DataFrame:
    """
    df : indicators.add_indicator_columns(df, rvol_length=78) 결과.
    """
    dead = _is_dead(df)
    bull = df["bull_aligned"].astype(bool)
    near_w = _near_ema20(df, 1.0)
    near_v = _near_ema20(df, 0.5)

    sig_w   = (df["stoch_k"] < 30) & near_w & (df["rvol"] >= 0.5) & ~dead
    sig_v   = (df["stoch_k"] < 15) & near_v & ~dead
    sig_s   = (df["bb_pctb"] > 95) & ~bull & ~dead
    sig_nos = (df["stoch_k"] > 75) & (df["mfi14"] > 75) & bull

    parts = [
        _to_pivots(df, sig_w,   "L5_W"),
        _to_pivots(df, sig_v,   "L5_V"),
        _to_pivots(df, sig_s,   "L5_S"),
        _to_pivots(df, sig_nos, "L5_NOS"),
    ]
    out = pd.concat(parts).sort_index()
    return _dedupe_with_cooldown(out, cooldown)


# ─────────────────────────────────────────────────────────────────────────
# 1m  Legacy
# ─────────────────────────────────────────────────────────────────────────

def find_legacy_1m(df: pd.DataFrame, *, cooldown: int = 15) -> pd.DataFrame:
    """
    df : indicators.add_indicator_columns(df, rvol_length=390) 결과.
    """
    dead = _is_dead(df)
    bull = df["bull_aligned"].astype(bool)
    near_v10 = _near_ema20(df, 1.0)
    near_w15 = _near_ema20(df, 1.5)
    near_w15b = _near_ema20(df, 0.7)

    sig_v10  = (df["stoch_k"] < 20) & near_v10  & (df["rvol"] > 1.2)              & ~dead
    sig_w15  = (df["stoch_k"] < 15) & near_w15  & (df["rvol"] > 1.2) & bull       & ~dead
    sig_bb7  = (df["bb_pctb"] < 10)             & (df["rvol"] > 0.8) & bull       & ~dead
    sig_w15b = (df["stoch_k"] < 20) & near_w15b                                   & ~dead
    sig_nos  = (df["stoch_k"] > 75) & (df["mfi14"] > 75) & bull

    parts = [
        _to_pivots(df, sig_v10,  "L1_V10"),
        _to_pivots(df, sig_w15,  "L1_W15"),
        _to_pivots(df, sig_bb7,  "L1_BB7"),
        _to_pivots(df, sig_w15b, "L1_W15B"),
        _to_pivots(df, sig_nos,  "L1_NOS"),
    ]
    out = pd.concat(parts).sort_index()
    return _dedupe_with_cooldown(out, cooldown)


if __name__ == "__main__":
    import os, time
    from indicators import add_indicator_columns
    DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"

    for tf, rvol_n, fn in [("5m", 78, find_legacy_5m), ("1m", 390, find_legacy_1m)]:
        print(f"\n=== {tf} Legacy ===")
        t0 = time.time()
        df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
        print(f"  rows: {len(df):,}  ({time.time()-t0:.1f}s load)")
        t1 = time.time()
        df = add_indicator_columns(df, rvol_length=rvol_n)
        print(f"  indicators OK  ({time.time()-t1:.1f}s)")
        t2 = time.time()
        piv = fn(df)
        print(f"  pivots: {len(piv):,}  ({time.time()-t2:.1f}s)")
        print(piv["kind"].value_counts().to_string())
