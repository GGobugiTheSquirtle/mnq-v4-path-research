# -*- coding: utf-8 -*-
"""
pivots_fractal — F (raw swing fractal)

규칙 (CLAUDE.md SR 섹션 + plan A):
  swing_high(i) = high[i] == max(high[i-N..i+N]) AND high[i] > high[i-1]
  swing_low(i)  = low[i]  == min(low[i-N..i+N])  AND low[i]  < low[i-1]

기본 N=5 (좌우 5봉). center 윈도라 lookahead 가 발생 — 분석 목적 OK.
백테스트 시 confirm 시점 (i+N) 사용 가능 — `confirm_offset_bars` 로 옵션화.

출력: DataFrame (index=pivot 의 ts_utc, cols=kind, price, idx)
  - kind  : 'high' | 'low'
  - price : pivot 의 high (high pivot) 또는 low (low pivot)
  - idx   : 원 df 의 row index (forward window slicing 용)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_fractals(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    df : open/high/low/close/volume 가지는 DataFrame (DatetimeIndex, sorted).
    n  : 좌/우 양쪽 lookback 봉수. 윈도 크기 = 2n+1.
    """
    win = 2 * n + 1
    h, l = df["high"], df["low"]
    h_max = h.rolling(win, center=True, min_periods=win).max()
    l_min = l.rolling(win, center=True, min_periods=win).min()

    # 동률 다중 occurrence 방지: 직전 바 대비 strictly higher/lower 일 때만
    is_high = (h == h_max) & (h > h.shift(1))
    is_low  = (l == l_min) & (l < l.shift(1))

    rows = []
    if is_high.any():
        sel = df[is_high]
        for ts, row in sel.iterrows():
            rows.append({"ts_utc": ts, "kind": "high",
                         "price": row["high"]})
    if is_low.any():
        sel = df[is_low]
        for ts, row in sel.iterrows():
            rows.append({"ts_utc": ts, "kind": "low",
                         "price": row["low"]})

    if not rows:
        return pd.DataFrame(columns=["ts_utc", "kind", "price", "idx"]).set_index("ts_utc")

    out = pd.DataFrame(rows).set_index("ts_utc").sort_index()
    # idx: forward slicing 용 row 인덱스 (df 의 0-based)
    pos = df.index.get_indexer(out.index)
    out["idx"] = pos
    return out


# 위 구현은 Python loop 라 느림. 5.8M rows 에서 vectorized 필요.
def find_fractals_fast(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """vectorized 버전 — find_fractals 와 동일 결과지만 5.8M rows OK."""
    win = 2 * n + 1
    h, l = df["high"], df["low"]
    h_max = h.rolling(win, center=True, min_periods=win).max()
    l_min = l.rolling(win, center=True, min_periods=win).min()

    is_high = ((h == h_max) & (h > h.shift(1))).fillna(False).to_numpy()
    is_low  = ((l == l_min) & (l < l.shift(1))).fillna(False).to_numpy()

    idx_arr = np.arange(len(df))
    high_idx = idx_arr[is_high]
    low_idx  = idx_arr[is_low]

    high_df = pd.DataFrame({
        "ts_utc": df.index[high_idx],
        "kind":  np.full(high_idx.shape, "high"),
        "price": h.to_numpy()[high_idx],
        "idx":   high_idx,
    })
    low_df = pd.DataFrame({
        "ts_utc": df.index[low_idx],
        "kind":  np.full(low_idx.shape, "low"),
        "price": l.to_numpy()[low_idx],
        "idx":   low_idx,
    })
    out = pd.concat([high_df, low_df], ignore_index=True)
    out = out.set_index("ts_utc").sort_index()
    return out


if __name__ == "__main__":
    import os, time
    DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"

    for tf, rvol_n in [("5m", 78), ("1m", 390)]:
        print(f"\n=== {tf} parquet ===")
        df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
        print(f"  rows: {len(df):,}")
        t0 = time.time()
        piv = find_fractals_fast(df, n=5)
        print(f"  pivots N=5: {len(piv):,}  ({time.time()-t0:.1f}s)")
        print(f"    high: {(piv['kind']=='high').sum():,}")
        print(f"    low : {(piv['kind']=='low').sum():,}")
        print(f"  ratio (pivots / bars): {len(piv)/len(df)*100:.2f}%")
        # 처음/마지막 pivot
        print(f"  first: {piv.index[0]}  ({piv.iloc[0]['kind']} @ {piv.iloc[0]['price']:.2f})")
        print(f"  last : {piv.index[-1]}  ({piv.iloc[-1]['kind']} @ {piv.iloc[-1]['price']:.2f})")
