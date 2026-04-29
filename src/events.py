# -*- coding: utf-8 -*-
"""
events — Retrospective event labeling

핵심 전환 (2026-04-29):
  forward prediction → retrospective profiling.
  과거 데이터에서 이미 발생한 변곡점/추세 이벤트들을 enumerate, 그 시점의 시장 상태(profile) 분석.

이벤트 종류:
  1. ZigZag pivot   : ≥ threshold×ATR 의 반전 — 모든 swing high/low (모집단)
  2. Inflection     : back leg 강 + forward leg 강 둘 다 — sharp 변곡점
  3. Trend leg end  : 한 방향 ≥ X×ATR over ≥ N bars — 강한 추세 종료점
  4. Trend leg start: 같은 leg 의 시작점 (= 직전 pivot)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# ZigZag (sequential — numpy loop)
# ─────────────────────────────────────────────────────────────────────────

def zigzag_pivots(df: pd.DataFrame, *, threshold_atr_mult: float = 1.5,
                  atr_col: str = "atr14") -> pd.DataFrame:
    """
    ZigZag 알고리즘으로 pivot 추출.
      threshold = threshold_atr_mult × ATR (extreme 시점 기준).
      direction 반대로 threshold 만큼 움직이면 직전 extreme 을 pivot 확정.

    반환 DataFrame columns:
      kind                     : 'high' | 'low'
      price                    : pivot price (high.max 또는 low.min)
      idx                      : df 의 row 인덱스 (forward 분석 용)
      leg_back_magnitude       : 직전 pivot → 이 pivot 까지 절대 가격 변화
      leg_back_duration        : 봉 수 (직전 pivot 과의 거리)
      leg_back_direction       : 'up' | 'down'
      leg_back_atr_at_start    : 직전 pivot 시점의 atr14 (정규화 용)
    """
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    atr = df[atr_col].to_numpy()
    idx_arr = np.arange(len(df))
    n = len(df)

    # warmup skip
    start = 0
    while start < n and (np.isnan(atr[start]) or atr[start] == 0):
        start += 1
    if start >= n - 1:
        return pd.DataFrame()

    pivots: list[dict] = []
    direction = 0
    extreme_idx = start
    extreme_price = h[start]
    last_pivot_idx = start
    last_pivot_price = (h[start] + l[start]) / 2.0  # neutral start

    for i in range(start + 1, n):
        a = atr[i]
        if np.isnan(a) or a == 0:
            continue

        if direction == 0:
            # bootstrap direction
            base_atr = atr[extreme_idx]
            if h[i] > extreme_price + threshold_atr_mult * base_atr * 0.5:
                direction = 1
                extreme_idx = i
                extreme_price = h[i]
            elif l[i] < extreme_price - threshold_atr_mult * base_atr * 0.5:
                direction = -1
                extreme_idx = i
                extreme_price = l[i]
            continue

        threshold = threshold_atr_mult * atr[extreme_idx]

        if direction == 1:  # uptrend, looking for reversal down
            if h[i] > extreme_price:
                extreme_idx = i
                extreme_price = h[i]
            elif (extreme_price - l[i]) >= threshold:
                pivots.append(dict(
                    ts_utc=df.index[extreme_idx],
                    kind="high",
                    price=float(extreme_price),
                    idx=int(extreme_idx),
                    leg_back_magnitude=float(abs(extreme_price - last_pivot_price)),
                    leg_back_duration=int(extreme_idx - last_pivot_idx),
                    leg_back_direction="up",
                    leg_back_atr_at_start=float(atr[last_pivot_idx]),
                ))
                last_pivot_idx = extreme_idx
                last_pivot_price = extreme_price
                direction = -1
                extreme_idx = i
                extreme_price = l[i]
        else:  # direction == -1
            if l[i] < extreme_price:
                extreme_idx = i
                extreme_price = l[i]
            elif (h[i] - extreme_price) >= threshold:
                pivots.append(dict(
                    ts_utc=df.index[extreme_idx],
                    kind="low",
                    price=float(extreme_price),
                    idx=int(extreme_idx),
                    leg_back_magnitude=float(abs(last_pivot_price - extreme_price)),
                    leg_back_duration=int(extreme_idx - last_pivot_idx),
                    leg_back_direction="down",
                    leg_back_atr_at_start=float(atr[last_pivot_idx]),
                ))
                last_pivot_idx = extreme_idx
                last_pivot_price = extreme_price
                direction = 1
                extreme_idx = i
                extreme_price = h[i]

    if not pivots:
        return pd.DataFrame()

    out = pd.DataFrame(pivots).set_index("ts_utc")
    return out


def add_forward_leg(zigzag: pd.DataFrame) -> pd.DataFrame:
    """zigzag pivot DF 에 leg_forward_* 메타 추가 (다음 pivot 까지)."""
    if zigzag.empty:
        return zigzag
    out = zigzag.copy()
    next_price = out["price"].shift(-1)
    next_idx   = out["idx"].shift(-1)
    out["leg_forward_magnitude"] = (next_price - out["price"]).abs()
    out["leg_forward_duration"]  = (next_idx - out["idx"]).fillna(0).astype("Int64")
    return out


# ─────────────────────────────────────────────────────────────────────────
# Event labelers
# ─────────────────────────────────────────────────────────────────────────

def label_inflection(zz_with_fwd: pd.DataFrame, *,
                     min_back_atr: float = 1.5,
                     min_forward_atr: float = 1.5) -> pd.DataFrame:
    """sharp 변곡: back leg 와 forward leg 모두 ≥ X×ATR (대칭 강도)."""
    if zz_with_fwd.empty:
        return zz_with_fwd
    atr_at_start = zz_with_fwd["leg_back_atr_at_start"].replace(0, np.nan)
    back_ratio = zz_with_fwd["leg_back_magnitude"] / atr_at_start
    fwd_ratio  = zz_with_fwd["leg_forward_magnitude"] / atr_at_start
    mask = (back_ratio >= min_back_atr) & (fwd_ratio >= min_forward_atr)
    out = zz_with_fwd[mask].copy()
    out["event_type"] = "inflection"
    out["leg_back_atr_ratio"] = back_ratio[mask]
    out["leg_forward_atr_ratio"] = fwd_ratio[mask]
    return out


def label_trend_leg_end(zz_with_fwd: pd.DataFrame, *,
                        min_atr: float = 2.5,
                        min_bars: int = 5,
                        max_bars: int = 200) -> pd.DataFrame:
    """
    강한 추세 leg 의 종료점.
      leg_back_magnitude ≥ X×ATR AND leg_back_duration ∈ [N, M] bars.
    이벤트 ts = leg 의 end pivot (즉 추세가 멈춘 곳).
    """
    if zz_with_fwd.empty:
        return zz_with_fwd
    atr_at_start = zz_with_fwd["leg_back_atr_at_start"].replace(0, np.nan)
    leg_atr = zz_with_fwd["leg_back_magnitude"] / atr_at_start
    dur = zz_with_fwd["leg_back_duration"]
    mask = (leg_atr >= min_atr) & (dur >= min_bars) & (dur <= max_bars)
    out = zz_with_fwd[mask].copy()
    out["event_type"] = "trend_end"
    out["leg_atr_ratio"] = leg_atr[mask]
    return out


if __name__ == "__main__":
    import os, time
    from indicators import add_indicator_columns
    from regime import attach_regime
    DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"

    print("=== 5m events smoke ===")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, "nq_5m.parquet"))
    df = add_indicator_columns(df, rvol_length=78)
    df = attach_regime(df, slope_window=288)
    print(f"  bars: {len(df):,}  ({time.time()-t0:.1f}s)")

    for thresh in (1.0, 1.5, 2.0):
        t1 = time.time()
        zz = zigzag_pivots(df, threshold_atr_mult=thresh)
        zz = add_forward_leg(zz)
        print(f"\n  ZigZag thresh={thresh}×ATR: {len(zz):,} pivots  ({time.time()-t1:.1f}s)")
        if zz.empty:
            continue
        leg_atr = zz["leg_back_magnitude"] / zz["leg_back_atr_at_start"].replace(0, np.nan)
        dur = zz["leg_back_duration"]
        print(f"    leg magnitude (×ATR): median {leg_atr.median():.2f}, p25 {leg_atr.quantile(0.25):.2f}, p75 {leg_atr.quantile(0.75):.2f}")
        print(f"    leg duration  (bars): median {dur.median():.0f}, p25 {dur.quantile(0.25):.0f}, p75 {dur.quantile(0.75):.0f}")

    # 본격 이벤트: thresh=1.5
    print("\n=== Events at thresh=1.5×ATR ===")
    zz = add_forward_leg(zigzag_pivots(df, threshold_atr_mult=1.5))
    inflect = label_inflection(zz, min_back_atr=1.5, min_forward_atr=1.5)
    trend_end = label_trend_leg_end(zz, min_atr=2.5, min_bars=5, max_bars=200)
    print(f"  inflection events  : {len(inflect):,}  (back+fwd 둘 다 ≥1.5×ATR)")
    print(f"  trend_end events   : {len(trend_end):,}  (one-leg ≥2.5×ATR over ≥5 bars)")
    print(f"  total ZigZag pivots: {len(zz):,}")
    overlap = inflect.index.intersection(trend_end.index)
    print(f"  overlap (inflection ∩ trend_end): {len(overlap):,}")
