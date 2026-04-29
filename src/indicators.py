# -*- coding: utf-8 -*-
"""
indicators — vectorized TA (Pine 룰 그대로)

Phase 2.2 — pivot extractor 와 regime tagger 가 공통으로 사용.

규칙:
- 모든 입력은 pd.Series (close/high/low/volume) — DataFrame 인덱스 보존.
- 윈도 함수는 ta.ema 등과 동일 결과 (Wilder/RMA — RSI/ATR 정확).
- look-ahead 차단: 현재 바 t 의 값은 close[0..t] 만 사용 (default rolling/ewm).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────

def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False, min_periods=length).mean()


# ─────────────────────────────────────────────────────────────────────────
# ATR (Wilder/RMA)
# ─────────────────────────────────────────────────────────────────────────

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr


def atr(h: pd.Series, l: pd.Series, c: pd.Series, length: int = 14) -> pd.Series:
    """Wilder smoothing — Pine ta.atr 동일."""
    tr = true_range(h, l, c)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


# ─────────────────────────────────────────────────────────────────────────
# RSI (Wilder)
# ─────────────────────────────────────────────────────────────────────────

def rsi(c: pd.Series, length: int = 14) -> pd.Series:
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ─────────────────────────────────────────────────────────────────────────
# MFI (Money Flow Index, hlc3)
# ─────────────────────────────────────────────────────────────────────────

def mfi(h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, length: int = 14) -> pd.Series:
    typ = (h + l + c) / 3.0
    mf = typ * v
    delta = typ.diff()
    pos_mf = mf.where(delta > 0, 0.0)
    neg_mf = mf.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(length, min_periods=length).sum()
    neg_sum = neg_mf.rolling(length, min_periods=length).sum()
    mr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + mr)


# ─────────────────────────────────────────────────────────────────────────
# Stochastic (Pine ta.stoch — close 기준)
# ─────────────────────────────────────────────────────────────────────────

def stoch_k(c: pd.Series, h: pd.Series, l: pd.Series, length: int = 14) -> pd.Series:
    hh = h.rolling(length, min_periods=length).max()
    ll = l.rolling(length, min_periods=length).min()
    rng = (hh - ll).replace(0, np.nan)
    return 100 * (c - ll) / rng


def stoch_d(k: pd.Series, length: int = 3) -> pd.Series:
    return k.rolling(length, min_periods=length).mean()


# ─────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────

def bbands(c: pd.Series, length: int = 20, mult: float = 2.0):
    mid = c.rolling(length, min_periods=length).mean()
    sd = c.rolling(length, min_periods=length).std(ddof=0)  # Pine ta.stdev = population
    up = mid + mult * sd
    lo = mid - mult * sd
    pctb = (c - lo) / (up - lo).replace(0, np.nan) * 100
    width_pct = (up - lo) / mid * 100
    return mid, up, lo, pctb, width_pct


# ─────────────────────────────────────────────────────────────────────────
# RVOL — Pine ta.sma(volume, N) 기반
# ─────────────────────────────────────────────────────────────────────────

def rvol(v: pd.Series, length: int) -> pd.Series:
    ma = v.rolling(length, min_periods=length).mean()
    return v / ma.replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────
# Vol-Z score (pine 1m_v3 BURN_X 사양: SMA 20 + STDEV 20)
# ─────────────────────────────────────────────────────────────────────────

def vol_z(v: pd.Series, length: int = 20) -> pd.Series:
    ma = v.rolling(length, min_periods=length).mean()
    sd = v.rolling(length, min_periods=length).std(ddof=0)
    return (v - ma) / sd.replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────
# Bundle helper — pivot extractor 가 한 번에 호출
# ─────────────────────────────────────────────────────────────────────────

def add_indicator_columns(df: pd.DataFrame, *, rvol_length: int) -> pd.DataFrame:
    """
    df: open/high/low/close/volume 가지는 DataFrame.
    out: 기본 indicator 컬럼 추가본 (in-place 아님).

    rvol_length = 1m 차트는 390 (1 RTH = 390 분),
                  5m 차트는 78  (1 RTH = 78 봉).
    """
    out = df.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]

    out["ema20"]  = ema(c, 20)
    out["ema50"]  = ema(c, 50)
    out["ema200"] = ema(c, 200)
    out["atr14"]  = atr(h, l, c, 14)
    out["atr_sma20"] = out["atr14"].rolling(20, min_periods=20).mean()
    out["atr_ratio"] = out["atr14"] / out["atr_sma20"].replace(0, np.nan)
    out["rsi14"]  = rsi(c, 14)
    out["mfi14"]  = mfi(h, l, c, v, 14)
    out["stoch_k"] = stoch_k(c, h, l, 14)
    out["stoch_d"] = stoch_d(out["stoch_k"], 3)
    bb_mid, bb_up, bb_lo, bb_pctb, bb_width = bbands(c, 20, 2.0)
    out["bb_up"], out["bb_lo"], out["bb_mid"] = bb_up, bb_lo, bb_mid
    out["bb_pctb"], out["bb_width_pct"] = bb_pctb, bb_width
    out["rvol"]   = rvol(v, rvol_length)
    out["vol_z"]  = vol_z(v, 20)

    out["bull_aligned"] = (out["ema20"] > out["ema50"]).astype(np.int8)
    out["ema_up"]   = ((out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema200"])).astype(np.int8)
    out["ema_down"] = ((out["ema20"] < out["ema50"]) & (out["ema50"] < out["ema200"])).astype(np.int8)

    return out


if __name__ == "__main__":
    # quick smoke test
    import os
    DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"
    print("[smoke] loading 5m parquet...")
    df = pd.read_parquet(os.path.join(DATA, "nq_5m.parquet"))
    print(f"  rows: {len(df):,}")
    print("[smoke] add_indicator_columns (rvol_length=78)...")
    out = add_indicator_columns(df, rvol_length=78)
    print(f"  added cols: {[c for c in out.columns if c not in df.columns]}")
    print("\n[smoke] tail 3:")
    print(out[["close", "ema20", "ema50", "atr14", "atr_ratio",
               "rsi14", "stoch_k", "bb_pctb", "rvol", "vol_z"]].tail(3).to_string())
