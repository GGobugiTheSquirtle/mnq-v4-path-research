# -*- coding: utf-8 -*-
"""
data_loader — FirstRateData NQ CSV → parquet 표준화

Phase 2.2 — v4_path_classifier_research

CSV schema (FirstRateData):
  Date;Time;Open;High;Low;Close;Volume
  - Date: DD/MM/YYYY  (EU 형식 — 2026-04-29 sample 검증 "14/12/2008")
  - Time: HH:MM
  - sep: ';'
  - 타임존: ET (US/Eastern, EDT/EST 자동 처리)
  - header: 없음

출력 parquet:
  index : DatetimeIndex (UTC, tz-aware)
  cols  : open, high, low, close, volume, kst_dt (datetime64[ns, Asia/Seoul]),
          kst_h, kst_m, kst_hm, dow_kst (Sunday=0..Saturday=6, KST 기준)
"""
from __future__ import annotations

import os
import sys
import time
import pandas as pd
import numpy as np

BASE = r"C:\Users\minb0\Desktop\Main folder\투자공부\backtest_data\nq_10y"
OUT  = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"

CSV_KW = dict(
    sep=";",
    header=None,
    names=["date", "time", "open", "high", "low", "close", "volume"],
    dtype={"open": np.float64, "high": np.float64, "low": np.float64,
           "close": np.float64, "volume": np.int64},
)


def load_csv(path: str) -> pd.DataFrame:
    """CSV 로드 → ET tz-aware datetime → UTC index → KST 컬럼 부여."""
    print(f"[load] {path}")
    t0 = time.time()
    df = pd.read_csv(path, **CSV_KW)
    print(f"  raw rows: {len(df):,}  ({time.time()-t0:.1f}s)")

    # datetime 결합 (ET 가정)
    dt_naive = pd.to_datetime(df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M")
    # ET (US/Eastern) → UTC
    dt_et = dt_naive.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    dt_utc = dt_et.dt.tz_convert("UTC")
    # NaT 행 제거 (DST 전환 ambiguous bar)
    mask = dt_utc.notna()
    if not mask.all():
        n_drop = (~mask).sum()
        print(f"  drop NaT (DST ambiguous): {n_drop}")
        df = df.loc[mask].copy()
        dt_utc = dt_utc.loc[mask]

    df.index = pd.DatetimeIndex(dt_utc, name="ts_utc")
    df = df.drop(columns=["date", "time"]).sort_index()

    # 중복 timestamp 제거 (FirstRateData 1m 에 4건 발견 — 동일 OHLCV 4배 반복)
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    if len(df) != n_before:
        print(f"  drop duplicate ts: {n_before - len(df)}")

    # KST 보조 컬럼
    kst = df.index.tz_convert("Asia/Seoul")
    df["kst_h"] = kst.hour.astype(np.int8)
    df["kst_m"] = kst.minute.astype(np.int8)
    df["kst_hm"] = (kst.hour * 100 + kst.minute).astype(np.int16)
    # DOW: 월=0..일=6 (pandas default). Thursday=3.
    df["dow_kst"] = kst.dayofweek.astype(np.int8)

    print(f"  span: {df.index[0]} → {df.index[-1]}")
    print(f"  cleaned rows: {len(df):,}  ({time.time()-t0:.1f}s)")
    return df


def save_parquet(df: pd.DataFrame, name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.parquet")
    t0 = time.time()
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    sz = os.path.getsize(path) / (1024 * 1024)
    print(f"[save] {path}  ({sz:.1f} MB, {time.time()-t0:.1f}s)")
    return path


def main():
    for tf in ("1m", "5m"):
        src = os.path.join(BASE, f"nq-{tf}_bk.csv")
        df = load_csv(src)
        save_parquet(df, f"nq_{tf}")
        # 간단 sanity
        print(f"  open range: {df['open'].min():.2f} ~ {df['open'].max():.2f}")
        print(f"  vol range : {df['volume'].min()} ~ {df['volume'].max():,}")
        print(f"  KST hist (10 시간대 RTH OPEN ±):")
        rth = df[df["kst_hm"].between(2200, 2300)]
        print(f"    KST 22:00-23:00 bars: {len(rth):,}")
        print()


if __name__ == "__main__":
    main()
