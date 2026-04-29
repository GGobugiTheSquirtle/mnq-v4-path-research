# -*- coding: utf-8 -*-
"""
compare — Random baseline vs event profile, Cohen's d ranking

목적:
  이벤트 시점의 feature 분포가 random 시점 분포와 통계적으로 얼마나 다른가?
  Cohen's d = (mean_event - mean_baseline) / pooled_stdev
  |d| < 0.2 = negligible, 0.2~0.5 = small, 0.5~0.8 = medium, > 0.8 = large
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_baseline(df_full: pd.DataFrame, *, n: int = 10000, seed: int = 42,
                    drop_warmup_bars: int = 500) -> pd.DataFrame:
    """랜덤 N개 row 선택. warmup (NaN heavy) 제외."""
    rng = np.random.default_rng(seed)
    valid_lo = drop_warmup_bars
    valid_hi = len(df_full) - 1
    idxs = rng.choice(np.arange(valid_lo, valid_hi), size=n, replace=False)
    out = df_full.iloc[idxs].copy()
    out["idx"] = idxs
    return out


def cohens_d(event_vals: np.ndarray, base_vals: np.ndarray) -> float:
    """Cohen's d (pooled stdev). NaN 제외."""
    e = event_vals[~np.isnan(event_vals)]
    b = base_vals[~np.isnan(base_vals)]
    if len(e) < 2 or len(b) < 2:
        return np.nan
    me, mb = e.mean(), b.mean()
    se, sb = e.std(ddof=1), b.std(ddof=1)
    ne, nb = len(e), len(b)
    pooled = np.sqrt(((ne - 1) * se ** 2 + (nb - 1) * sb ** 2) / (ne + nb - 2))
    if pooled == 0:
        return np.nan
    return (me - mb) / pooled


def discriminator_table(event_profile: pd.DataFrame,
                        base_profile: pd.DataFrame,
                        cols: list) -> pd.DataFrame:
    """
    각 feature 의 event vs baseline 비교.
    반환: DataFrame (feature, n_event, n_base, mean_event, mean_base,
                    median_event, median_base, cohens_d, abs_d, magnitude)
    """
    rows = []
    for c in cols:
        if c not in event_profile.columns or c not in base_profile.columns:
            continue
        ev = event_profile[c].to_numpy(dtype=float)
        ba = base_profile[c].to_numpy(dtype=float)
        d = cohens_d(ev, ba)
        if np.isnan(d):
            continue
        ev_clean = ev[~np.isnan(ev)]
        ba_clean = ba[~np.isnan(ba)]
        rows.append(dict(
            feature=c,
            n_event=len(ev_clean),
            n_base=len(ba_clean),
            mean_event=ev_clean.mean(),
            mean_base=ba_clean.mean(),
            median_event=np.median(ev_clean),
            median_base=np.median(ba_clean),
            cohens_d=d,
            abs_d=abs(d),
        ))
    out = pd.DataFrame(rows).sort_values("abs_d", ascending=False)
    out["magnitude"] = pd.cut(
        out["abs_d"], bins=[-np.inf, 0.1, 0.2, 0.5, 0.8, np.inf],
        labels=["negligible", "tiny", "small", "medium", "large"]
    )
    return out.reset_index(drop=True)


def categorical_compare(event_profile: pd.DataFrame,
                        base_profile: pd.DataFrame,
                        col: str) -> pd.DataFrame:
    """범주형 feature 의 event 빈도 vs baseline 빈도 비교."""
    e_counts = event_profile[col].value_counts(normalize=True) * 100
    b_counts = base_profile[col].value_counts(normalize=True) * 100
    df = pd.DataFrame({
        f"{col}_event_pct": e_counts,
        f"{col}_base_pct":  b_counts,
    }).fillna(0)
    df[f"{col}_lift"] = df[f"{col}_event_pct"] / df[f"{col}_base_pct"].replace(0, np.nan)
    return df.sort_values(f"{col}_lift", ascending=False)


if __name__ == "__main__":
    import os, time
    from indicators import add_indicator_columns
    from regime import attach_regime
    from events import zigzag_pivots, add_forward_leg, label_inflection, label_trend_leg_end
    from profiler import add_sr_and_shape, attach_profile, PROFILE_NUMERIC, PROFILE_CATEGORICAL

    DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"

    print("=== 5m: build features + events ===")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, "nq_5m.parquet"))
    df = add_indicator_columns(df, rvol_length=78)
    df = attach_regime(df, slope_window=288)
    df = add_sr_and_shape(df)
    print(f"  full df: {len(df):,} rows  ({time.time()-t0:.1f}s)")

    zz = add_forward_leg(zigzag_pivots(df, threshold_atr_mult=1.5))
    inf = label_inflection(zz, min_back_atr=2.0, min_forward_atr=2.0)
    tre = label_trend_leg_end(zz, min_atr=3.0, min_bars=5, max_bars=200)
    print(f"  inflection events: {len(inf):,}")
    print(f"  trend_end  events: {len(tre):,}")

    inf_p = attach_profile(inf, df)
    tre_p = attach_profile(tre, df)
    base   = attach_profile(random_baseline(df, n=20000), df)

    # split inflection into high vs low pivot
    inf_high = inf_p[inf_p["kind"] == "high"]
    inf_low  = inf_p[inf_p["kind"] == "low"]
    tre_high = tre_p[tre_p["kind"] == "high"]   # uptrend ended at high
    tre_low  = tre_p[tre_p["kind"] == "low"]    # downtrend ended at low

    cases = [
        ("INFLECTION HIGH (uptrend reverses)", inf_high),
        ("INFLECTION LOW  (downtrend reverses)", inf_low),
        ("TREND END HIGH (uptrend exhaustion)", tre_high),
        ("TREND END LOW  (downtrend exhaustion)", tre_low),
    ]

    for label, ev in cases:
        print(f"\n{'='*64}")
        print(f"{label}  (N={len(ev):,} vs baseline N={len(base):,})")
        print('='*64)
        tbl = discriminator_table(ev, base, PROFILE_NUMERIC)
        print("\n  TOP 12 discriminators (by |Cohen's d|):")
        print(tbl.head(12)[["feature", "mean_event", "mean_base",
                            "cohens_d", "magnitude"]].round(3).to_string(index=False))

        # categorical: regime
        print(f"\n  regime distribution (event vs baseline %):")
        cat = categorical_compare(ev, base, "regime")
        print(cat.round(2).to_string())

        # KST hour 분포
        print(f"\n  TOP-3 KST hour with highest event rate vs baseline:")
        kst_cmp = categorical_compare(ev, base, "kst_h")
        print(kst_cmp.head(3).round(2).to_string())
