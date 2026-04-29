# -*- coding: utf-8 -*-
"""
Phase 3.1 — 1m 전체 + 5m 비교, fingerprint JSON/CSV 저장.

산출:
  results_3_1/{tf}_{event}_discriminators.csv
  results_3_1/{tf}_{event}_kst.csv
  results_3_1/{tf}_{event}_regime.csv
  results_3_1/fingerprints.json   ← Phase 4 textbook 의 source
"""
from __future__ import annotations

import os
import json
import time
import pandas as pd
import numpy as np

from indicators import add_indicator_columns
from regime import attach_regime
from events import zigzag_pivots, add_forward_leg, label_inflection, label_trend_leg_end
from profiler import add_sr_and_shape, attach_profile, PROFILE_NUMERIC
from compare import random_baseline, discriminator_table, categorical_compare

DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"
OUT  = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\results_3_1"
os.makedirs(OUT, exist_ok=True)


def run_one_tf(tf: str, rvol_n: int, slope_w: int,
               zz_thresh: float = 1.5,
               inflection_atr: float = 2.0,
               trend_atr: float = 3.0,
               trend_min_bars: int = 5,
               n_baseline: int = 20000) -> dict:
    print(f"\n{'='*68}\n=== {tf}\n{'='*68}")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
    df = add_indicator_columns(df, rvol_length=rvol_n)
    df = attach_regime(df, slope_window=slope_w)
    df = add_sr_and_shape(df)
    print(f"  full df: {len(df):,} rows ({time.time()-t0:.1f}s)")

    t1 = time.time()
    zz = add_forward_leg(zigzag_pivots(df, threshold_atr_mult=zz_thresh))
    inf = label_inflection(zz, min_back_atr=inflection_atr, min_forward_atr=inflection_atr)
    tre = label_trend_leg_end(zz, min_atr=trend_atr, min_bars=trend_min_bars, max_bars=200)
    print(f"  zigzag: {len(zz):,}  inflection: {len(inf):,}  trend_end: {len(tre):,}  ({time.time()-t1:.1f}s)")

    inf_p = attach_profile(inf, df)
    tre_p = attach_profile(tre, df)
    base  = attach_profile(random_baseline(df, n=n_baseline), df)

    inf_high = inf_p[inf_p["kind"] == "high"]
    inf_low  = inf_p[inf_p["kind"] == "low"]
    tre_high = tre_p[tre_p["kind"] == "high"]
    tre_low  = tre_p[tre_p["kind"] == "low"]

    cases = [
        ("inflection_high", inf_high),
        ("inflection_low",  inf_low),
        ("trend_end_high",  tre_high),
        ("trend_end_low",   tre_low),
    ]

    summary = {
        "tf": tf,
        "rows": int(len(df)),
        "params": dict(zz_thresh=zz_thresh, inflection_atr=inflection_atr,
                       trend_atr=trend_atr, trend_min_bars=trend_min_bars,
                       n_baseline=n_baseline),
        "events": {},
    }
    for name, ev in cases:
        if len(ev) < 30:
            print(f"  [{name}] N={len(ev)} too small, skip")
            continue
        d_table = discriminator_table(ev, base, PROFILE_NUMERIC)
        d_table.to_csv(os.path.join(OUT, f"{tf}_{name}_discriminators.csv"), index=False)
        kst_cmp = categorical_compare(ev, base, "kst_h")
        kst_cmp.to_csv(os.path.join(OUT, f"{tf}_{name}_kst.csv"))
        reg_cmp = categorical_compare(ev, base, "regime")
        reg_cmp.to_csv(os.path.join(OUT, f"{tf}_{name}_regime.csv"))

        # JSON 으로 top-15 discriminator 만 추출 + KST top-3
        top_d = d_table.head(15)[["feature", "mean_event", "mean_base",
                                  "median_event", "median_base", "cohens_d", "magnitude"]]
        top_kst = kst_cmp.head(5).reset_index()
        # convert 'kst_h' index column name
        top_kst_records = []
        for _, r in top_kst.iterrows():
            top_kst_records.append({
                "kst_h": int(r["kst_h"]),
                "event_pct": float(r["kst_h_event_pct"]),
                "base_pct": float(r["kst_h_base_pct"]),
                "lift": float(r["kst_h_lift"]) if pd.notna(r["kst_h_lift"]) else None,
            })

        summary["events"][name] = {
            "n_event": int(len(ev)),
            "n_baseline": int(len(base)),
            "top_discriminators": top_d.to_dict("records"),
            "top_kst_hours": top_kst_records,
            "regime_lift": {str(k): float(v) for k, v in
                            reg_cmp["regime_lift"].dropna().to_dict().items()},
        }
        print(f"  [{name}] N={len(ev):,}  TOP3 discr: " +
              ", ".join(f"{r['feature']} d={r['cohens_d']:+.2f}"
                        for _, r in d_table.head(3).iterrows()))
    return summary


def cross_tf_consistency(s5m: dict, s1m: dict) -> dict:
    """5m top-15 discriminator 중 1m 에서도 |d| ≥ 0.5 비율 = consistency."""
    out = {}
    for ev_name in s5m["events"]:
        if ev_name not in s1m["events"]:
            continue
        top_5m = {r["feature"] for r in s5m["events"][ev_name]["top_discriminators"]}
        d_1m = {r["feature"]: r["cohens_d"] for r in s1m["events"][ev_name]["top_discriminators"]}
        # 1m 의 모든 features 모음 (top-15 안에 든 것만)
        consistent = sum(1 for f in top_5m if f in d_1m and abs(d_1m[f]) >= 0.5)
        out[ev_name] = dict(
            n_5m_top=len(top_5m),
            consistent_in_1m=consistent,
            consistency_pct=round(consistent / len(top_5m) * 100, 1),
        )
    return out


if __name__ == "__main__":
    s5m = run_one_tf("5m", rvol_n=78,  slope_w=288)
    s1m = run_one_tf("1m", rvol_n=390, slope_w=1440)

    cons = cross_tf_consistency(s5m, s1m)

    bundle = {"5m": s5m, "1m": s1m, "consistency_5m_top15_in_1m": cons}
    with open(os.path.join(OUT, "fingerprints.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n{'='*68}\n=== 5m vs 1m consistency (5m top-15 features 가 1m 에서도 |d|≥0.5)\n{'='*68}")
    for name, v in cons.items():
        print(f"  {name}: {v['consistent_in_1m']}/{v['n_5m_top']} = {v['consistency_pct']}%")

    # judgment
    pct_avg = sum(v["consistency_pct"] for v in cons.values()) / max(len(cons), 1)
    print(f"\nMean consistency: {pct_avg:.1f}%")
    print("Phase 3.1 PASS criterion: ≥70% — " +
          ("PASS ✓" if pct_avg >= 70 else f"BELOW (gap {70-pct_avg:.1f}%p)"))

    print(f"\nSaved: {OUT}")
