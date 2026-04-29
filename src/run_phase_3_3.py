# -*- coding: utf-8 -*-
"""
Phase 3.3 — Cluster sub-type 분석

목적:
  Inflection / Trend End 이벤트들이 단일 패턴인가, 여러 sub-type 으로 분리되나?
  k-means + silhouette 으로 cluster 안정성 확인.
  cluster 별 centroid 해석 + forward path label 분포 차이 측정.

판정:
  H4 PASS = silhouette ≥ 0.30 AND cluster 별 forward V/drift 비율 ±10%p 차이
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from indicators import add_indicator_columns
from regime import attach_regime
from events import zigzag_pivots, add_forward_leg, label_inflection, label_trend_leg_end
from profiler import add_sr_and_shape, attach_profile
from path_metrics import attach_path_metrics_multi, attach_labels_v2

DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"
OUT  = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\results_3_3"
os.makedirs(OUT, exist_ok=True)

CLUSTER_FEATURES = [
    "atr_ratio", "rsi14", "stoch_k", "bb_pctb", "bb_width_pct", "vol_z",
    "body_to_atr", "upper_wick_to_atr", "lower_wick_to_atr",
    "dist_to_hh20_atr", "dist_to_ll20_atr",
    "dist_to_ema20_atr", "dist_to_ema50_atr", "dist_to_ema200_atr",
    "pre_5_disp_atr", "pre_15_disp_atr", "pre_30_disp_atr",
]

LABEL_WINDOWS = [5, 15, 30]


def cluster_one_event(name: str, ev_with_path: pd.DataFrame,
                      *, k_list: list = (3, 4, 5),
                      sample_n: int = 30000, seed: int = 42) -> dict:
    print(f"\n--- {name}: N={len(ev_with_path):,} ---")
    df = ev_with_path.dropna(subset=CLUSTER_FEATURES + ["label_w15"])
    print(f"  after dropna: {len(df):,}")
    if len(df) < 1000:
        print("  too few, skip")
        return None

    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed)

    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES].values)

    silhouette_results = {}
    best = None
    for k in k_list:
        t0 = time.time()
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        lbl = km.fit_predict(X)
        # silhouette: 작은 sample 으로 (시간 절약)
        sil_n = min(5000, len(X))
        sil_idx = np.random.RandomState(seed).choice(len(X), sil_n, replace=False)
        sil = silhouette_score(X[sil_idx], lbl[sil_idx])
        silhouette_results[k] = float(sil)
        print(f"  k={k}: silhouette {sil:.3f}  ({time.time()-t0:.1f}s)")
        if best is None or sil > best[1]:
            best = (k, sil, lbl, km)

    k_best, sil_best, lbl_best, km_best = best
    df = df.copy()
    df["cluster"] = lbl_best

    # cluster centroid in original feature space
    centroids_z = km_best.cluster_centers_  # standardized
    centroid_df = pd.DataFrame(centroids_z, columns=CLUSTER_FEATURES,
                               index=[f"c{i}" for i in range(k_best)])

    # cluster 별 forward label 분포 (각 window)
    label_dists = {}
    for w in LABEL_WINDOWS:
        col = f"label_w{w}"
        if col not in df.columns:
            continue
        ct = pd.crosstab(df["cluster"], df[col], normalize="index") * 100
        label_dists[f"w{w}"] = ct.round(2).to_dict()

    # cluster 별 size + 핵심 feature mean
    summary = []
    for c in range(k_best):
        sub = df[df["cluster"] == c]
        row = dict(cluster=c, n=len(sub))
        for f in CLUSTER_FEATURES:
            row[f] = float(sub[f].mean())
        for w in LABEL_WINDOWS:
            col = f"label_w{w}"
            if col not in df.columns:
                continue
            v = (sub[col] == "V").sum() / len(sub) * 100
            d = (sub[col] == "drift").sum() / len(sub) * 100
            f_ = (sub[col] == "flat").sum() / len(sub) * 100
            row[f"w{w}_V_pct"] = round(v, 2)
            row[f"w{w}_drift_pct"] = round(d, 2)
            row[f"w{w}_flat_pct"] = round(f_, 2)
        summary.append(row)

    return dict(
        n_total=int(len(df)),
        k_best=int(k_best),
        silhouette_best=float(sil_best),
        silhouette_all=silhouette_results,
        cluster_summary=summary,
        centroids_z=centroid_df.round(3).to_dict(),
        label_dists=label_dists,
    )


def run_tf(tf: str, rvol_n: int, slope_w: int) -> dict:
    print(f"\n{'='*72}\n=== {tf} cluster\n{'='*72}")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
    df = add_indicator_columns(df, rvol_length=rvol_n)
    df = attach_regime(df, slope_window=slope_w)
    df = add_sr_and_shape(df)
    print(f"  full df: {len(df):,} rows ({time.time()-t0:.1f}s)")

    zz = add_forward_leg(zigzag_pivots(df, threshold_atr_mult=1.5))
    inf = label_inflection(zz, min_back_atr=2.0, min_forward_atr=2.0)
    tre = label_trend_leg_end(zz, min_atr=3.0, min_bars=5, max_bars=200)
    inf_p = attach_profile(inf, df)
    tre_p = attach_profile(tre, df)

    # forward path label 부여 (cluster 분석 시 outcome 비교 용)
    print(f"  attaching path metrics...")
    inf_p = attach_path_metrics_multi(inf_p, df, n_bars_list=LABEL_WINDOWS)
    inf_p = attach_labels_v2(inf_p, LABEL_WINDOWS)
    tre_p = attach_path_metrics_multi(tre_p, df, n_bars_list=LABEL_WINDOWS)
    tre_p = attach_labels_v2(tre_p, LABEL_WINDOWS)
    print(f"  inf: {len(inf_p):,}  tre: {len(tre_p):,}")

    cases = [
        ("inflection_high", inf_p[inf_p["kind"] == "high"]),
        ("inflection_low",  inf_p[inf_p["kind"] == "low"]),
        ("trend_end_high",  tre_p[tre_p["kind"] == "high"]),
        ("trend_end_low",   tre_p[tre_p["kind"] == "low"]),
    ]
    results = {}
    for name, ev in cases:
        r = cluster_one_event(name, ev)
        if r:
            results[name] = r
    return results


if __name__ == "__main__":
    # 5m 만 (1m 은 너무 큼, sklearn 시간 폭증). 5m N 충분.
    res_5m = run_tf("5m", rvol_n=78, slope_w=288)

    bundle = {"5m": res_5m}
    with open(os.path.join(OUT, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str, ensure_ascii=False)

    # 종합 verdict 표
    print(f"\n{'='*72}\n=== H4 Cluster verdict\n{'='*72}")
    print(f"{'event':<22}{'k_best':>8}{'silhouette':>13}{'min_w15_V':>12}{'max_w15_V':>12}{'spread_w15':>12}")
    for name, r in res_5m.items():
        sil = r["silhouette_best"]
        v_pcts = [s.get("w15_V_pct", 0) for s in r["cluster_summary"]]
        v_min, v_max = min(v_pcts), max(v_pcts)
        spread = v_max - v_min
        verdict = "PASS" if sil >= 0.30 and spread >= 10 else "MARGINAL" if sil >= 0.20 or spread >= 5 else "FAIL"
        print(f"{name:<22}{r['k_best']:>8}{sil:>13.3f}{v_min:>12.1f}{v_max:>12.1f}{spread:>12.1f}  {verdict}")

    print(f"\nSaved: {OUT}")
