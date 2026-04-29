# -*- coding: utf-8 -*-
"""
Phase 3.2 — Pre-only Features AUC

목적:
  실시간 운용 가능한 features (forward/leg metadata 제외) 만으로 분류기 학습.
  AUC = "이 바가 inflection/trend_end 인가?" 를 t0 시점 정보만으로 분류.
  H1: Inflection AUC ≥ 0.80 / H2: Trend End AUC ≥ 0.85 / H3: Pre-only AUC ≥ 0.70

설계:
  - features: PROFILE_NUMERIC 만 (leg_back_*, leg_forward_*, kind 제외)
  - train: 2008-12 ~ 2025-04-15
  - test : 2025-04-15 ~ 2026-04-15  (OOS 12개월)
  - 모델: LogisticRegression (interpretable) + GradientBoostingClassifier
  - 클래스 균형: events vs baseline 동일 N 샘플링
  - reproducibility: seed=42
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# unbuffered stdout — background task 진행 가시성
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve

from indicators import add_indicator_columns
from regime import attach_regime
from events import zigzag_pivots, add_forward_leg, label_inflection, label_trend_leg_end
from profiler import add_sr_and_shape, attach_profile
from compare import random_baseline

DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"
OUT  = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\results_3_2"
os.makedirs(OUT, exist_ok=True)

# Pre-only features — forward/leg metadata 제외
PRE_ONLY = [
    "atr_ratio", "rsi14", "mfi14", "stoch_k", "stoch_d",
    "bb_pctb", "bb_width_pct", "rvol", "vol_z",
    "slope_pct", "above_pct",
    "body_to_atr", "upper_wick_to_atr", "lower_wick_to_atr",
    "dist_to_hh20_atr", "dist_to_ll20_atr",
    "dist_to_ema20_atr", "dist_to_ema50_atr", "dist_to_ema200_atr",
    "pre_5_disp_atr", "pre_15_disp_atr", "pre_30_disp_atr",
]


def precision_at_top_k(y_true, p_score, k_pct: float):
    """상위 k_pct% 확률 예측의 precision."""
    n = len(y_true)
    k = max(1, int(n * k_pct))
    top_idx = np.argsort(p_score)[::-1][:k]
    prec = y_true[top_idx].sum() / k
    return float(prec), int(k)


def evaluate_event(event_df: pd.DataFrame, baseline_df: pd.DataFrame,
                   features: list, time_split: pd.Timestamp,
                   max_per_class: int = 20000, seed: int = 42) -> dict:
    """
    event_df, baseline_df: profile 부여된 DataFrame (DatetimeIndex).
    Returns: AUC + feature importance dict.
    """
    pos = event_df[features].copy()
    pos["label"] = 1
    pos["ts"] = event_df.index
    neg = baseline_df[features].copy()
    neg["label"] = 0
    neg["ts"] = baseline_df.index

    full = pd.concat([pos, neg], ignore_index=True).dropna(subset=features)

    train = full[full["ts"] < time_split]
    test  = full[full["ts"] >= time_split]
    if len(test) < 100 or test["label"].sum() < 30 or test["label"].sum() == len(test):
        return None

    # 클래스 균형: 각 split 안에서 max_per_class 까지만
    rng = np.random.default_rng(seed)

    def balance(d):
        pos_d = d[d["label"] == 1]
        neg_d = d[d["label"] == 0]
        n = min(len(pos_d), len(neg_d), max_per_class)
        if n < 50:
            return d  # too small, no balance
        return pd.concat([
            pos_d.sample(n, random_state=seed),
            neg_d.sample(n, random_state=seed),
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    train_b = balance(train)
    test_b  = balance(test)

    X_tr, y_tr = train_b[features].values, train_b["label"].values
    X_te, y_te = test_b[features].values,  test_b["label"].values

    # Logistic
    pipe_lr = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])
    pipe_lr.fit(X_tr, y_tr)
    p_lr_tr = pipe_lr.predict_proba(X_tr)[:, 1]
    p_lr_te = pipe_lr.predict_proba(X_te)[:, 1]
    auc_lr_tr = roc_auc_score(y_tr, p_lr_tr)
    auc_lr_te = roc_auc_score(y_te, p_lr_te)

    # GBM
    gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                      learning_rate=0.1, random_state=seed)
    gbm.fit(X_tr, y_tr)
    p_gbm_tr = gbm.predict_proba(X_tr)[:, 1]
    p_gbm_te = gbm.predict_proba(X_te)[:, 1]
    auc_gbm_tr = roc_auc_score(y_tr, p_gbm_tr)
    auc_gbm_te = roc_auc_score(y_te, p_gbm_te)

    # Top-K precision
    prec_top10_lr,  k10  = precision_at_top_k(y_te, p_lr_te,  0.10)
    prec_top10_gbm, _    = precision_at_top_k(y_te, p_gbm_te, 0.10)
    prec_top5_gbm,  k5   = precision_at_top_k(y_te, p_gbm_te, 0.05)

    importances = pd.Series(gbm.feature_importances_, index=features).sort_values(ascending=False)
    coefs = pd.Series(pipe_lr.named_steps["lr"].coef_[0], index=features)
    coefs_abs = coefs.abs().sort_values(ascending=False)

    return dict(
        n_train=int(len(train_b)),
        n_test=int(len(test_b)),
        n_train_pos=int(train_b["label"].sum()),
        n_test_pos=int(test_b["label"].sum()),
        auc_lr_train=float(auc_lr_tr),
        auc_lr_test=float(auc_lr_te),
        auc_gbm_train=float(auc_gbm_tr),
        auc_gbm_test=float(auc_gbm_te),
        overfit_gap_gbm=float(auc_gbm_tr - auc_gbm_te),
        prec_top10_lr=prec_top10_lr,
        prec_top10_gbm=prec_top10_gbm,
        prec_top5_gbm=prec_top5_gbm,
        k_top10=int(k10),
        k_top5=int(k5),
        gbm_importances=importances.to_dict(),
        lr_coef_abs=coefs_abs.to_dict(),
        lr_coef_signed=coefs.to_dict(),
    )


def run_tf(tf: str, rvol_n: int, slope_w: int,
           inflection_atr: float = 2.0,
           trend_atr: float = 3.0,
           n_baseline: int = 80000) -> dict:
    print(f"\n{'='*72}\n=== {tf} pre-only AUC\n{'='*72}")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
    df = add_indicator_columns(df, rvol_length=rvol_n)
    df = attach_regime(df, slope_window=slope_w)
    df = add_sr_and_shape(df)
    print(f"  full df: {len(df):,} rows ({time.time()-t0:.1f}s)")

    zz = add_forward_leg(zigzag_pivots(df, threshold_atr_mult=1.5))
    inf = label_inflection(zz, min_back_atr=inflection_atr, min_forward_atr=inflection_atr)
    tre = label_trend_leg_end(zz, min_atr=trend_atr, min_bars=5, max_bars=200)
    inf_p = attach_profile(inf, df)
    tre_p = attach_profile(tre, df)
    base  = attach_profile(random_baseline(df, n=n_baseline, seed=42), df)
    print(f"  events inf={len(inf_p):,} tre={len(tre_p):,} base={len(base):,}")

    TIME_SPLIT = pd.Timestamp("2025-04-15", tz="UTC")
    cases = [
        ("inflection_high", inf_p[inf_p["kind"] == "high"]),
        ("inflection_low",  inf_p[inf_p["kind"] == "low"]),
        ("trend_end_high",  tre_p[tre_p["kind"] == "high"]),
        ("trend_end_low",   tre_p[tre_p["kind"] == "low"]),
    ]
    results = {}
    for name, ev in cases:
        t1 = time.time()
        r = evaluate_event(ev, base, PRE_ONLY, TIME_SPLIT)
        if r is None:
            print(f"  [{name}] insufficient test data, skip")
            continue
        results[name] = r
        print(f"\n  [{name}] N_event={len(ev):,}  ({time.time()-t1:.1f}s)")
        print(f"    test  AUC: LR {r['auc_lr_test']:.3f} | GBM {r['auc_gbm_test']:.3f}  "
              f"(overfit gap {r['overfit_gap_gbm']:+.3f})")
        print(f"    train AUC: LR {r['auc_lr_train']:.3f} | GBM {r['auc_gbm_train']:.3f}")
        print(f"    precision @ top-10%: LR {r['prec_top10_lr']:.3f} | GBM {r['prec_top10_gbm']:.3f}")
        print(f"    precision @ top-5%:  GBM {r['prec_top5_gbm']:.3f}")
        top_imp = sorted(r["gbm_importances"].items(), key=lambda x: -x[1])[:5]
        print(f"    GBM top-5 importance: " +
              ", ".join(f"{k} ({v:.3f})" for k, v in top_imp))
    return results


if __name__ == "__main__":
    # 1m 만 우선 (5.8M, 더 많은 events). 5m 은 backup.
    res_1m = run_tf("1m", rvol_n=390, slope_w=1440, n_baseline=80000)
    res_5m = run_tf("5m", rvol_n=78,  slope_w=288,  n_baseline=80000)

    bundle = {"1m": res_1m, "5m": res_5m}
    with open(os.path.join(OUT, "auc_pre_only.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str, ensure_ascii=False)

    # Verdict 표
    print(f"\n{'='*72}\n=== H1/H2/H3 verdict\n{'='*72}")
    print(f"{'event':<20}{'TF':>4}{'AUC_LR':>9}{'AUC_GBM':>10}{'overfit':>10}{'prec@10%':>10}")
    for tf, results in [("1m", res_1m), ("5m", res_5m)]:
        for name, r in results.items():
            print(f"{name:<20}{tf:>4}{r['auc_lr_test']:>9.3f}{r['auc_gbm_test']:>10.3f}"
                  f"{r['overfit_gap_gbm']:>+10.3f}{r['prec_top10_gbm']:>10.3f}")

    # H1/H2/H3 판정
    print(f"\nH1 Inflection AUC ≥ 0.80:")
    for tf, results in [("1m", res_1m), ("5m", res_5m)]:
        for name in ("inflection_high", "inflection_low"):
            if name in results:
                auc = results[name]["auc_gbm_test"]
                v = "PASS ✓" if auc >= 0.80 else f"FAIL ({0.80-auc:+.3f})"
                print(f"  {tf} {name}: {auc:.3f}  {v}")
    print(f"\nH2 Trend End AUC ≥ 0.85:")
    for tf, results in [("1m", res_1m), ("5m", res_5m)]:
        for name in ("trend_end_high", "trend_end_low"):
            if name in results:
                auc = results[name]["auc_gbm_test"]
                v = "PASS ✓" if auc >= 0.85 else f"FAIL ({0.85-auc:+.3f})"
                print(f"  {tf} {name}: {auc:.3f}  {v}")
    print(f"\nH3 Pre-only AUC ≥ 0.70 (모든 event):")
    all_aucs = []
    for tf, results in [("1m", res_1m), ("5m", res_5m)]:
        for name, r in results.items():
            all_aucs.append((tf, name, r["auc_gbm_test"]))
    min_auc = min(x[2] for x in all_aucs) if all_aucs else 0
    print(f"  min AUC (all {len(all_aucs)} cases): {min_auc:.3f}  " +
          ("PASS ✓" if min_auc >= 0.70 else f"FAIL ({0.70-min_auc:+.3f})"))

    print(f"\nSaved: {OUT}")
