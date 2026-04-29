# -*- coding: utf-8 -*-
"""
Phase 3.5 + 3.6 — Tier A 재검증 (H5) + Conditional Probability 표

3.5 (H5):
  OB pivots (R41A Bull / R42A Bear) 의 Tier A vs Tier B fingerprint 비교
  Tier A 정의:
    Bull: KST slot 21:00-22:30 AND atr_ratio > 1.3
    Bear: KST slot 22:30-23:00 (RTH open) AND atr_ratio > 1.0
  비교: forward 5/15/30봉 mean disp + WR + sharpe

3.6 (Conditional probability 표):
  cell = pivot_kind × kst_slot × atr_bucket × pre_dir
  각 cell 의 V/W/drift/Λ/N/flat 비율 + forward w15 EV + N
  STOP 룰 = V+drift < 5% AND N >= 50 cell
  GO 룰   = V+drift > 25% AND N >= 50 cell
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

from indicators import add_indicator_columns
from regime import attach_regime
from events import zigzag_pivots, add_forward_leg, label_inflection, label_trend_leg_end
from pivots_ob import find_ob_origins
from profiler import add_sr_and_shape, attach_profile
from path_metrics import attach_path_metrics_multi, attach_labels_v2

DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"
OUT_35 = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\results_3_5"
OUT_36 = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\results_3_6"
os.makedirs(OUT_35, exist_ok=True)
os.makedirs(OUT_36, exist_ok=True)


# ────────────────────────────────────────────────────────────
# Phase 3.5 — Tier A H5
# ────────────────────────────────────────────────────────────

def phase_3_5(tf: str, rvol_n: int, slope_w: int) -> dict:
    print(f"\n{'='*72}\n=== 3.5 H5 Tier A ({tf})\n{'='*72}")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
    df = add_indicator_columns(df, rvol_length=rvol_n)
    df = attach_regime(df, slope_window=slope_w)
    df = add_sr_and_shape(df)
    print(f"  full df: {len(df):,} rows ({time.time()-t0:.1f}s)")

    # OB origins (5m=Bull 권장, 1m=Bear 권장)
    ob = find_ob_origins(df)
    ob_p = attach_profile(ob, df)
    # forward path metric
    ob_p = attach_path_metrics_multi(ob_p, df, n_bars_list=[5, 15, 30])
    ob_p = attach_labels_v2(ob_p, [5, 15, 30])
    print(f"  OB origins: Bull={len(ob_p[ob_p['kind']=='O_BULL']):,} Bear={len(ob_p[ob_p['kind']=='O_BEAR']):,}")

    # Tier 분류
    ob_p = ob_p.copy()
    bull_in_slot = ob_p["kst_h"].isin([21]) | ((ob_p["kst_h"] == 22) & (ob_p["kst_m"] < 30))
    bear_in_slot = (ob_p["kst_h"] == 22) & (ob_p["kst_m"] >= 30) | (ob_p["kst_h"] == 23)
    # actually fix: Bear slot 22:30-23:00 → kst_h=22 & kst_m>=30
    bear_in_slot = (ob_p["kst_h"] == 22) & (ob_p["kst_m"] >= 30)

    is_bull = ob_p["kind"] == "O_BULL"
    is_bear = ob_p["kind"] == "O_BEAR"
    bull_tier_a = is_bull & bull_in_slot & (ob_p["atr_ratio"] >= 1.3)
    bear_tier_a = is_bear & bear_in_slot & (ob_p["atr_ratio"] >= 1.0)
    bull_tier_b = is_bull & ~bull_tier_a
    bear_tier_b = is_bear & ~bear_tier_a

    ob_p.loc[bull_tier_a, "tier"] = "Bull_A"
    ob_p.loc[bull_tier_b, "tier"] = "Bull_B"
    ob_p.loc[bear_tier_a, "tier"] = "Bear_A"
    ob_p.loc[bear_tier_b, "tier"] = "Bear_B"

    print(f"  Bull_A {bull_tier_a.sum():,} / Bull_B {bull_tier_b.sum():,}")
    print(f"  Bear_A {bear_tier_a.sum():,} / Bear_B {bear_tier_b.sum():,}")

    # forward disp 계산 (close[+15]-close[+0]) × direction
    cl = df["close"].to_numpy()
    L = len(df)

    def compute_forward(rows, direction, n_bars):
        idxs = rows["idx"].to_numpy()
        disps = []
        for i in idxs:
            if i + n_bars >= L:
                continue
            d = (cl[i + n_bars] - cl[i]) * direction
            disps.append(d)
        return np.array(disps)

    results = {}
    for tier_name, mask, direction in [
        ("Bull_A", bull_tier_a, +1),
        ("Bull_B", bull_tier_b, +1),
        ("Bear_A", bear_tier_a, -1),
        ("Bear_B", bear_tier_b, -1),
    ]:
        if mask.sum() == 0:
            continue
        rows = ob_p[mask]
        tier_res = dict(N=int(len(rows)))
        for w in (5, 15, 30):
            disps = compute_forward(rows, direction, w)
            if len(disps) == 0:
                continue
            tier_res[f"w{w}"] = dict(
                N=int(len(disps)),
                mean_disp=float(disps.mean()),
                median_disp=float(np.median(disps)),
                std=float(disps.std()),
                sharpe=float(disps.mean() / (disps.std() + 1e-9)),
                win_rate=float((disps > 0).mean() * 100),
            )
        # path label 분포
        for w in (5, 15, 30):
            col = f"label_w{w}"
            if col in rows.columns:
                vc = rows[col].value_counts(normalize=True) * 100
                tier_res[f"label_dist_w{w}"] = vc.round(2).to_dict()
        results[tier_name] = tier_res
        print(f"\n  [{tier_name}] N={len(rows):,}")
        for w in (5, 15, 30):
            r = tier_res.get(f"w{w}", {})
            if r:
                print(f"    w{w}: mean {r['mean_disp']:+.2f} pt "
                      f"sharpe {r['sharpe']:+.3f} WR {r['win_rate']:.1f}%")

    # H5 verdict: A vs B sharpe diff
    print(f"\n  H5 verdict — Tier A vs Tier B sharpe diff:")
    for pair_a, pair_b in [("Bull_A", "Bull_B"), ("Bear_A", "Bear_B")]:
        if pair_a in results and pair_b in results:
            for w in (5, 15, 30):
                a = results[pair_a].get(f"w{w}", {})
                b = results[pair_b].get(f"w{w}", {})
                if a and b:
                    diff = a["sharpe"] - b["sharpe"]
                    print(f"    {pair_a} vs {pair_b} w{w}: "
                          f"A={a['sharpe']:+.3f} B={b['sharpe']:+.3f} "
                          f"diff={diff:+.3f}")
    return results


# ────────────────────────────────────────────────────────────
# Phase 3.6 — Conditional Probability 표
# ────────────────────────────────────────────────────────────

def kst_slot_bucket(kst_hm: int) -> str:
    """KST hh*100+mm 을 5 버킷 으로."""
    if 1900 <= kst_hm < 2100:
        return "EU_late"
    if 2100 <= kst_hm < 2230:
        return "NY_pre_open"
    if 2230 <= kst_hm < 2300:
        return "NY_open"
    if 2300 <= kst_hm < 100:
        return "NY_burst"
    if 100 <= kst_hm < 500:
        return "Asia_late"
    return "Other"


def atr_bucket(atr_ratio: float) -> str:
    if pd.isna(atr_ratio):
        return "NA"
    if atr_ratio < 0.8:
        return "low"
    if atr_ratio < 1.2:
        return "mid"
    return "high"


def phase_3_6(tf: str, rvol_n: int, slope_w: int) -> dict:
    print(f"\n{'='*72}\n=== 3.6 Conditional Probability 표 ({tf})\n{'='*72}")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
    df = add_indicator_columns(df, rvol_length=rvol_n)
    df = attach_regime(df, slope_window=slope_w)
    df = add_sr_and_shape(df)
    print(f"  full df: {len(df):,} rows ({time.time()-t0:.1f}s)")

    # ZigZag inflection events 가 모집단
    zz = add_forward_leg(zigzag_pivots(df, threshold_atr_mult=1.5))
    inf = label_inflection(zz, min_back_atr=2.0, min_forward_atr=2.0)
    inf_p = attach_profile(inf, df)
    inf_p = attach_path_metrics_multi(inf_p, df, n_bars_list=[15])
    inf_p = attach_labels_v2(inf_p, [15])
    print(f"  inflection events: {len(inf_p):,}")

    # cell 부여
    inf_p = inf_p.copy()
    inf_p["slot"] = inf_p["kst_hm"].apply(kst_slot_bucket)
    inf_p["atr_bin"] = inf_p["atr_ratio"].apply(atr_bucket)
    inf_p["pre_dir"] = np.where(inf_p["pre_5_disp_atr"] > 0.3, "up",
                       np.where(inf_p["pre_5_disp_atr"] < -0.3, "down", "flat"))

    # 그룹: kind × slot × atr_bin × pre_dir
    grp_cols = ["kind", "slot", "atr_bin", "pre_dir"]
    grouped = inf_p.groupby(grp_cols)

    # forward disp (high → direction=-1, low → +1)
    cl = df["close"].to_numpy()
    L = len(df)

    def fwd_disp(row, n=15):
        i = int(row["idx"])
        if i + n >= L:
            return np.nan
        direction = +1 if row["kind"] == "low" else -1
        return (cl[i + n] - cl[i]) * direction

    inf_p["fwd_w15_disp"] = inf_p.apply(fwd_disp, axis=1)

    rows = []
    for keys, grp in grouped:
        kind, slot, atr_bin, pre_dir = keys
        n = len(grp)
        if n < 30:
            continue
        labels = grp["label_w15"]
        v_pct = (labels == "V").mean() * 100
        w_pct = (labels == "W").mean() * 100
        drift_pct = (labels == "drift").mean() * 100
        flat_pct = (labels == "flat").mean() * 100
        n_pct = (labels == "N").mean() * 100
        lambda_pct = (labels == "Λ").mean() * 100
        fwd = grp["fwd_w15_disp"].dropna()
        rows.append({
            "kind": kind, "slot": slot, "atr_bin": atr_bin, "pre_dir": pre_dir,
            "N": n,
            "V_pct": round(v_pct, 1),
            "W_pct": round(w_pct, 1),
            "drift_pct": round(drift_pct, 1),
            "Lambda_pct": round(lambda_pct, 1),
            "N_pct": round(n_pct, 1),
            "flat_pct": round(flat_pct, 1),
            "v_drift_sum": round(v_pct + drift_pct, 1),
            "w15_mean_disp": round(float(fwd.mean()), 2) if len(fwd) else None,
            "w15_sharpe": round(float(fwd.mean() / (fwd.std() + 1e-9)), 3) if len(fwd) > 1 else None,
            "w15_wr": round(float((fwd > 0).mean() * 100), 1) if len(fwd) else None,
        })

    table = pd.DataFrame(rows).sort_values("v_drift_sum", ascending=False)
    print(f"  cells (N≥30): {len(table):,}")

    # GO / STOP 추출
    go_cells = table[(table["v_drift_sum"] >= 25) & (table["N"] >= 50)].copy()
    stop_cells = table[(table["v_drift_sum"] < 5) & (table["N"] >= 50)].copy()
    high_ev_cells = table[(table["w15_sharpe"].fillna(-9) > 0.05) & (table["N"] >= 50)].copy()
    bad_ev_cells = table[(table["w15_sharpe"].fillna(0) < -0.05) & (table["N"] >= 50)].copy()

    print(f"\n  GO cells (V+drift ≥ 25%, N≥50): {len(go_cells):,}")
    if len(go_cells):
        print(go_cells.head(10).to_string(index=False))

    print(f"\n  STOP cells (V+drift < 5%, N≥50): {len(stop_cells):,}")
    if len(stop_cells):
        print(stop_cells.head(10).to_string(index=False))

    print(f"\n  HIGH EV cells (sharpe > 0.05, N≥50): {len(high_ev_cells):,}")
    if len(high_ev_cells):
        print(high_ev_cells.sort_values("w15_sharpe", ascending=False).head(10).to_string(index=False))

    print(f"\n  BAD EV cells (sharpe < -0.05, N≥50): {len(bad_ev_cells):,}")
    if len(bad_ev_cells):
        print(bad_ev_cells.sort_values("w15_sharpe").head(10).to_string(index=False))

    # 저장
    table.to_csv(os.path.join(OUT_36, f"{tf}_conditional_table.csv"), index=False)
    go_cells.to_csv(os.path.join(OUT_36, f"{tf}_GO_cells.csv"), index=False)
    stop_cells.to_csv(os.path.join(OUT_36, f"{tf}_STOP_cells.csv"), index=False)
    high_ev_cells.to_csv(os.path.join(OUT_36, f"{tf}_HIGH_EV_cells.csv"), index=False)

    return dict(
        n_cells=len(table),
        n_go=len(go_cells),
        n_stop=len(stop_cells),
        n_high_ev=len(high_ev_cells),
        n_bad_ev=len(bad_ev_cells),
        top_go=go_cells.head(10).to_dict("records"),
        top_high_ev=high_ev_cells.sort_values("w15_sharpe", ascending=False).head(10).to_dict("records"),
        top_bad_ev=bad_ev_cells.sort_values("w15_sharpe").head(10).to_dict("records"),
    )


if __name__ == "__main__":
    print(f"{'#'*72}\n# Phase 3.5 — H5 Tier A 재검증\n{'#'*72}")
    res_35_5m = phase_3_5("5m", rvol_n=78,  slope_w=288)
    res_35_1m = phase_3_5("1m", rvol_n=390, slope_w=1440)
    with open(os.path.join(OUT_35, "tier_a_results.json"), "w", encoding="utf-8") as f:
        json.dump({"5m": res_35_5m, "1m": res_35_1m}, f, indent=2, default=str, ensure_ascii=False)

    print(f"\n{'#'*72}\n# Phase 3.6 — Conditional Probability 표\n{'#'*72}")
    res_36_5m = phase_3_6("5m", rvol_n=78,  slope_w=288)
    res_36_1m = phase_3_6("1m", rvol_n=390, slope_w=1440)
    with open(os.path.join(OUT_36, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"5m": res_36_5m, "1m": res_36_1m}, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nSaved 3.5: {OUT_35}")
    print(f"Saved 3.6: {OUT_36}")
