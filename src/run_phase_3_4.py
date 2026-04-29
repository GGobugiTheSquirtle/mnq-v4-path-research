# -*- coding: utf-8 -*-
"""
Phase 3.4 — Pine 룰 후보 → ZigZag event hit rate + forward EV

룰 정의 (Phase 3.2 GBM top features 기반, Pine 호환):
  inflection_high (mean reversion short):
    upper_wick_to_atr > 0.5 AND pre_5_disp_atr > 0.8 AND
    stoch_k > 65 AND bb_pctb > 70
  inflection_low (mean reversion long):
    lower_wick_to_atr > 0.5 AND pre_5_disp_atr < -0.8 AND
    stoch_k < 35 AND bb_pctb < 30
  trend_end_high (강상승 종료, exit/short signal):
    stoch_d > 80 AND pre_5_disp_atr > 1.5 AND
    upper_wick_to_atr > 0.4 AND dist_to_ema20_atr > 1.0
  trend_end_low (강하락 종료, exit/long signal):
    stoch_d < 20 AND pre_5_disp_atr < -1.5 AND
    lower_wick_to_atr > 0.4 AND dist_to_ema20_atr < -1.0

판정:
  hit rate ≥ 30% (baseline ~0.5-2% lift 15-60x)
  forward 5/15/30봉 EV: hit rate × avg_disp 기반

KST 시간대 필터 옵션 (HOT_SLOTS): 21, 22, 15시
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
from profiler import add_sr_and_shape

DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"
OUT  = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\results_3_4"
os.makedirs(OUT, exist_ok=True)

KST_HOT_SLOTS = (21, 22, 15)


def define_rules(df: pd.DataFrame, *, with_kst_filter: bool = False) -> dict:
    """4 룰의 boolean Series."""
    base_filter = pd.Series(True, index=df.index)
    if with_kst_filter:
        base_filter = df["kst_h"].isin(KST_HOT_SLOTS)

    rules = {
        "inflection_high": (
            (df["upper_wick_to_atr"] > 0.5) &
            (df["pre_5_disp_atr"] > 0.8) &
            (df["stoch_k"] > 65) &
            (df["bb_pctb"] > 70) &
            base_filter
        ).fillna(False),
        "inflection_low": (
            (df["lower_wick_to_atr"] > 0.5) &
            (df["pre_5_disp_atr"] < -0.8) &
            (df["stoch_k"] < 35) &
            (df["bb_pctb"] < 30) &
            base_filter
        ).fillna(False),
        "trend_end_high": (
            (df["stoch_d"] > 80) &
            (df["pre_5_disp_atr"] > 1.5) &
            (df["upper_wick_to_atr"] > 0.4) &
            (df["dist_to_ema20_atr"] > 1.0) &
            base_filter
        ).fillna(False),
        "trend_end_low": (
            (df["stoch_d"] < 20) &
            (df["pre_5_disp_atr"] < -1.5) &
            (df["lower_wick_to_atr"] > 0.4) &
            (df["dist_to_ema20_atr"] < -1.0) &
            base_filter
        ).fillna(False),
    }
    return rules


def hit_rate(rule_mask: pd.Series, event_idxs: set, tolerance: int = 2) -> dict:
    """룰 trigger 시점 ±tolerance 내 event idx 가 있나? hit rate."""
    trigger_idxs = np.where(rule_mask.values)[0]
    if len(trigger_idxs) == 0:
        return dict(n_triggers=0, n_hits=0, hit_rate=0.0)
    hits = 0
    for ti in trigger_idxs:
        for offset in range(-tolerance, tolerance + 1):
            if (ti + offset) in event_idxs:
                hits += 1
                break
    return dict(
        n_triggers=int(len(trigger_idxs)),
        n_hits=int(hits),
        hit_rate=float(hits / len(trigger_idxs)),
    )


def forward_ev(df: pd.DataFrame, rule_mask: pd.Series,
               n_bars_list=(5, 15, 30), direction: int = 1) -> dict:
    """
    룰 trigger 후 forward N봉 평균 disp + sharpe.
    direction: +1 = 룰이 long entry (forward up 기대), -1 = short entry.
    sharpe = mean(disp) / std(disp) × √252 (간이)
    """
    trigger_idxs = np.where(rule_mask.values)[0]
    n_total = len(df)
    out = {}
    for n in n_bars_list:
        disps = []
        for ti in trigger_idxs:
            if ti + n >= n_total:
                continue
            disp = df["close"].iloc[ti + n] - df["close"].iloc[ti]
            disps.append(direction * disp)  # direction 보정
        if not disps:
            out[f"w{n}"] = None
            continue
        arr = np.array(disps)
        out[f"w{n}"] = dict(
            n=int(len(arr)),
            mean_disp=float(arr.mean()),
            median_disp=float(np.median(arr)),
            std=float(arr.std()),
            sharpe=float(arr.mean() / (arr.std() + 1e-9)),  # daily sharpe (근사)
            win_rate=float((arr > 0).mean() * 100),
        )
    return out


def run_tf(tf: str, rvol_n: int, slope_w: int) -> dict:
    print(f"\n{'='*72}\n=== {tf}\n{'='*72}")
    t0 = time.time()
    df = pd.read_parquet(os.path.join(DATA, f"nq_{tf}.parquet"))
    df = add_indicator_columns(df, rvol_length=rvol_n)
    df = attach_regime(df, slope_window=slope_w)
    df = add_sr_and_shape(df)
    print(f"  full df: {len(df):,} rows ({time.time()-t0:.1f}s)")

    zz = add_forward_leg(zigzag_pivots(df, threshold_atr_mult=1.5))
    inf = label_inflection(zz, min_back_atr=2.0, min_forward_atr=2.0)
    tre = label_trend_leg_end(zz, min_atr=3.0, min_bars=5, max_bars=200)

    # event idx sets — kind 별 분리
    event_sets = {
        "inflection_high": set(inf[inf["kind"] == "high"]["idx"].tolist()),
        "inflection_low":  set(inf[inf["kind"] == "low"]["idx"].tolist()),
        "trend_end_high":  set(tre[tre["kind"] == "high"]["idx"].tolist()),
        "trend_end_low":   set(tre[tre["kind"] == "low"]["idx"].tolist()),
    }
    print(f"  event sets: " + ", ".join(f"{k}={len(v):,}" for k, v in event_sets.items()))

    # baseline hit rate (random bars vs event)
    baseline_n = 50000
    rng = np.random.default_rng(42)
    rand_idxs = rng.choice(len(df) - 30, baseline_n, replace=False)
    baselines = {}
    for ev_name, ev_set in event_sets.items():
        hits = 0
        for ti in rand_idxs:
            for offset in range(-2, 3):
                if (ti + offset) in ev_set:
                    hits += 1
                    break
        baselines[ev_name] = hits / baseline_n
    print(f"  baseline hit rate (random ±2):")
    for k, v in baselines.items():
        print(f"    {k}: {v*100:.2f}%")

    # 룰 평가 — KST filter ON / OFF 둘 다
    results = {}
    for kst_filter in [False, True]:
        suffix = "kst_filter" if kst_filter else "no_filter"
        rules = define_rules(df, with_kst_filter=kst_filter)
        for rule_name, mask in rules.items():
            event_set = event_sets[rule_name]
            hr = hit_rate(mask, event_set, tolerance=2)
            direction = +1 if "low" in rule_name else -1  # _low 는 long entry, _high 는 short
            ev = forward_ev(df, mask, direction=direction)
            base_hr = baselines[rule_name]
            lift = hr["hit_rate"] / base_hr if base_hr > 0 else None
            print(f"\n  [{rule_name}, {suffix}]")
            print(f"    triggers: {hr['n_triggers']:,} ({hr['n_triggers']/len(df)*100:.3f}% of bars)")
            print(f"    hits (event ±2 bars): {hr['n_hits']:,} → hit_rate {hr['hit_rate']*100:.1f}%")
            print(f"    baseline hit rate: {base_hr*100:.2f}%  →  LIFT {lift:.1f}x" if lift else "")
            print(f"    forward EV (direction={direction:+d}):")
            for w in (5, 15, 30):
                e = ev[f"w{w}"]
                if e:
                    print(f"      w{w}: mean {e['mean_disp']:+.2f} pt, "
                          f"sharpe {e['sharpe']:+.3f}, WR {e['win_rate']:.1f}% (N={e['n']:,})")
            results[f"{rule_name}__{suffix}"] = dict(
                hit_rate=hr,
                baseline_hit_rate=float(base_hr),
                lift=float(lift) if lift else None,
                forward_ev=ev,
            )
    return dict(results=results, baselines=baselines, total_bars=int(len(df)))


if __name__ == "__main__":
    res_5m = run_tf("5m", rvol_n=78,  slope_w=288)
    res_1m = run_tf("1m", rvol_n=390, slope_w=1440)

    bundle = {"5m": res_5m, "1m": res_1m}
    with open(os.path.join(OUT, "rule_backtest.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str, ensure_ascii=False)

    # verdict 표 (no_filter)
    print(f"\n{'='*72}\n=== Phase 3.4 verdict (KST filter OFF)\n{'='*72}")
    print(f"{'rule':<20}{'TF':>4}{'triggers':>10}{'%bars':>8}{'hit_rate':>10}{'lift':>8}{'EV w15':>10}{'sharpe':>9}")
    for tf, res in [("5m", res_5m), ("1m", res_1m)]:
        for rule_name in ("inflection_high", "inflection_low", "trend_end_high", "trend_end_low"):
            key = f"{rule_name}__no_filter"
            if key not in res["results"]:
                continue
            r = res["results"][key]
            tot = res["total_bars"]
            ev15 = r["forward_ev"].get("w15", {}) or {}
            mean_15 = ev15.get("mean_disp", 0) if ev15 else 0
            sh_15 = ev15.get("sharpe", 0) if ev15 else 0
            print(f"{rule_name:<20}{tf:>4}{r['hit_rate']['n_triggers']:>10,}"
                  f"{r['hit_rate']['n_triggers']/tot*100:>7.3f}%"
                  f"{r['hit_rate']['hit_rate']*100:>9.1f}%"
                  f"{r['lift']:>7.1f}x" if r['lift'] else f"{'-':>8}"
                  f"{mean_15:>+10.2f}{sh_15:>+9.3f}")

    # 판정
    print(f"\nPASS criterion: hit_rate ≥ 30% (lift ≥ 15x baseline)")
    pass_count = 0
    total = 0
    for tf, res in [("5m", res_5m), ("1m", res_1m)]:
        for rule_name in ("inflection_high", "inflection_low", "trend_end_high", "trend_end_low"):
            key = f"{rule_name}__no_filter"
            if key not in res["results"]:
                continue
            total += 1
            if res["results"][key]["hit_rate"]["hit_rate"] >= 0.30:
                pass_count += 1
    print(f"  PASS: {pass_count}/{total} cases")

    print(f"\nSaved: {OUT}")
