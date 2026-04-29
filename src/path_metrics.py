# -*- coding: utf-8 -*-
"""
path_metrics — pre/forward window metrics + classifier

설계 원칙:
  - Pre-metrics  : Pine 호환 가능 (OHLC + ATR 만)
  - Fwd-metrics  : Python backtest only — Pine 실시간 사용 불가
  - Classifier   : 2단계
       label_full : V/W/Λ/N/drift/flat (retrospective, analysis only)
       binary    : is_v_form / is_drift  (pre+context만, Pine 옮길 수 있음 — 결과 룰만)

규칙 (plan A~C 반영):
  N = 5 (1m → 5min, 5m → 25min)
  pre window  = bars [t0-N+1 ... t0]   (현재 봉 포함, 직전 5봉)
  fwd window  = bars [t0+1 ... t0+N]   (직후 5봉)

Intensity 메트릭 4종:
  net_disp     = close_end - close_start
  path_eff     = |net_disp| / sum(|HL|)        ∈ [0,1]
  velocity     = net_disp / N                  (pt per bar)
  acceleration = fwd_velocity - pre_velocity   (Δvelocity)

Shape classifier (6 cat) — fwd window 만:
  V    : pre_net > 0 (up)  AND fwd_net < -|pre_net|*0.7  (sharp reversal down)
         pre_net < 0 (dn)  AND fwd_net > +|pre_net|*0.7  (sharp reversal up)
  W    : fwd 의 반대 reversal AND fwd path 안에 1개 이상 turn (low→high→low or high→low→high)
  Λ    : fwd same direction, but path_eff < 0.5 AND end retrace > 30%
  N    : fwd opposite sign at end vs start, but path_eff < 0.5
  drift: fwd same direction AND path_eff >= 0.7
  flat : |fwd_net| < 0.5 * atr14_at_t0
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
# Window slicing helpers
# ─────────────────────────────────────────────────────────────────────────

def _slice_window(df: pd.DataFrame, idx: int, start_offset: int, end_offset: int):
    """df rows [idx+start_offset .. idx+end_offset] (inclusive). Bound check."""
    lo = idx + start_offset
    hi = idx + end_offset
    if lo < 0 or hi >= len(df):
        return None
    return df.iloc[lo : hi + 1]


def _path_metrics_block(o, h, l, c) -> dict:
    """단일 window 의 metric 계산 (numpy arrays)."""
    if len(c) == 0:
        return dict(net_disp=0.0, path_eff=0.0, velocity=0.0,
                    n_turns=0, max_excursion=0.0, end_retrace_pct=0.0)
    n = len(c)
    net_disp = float(c[-1] - c[0])
    # path length: bar 별 |HL|
    hl = h - l
    path_len = float(hl.sum())
    path_eff = abs(net_disp) / path_len if path_len > 0 else 0.0
    velocity = net_disp / n if n > 0 else 0.0

    # turns: close diff sign changes
    diffs = np.diff(c)
    if len(diffs) > 1:
        signs = np.sign(diffs)
        signs = signs[signs != 0]  # ignore flat
        n_turns = int(((signs[1:] != signs[:-1]).sum()))
    else:
        n_turns = 0

    # excursion: |max(close) - close[0]| or |close[0] - min(close)| 중 큰
    max_up = float(c.max() - c[0])
    max_dn = float(c[0] - c.min())
    max_excursion = max(max_up, max_dn)

    # end retrace: 추세 끝부분의 후퇴 비율
    if net_disp > 0:
        peak = c.max()
        end_retrace = (peak - c[-1]) / (peak - c[0]) if peak > c[0] else 0.0
    elif net_disp < 0:
        trough = c.min()
        end_retrace = (c[-1] - trough) / (c[0] - trough) if c[0] > trough else 0.0
    else:
        end_retrace = 0.0
    end_retrace_pct = float(np.clip(end_retrace, 0, 1) * 100)

    return dict(
        net_disp=net_disp,
        path_eff=path_eff,
        velocity=velocity,
        n_turns=n_turns,
        max_excursion=max_excursion,
        end_retrace_pct=end_retrace_pct,
    )


# ─────────────────────────────────────────────────────────────────────────
# 메인 — pivot DataFrame + bar DataFrame → metric 컬럼 추가
# ─────────────────────────────────────────────────────────────────────────

def attach_path_metrics(pivots: pd.DataFrame, df_bars: pd.DataFrame,
                        *, n_bars: int = 5,
                        atr_col: str = "atr14") -> pd.DataFrame:
    """
    pivots : pivots_*.find_*() 결과 (idx 컬럼 보유, 옵션 origin_idx)
    df_bars: indicators.add_indicator_columns() 결과 (atr14 포함)
    n_bars : pre/fwd 각각 봉수 (default 5)

    반환: pivots DataFrame 에 아래 컬럼 추가 ─
      pre_net_disp, pre_path_eff, pre_velocity, pre_max_excursion,
      fwd_net_disp, fwd_path_eff, fwd_velocity, fwd_n_turns,
      fwd_max_excursion, fwd_end_retrace_pct,
      atr_at_t0, acceleration
    """
    if pivots.empty:
        return pivots.copy()

    n = len(pivots)
    o_arr = df_bars["open"].to_numpy()
    h_arr = df_bars["high"].to_numpy()
    l_arr = df_bars["low"].to_numpy()
    c_arr = df_bars["close"].to_numpy()
    atr_arr = df_bars[atr_col].to_numpy()
    L = len(df_bars)

    # 결과 배열
    cols = {k: np.full(n, np.nan, dtype=np.float64) for k in [
        "pre_net_disp", "pre_path_eff", "pre_velocity", "pre_max_excursion",
        "fwd_net_disp", "fwd_path_eff", "fwd_velocity", "fwd_max_excursion",
        "fwd_end_retrace_pct", "atr_at_t0", "acceleration",
    ]}
    fwd_n_turns = np.full(n, -1, dtype=np.int32)

    pivot_idx_arr = pivots["idx"].to_numpy()

    for i, t0 in enumerate(pivot_idx_arr):
        # boundary check
        if t0 - (n_bars - 1) < 0 or t0 + n_bars >= L:
            continue

        pre_lo, pre_hi = t0 - n_bars + 1, t0  # inclusive
        fwd_lo, fwd_hi = t0 + 1, t0 + n_bars

        pre = _path_metrics_block(
            o_arr[pre_lo:pre_hi+1], h_arr[pre_lo:pre_hi+1],
            l_arr[pre_lo:pre_hi+1], c_arr[pre_lo:pre_hi+1])
        fwd = _path_metrics_block(
            o_arr[fwd_lo:fwd_hi+1], h_arr[fwd_lo:fwd_hi+1],
            l_arr[fwd_lo:fwd_hi+1], c_arr[fwd_lo:fwd_hi+1])

        cols["pre_net_disp"][i]      = pre["net_disp"]
        cols["pre_path_eff"][i]      = pre["path_eff"]
        cols["pre_velocity"][i]      = pre["velocity"]
        cols["pre_max_excursion"][i] = pre["max_excursion"]

        cols["fwd_net_disp"][i]      = fwd["net_disp"]
        cols["fwd_path_eff"][i]      = fwd["path_eff"]
        cols["fwd_velocity"][i]      = fwd["velocity"]
        cols["fwd_max_excursion"][i] = fwd["max_excursion"]
        cols["fwd_end_retrace_pct"][i] = fwd["end_retrace_pct"]
        fwd_n_turns[i] = fwd["n_turns"]

        cols["atr_at_t0"][i]    = atr_arr[t0]
        cols["acceleration"][i] = fwd["velocity"] - pre["velocity"]

    out = pivots.copy()
    for k, v in cols.items():
        out[k] = v
    out["fwd_n_turns"] = fwd_n_turns
    return out


# ─────────────────────────────────────────────────────────────────────────
# Classifier v1 (6-class label) — 보존 (legacy)
# ─────────────────────────────────────────────────────────────────────────

V_RATIO = 0.7        # V/W reversal magnitude threshold
DRIFT_EFF = 0.7      # drift path_eff threshold
LAMBDA_RETRACE = 30  # Λ end retrace % threshold
FLAT_ATR_MULT = 0.5  # flat = |fwd_net| < 0.5×ATR

# ─────────────────────────────────────────────────────────────────────────
# Classifier v2 (window-adaptive, relaxed) — 2026-04-29 사용자 요청
#   - pre/fwd magnitude 비교 폐기 → "확실한 변곡" 만 보면 OK
#   - √N scaling 으로 window 별 threshold 자동 조정
#   - drift_eff 0.7 → 0.5 완화
# ─────────────────────────────────────────────────────────────────────────

DRIFT_EFF_V2 = 0.30        # v3: 0.5 → 0.3 (window 길어지면 path_eff 자연 떨어져서 완화)
V_TURN_RATE_MAX = 0.40     # v3 V vs W: n_turns/(n_bars-1) < 0.4 = V (단방향), 그 외 W

def boundaries_v2(n_bars: int, *,
                  flat_base: float = 0.4,
                  v_drift_base: float = 0.8) -> dict:
    """√N scaling — n=5 base 기준. v3 (2026-04-29 발견): drift_eff 0.5→0.3, V vs W = turn rate."""
    scale = (n_bars / 5.0) ** 0.5
    return dict(
        flat_max=flat_base * scale,
        v_drift_min=v_drift_base * scale,
        drift_eff=DRIFT_EFF_V2,
        v_turn_rate_max=V_TURN_RATE_MAX,
    )


def classify_path(row) -> str:
    """row: pivot DataFrame row with path metrics (attach_path_metrics 결과)."""
    pre = row.get("pre_net_disp", np.nan)
    fwd = row.get("fwd_net_disp", np.nan)
    fwd_eff = row.get("fwd_path_eff", np.nan)
    fwd_turns = row.get("fwd_n_turns", -1)
    fwd_retrace = row.get("fwd_end_retrace_pct", np.nan)
    atr = row.get("atr_at_t0", np.nan)

    if any(pd.isna(x) for x in [pre, fwd, fwd_eff, atr]):
        return "unknown"

    # flat 우선
    if abs(fwd) < FLAT_ATR_MULT * atr:
        return "flat"

    # reversal threshold (V/W) — fwd 가 pre 와 반대 + 충분히 큼
    pre_abs = abs(pre)
    is_reversal_strong = (
        pre_abs > 0 and
        np.sign(fwd) != np.sign(pre) and
        abs(fwd) >= V_RATIO * pre_abs
    )

    if is_reversal_strong:
        # W = fwd 의 turn 1+ (중간 reversal 있음)
        if fwd_turns >= 1:
            return "W"
        else:
            return "V"

    # 같은 방향
    if np.sign(fwd) == np.sign(pre) and pre_abs > 0:
        if fwd_eff >= DRIFT_EFF:
            return "drift"
        if fwd_retrace >= LAMBDA_RETRACE:
            return "Λ"

    # opposite sign (마지막 봉만 반대) but not strong reversal
    if pre_abs > 0 and np.sign(fwd) != np.sign(pre):
        if fwd_eff < 0.5:
            return "N"

    # default
    return "Λ" if pre_abs > 0 and np.sign(fwd) == np.sign(pre) else "flat"


def attach_label(pivots_with_metrics: pd.DataFrame) -> pd.DataFrame:
    out = pivots_with_metrics.copy()
    out["path_label"] = out.apply(classify_path, axis=1)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Multi-window (v2) — pivot 당 여러 N 값 path metric + label 부여
# ─────────────────────────────────────────────────────────────────────────

def attach_path_metrics_multi(pivots: pd.DataFrame, df_bars: pd.DataFrame,
                              *, n_bars_list: list,
                              atr_col: str = "atr14") -> pd.DataFrame:
    """
    여러 window 동시 측정. 결과 컬럼 prefix = "w{N}_".
      e.g., w5_pre_net_disp, w15_fwd_path_eff, w30_fwd_n_turns 등.
    atr_at_t0 는 한 번만 부여.
    """
    out = pivots.copy()
    first = True
    for n in n_bars_list:
        single = attach_path_metrics(pivots, df_bars, n_bars=n, atr_col=atr_col)
        prefix = f"w{n}_"
        cols = ["pre_net_disp", "pre_path_eff", "pre_velocity", "pre_max_excursion",
                "fwd_net_disp", "fwd_path_eff", "fwd_velocity", "fwd_max_excursion",
                "fwd_end_retrace_pct", "fwd_n_turns", "acceleration"]
        for c in cols:
            if c in single.columns:
                out[prefix + c] = single[c].to_numpy()
        if first:
            out["atr_at_t0"] = single["atr_at_t0"].to_numpy()
            first = False
    return out


def classify_path_v2_row(row, n_bars: int) -> str:
    """row 의 w{n_bars}_* 컬럼으로 v2 분류."""
    pre = row.get(f"w{n_bars}_pre_net_disp", np.nan)
    fwd = row.get(f"w{n_bars}_fwd_net_disp", np.nan)
    fwd_eff = row.get(f"w{n_bars}_fwd_path_eff", np.nan)
    fwd_turns = row.get(f"w{n_bars}_fwd_n_turns", -1)
    atr = row.get("atr_at_t0", np.nan)

    if any(pd.isna(x) for x in [pre, fwd, fwd_eff, atr]):
        return "unknown"

    b = boundaries_v2(n_bars)
    flat_max    = b["flat_max"]    * atr
    v_drift_min = b["v_drift_min"] * atr
    drift_eff   = b["drift_eff"]

    if abs(fwd) < flat_max:
        return "flat"

    if pre == 0:
        return "drift" if fwd_eff >= drift_eff and abs(fwd) >= v_drift_min else "Λ"

    same_dir = np.sign(fwd) == np.sign(pre)
    if same_dir:
        if fwd_eff >= drift_eff and abs(fwd) >= v_drift_min:
            return "drift"
        return "Λ"
    # opposite
    if abs(fwd) >= v_drift_min:
        return "V" if fwd_turns < 2 else "W"
    return "N"


def classify_v2_vectorized(df_multi: pd.DataFrame, n_bars: int) -> pd.Series:
    """vectorized v2 classifier (apply 보다 100배 빠름)."""
    pre   = df_multi[f"w{n_bars}_pre_net_disp"]
    fwd   = df_multi[f"w{n_bars}_fwd_net_disp"]
    fwd_e = df_multi[f"w{n_bars}_fwd_path_eff"]
    fwd_t = df_multi[f"w{n_bars}_fwd_n_turns"]
    atr   = df_multi["atr_at_t0"]

    b = boundaries_v2(n_bars)
    flat_max    = b["flat_max"]    * atr
    v_drift_min = b["v_drift_min"] * atr
    drift_eff   = b["drift_eff"]

    labels = pd.Series("unknown", index=df_multi.index, dtype=object)
    valid = pre.notna() & fwd.notna() & fwd_e.notna() & atr.notna()

    fwd_abs = fwd.abs()
    mask_flat = valid & (fwd_abs < flat_max)
    labels[mask_flat] = "flat"

    rest = valid & ~mask_flat
    same_dir = (np.sign(fwd) == np.sign(pre)) & (pre != 0)
    opp_dir  = (np.sign(fwd) != np.sign(pre)) & (pre != 0)
    pre_zero = (pre == 0) & rest

    drift_cond = (fwd_e >= drift_eff) & (fwd_abs >= v_drift_min)

    labels[rest & same_dir &  drift_cond] = "drift"
    labels[rest & same_dir & ~drift_cond] = "Λ"
    labels[pre_zero &  drift_cond] = "drift"
    labels[pre_zero & ~drift_cond] = "Λ"
    # v3 V vs W: turn_rate = n_turns / (n_bars-1) < 0.4 → V (단방향), ≥ 0.4 → W
    turn_rate_max = b["v_turn_rate_max"]
    max_diffs = max(n_bars - 1, 1)
    turn_rate = fwd_t / max_diffs
    labels[rest & opp_dir & (fwd_abs >= v_drift_min) & (turn_rate <  turn_rate_max)] = "V"
    labels[rest & opp_dir & (fwd_abs >= v_drift_min) & (turn_rate >= turn_rate_max)] = "W"
    labels[rest & opp_dir & (fwd_abs <  v_drift_min)] = "N"
    return labels


def attach_labels_v2(pivots_multi: pd.DataFrame, n_bars_list: list) -> pd.DataFrame:
    """각 window 별 label 컬럼 추가 (label_w{N})."""
    out = pivots_multi.copy()
    for n in n_bars_list:
        out[f"label_w{n}"] = classify_v2_vectorized(out, n)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Binary signals (Pine 호환 가능 — pre + at-pivot context 만)
# ─────────────────────────────────────────────────────────────────────────

def attach_binary_signals(pivots_with_metrics: pd.DataFrame,
                          *, atr_intensity_min: float = 1.0) -> pd.DataFrame:
    """
    Pine 에서 t0 시점에 즉시 계산 가능한 boolean signals.
    실시간 사용 가능.

      is_strong_pre  : |pre_net_disp| >= atr_intensity_min × atr14
                       → "직전 5봉 충분히 강한 추세 또는 hammered drop"
      pre_direction  : +1 (up) / -1 (down) / 0 (flat)
    """
    out = pivots_with_metrics.copy()
    pre = out["pre_net_disp"]
    atr = out["atr_at_t0"]
    out["is_strong_pre"] = (pre.abs() >= atr_intensity_min * atr).fillna(False).astype(bool)
    out["pre_direction"] = np.sign(pre).fillna(0).astype(np.int8)
    return out


if __name__ == "__main__":
    import os, time
    from indicators import add_indicator_columns
    from pivots_fractal import find_fractals_fast
    from pivots_legacy import find_legacy_5m, find_legacy_1m
    from pivots_ob import find_ob_origins

    DATA = r"C:\Users\minb0\Desktop\Main folder\투자공부\공부\해외선물_나스닥_NQ\research\v4_path\data"

    # 5m smoke
    print("=== 5m path metrics smoke ===")
    df5 = pd.read_parquet(os.path.join(DATA, "nq_5m.parquet"))
    df5 = add_indicator_columns(df5, rvol_length=78)
    print(f"  bars: {len(df5):,}")

    # F (fractal) 만 — 가장 단순
    t0 = time.time()
    piv_f = find_fractals_fast(df5, n=5)
    print(f"  F pivots: {len(piv_f):,}  ({time.time()-t0:.1f}s)")
    t1 = time.time()
    piv_f = attach_path_metrics(piv_f, df5, n_bars=5)
    piv_f = attach_label(piv_f)
    piv_f = attach_binary_signals(piv_f)
    print(f"  metrics + label + binary  ({time.time()-t1:.1f}s)")

    valid = piv_f.dropna(subset=["fwd_net_disp"])
    print(f"  valid (window inside data): {len(valid):,} / {len(piv_f):,}")
    print("\n  path_label distribution (5m fractal):")
    print(valid["path_label"].value_counts().to_string())
    print("\n  is_strong_pre rate:", f"{valid['is_strong_pre'].mean()*100:.1f}%")
    print("\n  cross: path_label x is_strong_pre")
    ct = pd.crosstab(valid["path_label"], valid["is_strong_pre"], normalize="columns") * 100
    print(ct.round(1).to_string())

    # OB origin (5m Bull) — 가장 narrow
    print("\n=== 5m OB origin path metrics ===")
    piv_o = find_ob_origins(df5)
    piv_o = attach_path_metrics(piv_o, df5, n_bars=5)
    piv_o = attach_label(piv_o)
    valid_o = piv_o.dropna(subset=["fwd_net_disp"])
    print(f"  pivots: {len(piv_o):,}, valid: {len(valid_o):,}")
    print("  path_label by kind:")
    print(pd.crosstab(valid_o["kind"], valid_o["path_label"],
                       normalize="index").round(3).to_string())
