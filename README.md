# NQ 변곡점 / 강추세 Fingerprint 연구 (v4_path)

> **NASDAQ-100 E-mini Futures (NQ) 17.4년 1m+5m 데이터 기반 변곡점·강추세 종료 식별 + Pine 호환 트레이딩 룰 추출.**

📖 **[교재 (HTML)](https://ggobugithesquirtle.github.io/mnq-v4-path-research/)** · 📄 [Research Report (Markdown)](RESEARCH_REPORT.md)

---

## TL;DR

- **데이터**: FirstRateData NQ 1m (5,809,378 bars) + 5m (1,217,620 bars), 2008-12 ~ 2026-04 (17.4년)
- **방법**: ZigZag 기반 변곡점·추세종료 enumeration → 22 features fingerprint → Cohen's d / GBM AUC / cell-conditional probability
- **결과**:
  - **변곡점·추세종료 식별 AUC 0.875~0.954** (Pre-only features, OOS 12개월)
  - **Top features = bar wick + pre 5봉 disp + Stoch D** (단 2~3개로 65~70% importance)
  - **단순 룰 trigger forward EV ≈ 0** — entry 무수익
  - **Cell-conditional (KST × ATR × pre_dir) 결합 시 WR 75~89%, sharpe +0.3~0.6** ← 진짜 edge
  - Cluster sub-type 분리 안 됨 → 단일 GBM 으로 충분
  - OB Tier A "Sharpe 4.69" 의 진짜 edge = ATM exit 결합 효과

---

## Repository Structure

```
mnq-v4-path-research/
├── README.md                    ← 이 파일
├── RESEARCH_REPORT.md           ← 상세 보고서 (한국어, ~ 400줄)
├── LICENSE                      ← MIT
├── docs/
│   ├── index.html               ← 교재 (단일 SPA, GitHub Pages root)
│   └── .nojekyll                ← Jekyll 충돌 방지
├── src/                         ← Python 분석 코드
│   ├── data_loader.py           ── CSV → parquet
│   ├── indicators.py            ── EMA/ATR/Stoch/BB/RSI/MFI vectorized
│   ├── pivots_fractal.py        ── F (raw fractal)
│   ├── pivots_legacy.py         ── L (Pine 신호)
│   ├── pivots_ob.py             ── O (OB origin BOS)
│   ├── path_metrics.py          ── multi-window classifier
│   ├── regime.py                ── BULL/BEAR/MIXED
│   ├── events.py                ── ZigZag + inflection + trend_end
│   ├── profiler.py              ── SR + bar shape + indicator snapshot
│   ├── compare.py               ── random baseline + Cohen's d
│   └── run_phase_3_*.py         ── Phase 3.1 ~ 3.6 실행 스크립트
└── results/
    ├── results_3_1/             ── Cross-TF fingerprint consistency
    ├── results_3_2/             ── Pre-only AUC
    ├── results_3_3/             ── Cluster
    ├── results_3_4/             ── Rule hit rate
    ├── results_3_5/             ── OB Tier A H5
    └── results_3_6/             ── Conditional probability cells
```

---

## Reproduce

### 환경

- Python 3.12
- pandas 2.3.3 / numpy 2.4 / pyarrow 20.0 / scikit-learn 1.6
- (Windows 권장: `PYTHONUTF8=1`)

```bash
pip install pandas numpy pyarrow scikit-learn
```

### 데이터

NQ 1m + 5m FirstRateData CSV 가 필요합니다 (재배포 불가). `data_loader.py` 의 `BASE` 경로를 본인 CSV 위치로 수정 후:

```bash
cd src
python data_loader.py            # CSV → parquet (1회)
python run_phase_3_1.py          # Cross-TF consistency (~1 min)
python run_phase_3_2.py          # Pre-only AUC (~5 min)
python run_phase_3_3.py          # Cluster (~30 sec)
python run_phase_3_4.py          # Rule hit rate (~3 min)
python run_phase_3_5_6.py        # Tier A + Conditional probability (~3 min)
```

각 단계 결과는 `results/results_3_*/` 폴더에 JSON/CSV 로 저장됩니다.

---

## Pine 룰 (v5 후보)

전체 코드는 [RESEARCH_REPORT.md §4](RESEARCH_REPORT.md#4-pine-룰-추출-v5-후보) 또는 [교재 §4](https://ggobugithesquirtle.github.io/mnq-v4-path-research/#pine) 참고.

### 핵심 detector

```pine
inflection_high =
    upper_wick > 0.5 * atr14 and
    (close - close[5]) > 0.8 * atr14 and
    stoch_k > 65 and bb_pctb > 70

trend_end_high =
    stoch_d > 80 and
    (close - close[5]) > 1.5 * atr14 and
    upper_wick > 0.4 * atr14 and
    (close - ema20) > 1.0 * atr14
```

### Cell-conditional GO filter (실제 entry 신호)

```pine
go_long_high_wr =
    kind == "low" and slot == "Other" and
    atr_bin == "low" and pre_dir == "down"
    // → V+drift 40.9%, WR 77.8%, sharpe +0.426 (N=1761, 1m)
```

---

## Hypotheses Verdict

| 가설 | 기준 | 결과 | 판정 |
|---|---|---|---|
| H1 Inflection AUC | ≥ 0.80 | min 0.875 | ✅ PASS |
| H2 Trend End AUC | ≥ 0.85 | min 0.934 | ✅ PASS |
| H3 Pre-only AUC | ≥ 0.70 | min 0.875 | ✅ PASS |
| H4 Cluster sub-type | silhouette ≥ 0.30 | max 0.191 | ❌ NOGO (좋은 의미) |
| H5 OB Tier A | A > B sharpe | Bear 양 TF, Bull 1m | 🟡 PARTIAL |

---

## License

MIT — see [LICENSE](LICENSE).

코드 + 콘텐츠 모두 MIT. 인용/재사용 자유, 변경 가능, 보증 없음.

---

## Caveats

- **Forward path 본질적 noise** — 변곡 식별 ≠ EV 보장
- **regime 정의 한계** — EMA200 24h slope 가 too long-term, 모든 이벤트 무차별
- **시간 가중 미적용** — 5y half-life decay 아님, flat
- **Tick data 미사용** — 1m 내부 SL/TP 모호성 미측정
- **Pine 룰 backtest 미실행** — ATM/exit 결합 backtest 별도 단계

---

*v4_path 연구 — 2026-04-29 완료. NQ 17.4년 / 5,809,378 1m bars.*
