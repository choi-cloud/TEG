# T28 보고 — 순수 X 코사인 프로토타입 (학습 없음)

지시서 `instructions/T28_T29_QUEUE.md` (T28 단위) · 브랜치 `trainfree_baselines` · H0 `2967476` · 표 전체는 `reports/T28_summary.md`

## 1. 변경 파일

| 파일 | 내용 |
|---|---|
| `tools/analyze_t28.py` (새 파일) | `load_eps`, `run(ds)`, `t22_table2`, `report()`. 재사용: `fusion_baseline.diffusion_features_hops`, `diffusion_features`, `cosine_probe_acc`; `analyze_t22.fixed_teg`, `mean_sd`, `paired`, `run_dir`; `analyze_t06b.episodes`, `fewshot_acc` |
| `reports/T28_xproto.md`, `reports/T28_summary.md` | 보고서 |

- 학습 코드와 기존 `tools/*.py`는 고치지 않았다.
- 새 난수는 없다. probe 에피소드는 기존 1000 + seed 오프셋을 쓴다.
- GPU는 쓰지 않았다.

## 2. 실행 명령과 출력

```
python tools/analyze_t28.py run Amazon_clothing      # 3개 데이터셋 동시 실행, CPU
python tools/analyze_t28.py run dblp
python tools/analyze_t28.py run Amazon_electronics
python tools/analyze_t28.py report
```

- 실행 시간은 58 s(Clothing), 62 s(dblp), 63 s(Electronics)이다. 로그 생성 시각부터 JSON 기록 시각까지로 쟀다.
- `results/T28/` 디스크 사용량은 368K이다.

T1 — 고정 test 정확도 (seed 0–9, 각 200 에피소드, seed 평균 ± sd)

| 데이터셋 | shot | X0 | X1 | D | TEG | C_all |
|---|---|---|---|---|---|---|
| Amazon_clothing | 1 | 71.93 ± 0.72 | 82.19 ± 0.61 | 85.22 ± 0.49 | 81.78 ± 1.34 | 87.79 ± 0.48 |
| Amazon_clothing | 5 | 87.63 ± 0.48 | 92.38 ± 0.35 | 93.13 ± 0.50 | 90.37 ± 0.99 | 92.78 ± 0.46 |
| dblp | 1 | 48.35 ± 1.27 | 68.38 ± 1.05 | 72.43 ± 1.23 | 74.44 ± 1.20 | 78.19 ± 1.04 |
| dblp | 5 | 68.29 ± 0.53 | 83.74 ± 0.52 | 85.60 ± 0.75 | 84.23 ± 1.02 | 86.24 ± 0.58 |
| Amazon_electronics | 1 | 73.38 ± 0.70 | 73.52 ± 0.79 | 75.30 ± 0.88 | 73.59 ± 1.20 | 81.82 ± 1.81 |
| Amazon_electronics | 5 | 87.76 ± 0.73 | 87.66 ± 0.80 | 88.50 ± 0.74 | 86.92 ± 1.17 | 90.84 ± 0.81 |

T2 — 짝지은 차 (%p). 값은 에피소드 2,000개의 평균 ± SE이고, 괄호 안은 (양수 비율, seed 부호 일치/10)이다.

| 데이터셋 | shot | X0 − D | X0 − TEG | X0 − C_all | X1 − D |
|---|---|---|---|---|---|
| Amazon_clothing | 1 | -13.29 ± 0.24 (0.07, 10/10) | -9.85 ± 0.28 (0.17, 10/10) | -15.87 ± 0.28 (0.07, 10/10) | -3.04 ± 0.15 (0.18, 10/10) |
| Amazon_clothing | 5 | -5.50 ± 0.16 (0.11, 10/10) | -2.75 ± 0.18 (0.26, 10/10) | -5.15 ± 0.18 (0.15, 10/10) | -0.74 ± 0.09 (0.17, 10/10) |
| dblp | 1 | -24.08 ± 0.30 (0.02, 10/10) | -26.09 ± 0.33 (0.03, 10/10) | -29.84 ± 0.32 (0.01, 10/10) | -4.05 ± 0.17 (0.20, 10/10) |
| dblp | 5 | -17.32 ± 0.23 (0.03, 10/10) | -15.94 ± 0.25 (0.05, 10/10) | -17.95 ± 0.24 (0.02, 10/10) | -1.87 ± 0.11 (0.19, 10/10) |
| Amazon_electronics | 1 | -1.92 ± 0.25 (0.34, 10/10) | -0.21 ± 0.30 (0.42, 5/10) | -8.44 ± 0.28 (0.19, 10/10) | -1.78 ± 0.18 (0.30, 10/10) |
| Amazon_electronics | 5 | -0.74 ± 0.16 (0.33, 9/10) | +0.84 ± 0.19 (0.43, 8/10) | -3.08 ± 0.18 (0.24, 10/10) | -0.85 ± 0.11 (0.26, 10/10) |

T3(기록, 표현 수준 probe)는 `reports/T28_summary.md`에 있다.

판정값 (계산값만)

| 판정 | 정의 | 값 |
|---|---|---|
| M1 | acc(X0) > acc(D) | 0/6 |
| M2 | X0 − D 분명한 양수 / 분명한 음수 | 0/6 / 6/6 |
| M3 | acc(X0) > acc(TEG) | 1/6 (Amazon_electronics 5) |
| M4 (기록) | acc(X0) > acc(C_all); acc(X1) > acc(D); T3 acc_test(X0) > acc_test(S3) | 0/6; 0/6; 2/6 (Amazon_electronics 1, 5) |

사전 등록 계산값

| 예측 | 계산값 |
|---|---|
| P144 | M1 = 0/6 |
| P145 | 분명한 음수인 dblp 칸: 1-shot −24.08 ± 0.30, 5-shot −17.32 ± 0.23 (2/2) |
| P146 | M3 = 1/6 (Amazon_electronics 5) |

## 3. 완료 조건

| 조건 | 결과 | 충족 |
|---|---|---|
| Ga 선행 | `base` npz 60/60, 선택 λ `infonce_all` npz 60/60, T21 `analysis_<ds>.json` 3/3 | 예 |
| Gb 재현 (D·TEG·C_all 18값, \|Δ\| ≤ 0.005 %p) | 최대 \|Δ\| 0.0040 (T22 표 T2는 소수 둘째 자리) | 예 |
| Gc probe 동일 (S0 54값, \|Δ\| ≤ 1e-6) | 최대 \|Δ\| 0 | 예 |
| Gd 변경 범위 | `git diff --stat 2967476 <보고서 커밋>` = `tools/analyze_t28.py`, `reports/T28_summary.md`, `reports/T28_xproto.md` | 예 (부기에 확인값 기록) |
| Ge 유한 | 특징·지표 모두 유한 | 예 |
| 표 T1–T3, 판정 M1–M4, P144–P146 계산값 | `reports/T28_summary.md` | 예 |

## 4. 이상 징후

1. **Gb 비교 방식:** T22 표 T2의 값이 소수 둘째 자리까지라서 비교 대상은 반올림된 값이다. T22의 D는 `evaluate_npz`의 `acc_D`(s·cos의 log_softmax argmax)이고, T28의 D는 `cosine_probe_acc`(cos argmax)이다. 18값 모두 \|Δ\| ≤ 0.004로 기준 안이다.
2. **T3 probe 분할 (승인된 판단 1):** T21과 같이 T09 `base`의 `split.json`·`labels.npy`(seed 0–2)를 썼다.
3. **에피소드 순서:** 모든 고정 test 에피소드에서 `query_y`가 클래스 순(0,…,0,1,…,4)임을 확인했다. `cosine_probe_acc`는 이 순서를 전제로 한다.

## 부기 (2026-10-21, push)

- 보고서 커밋 `4100884`, 태그 `t28`(→ `4100884`), `origin/trainfree_baselines`와 태그 push 완료. force 없음.
- Gd 확인값: `git diff --stat 2967476 4100884` = `reports/T28_summary.md`(102+), `reports/T28_xproto.md`(84+), `tools/analyze_t28.py`(229+), 3 files.
- 이 부기는 T29 He(`git diff --stat t28 <T29 보고서 커밋>`)의 범위 밖에 두기 위해 T29 태그 뒤에 커밋했다.
