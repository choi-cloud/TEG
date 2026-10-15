# T23c — 일치 수 c의 추가 신호: 노드 쌍과 에피소드 안 support 쌍 (학습 없음)

2026-10-15 · 브랜치 `hier_purity` · H0 `14d9dfe` · 선행 `t23b`(`cb3fed1`) · 요약표 `reports/T23c_summary.md`

## 1. 변경 파일과 위치 (파일 단위 계획)

| 파일 | 내용 |
|---|---|
| `tools/t23.py` | T23c 함수: `agree_counts`, `fit_auc`, `make_episodes`, `episode_auc`, `c_run`, `c_gates`, `c_report`(T23b 커밋 시점에 이미 포함, 이 지시서에서 실행). 이 커밋에서 코드 변경 없음 |

- 재사용: `restricted_knn`(T23b, `tools/analyze_t19.py:41` `knn_sorted` 동점 규칙)으로 E top-50 이웃, `analyze_t19.py:80/112/118` `normed`·`split_of`·`group_ids`, T23b `labels_eff`·`shuf_of`(17300 + s). `sklearn.linear_model.LogisticRegression`(lbfgs, C = 1.0, max_iter = 1000, base 쌍 평균·표준편차로 표준화, `decision_function`으로 AUC), `sklearn.metrics.roc_auc_score`. 에피소드 AUC = support마다 나머지 24개 점수의 평균 순위(`scipy.stats.rankdata`, 동점 평균)로 낸 Mann–Whitney AUC.
- 에피소드: `np.random.default_rng(17100 + s)`(test) / `17200 + s`(valid)에서 후보 클래스(레이블 노드 ≥ 5) 5개 비복원 → 클래스마다 노드 5개 비복원, 200개. seed 0 에피소드 노드 id는 `results/T23/T23c/<cell>/episodes_seed0.npz`.

## 2. 실행 명령과 출력

| 명령 | 시간 | 출력 |
|---|---|---|
| `python tools/t23.py c-gates` (CPU) | 05:05:4x | `results/T23/gates_c.json` |
| `python tools/t23.py c-run <ds> <k>` × 6 (GPU 6개, UUID) | 05:09:1x–05:12:37 (3.4분; 프로세스 115–193 s) | `results/T23/T23c/<cell>/seed<s>.csv` 18개 |
| `python tools/t23.py c-report` | < 1분 | `reports/T23c_summary.md`, `results/T23/judgements_c.json` |

- 3절 계산 순서 ①–⑥ 전부 실행(시간으로 뺀 항목 없음).

### 게이트

| 게이트 | 결과 | 충족 |
|---|---|---|
| Ia 선행 | `t23b` 있음 | 예 |
| Ib 쌍 일치 | C-1 top-50을 K = 10으로 자른 test 같은-레이블 비율 − T23b u_10: 36개(셀 6 × seed 3 × E 2) 최대 \|Δ\| = 0 | 예 |
| Ic 일치 수 | 합성 50노드(무작위 아님) 2500쌍 c·s 불일치 0. H 실제 배정(AX, Amazon_clothing) 결정적 쌍 200000개에서 c = 가장 깊은 공유 수준, 불일치 0 | 예 |
| Id 에피소드 | 에피소드마다 클래스 5개가 모두 해당 그룹 클래스, 노드 25개 서로 다름, 클래스마다 5개: 전부 충족. 후보 클래스 수 test 20 / 30 / 40, valid 17 / 27 / 36 | 예 |
| Ie 유한 | AUC 전부 유한, 회귀 미수렴 0 | 예 |
| If 변경 범위 | 부기에 기록 | — |

### 판정과 사전 등록 (값만)

| 판정 | 값 |
|---|---|
| S1 ΔAUC_multi(TEGe, AX) 분명히 양수 | 6/6 |
| S2 ΔAUC_multi − ΔAUC_single (TEGe, AX) | 6/6 |
| S3 ΔAUC_multi(TEGe, SELF) | 6/6 |
| S4 ΔAUC_multi(TEGe, AX) − ΔAUC_multi(CONe, AX) | 6/6 |
| S5 ΔAUC_ep(TEGe, AX) | 6/6 |
| S6 (기록) ΔAUC_multi(TEGe, X / rand1 / SHUF) | 6/6 / 6/6 / 0/6; β* 분포 {0: 17, 0.05: 5, 0.1: 12, 0.2: 28, 0.5: 82} |

| 예측 | 조건 | 관측 | 일치 |
|---|---|---|---|
| P121 | S1 ≥ 4/6 | 6/6 | 일치 |
| P122 | S2 ≥ 4/6 | 6/6 | 일치 |
| P123 | S3 ≤ 1/6 | 6/6 | 어긋남 |
| P124 | S4 ≥ 4/6 | 6/6 | 일치 |
| P125 | S5 ≥ 4/6 | 6/6 | 일치 |

- C1 ΔAUC_multi(TEGe, AX): +0.0237, +0.0292, +0.0566, +0.0441, +0.0699, +0.0539 (clothing 1, 5 / dblp 1, 5 / electronics 1, 5). AUC_cos(TEGe, test 쌍) 0.519–0.626.
- C4 ΔAUC_ep(TEGe, AX): +0.0179, +0.0161, +0.0039, +0.0021, +0.0119, +0.0099.

## 3. 완료 조건 대비

| 조건 | 결과 | 충족 |
|---|---|---|
| run 18개 결과 | 18/18 | 예 |
| 표 C1–C5 | `reports/T23c_summary.md` | 예 |
| S1–S6, P121–P125 | 위 표 | 예 |
| 게이트 Ia–Ie | 충족 | 예 |

## 4. 이상 징후

1. β* = 0.5(격자 최대)가 144개 (E, V, F, 셀) 중 82개다. (TEGe, AX, H)는 dblp를 포함해 6셀 모두 β* = 0.5다(C4).
2. SHUF의 ΔAUC_multi·ΔAUC_single이 6셀 모두 \|값\| < 0.00005다(C1).
3. 지시서 §2 C-2는 1-shot 셀에도 같은 5-shot 에피소드를 쓴다. 에피소드·AX 배정이 shot과 무관해 AUC(S1)(V = AX 등 학습 없는 뷰)가 1-shot·5-shot 셀에서 같다.
4. Ic(b)의 쌍 목록은 난수 없이 `(i·7919) mod N`, `(i·104729 + 13) mod N`(i < 200000)으로 만들었다(지시서에 표본 방식 없음; 처음 작성한 `default_rng(17300)` 표본은 SHUF 오프셋과 겹쳐 실행 전에 바꿨다).
5. C3(그림용 표)의 빈 칸(쌍 0개)은 "—"로 적었다.
