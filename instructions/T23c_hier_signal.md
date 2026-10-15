# T23c — 일치 수 c의 추가 신호: 노드 쌍과 에피소드 안 support 쌍 (학습 없음)

2026-10-15 · 작성: 채팅 Claude · 모드: auto mode(`T23_QUEUE.md`) · 브랜치 `hier_purity` · 태그 `t23c` · 선행: `t23b`

## 0. 목적과 검증 가능한 주장

T23b는 "클러스터로 이웃을 거르면 순도가 오르는가"를 봤다. 이 지시서는 같은 질문을 **연속 점수**로 본다.

- **C-1 노드 쌍.** 임베딩 코사인이 같은 두 이웃 쌍 중, 일치 수 c(같은 클러스터에 속하는 해상도 수, 0–5)가 큰 쌍이 같은 클래스일 확률이 더 높은가. 임베딩 거리만 쓴 판별력(AUC)에 c를 더했을 때의 증가 ΔAUC로 잰다.
- **C-2 에피소드 안 support 쌍.** 실제 test 클래스 5-way 5-shot 에피소드에서, support 하나가 나머지 24개 중 같은 클래스 4개를 얼마나 잘 골라내는가. 점수 = 코사인 / 코사인 + β·c. task 안 attention이 쓸 수 있는 정보인지를 직접 본다.

주장은 4절 표, 5절 판정값, 6절 사전 등록 대조다. 해석은 하지 않는다.

## 1. 착수

1. 태그 `t23b` 확인. 배정 파일과 T23b 결과를 그대로 쓴다.
2. `tools/t23.py`에 하위 명령을 더한다. 로지스틱 회귀는 `sklearn.linear_model.LogisticRegression`(lbfgs, C = 1.0, max_iter = 1000, 특징 표준화). AUC는 `sklearn.metrics.roc_auc_score`(동점은 Mann–Whitney 0.5 처리).

## 2. 정의

공통: E ∈ {TEGe, CONe}, V ∈ {X, AX, A2X, rand1, SELF, SHUF}(T23b와 같은 정의), F ∈ {H, M}.
- c_V^F(u, v) = Σ_{ℓ=1..5} 1[같은 클러스터]. 0–5.
- s_V^F(u, v) = 1[ℓ = 5에서 같은 클러스터] (단일 해상도 대조).

**C-1 노드 쌍.**
- 쌍 집합 P_g: 그룹 g의 anchor u마다 E에서 전체 노드 대상 top-50 이웃 v(자기 제외, T19a 동점 규칙). 정답 y = 1[y_u = y_v](레이블 없는 v는 0).
- AUC_cos = test 쌍에서 코사인만으로 낸 AUC(회귀 없음).
- AUC_cos+c = base 쌍으로 [cos, c]에 맞춘 회귀를 test 쌍에 적용한 AUC. AUC_cos+s도 같은 방식.
- **ΔAUC_multi = AUC_cos+c − AUC_cos**, ΔAUC_single = AUC_cos+s − AUC_cos.
- valid 쌍의 같은 값은 기록한다.
- 그림용 표: E = TEGe, V = AX, F = H, seed 0, test 쌍에서 코사인 10분위 × c(0–5) → 같은-클래스 비율과 쌍 수.

**C-2 에피소드.**
- test 에피소드 200개: `np.random.default_rng(17100 + s)`로 test 클래스 5개(비복원) → 클래스마다 노드 5개(비복원). 레이블 없는 노드는 후보에서 뺀다. 5개 미만인 클래스는 후보 클래스에서 뺀다.
- valid 에피소드 200개: 같은 방식, `default_rng(17200 + s)`. valid 클래스가 5개 미만이면 β 선택을 test가 아닌 base 에피소드(`default_rng(17200 + s)`, base 클래스)로 바꾸고 이상 징후에 적는다.
- 1-shot 셀의 임베딩에도 같은 5-shot 에피소드를 쓴다(분석 전용, shot은 임베딩 출처만 뜻한다).
- 에피소드마다 support 25개. support i마다 나머지 24개를 점수로 정렬해 같은 클래스 4개 대 다른 클래스 20개의 AUC를 내고, 에피소드·support 평균을 낸다.
- 점수: S0 = cos_E, S1 = c_V^F, S2(β) = cos_E + β·c_V^F / 5, β ∈ {0, 0.05, 0.1, 0.2, 0.5}.
- β\* = valid 에피소드에서 S2 AUC의 seed 평균이 최대인 β, 동률이면 작은 β. (E, V, F, 셀)마다.
- **ΔAUC_ep = AUC(S2(β\*)) − AUC(S0)**, test 에피소드.

**"분명".** T23b와 같다(seed n = 3, 평균 > 0, > 2·SE, 부호 3/3).

## 3. 실행

- 프로세스 6개 = (데이터셋, shot).
- 결과 `results/T23/T23c/<ds>_5w<k>s/seed<s>.csv`. 쌍 목록과 에피소드 목록은 저장하지 않는다(에피소드 노드 id는 seed 0만 `episodes_seed0.npz`로 남긴다).
- **계산 순서(시간이 모자라면 뒤에서부터 빠진다):** ① C-1, E = TEGe, F = H, V 전부 → ② C-2, E = TEGe, F = H, V ∈ {AX, X, SELF, SHUF} → ③ C-1, E = CONe, F = H → ④ C-2, E = CONe → ⑤ F = M 전부 → ⑥ 그림용 표.

## 4. 산출 표 (`reports/T23c_summary.md`)

| 표 | 내용 |
|---|---|
| C1 | **주 표.** C-1, test, F = H: 행 = (E, V), 열 = 셀 6개. 값 = AUC_cos, ΔAUC_multi, ΔAUC_single(seed 평균 ± SE, 부호 일치/3) |
| C2 | C1과 같은 형식, F = M |
| C3 | 그림용: 코사인 10분위 × c → 같은-클래스 비율(쌍 수). 셀 6개 |
| C4 | C-2, F = H: 행 = (E, V), 열 = 셀. 값 = AUC(S0), AUC(S1), AUC(S2(β\*)), β\*, ΔAUC_ep |
| C5 | C-2의 β 곡선(E = TEGe, V = AX, F = H): β별 valid·test AUC |

## 5. 판정 (값만. 단위 = 셀 6개, test, F = H)

| 판정 | 정의 |
|---|---|
| S1 | ΔAUC_multi(TEGe, AX)가 분명히 양수인 셀 수 |
| S2 | ΔAUC_multi(TEGe, AX) − ΔAUC_single(TEGe, AX)가 분명히 양수인 셀 수 |
| S3 | ΔAUC_multi(TEGe, SELF)가 분명히 양수인 셀 수 |
| S4 | ΔAUC_multi(TEGe, AX) − ΔAUC_multi(CONe, AX)가 분명히 양수인 셀 수 |
| S5 | ΔAUC_ep(TEGe, AX)가 분명히 양수인 셀 수 |
| S6 (기록) | ΔAUC_multi(TEGe, X), ΔAUC_multi(TEGe, rand1), ΔAUC_multi(TEGe, SHUF)가 분명히 양수인 셀 수; C-2의 β\* 분포 |

## 6. 사전 등록 (수정 금지, 어긋나면 날짜 있는 부기)

| 예측 | 내용 |
|---|---|
| P121 | S1 ≥ 4/6 |
| P122 | S2 ≥ 4/6 |
| P123 | S3 ≤ 1/6 |
| P124 | S4 ≥ 4/6 |
| P125 | S5 ≥ 4/6 |

## 7. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ia 선행 | `t23b` 존재 | 시작하지 않음 |
| Ib 쌍 일치 | C-1의 top-50 쌍에서 K = 10까지 자른 같은-레이블 비율의 test 평균이 T23b u_10(같은 run)과 \|Δ\| ≤ 1e-9 | 멈추고 보고 |
| Ic 일치 수 정확성 | 합성 배정(노드 50개, 무작위 아님)에서 c, s를 전수 비교로 확인. H 계열 실제 배정 하나에서 c(u, v) = (가장 깊은 공유 수준)과 같다(중첩이므로) | 멈추고 보고 |
| Id 에피소드 | 에피소드마다 클래스 5개가 모두 test(valid) 클래스이고, 노드 25개가 서로 다르며 클래스마다 5개 | 멈추고 보고 |
| Ie 유한 | 모든 AUC 유한. 회귀가 수렴하지 않은 경우 기록 | 해당 값 제외, 기록 |
| If 변경 범위 | `git diff --stat <H0> <보고서 커밋>`이 `tools/t23.py`, `reports/`, `instructions/` 안 | 멈추고 보고 |

## 8. 시간이 모자랄 때

T0 + 55분까지 끝나지 않으면 3절 순서에서 끝난 것까지만 표로 내고, 빠진 표와 판정은 "미실행(시간)"으로 적는다. 판정 S1·S3·S5와 P121·P123·P125는 ①·②만으로 낼 수 있다.

## 9. 보고

- `reports/T23c_hier_signal.md`(CLAUDE.md §6 형식), `reports/T23c_summary.md`.
- 커밋 → 태그 `t23c` → push.
- 이어서 `reports/T23_queue_summary.md`(큐 §4)를 쓰고 커밋·push한 뒤 멈추고 다음 지시를 기다린다.
