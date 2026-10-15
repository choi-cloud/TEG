# T23b — 클러스터에서 고른 support의 레이블 일치 (학습 없음)

2026-10-15 · 작성: 채팅 Claude · 모드: auto mode(`T23_QUEUE.md`) · 브랜치 `hier_purity` · 태그 `t23b` · 선행: `t23a`

## 0. 목적과 검증 가능한 주장

두 가지를 잰다.

- **B-1 클러스터 support 순도.** anchor와 같은 클러스터에서 support를 무작위로 고를 때, 그 support가 anchor와 같은 클래스일 확률. 클러스터를 pseudo-class로 쓰는 태스크 생성(CACTUs·C2FPS류)의 품질이다.
- **B-2 제한 kNN.** 고정 임베딩(TEGe, CONe)의 코사인 top-K 이웃을 **다른 뷰 V의 같은 클러스터 안으로 제한**하면 순도가 오르는가. 오르면 V의 클러스터가 임베딩 거리에 없는 같은-클래스 정보를 담고 있다는 직접 증거다. V = 임베딩 자신(SELF)과 무작위 재배정(SHUF)이 대조군이다.

주장은 4절 표, 5절 판정값, 6절 사전 등록 대조다. 해석은 하지 않는다.

## 1. 착수

1. 태그 `t23a`가 있고 `results/T23/clusters/`의 배정 54개(H 계열을 뺐다면 M만)가 있는지 확인한다.
2. 코드는 `tools/t23.py`에 하위 명령을 더한다. kNN 동점 규칙은 T19a와 같다(`analyze_t19.knn_sorted`: GPU float64, 유사도 내림차순 → 노드 인덱스 오름차순). 재사용한 `파일:줄`을 보고서 1절에 적는다.

## 2. 정의

공통: anchor u는 그룹 g ∈ {base, valid, test}의 노드 전부(표본 없음). 그룹·레이블은 그 run의 `split.json`, `labels.npy`. 어느 클래스 목록에도 없는 노드(T15 J절)는 "다른 클래스"로 센다. 학습 없는 뷰도 seed별 그룹 분할로 계산한다(T19a와 같음). C_V^ℓ(u) = 뷰 V, 계열 F, 수준 ℓ에서 u가 속한 클러스터.

**B-1.** p(u; V, F, ℓ) = |{v ∈ C(u) \ {u} : y_v = y_u}| / (|C(u)| − 1). 크기 1인 클러스터의 anchor는 빼고 그 비율을 기록한다. 그룹 평균을 낸다.
- V ∈ {X, AX, A2X, rand1, TEGe, CONe}, F ∈ {M, H}, ℓ = 1–5.

**B-2.** 고정 임베딩 E ∈ {TEGe, CONe}(같은 run), K ∈ {5, 10}.
- u_K(u) = E에서 전체 노드 대상 top-K 이웃의 같은-레이블 비율(T19a `pur_all`과 같은 값).
- r_K(u; V, F, ℓ) = E 코사인 top-K를 C_V^ℓ(u) \ {u} 안에서만 고른 같은-레이블 비율. 구현: 유사도 행렬 블록에서 클러스터가 다른 열과 자기 자신을 −∞로 두고 top-K. 멤버가 K보다 적으면 있는 멤버 전부로 계산하고(그 비율 `frac_short` 기록), 멤버가 없으면 뺀다.
- **gain(E, V, F, ℓ, K) = 그룹 평균 r_K − 그룹 평균 u_K**(같은 anchor 집합).
- V ∈ {X, AX, A2X, rand1, SELF, SHUF}.
  - SELF = E 자신의 클러스터(같은 run의 TEGe 또는 CONe 배정).
  - SHUF = AX 배정을 수준마다 노드에 무작위로 다시 붙인 것(클러스터 크기 보존). `np.random.default_rng(17300 + s)`로 수준 1–5 순서대로 `permutation(N)`을 한 번씩 뽑는다. H·M 각각에 같은 순열을 쓴다.
  - 학습 없는 뷰의 배정은 `seed0`, rand1은 그 run의 seed를 쓴다.
- 효율: E의 유사도 블록은 anchor 블록마다 한 번만 계산하고, 그 블록에 (V, F, ℓ) 마스크를 차례로 건다.

**수준 선택 ℓ\*.** (E, V, F, 데이터셋, shot)마다 valid 그룹 gain(K = 10)의 seed 평균이 최대인 ℓ. 동률이면 작은 ℓ(거친 쪽). test 그룹 값은 ℓ\*에서 보고한다.

**"분명".** 단위는 seed(0–2, n = 3). 평균 > 0, 평균 > 2·SE, 부호 3/3. 차이(예: TEGe − CONe)는 seed끼리 짝지어 낸다. anchor 단위 SE는 기록으로만 병기한다.

## 3. 실행

- 프로세스 6개 = (데이터셋, shot). GPU UUID 지정, 동시 6개 이하.
- run별 결과를 `results/T23/T23b/<ds>_5w<k>s/seed<s>.csv`(긴 형식: E, V, F, ℓ, K, group, 값 열)로 저장한다. 이웃 인덱스는 저장하지 않는다.
- **계산 순서(시간이 모자라면 뒤에서부터 빠진다):** ① B-2, E = TEGe, F = H, K = 10 → ② B-2, E = CONe, F = H, K = 10 → ③ B-1 전부 → ④ B-2, F = M → ⑤ B-2, K = 5.

## 4. 산출 표 (`reports/T23b_summary.md`)

| 표 | 내용 |
|---|---|
| B1 | B-1 순도: 행 = 뷰 × 계열, 열 = 그룹 × ℓ(1–5). 셀별(seed 평균 ± sd). 각 ℓ의 클러스터 크기 중앙값을 같은 표 아래에 적는다 |
| B2 | 기준 kNN 순도 u_K(K = 5, 10)와, 참고용 `pur_all`(K = 20, 50): E ∈ {X, AX, TEGe, CONe} × 그룹. 셀별 |
| B3 | **주 표.** test 그룹, K = 10, F = H: 행 = (E, V), 열 = 셀 6개. 값 = ℓ\*에서의 gain(seed 평균 ± SE, 부호 일치/3), ℓ\*, frac_short |
| B4 | B3과 같은 형식, F = M |
| B5 | 모든 ℓ에서의 gain(E = TEGe, F = H, K = 10, test, valid). 수준별 곡선용 |
| B6 | K = 5에서 B3과 같은 형식(기록) |

## 5. 판정 (값만. 단위 = (데이터셋, shot) 6개. test 그룹, K = 10, F = H, ℓ\*)

| 판정 | 정의 |
|---|---|
| R1 | gain(TEGe, AX)가 분명히 양수인 셀 수 |
| R2 | gain(TEGe, X)가 분명히 양수인 셀 수 |
| R3 | gain(TEGe, SELF)가 분명히 양수인 셀 수 |
| R4 | gain(TEGe, SHUF)가 분명히 음수인 셀 수(평균 < 0, \|평균\| > 2·SE, 부호 3/3) |
| R5 | gain(TEGe, AX) − gain(CONe, AX)가 분명히 양수인 셀 수 |
| R6 | gain(TEGe, rand1)가 분명히 양수인 셀 수 |
| R7 (기록) | gain_H(TEGe, AX) − gain_M(TEGe, AX)가 분명히 양수인 셀 수; ℓ\*의 분포 |

## 6. 사전 등록 (수정 금지, 어긋나면 날짜 있는 부기)

| 예측 | 내용 |
|---|---|
| P116 | R1 ≥ 4/6 |
| P117 | R3 ≤ 1/6 |
| P118 | R4 = 6/6 |
| P119 | R5 ≥ 4/6 |
| P120 | R6 ≤ 2/6 |

## 7. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ha 선행 | `t23a` 존재, 배정 파일 존재 | 시작하지 않음 |
| Hb 기준 일치 | u_10(TEGe, Amazon_clothing 1-shot seed 0, test)이 `analyze_t19`의 `pur_all` 계산(같은 임베딩)과 \|Δ\| ≤ 1e-9 | 멈추고 보고 |
| Hc 제한 kNN 정확성 | (a) 모든 노드를 한 클러스터로 둔 마스크에서 r_10 = u_10(위 run, \|Δ\| ≤ 1e-9). (b) 노드 200개 결정적 합성 자료(생성기 17000)에서 r_K를 anchor별 전수 정렬과 비교해 완전 일치 | 멈추고 보고 |
| Hd B-1 정확성 | 합성 자료에서 p(u)를 전수 계산과 비교해 완전 일치 | 멈추고 보고 |
| He 유한 | 모든 값 유한 | 해당 run 제외, 기록 |
| Hf 변경 범위 | `git diff --stat <H0> <보고서 커밋>`이 `tools/t23.py`, `reports/`, `instructions/` 안 | 멈추고 보고 |

## 8. 보고

- `reports/T23b_support_purity.md`(CLAUDE.md §6 형식), `reports/T23b_summary.md`.
- 커밋 → 태그 `t23b` → push. 그다음 T23c로 넘어간다.
