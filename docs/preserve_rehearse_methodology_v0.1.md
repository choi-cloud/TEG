# Preserve & Rehearse (가칭) Methodology 요약

TEG(Kim et al., KDD 2023)를 기반으로 한 graph few-shot node classification 설계안이다(v0.1, 2026-10-06; v0 대비 변경: 가짜 클래스 core 선택 규칙 제거 — 클러스터를 그대로 사용).
핵심 아이디어: **인코더는 레이블 없는 노드들의 확산 특징 이웃 구조를 보존하고(Preserve, 이웃 분포 KL 정규화), task 처리부(EGNN·프로토타입 거리)는 레이블 없는 영역에서 만든 가짜 클래스 에피소드로 연습한다(Rehearse).**
학습 경로는 TEG 원본에 손실 항 1개(보존)와 에피소드 출처 1개(가짜 클래스)를 더한 것이며, 추론 경로는 TEG 원본과 같다.

### 기호

- $\mathcal{G}=(\mathcal{V},\mathcal{E})$, 특징 $X \in \mathbb{R}^{|\mathcal{V}|\times F}$, $\hat{A} = \tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}$
- $\mathcal{C}_{base}$: base 클래스, $\mathcal{V}_{base}$: base 레이블 노드, $n_{med} = \mathrm{median}_{c\in\mathcal{C}_{base}} |\mathcal{V}_c|$
- $\mathcal{U}$: 보존·가짜 클래스용 노드 풀. 설정 $\mathcal{U}_{nb} = \mathcal{V}\setminus\mathcal{V}_{base}$ 또는 $\mathcal{U}_{all} = \mathcal{V}$
- $f_\theta$: TEG GCN 인코더, $z_i = \mathrm{LN}(f_\theta(X,\hat{A}))_i \in \mathbb{R}^{64}$ (EGNN 입력 표현)
- $g_\phi$: TEG task embedder(EGNN) + 프로토타입 분류기

---

## Phase 0. 학습 전 사전 계산 (1회, 고정)

### 0-1. 교사 특징 구성

- **Input**:
  - $X$, $\hat{A}$, 확산 차수 $k=2$

- **Process**:
  - $H^{T} = \hat{A}^{k} X$ (희소 곱)
  - 행 정규화 $\bar{h}_i = h_i / \lVert h_i \rVert_2$

- **Output**:
  - 교사 특징 $\bar{H}^{T} \in \mathbb{R}^{|\mathcal{V}|\times F}$ (학습 중 gradient 없음)

---

### 0-2. 가짜 클래스 구성 (Rehearse 재료)

- **Input**:
  - $\bar{H}^{T}$, 노드 풀 $\mathcal{U}$, $n_{med}$, $N$, $K$, $Q$, 전용 난수 시드

- **Process**:
  - 클러스터 수 $M = \lfloor |\mathcal{U}| / n_{med} \rfloor$; $\{\bar{h}_i\}_{i\in\mathcal{U}}$에 cosine k-means($M$개) → 클러스터 $\mathcal{S}_1,\dots,\mathcal{S}_M$
  - $|\mathcal{S}_m| < K+Q$인 클러스터 제거, 나머지를 그대로 가짜 클래스 $P_m = \mathcal{S}_m$로 사용 → $\mathcal{P} = \{P_1,\dots,P_{M'}\}$
  - $M' \ge N$ 확인(미충족 시 구성 실패로 기록)

- **Output**:
  - 가짜 클래스 집합 $\mathcal{P}$ (노드 ID 목록, 학습 전 고정)

---

## Phase 1. 학습 스텝 (매 step 반복)

### 1-1. 에피소드 샘플링

- **Input**:
  - $\mathcal{C}_{base}$, $\mathcal{P}$, $N$, $K$, query 수 $Q$, 가짜 에피소드 확률 $p$

- **Process**:
  - $u \sim \mathrm{Bernoulli}(p)$로 에피소드 출처 결정
  - $u=0$: TEG 원본 규칙으로 base 에피소드 $e$ 생성($\mathcal{C}_{base}$에서 $N$개 클래스)
  - $u=1$: $\mathcal{P}$에서 $N$개 무작위 선택, 각 $P_m$에서 support $K$·query $Q$개 비복원 추출, 라벨 = 에피소드 내 가짜 클래스 순번

- **Output**:
  - 에피소드 $e=(\mathcal{S}_e, \mathcal{Q}_e, y_e)$, 출처 플래그 $u$

---

### 1-2. 인코딩 및 task 처리 (TEG 원본)

- **Input**:
  - $X$, $\hat{A}$, 에피소드 $e$

- **Process**:
  - 전체 노드 인코딩 $Z = \mathrm{LN}(f_\theta(X,\hat{A}))$
  - $\mathcal{S}_e \cup \mathcal{Q}_e$로 task 그래프 구성, $g_\phi$(EGNN, virtual anchor 구조 특징 포함)로 좌표 갱신
  - 갱신 좌표로 프로토타입 계산, query 분류 → TEG 손실 $\mathcal{L}_{TEG}(e) = \gamma\,\mathcal{L}_N + (1-\gamma)\,\mathcal{L}_G$

- **Output**:
  - $Z$, 에피소드 손실 $\mathcal{L}_{TEG}(e)$

---

### 1-3. 이웃 분포 보존 손실 (Preserve)

- **Input**:
  - $Z$, $\bar{H}^{T}$, $\mathcal{U}$, 배치 크기 $m$, 온도 $\tau$

- **Process**:
  - $\mathcal{U}$에서 $m$개 노드 $\mathcal{B}$ 비복원 추출
  - $S^{T}_{ij} = \cos(\bar{h}_i,\bar{h}_j)$, $S^{S}_{ij} = \cos(z_i,z_j)$, $i,j\in\mathcal{B}$, 대각($i=j$) 제외
  - $p^{T}_{i\cdot} = \mathrm{softmax}_{j\ne i}(S^{T}_{ij}/\tau)$, $p^{S}_{i\cdot} = \mathrm{softmax}_{j\ne i}(S^{S}_{ij}/\tau)$
  - $\mathcal{L}_{pres} = \frac{1}{m}\sum_{i\in\mathcal{B}} \mathrm{KL}\left(p^{T}_{i\cdot}\,\Vert\,p^{S}_{i\cdot}\right)$ (gradient는 $z$ 쪽으로만)

- **Output**:
  - 보존 손실 $\mathcal{L}_{pres}$

---

## Training Objective / Optimization

### 전체 손실 및 모델 선택

- **Input**:
  - $\mathcal{L}_{TEG}(e)$, $u$, $\mathcal{L}_{pres}$, 가중치 $\beta$, $\lambda$

- **Process**:
  - step 손실 $\mathcal{L} = w_u\,\mathcal{L}_{TEG}(e) + \lambda\,\mathcal{L}_{pres}$, $w_0 = 1$, $w_1 = \beta$
  - $\theta, \phi$를 TEG 원본 optimizer·학습률·에폭 수로 갱신
  - valid 평가·모델 선택은 TEG 원본 규칙(`acc_valid == best_acc_valid`)
  - 구성 플래그: $p=0$ → Rehearse 끔, $\lambda=0$ → Preserve 끔, $\mathcal{U}\in\{\mathcal{U}_{nb},\mathcal{U}_{all}\}$

- **Output**:
  - 학습된 $\theta^{*}, \phi^{*}$

---

## Phase 2. 추론

### 2-1. test 에피소드 분류 (TEG 원본)

- **Input**:
  - $\theta^{*}, \phi^{*}$, test 클래스 에피소드(support $K$개 라벨)

- **Process**:
  - 1-2와 같은 인코딩·task 그래프·EGNN·프로토타입 분류
  - $\mathcal{P}$, $\bar{H}^{T}$, $\mathcal{L}_{pres}$ 미사용

- **Output**:
  - query 노드 예측 라벨

---

## 하이퍼파라미터 (v0.1 기본값)

| 기호 | 의미 | v0.1 값 |
|---|---|---|
| $k$ | 교사 확산 차수 | 2 |
| $\tau$ | 보존 손실 온도 | 0.1 |
| $m$ | 보존 배치 크기 | 1,024 |
| $\lambda$ | 보존 가중치 | {0.1, 1, 10} 중 valid 선택 |
| $p$ | 가짜 에피소드 확률 | 0(E1), 0.5 |
| $\beta$ | 가짜 에피소드 손실 가중치 | 1 |
| $M$ | 가짜 클래스 수 | $\lfloor |\mathcal{U}| / n_{med} \rfloor$ |

---

## 전체 파이프라인 요약
```
- X, Â, base 레이블
  - ↓
- [0-1] 교사 특징 H̄ᵀ = norm(Â²X)            (1회, 고정)
  - ↓
- [0-2] U 위 cosine k-means → 가짜 클래스 𝒫  (1회, 고정)
  - ↓
- 매 step:
  - [1-1] 에피소드 출처 u ~ Bernoulli(p): base 에피소드 | 가짜 클래스 에피소드
  - [1-2] Z = LN(GCN(X, Â)) → task 그래프 → EGNN → 프로토타입 → 𝓛_TEG(e)
  - [1-3] U에서 m개 추출 → 이웃 분포 KL(p_T ‖ p_S) → 𝓛_pres
  - 𝓛 = w_u·𝓛_TEG(e) + λ·𝓛_pres → θ, φ 갱신
  - ↓
- valid 기준 모델 선택 (TEG 원본 규칙)
  - ↓
- [2-1] test 에피소드: TEG 원본 추론 → query 라벨
```
