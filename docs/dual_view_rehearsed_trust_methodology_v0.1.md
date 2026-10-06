# Dual-View Rehearsed Trust (가칭) Methodology 요약

TEG(Kim et al., KDD 2023)를 기반으로 한 graph few-shot node classification 설계안이다(v0.1, 2026-10-06; v0 대비 변경: ① 가짜 클래스 core 선택 규칙 제거 — 클러스터를 그대로 사용, ② 판단기를 특징 2개 로지스틱 회귀로 축소).
핵심 아이디어: **task마다 학습된 관점(TEG logit)과 압축 이전의 확산 관점(Â²X 코사인 logit)을 신뢰도 α로 결합하고, α를 정하는 판단기는 인코더가 본 적 없는 가짜 클래스 에피소드에서만 학습한다(인코더·EGNN 고정).**
학습은 2단계(TEG 원본 학습 → 판단기 사후 보정)이며, 1단계는 TEG 원본과 같다.

### 기호

- $\mathcal{G}=(\mathcal{V},\mathcal{E})$, 특징 $X \in \mathbb{R}^{|\mathcal{V}|\times F}$, $\hat{A} = \tilde{D}^{-1/2}(A+I)\tilde{D}^{-1/2}$
- $\mathcal{C}_{base}$: base 클래스, $\mathcal{V}_{base}$: base 레이블 노드, $n_{med} = \mathrm{median}_{c\in\mathcal{C}_{base}} |\mathcal{V}_c|$
- $\mathcal{U}$: 가짜 클래스용 노드 풀. 설정 $\mathcal{U}_{nb} = \mathcal{V}\setminus\mathcal{V}_{base}$ 또는 $\mathcal{U}_{all} = \mathcal{V}$
- $f_\theta$: TEG GCN 인코더, $g_\phi$: TEG task embedder(EGNN) + 유클리드 프로토타입 분류기
- 관점 $v \in \{L, D\}$: $L$ = 학습된 관점(TEG), $D$ = 확산 관점
- 에피소드: $N$-way $K$-shot, 클래스당 query $Q$개, query 집합 $\mathcal{Q}_e$

---

## Phase 0. 학습 전 사전 계산 (1회, 고정)

### 0-1. 확산 관점 특징 구성

- **Input**:
  - $X$, $\hat{A}$, 확산 차수 $k=2$

- **Process**:
  - $H^{D} = \hat{A}^{k} X$ (희소 곱)
  - 행 정규화 $\bar{h}_i = h_i / \lVert h_i \rVert_2$

- **Output**:
  - 확산 특징 $\bar{H}^{D} \in \mathbb{R}^{|\mathcal{V}|\times F}$ (학습 파라미터 없음)

---

### 0-2. 가짜 클래스 구성 (Rehearse 재료)

- **Input**:
  - $\bar{H}^{D}$, 노드 풀 $\mathcal{U}$, $n_{med}$, $N$, $K$, $Q$, 전용 난수 시드

- **Process**:
  - 클러스터 수 $M = \lfloor |\mathcal{U}| / n_{med} \rfloor$; $\{\bar{h}_i\}_{i\in\mathcal{U}}$에 cosine k-means($M$개) → 클러스터 $\mathcal{S}_1,\dots,\mathcal{S}_M$
  - $|\mathcal{S}_m| < K+Q$인 클러스터 제거, 나머지를 그대로 가짜 클래스 $P_m = \mathcal{S}_m$로 사용 → $\mathcal{P} = \{P_1,\dots,P_{M'}\}$
  - $M' \ge N$ 확인(미충족 시 구성 실패로 기록)

- **Output**:
  - 가짜 클래스 집합 $\mathcal{P}$ (노드 ID 목록, 학습 전 고정)

---

## Phase 1. 인코더·task 처리부 학습 (TEG 원본)

### 1-1. base 에피소드 학습 및 체크포인트 선택

- **Input**:
  - $X$, $\hat{A}$, $\mathcal{C}_{base}$, TEG 원본 하이퍼파라미터

- **Process**:
  - TEG 원본 규칙으로 base 에피소드 생성, $\mathcal{L}_{TEG} = \gamma\,\mathcal{L}_N + (1-\gamma)\,\mathcal{L}_G$로 $\theta, \phi$ 갱신
  - valid 평가 후 원본 모델 선택 규칙(`acc_valid == best_acc_valid`)으로 체크포인트 갱신

- **Output**:
  - 고정 파라미터 $\theta^{*}, \phi^{*}$

---

## Phase 2. 신뢰 판단기 보정 (Rehearse, $\theta^{*}, \phi^{*}$ 고정)

### 2-1. 가짜 클래스 에피소드 샘플링

- **Input**:
  - $\mathcal{P}$, $N$, $K$, $Q$

- **Process**:
  - $\mathcal{P}$에서 $N$개 무작위 선택
  - 각 $P_m$에서 support $K$개·query $Q$개 비복원 추출, 라벨 = 에피소드 내 가짜 클래스 순번

- **Output**:
  - 가짜 에피소드 $e=(\mathcal{S}_e, \mathcal{Q}_e, y_e)$

---

### 2-2. 두 관점 logit 계산

- **Input**:
  - 에피소드 $e$, $\theta^{*}, \phi^{*}$, $\bar{H}^{D}$

- **Process**:
  - 관점 $L$: TEG 원본 forward(인코딩 → task 그래프 → EGNN) 후 좌표 $x_i$, 프로토타입 $\pi^{L}_c$, $\ell^{L}_{q,c} = -\lVert x_q - \pi^{L}_c \rVert_2^2$
  - 관점 $D$: $\pi^{D}_c = \mathrm{norm}\big(\frac{1}{K}\sum_{i\in\mathcal{S}_{e,c}} \bar{h}_i\big)$, $\ell^{D}_{q,c} = \cos(\bar{h}_q, \pi^{D}_c)$
  - 관점별 확률 $p^{L}_{q} = \mathrm{softmax}_c(\ell^{L}_{q,c}/T_L)$, $p^{D}_{q} = \mathrm{softmax}_c(s\,\ell^{D}_{q,c})$

- **Output**:
  - $\ell^{L}, \ell^{D} \in \mathbb{R}^{|\mathcal{Q}_e|\times N}$, $p^{L}, p^{D}$, 프로토타입 $\pi^{L}, \pi^{D}$

---

### 2-3. task 특징 추출

- **Input**:
  - $p^{L}, p^{D}$

- **Process**:
  - 확신도 차 $\Delta\mathrm{conf} = \frac{1}{|\mathcal{Q}_e|}\sum_q \big(\max_c p^{L}_{q,c} - \max_c p^{D}_{q,c}\big)$
  - 관점 간 일치율 $\mathrm{agr} = \frac{1}{|\mathcal{Q}_e|}\sum_q \mathbb{1}[\arg\max_c p^{L}_{q,c} = \arg\max_c p^{D}_{q,c}]$
  - 특징 벡터 $u_e = [\Delta\mathrm{conf}, \mathrm{agr}] \in \mathbb{R}^{2}$, stop-gradient

- **Output**:
  - task 특징 $u_e$

---

### 2-4. 신뢰도 산출 및 두 관점 결합

- **Input**:
  - $u_e$, $\ell^{L}, \ell^{D}$, 판단기 파라미터 $\psi = (w \in \mathbb{R}^{2}, b)$, $T_L$, $s$

- **Process**:
  - $\alpha_e = \sigma(w^{\top} u_e + b) \in (0,1)$ (로지스틱 회귀, task당 스칼라)
  - $\tilde{\ell}_{q,c} = \alpha_e \log \mathrm{softmax}_c(\ell^{L}_{q,c}/T_L) + (1-\alpha_e)\log \mathrm{softmax}_c(s\,\ell^{D}_{q,c})$
  - $\hat{p}_{q} = \mathrm{softmax}_c(\tilde{\ell}_{q,c})$, 예측 $\hat{y}_q = \arg\max_c \hat{p}_{q,c}$

- **Output**:
  - 결합 확률 $\hat{p}$, 신뢰도 $\alpha_e$

---

## Training Objective / Optimization

### 보정 손실 및 판단기 선택

- **Input**:
  - 가짜 에피소드의 $\hat{p}$, 가짜 라벨 $y_e$, valid 에피소드

- **Process**:
  - $\mathcal{L}_{cal} = -\frac{1}{|\mathcal{Q}_e|}\sum_{q\in\mathcal{Q}_e} \log \hat{p}_{q, y_q}$
  - 갱신 대상 $\{\psi, T_L, s\}$ 한정; $\theta^{*}, \phi^{*}$ 고정(gradient 차단), Adam
  - $n_{eval}$ step마다 valid 에피소드(TEG 원본 생성 규칙)에서 결합 정확도 계산, 최고값 시점의 $\{\psi, T_L, s\}$ 보존
  - 구성 플래그: 판단기 학습 에피소드 출처 $\in$ {가짜($\mathcal{U}_{nb}$), 가짜($\mathcal{U}_{all}$), base, held-out base}, 고정 $\alpha$ 모드

- **Output**:
  - 보정된 $\psi^{*}, T_L^{*}, s^{*}$

---

## Phase 3. 추론

### 3-1. test 에피소드 결합 분류

- **Input**:
  - $\theta^{*}, \phi^{*}, \psi^{*}, T_L^{*}, s^{*}$, $\bar{H}^{D}$, test 에피소드(support $K$개 라벨)

- **Process**:
  - 2-2 → 2-3 → 2-4를 test 에피소드에 그대로 적용
  - $\mathcal{P}$, $\mathcal{L}_{cal}$ 미사용

- **Output**:
  - query 노드 예측 라벨, 에피소드별 $\alpha_e$

---

## 하이퍼파라미터 (v0.1 기본값)

| 기호 | 의미 | v0.1 값 |
|---|---|---|
| $k$ | 확산 차수 | 2 |
| $M$ | 가짜 클래스 수 | $\lfloor |\mathcal{U}| / n_{med} \rfloor$ |
| $\psi$ | 판단기 | 로지스틱 회귀 $(w \in \mathbb{R}^2, b)$, 초기값 $w=0$, $b=0$ ($\alpha=0.5$) |
| $T_L$, $s$ | 관점별 온도·척도 초기값 | 1, 10 |
| 학습률 | Adam | $10^{-3}$ |
| 보정 step 수 | 가짜 에피소드 수 | 2,000 |
| $n_{eval}$ | valid 평가 주기 | 100 step |

---

## 전체 파이프라인 요약
```
- X, Â, base 레이블
  - ↓
- [0-1] 확산 특징 H̄ᴰ = norm(Â²X)                    (1회, 고정)
- [0-2] U 위 cosine k-means → 가짜 클래스 𝒫          (1회, 고정)
  - ↓
- [1-1] TEG 원본 학습(base 에피소드) → valid 선택 → θ*, φ* 고정
  - ↓
- 보정 step 반복 (θ*, φ* 고정):
  - [2-1] 𝒫에서 가짜 에피소드
  - [2-2] 관점 L: TEG logit / 관점 D: Â²X 코사인 logit
  - [2-3] task 특징 u_e = [확신도 차, 관점 간 일치율]
  - [2-4] α_e = σ(wᵀu_e + b) → log-확률 가중 결합
  - 𝓛_cal → ψ, T_L, s 갱신; valid 결합 정확도로 선택
  - ↓
- [3-1] test 에피소드: 2-2 → 2-3 → 2-4 → query 라벨, α_e
```
