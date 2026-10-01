# 관계 교과서(Relation Textbook) Methodology 요약 — 설계 초안 v0.1 (2026-10-01)

> v0.1 변경: 쓰기·보정 대상을 support↔support와 support↔query 두 관계로 확장(Q3 수정, Choi 합의). 첫 구현(T03a·T03b)은 support↔query만 적용하고 메모리는 평가 시점에만 읽는다 — 아래 부록 B.

- training 전체의 샘플 쌍 관계를 메모리("교과서")에 쌓고, test 에피소드에서는 support 쌍마다 유사 관계를 조회해 관계를 보정한 뒤 support 표현을 갱신하고 예측한다("오픈북").
- 핵심 idea: **공유 인코더 = 에피소드 공통(불변) 패턴, 메모리 = 에피소드 고유·드문 관계 패턴.** base 노드는 지도 에피소드로, base ∪ test 노드는 라벨 없는 에피소드로 관계를 공급한다.
- 기호는 구현 틀(TEG, Kim et al., KDD 2023)과 맞춘다. 각 항목 끝 태그: **[확정]** 합의됨 / **[1차]** 첫 구현용 기본값, 진단 후 변경 가능 / **[미정]** 설계 미결.

---

## Phase 0. 사전 계산 (학습 전 1회)

### 0-1. 구조 특징 생성

- **Input**:
  - 그래프 $\mathcal{G}=(\mathcal{V},\mathcal{E})$, 인접행렬 $\mathbf{A}$, 가상 anchor 수 $k$
- **Process**:
  - 가상 anchor 노드 $\mathcal{V}_\alpha=\{v_{\alpha_1},\dots,v_{\alpha_k}\}$ 생성, anchor $i$는 각 노드와 확률 $1/2^i$로 연결
  - $\mathbf{H}^{(s)}_v=\big(s(v,v_{\alpha_1}),\dots,s(v,v_{\alpha_k})\big)$, $s(v,u)=1/(d^{sp}(v,u)+1)$
  - 학습·평가 전 과정에서 고정 사용 [확정: TEG Eq.3 그대로]
- **Output**:
  - 구조 특징 $\mathbf{H}^{(s)}\in\mathbb{R}^{|\mathcal{V}|\times k}$ (인코더와 무관)

### 0-2. 노드 유사도 행렬 (라벨 없는 에피소드용)

- **Input**:
  - 원 특징 $\mathbf{X}$ 또는 $\mathbf{A}$
- **Process**:
  - feature 기준: $\mathbf{S}_{uv}=\cos(\mathbf{x}_u,\mathbf{x}_v)$, 또는 구조 기준: $\mathbf{S}=\sum_k\theta_k\mathbf{T}^k$ (PPR diffusion)
  - 노드별 Top-$Q$ 유사 노드 목록 저장
  - 두 기준 중 선택 [미정]
- **Output**:
  - 노드별 Top-$Q$ 유사 노드 목록

---

## Phase 1. 에피소드 생성

### 1-1. 지도 에피소드

- **Input**:
  - base 클래스 $C_b$와 라벨, $N$, $K$, $M$
- **Process**:
  - $C_b$에서 $N$개 클래스 균등 추출
  - 클래스별 support $K$개, query $M$개 추출, 실제 라벨 사용
- **Output**:
  - $\mathcal{T}^{sup}=\{S,Q\}$, 노드 $\subset C_b$

### 1-2. 라벨 없는 에피소드

- **Input**:
  - 전체 노드 $\mathcal{V}$ (base ∪ test 노드, 라벨 미사용), 0-2 목록, $N$, $Q$
- **Process**:
  - $\mathcal{V}$에서 support 노드 $N$개 무작위 추출, 노드마다 서로 다른 의사 라벨 부여 ($K=1$)
  - 각 support 노드의 Top-$Q$ 유사 노드를 같은 의사 라벨의 query로 지정
  - base 노드 간 같은 라벨 마스킹 없음 [1차: 3-i 선택지 B]
- **Output**:
  - $\mathcal{T}^{uns}=\{S,Q\}$, 노드 $\subset C_b\cup C_t$ (base–test 혼합 쌍 포함)

### 1-3. 에피소드 스케줄

- **Input**:
  - 1-1, 1-2 생성기
- **Process**:
  - 학습 스텝마다 $\mathcal{T}^{sup}$와 $\mathcal{T}^{uns}$를 1:1로 번갈아 공급 [1차]
- **Output**:
  - 학습 에피소드 열 $\{\mathcal{T}_t\}_{t=1}^{T}$

---

## Phase 2. 관계 계산

### 2-1. 의미 특징

- **Input**:
  - $\mathbf{X}$, $\mathbf{A}$
- **Process**:
  - $\mathbf{H}^{(l)}=\mathrm{GNN}_\theta(\mathbf{X},\mathbf{A})$
- **Output**:
  - 의미 특징(좌표) $\mathbf{H}^{(l)}\in\mathbb{R}^{|\mathcal{V}|\times d_l}$

### 2-2. 쌍 관계

- **Input**:
  - 현재 에피소드의 노드 쌍 $(i,j)$, $\mathbf{h}^{(s)}_i,\mathbf{h}^{(s)}_j,\mathbf{h}^{(l)}_i,\mathbf{h}^{(l)}_j$
- **Process**:
  - $\mathbf{r}_{ij}=\phi_m\big(\mathbf{h}^{(s)}_i,\mathbf{h}^{(s)}_j,\|\mathbf{h}^{(l)}_i-\mathbf{h}^{(l)}_j\|^2\big)$ (TEG Eq.4의 첫 층 메시지)
  - 라벨 미사용 [확정]
  - 첫 층에서만 계산 → 다른 노드의 영향을 받지 않는 쌍 관계 [1차]
- **Output**:
  - 관계 표현 $\mathbf{r}_{ij}\in\mathbb{R}^{d_l}$

---

## Phase 3. 메모리 쓰기 (교과서)

### 3-1. 메모리 구조

- **Input**:
  - 용량 $B$
- **Process**:
  - $\mathcal{M}=\{(\mathbf{k}_m,\mathbf{v}_m,a_m,u_m)\}_{m=1}^{|\mathcal{M}|}$, $|\mathcal{M}|\le B$ ($a_m$: 기록 시점, $u_m$: 조회 횟수)
  - 키: $\mathbf{k}_{ij}=\psi(\mathbf{h}^{(s)}_i,\mathbf{h}^{(s)}_j)$, 순서 무관 대칭형, 인코더와 무관 [확정: staleness 선택지 A]
  - 값 $\mathbf{v}$의 형태(관계 표현 자체 / 학습되는 보정량) [미정]
- **Output**:
  - 빈 메모리 $\mathcal{M}$

### 3-2. 쓰기

- **Input**:
  - 에피소드 $\mathcal{T}_t$(지도·라벨 없음 모두)의 support 쌍 관계 $\{\mathbf{r}_{ij}\}$
- **Process**:
  - 각 support 쌍에 대해 $(\mathbf{k}_{ij},\mathbf{v}_{ij},a=t,u=0)$ 추가
  - 쓰기 대상 쌍: support↔support, support↔query [확정: Q3 수정]. 첫 구현은 support↔query만 [1차]
- **Output**:
  - 갱신된 $\mathcal{M}$

### 3-3. 망각

- **Input**:
  - $\mathcal{M}$, $B$
- **Process**:
  - $|\mathcal{M}|>B$이면 $a_m$이 가장 오래된 항목(또는 $u_m$ 최소 항목)부터 삭제 [1차: 3-ii]
- **Output**:
  - $|\mathcal{M}|\le B$인 메모리

---

## Phase 4. 조회와 보정 (오픈북)

### 4-1. 조회

- **Input**:
  - 현재 에피소드의 보정 대상 쌍 $(i,j)$, $\mathbf{k}_{ij}$, $\mathcal{M}$, $k_{ret}$
- **Process**:
  - $\mathrm{sim}(\mathbf{k}_{ij},\mathbf{k}_m)$ 상위 $k_{ret}$개 항목 $\mathcal{N}_{ij}$ 선택
  - 가중치 $w_m=\mathrm{softmax}_{m\in\mathcal{N}_{ij}}\big(\mathrm{sim}(\mathbf{k}_{ij},\mathbf{k}_m)/\tau\big)$
  - 선택된 항목의 $u_m\leftarrow u_m+1$
- **Output**:
  - 조회 결과 $\bar{\mathbf{v}}_{ij}=\sum_{m\in\mathcal{N}_{ij}}w_m\mathbf{v}_m$

### 4-2. 관계 보정

- **Input**:
  - $\mathbf{r}_{ij}$, $\bar{\mathbf{v}}_{ij}$
- **Process**:
  - $\tilde{\mathbf{r}}_{ij}=g(\mathbf{r}_{ij},\bar{\mathbf{v}}_{ij})$
  - $g$의 형태(가중합 $(1-\beta)\mathbf{r}_{ij}+\beta\bar{\mathbf{v}}_{ij}$ / 게이트 / MLP) [미정]
  - 보정 대상: support↔query 쌍(양방향) [확정, 우선 구현], support↔support 쌍 [확정, 후속 — TEG task 그래프에 간선 추가 필요]
- **Output**:
  - 보정된 관계 $\tilde{\mathbf{r}}_{ij}$

### 4-3. support·query 표현 갱신

- **Input**:
  - support·query 좌표 $\mathbf{h}^{(l)}$, 보정 관계 $\tilde{\mathbf{r}}_{ij}$
- **Process**:
  - $\mathbf{h}^{(l),\lambda+1}_i=\mathbf{h}^{(l),\lambda}_i+\frac{1}{C}\sum_{j\neq i}(\mathbf{h}^{(l),\lambda}_i-\mathbf{h}^{(l),\lambda}_j)\,\phi_l(\tilde{\mathbf{r}}_{ij})$ (TEG Eq.5에 보정 관계 대입)
  - support는 query와의 관계로, query는 support와의 관계로 갱신 [확정: Q3 수정]. 층 수 [미정]
- **Output**:
  - support·query 최종 표현 $\mathbf{z}^{(l)}$

---

## Phase 5. 예측

### 5-1. 프로토타입 분류

- **Input**:
  - support 최종 표현 $\mathbf{z}^{(l)}$, query 표현 $\mathbf{z}^{(l)}_q$
- **Process**:
  - $\mathbf{p}_c=\frac{1}{K}\sum_{i=1}^{K}\mathbf{z}^{(l)}_{c,i}$
  - $p(c\mid\mathbf{z}^{(l)}_q)=\dfrac{\exp(-d(\mathbf{z}^{(l)}_q,\mathbf{p}_c))}{\sum_{c'}\exp(-d(\mathbf{z}^{(l)}_q,\mathbf{p}_{c'}))}$, $d$ = 제곱 유클리드 거리
- **Output**:
  - query별 클래스 확률, 예측 $\hat{y}_q=\arg\max_c p(c\mid\mathbf{z}^{(l)}_q)$

---

## Training Objective / Optimization

### 에피소드 손실과 교대 학습

- **Input**:
  - 5-1의 확률, 지도 에피소드의 실제 라벨 또는 라벨 없는 에피소드의 의사 라벨
- **Process**:
  - $\mathcal{L}_{sup}=-\sum_{q}\log p(y_q\mid\mathbf{z}^{(l)}_q)$ ($\mathcal{T}^{sup}$), $\mathcal{L}_{uns}$ 동일 형식 ($\mathcal{T}^{uns}$, 의사 라벨)
  - 스텝마다 해당 에피소드 손실로 $\theta,\phi$ 갱신 (1-3 교대)
  - 계측: $\cos(\nabla\mathcal{L}_{sup},\nabla\mathcal{L}_{uns})$ 기록 [1차: 3-i B]
  - 메모리 값으로 기울기가 흐르는지 여부, TEG의 $\mathcal{L}_G$ 유지 여부 [미정]
- **Output**:
  - 학습된 $\theta,\phi$, 학습 종료 시점의 $\mathcal{M}$

### Meta-test

- **Input**:
  - test 에피소드(클래스 $\subset C_t$, 라벨 있는 support), 고정된 $\mathcal{M}$
- **Process**:
  - Phase 2 → 4 → 5 수행, 메모리 쓰기 없음 [1차]
- **Output**:
  - query 예측

---

## 전체 파이프라인 요약

```
- [학습 전] G, X, A
  - ↓
- 0-1 구조 특징 H^(s)  /  0-2 노드 유사도 Top-Q
  - ↓
- [학습 루프] 1-3 교대 공급: 지도 에피소드(C_b) ↔ 라벨 없는 에피소드(C_b ∪ C_t)
  - ↓
- 2-1 의미 특징 H^(l) → 2-2 쌍 관계 r_ij
  - ↓
- 3-2 메모리 쓰기 → 3-3 망각
  - ↓
- 4-1 조회 → 4-2 관계 보정 → 4-3 support 표현 갱신
  - ↓
- 5-1 프로토타입 분류 → 손실 → 갱신
  - ↓
- [meta-test] 고정 메모리로 2 → 4 → 5
  - ↓
- query 예측
```

---

## 부록. 미정 항목

| 위치 | 항목 | 비고 |
|---|---|---|
| 0-2 | feature 기준 / 구조 기준 유사도 | 데이터셋별로 다를 수 있음 |
| 3-1 | 값 $\mathbf{v}$의 형태 | "test에서 직접 계산할 수 없는 것"이어야 메모리가 의미를 가짐 |
| 4-2 | 보정 함수 $g$ | 첫 구현은 가중합 |
| 4-3 | 층 수, support↔support 간선 추가 시 대조군 | TEG task 그래프는 support↔query 간선만 가짐(코드 확인) |
| 학습 | 메모리로의 기울기, $\mathcal{L}_G$ 유지 | 첫 구현은 메모리 비학습(사후 구축) |

## 부록 B. 첫 구현(T03a·T03b) 범위

| 모듈 | 첫 구현 |
|---|---|
| 0-2, 1-2, 1-3 (라벨 없는 에피소드) | 미구현 |
| 2-2 쌍 관계 | TEG `gcl_0.msg_model` 출력 |
| 3-1 키 | 평가 시점 현재 모델로 재계산한 첫 층 메시지 (staleness 없음) |
| 3-2 쓰기 | 학습 에피소드(에폭 ≥ 1)의 support↔query 쌍, 양방향, 노드 ID + base 정답 같은 클래스 여부 |
| 3-3 망각 | 미구현 (전부 보관) |
| 4-2 보정 | arm A: 메시지 가중합 / arm B: 첫 층 좌표 갱신 가중치를 p̂로 조정 |
| 학습 | 메모리는 학습에 관여하지 않음 |
