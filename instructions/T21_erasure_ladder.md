# T21 — 지움의 사다리: 학습 0 → 에피소드 없는 base 지도 → 에피소드(EGNN 없음) → TEG

2026-10-14 · 작성: 채팅 Claude · 모드: 채팅 지시문을 따른다 · 브랜치 `erasure_ladder`(`m1_sensor_check` HEAD에서 생성) · 태그 `t21` · 선행: `t19b`

## 0. 목적과 검증 가능한 주장

**배경.** T06b·T09에서 학습 없는 Â²X 프로토타입(D)이 TEG 출력보다 학습 미사용 클래스(valid ∪ test)에서 같거나 낫다. 이것이 (i) 에피소드 학습 고유의 효과인지, (ii) EGNN 고유의 효과인지, (iii) base 레이블로 하는 지도 학습 일반의 효과인지를 가른다. 같은 입력(Â²X), 같은 head(코사인 프로토타입), 같은 probe 에피소드를 두고 표현만 네 단계로 바꾼다.

| 단계 | 표현 | 학습 | 출처 |
|---|---|---|---|
| S0 | Â²X (행 정규화) | 없음 | `fusion_baseline.diffusion_features_hops(ds, 2)` |
| S1 | Â²X → MLP(64) 은닉층 | base 노드 전부, 교차엔트로피, 에피소드 없음 | **이 지시서에서 새로 학습** |
| S2 | TEG γ = 0 GCN 출력 | 에피소드, EGNN 손실 없음 | **이 지시서에서 새로 실행** |
| S3 | TEG 원본 GCN 출력 | 에피소드 + EGNN | T09 `base` seed 0–2 `emb_best.npy` |
| 참고 | 대조 (T09 `infonce` 선택 λ) | S3 + 보조 손실 | T09 |

덧붙여 head만 바꾸는 비교를 한 칸 넣는다. S0 위에서 (a) 코사인 프로토타입과 (b) support만으로 맞춘 로지스틱 회귀의 정확도를 같은 에피소드에서 잰다. 발표된 GFSL 논문의 SGC·GCN 기준선(support 몇 개로 학습한 선형 head)이 약한 이유가 head인지 특징인지를 우리 데이터에서 보기 위한 것이다.

**주장.** 4절의 표와 5절의 판정값, 6절의 사전 등록 대조를 산출한다. 해석은 하지 않는다. 레이블 사용: base 클래스 노드의 레이블은 S1 학습에 쓴다(벤치마크 규약상 base 노드는 전부 레이블이 있다). valid·test 레이블은 평가 지표에만 쓴다.

## 1. 착수

1. 현재 브랜치가 `m1_sensor_check`이고 `t19b` 태그가 있는지 확인한다. `git status`에는 사용자가 수정한 `CLAUDE.md`(modified)와 이 지시서 `instructions/T21_erasure_ladder.md`(untracked)만 보여야 한다. 다른 것이 보이면 멈추고 보고한다.
2. `git switch -c erasure_ladder` (현재 `m1_sensor_check` HEAD에서. 작업 트리 변경은 그대로 따라온다).
3. `CLAUDE.md`와 이 지시서를 H0 커밋으로 넣는다. Agent는 CLAUDE.md를 고치지 않는다. H0 해시가 He 게이트의 "착수 HEAD"다.
4. 코드 작성 전 파일 단위 계획을 보이고 승인을 기다린다(CLAUDE.md §2).

## 2. 입력과 단위

- 데이터셋 3개(`Amazon_clothing`, `dblp`, `Amazon_electronics`), seed 0–2.
- 클래스 분할(base / valid / test)은 TEG `utils.load_data`가 seed별로 만드는 것을 그대로 쓴다. S3의 T09 run이 쓴 분할과 같아야 한다(`split.json` 또는 T09 run의 기록과 대조해 Hb에서 확인).
- probe 에피소드: `analyze_t06b.episodes(random.Random(1000 + seed), classes, by_class, k)`, 그룹(base, valid, test)별로 한 번 뽑아 모든 표현이 공유한다. k ∈ {1, 5}, 에피소드 수는 `analyze_t06b.N_EPI`. T12 표 2와 같은 방식이다.
- 모든 표현은 probe 전에 행 L2 정규화한다.
  - (2026-10-14 부기) 적용하지 않음. T12와 같이 정규화 없이 fewshot_acc에 넣는다. S0은 입력 자체가 행 정규화돼 있다.

## 3. 실행

### 3.1 S1 — Â²X 위 base-CE MLP (새 코드 `tools/ladder_t21.py`)

- 입력 Â²X(행 정규화, `diffusion_features_hops(ds, 2)`). 모델 `Linear(d, 64) → ReLU → Linear(64, |base|)`. 손실: base 클래스 노드 **전부**에 대한 교차엔트로피(full batch). Adam lr 5e-3, weight decay 5e-4(TEG 기본값과 같음). 200 에폭.
- 초기화 난수: `torch.Generator().manual_seed(16000 + seed)` (새 오프셋 16000, 등록부에 추가). 전역 `random`·`torch` 기본 생성기는 건드리지 않는다.
- 스냅샷: 에폭 0(학습 전), 25, 50, 100, 200에서 은닉층 출력 h = ReLU(W₁Â²X + b₁)(전체 노드, 64차원)를 `results/T21/mlp/<ds>/seed<s>/h_ep<e>.npy`로 저장한다. 에폭별 base 학습 정확도를 `run.json`에 적는다.
- **주 표현 = 에폭 200.** 에폭 0은 학습 없는 대조군(무작위 사영)이다.
- 장치: GPU 하나면 충분하다. 수 분 안에 끝난다.

### 3.2 S2 — TEG γ = 0 (기존 코드, 플래그만)

- `tools/run_t09.py`를 복사한 `tools/run_t21.py`로 3 데이터셋 × 5-way {1, 5}-shot × seed 0–2 = 18 run. 명령은 T09 `base`와 같고 `--gamma 0`만 더한다: `--pres_lambda 0 --dump_logits --dump_emb --fixed_eval --out_dir results/T21/gamma0/<ds>_5w<k>s/seed<s>`.
- 코드 변경 없음. `--gamma 0`은 기존 플래그다(`argument.py:30`).
- 동시 실행 ≤ 6, GPU 3번 제외, UUID 지정(CLAUDE.md §4).

### 3.3 S3·참고 — 기존 결과

- T09 `base` seed 0–2와 `infonce`(선택 λ, `reports/T09_summary.md` 표 2) seed 0–2의 `emb_best.npy`. 없으면 멈추고 보고한다(삭제됐을 수 있다).

### 3.4 측정 (`tools/analyze_t21.py`)

표현 V ∈ {S0, S1(ep 0/25/50/100/200), S2, S3, 대조}, 데이터셋, seed, k ∈ {1, 5}마다:

| 기호 | 정의 | 레이블 |
|---|---|---|
| acc_g(V, k) | 그룹 g ∈ {base, valid, test}의 probe 에피소드에서 코사인 프로토타입 정확도(%). `analyze_t06b.fewshot_acc` | 평가만 |
| pur_g(V) | 그룹 g 노드 안에서 top-10 코사인 kNN 순도. `analyze_t14.knn_purity(V, nodes_g, y)` | 평가만 |
| acc_test^LR(S0, k) | S0 위에서, 각 probe 에피소드의 support만으로 `sklearn.linear_model.LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)`을 맞춰 query를 분류한 정확도(%). 에피소드는 acc_test(S0, k)와 같다 | 평가만 |

- S2는 shot별 run이 따로 있으므로 k = 1 probe에는 1-shot run의 임베딩을, k = 5에는 5-shot run의 임베딩을 쓴다. S3·대조도 같다(T09 run이 shot별). S0·S1은 shot과 무관하므로 같은 표현을 두 k에 쓴다.
- 무작위성: probe 에피소드(1000 + seed), S1 초기화(16000 + seed)뿐이다. LogisticRegression은 lbfgs라 결정적이다.

## 4. 산출 표 (`reports/T21_summary.md`)

| 표 | 내용 |
|---|---|
| T1 | 사다리: 데이터셋 × k × 표현(S0, S1 ep200, S2, S3, 대조) → acc_base, acc_valid, acc_test (seed 평균 ± sd) |
| T2 | S1 궤적: 데이터셋 × 에폭(0, 25, 50, 100, 200) → base 학습 정확도, acc_base, acc_test(k = 5), pur_test |
| T3 | 순도: 데이터셋 × 표현 → pur_base, pur_valid, pur_test (S2·S3·대조는 5-shot run) |
| T4 | head 비교: 데이터셋 × k → acc_test(S0) 프로토타입 vs 로지스틱 회귀, seed 짝지은 차(평균 ± SE, 부호 일치/3) |
| T5 | 짝지은 차(seed 단위, 평균 ± SE, 부호 일치/3): acc_test(S1) − acc_test(S0), acc_test(S2) − acc_test(S0), acc_test(S2) − acc_test(S3), acc_base(S1) − acc_base(S0) |
| 그림 1 | 데이터셋별 막대: acc_test(k = 5)를 S0 → S1 ep0 → S1 ep200 → S2 → S3 → 대조 순으로. `reports/T21_ladder.png` |

## 5. 판정 (값만. 단위 (데이터셋, k) 6개, seed 평균 기준. 순도는 데이터셋 3개 기준)

| 판정 | 정의 |
|---|---|
| J1 | acc_test(S1 ep200) < acc_test(S0)인 수 (/6) |
| J2 | acc_test(S2) < acc_test(S0)인 수 (/6) |
| J3 | acc_base(S1 ep200) > acc_base(S0)인 수 (/6) |
| J4 | acc_test^LR(S0) < acc_test(S0)인 수 (/6), 그중 k = 1에서의 수 (/3) |
| J5 | pur_test(S1 ep200) < pur_test(S0)인 수 (/3); pur_test(S2) < pur_test(S0)인 수 (/3) |
| J6 (기록) | acc_test(S2) − acc_test(S3)의 짝지은 차 6칸 |

## 6. 사전 등록 (수정 금지, 어긋나면 날짜 있는 부기)

| 예측 | 내용 | 뜻(채팅 해석용, Agent는 판정값만 적음) |
|---|---|---|
| P107 | J1 ≥ 5/6 | 에피소드 없이 base 지도만 해도 test 구분력이 S0보다 떨어진다 |
| P108 | J2 ≥ 5/6 | EGNN 없이도 떨어진다 |
| P109 | J3 ≥ 5/6 | 대신 base는 좋아진다(맞교환) |
| P110 | J4 ≥ 4/6, k = 1에서 3/3 | 같은 특징 위에서 support로 맞춘 선형 head가 프로토타입보다 못하다 |
| P111 | J5 둘 다 3/3 | 순도에서도 같은 방향 |

## 7. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ha 선행 | `t19b` 존재. T09 `base`·`infonce` seed 0–2의 `emb_best.npy` 존재 | 시작하지 않음 |
| Hb 분할 일치 | seed별 base/valid/test 클래스 집합이 T09 run의 기록(`run.json` 또는 `split.json`)과 같다 | 멈추고 보고 |
| Hc 구현 | (1) S1 에폭 200의 base 학습 정확도 ≥ 80 % (3 데이터셋 모두). (2) `fewshot_acc`로 잰 S0(Â²X)·S3(원본)의 acc_test(k = 5, seed 0–2 평균)가 `reports/T12_summary.md` 표 2의 같은 칸과 ±0.5 %p 안(같은 에피소드·같은 probe이므로 일치해야 한다) | 멈추고 보고 |
| Hd γ = 0 회귀 | 18 run의 `run.json`에 `gamma: 0.0`. `test_acc_at_best_valid`가 기록됨 | 멈추고 보고 |
| He 변경 범위 | `git diff --stat <H0> <보고서 커밋>`의 파일이 `tools/ladder_t21.py`, `tools/run_t21.py`, `tools/analyze_t21.py`, `reports/`, `instructions/` 안에만 있다 | 멈추고 보고 |
| Hf 유한 | 모든 지표가 유한 | 해당 run 제외, 기록 |

- (2026-10-14 부기, Hc(1)) 80 % 기준은 5-way 에피소드 정확도 감각으로 적은 것이며 80·91-way 전체 분류에는 맞지 않는다. 대체 기준: 세 데이터셋 모두 (i) 최종 에폭 base 학습 정확도가 에폭 0보다 40 %p 이상 높고 (ii) 학습 손실이 에폭 0 대비 단조 감소(스냅샷 에폭 기준).

## 8. 보고

- `reports/T21_ladder.md`(§6 형식), `reports/T21_summary.md`, `reports/T21_ladder.png`.
- 커밋 → 태그 `t21` → push → 부기 커밋.
- 실행 시간(분), 디스크 사용량(`results/T21/`)을 적는다.
- `results/T21/gamma0/*/emb_best.npy`는 보고서 커밋·push·태그 뒤에만 지울 수 있다(CLAUDE.md §7). 이 지시서에서는 지우지 않는다.
