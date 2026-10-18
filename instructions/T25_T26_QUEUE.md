# T25 → T26 큐 — 차원 대조 (T25: PCA, 학습 없음 · T26: ogbn-arxiv에서 D vs GPN)

2026-10-18 · 작성: 채팅 Claude · 브랜치 `dim_control`(`gpn_backbone` HEAD `95e3647`에서 생성) · 태그 `t25`, `t26` · 선행: `t21`, `t22`, `t24` · 사전 등록 T25 P131–P134, T26 P135–P138

## 0. 이 큐의 규칙 (작업·커밋 단위 분리)

- **T25와 T26은 별개의 작업 단위다.** 코드 파일, 결과 디렉터리, 보고서, 커밋, 태그를 모두 따로 둔다. 한 커밋에 두 단위의 파일을 섞지 않는다.

  | 단위 | 새 코드 | 결과 | 보고서 | 태그 |
  |---|---|---|---|---|
  | T25 | `tools/analyze_t25.py` | `results/T25/` | `reports/T25_pca.md`, `reports/T25_summary.md` | `t25` |
  | T26 | `tools/run_t26.py`, `tools/analyze_t26.py` | `results/T26/` | `reports/T26_arxiv.md`, `reports/T26_summary.md`, `reports/T26_ladder.png` | `t26` |

- 순서: H0 커밋(CLAUDE.md + 이 지시서) → **T25** 구현·실행·보고서 커밋 → 태그 `t25` → push → **T26** 구현·실행·보고서 커밋 → 태그 `t26` → push → 큐 요약.
- **멈춤은 두 번만 한다.** (1) H0 커밋 뒤, 두 단위의 파일 단위 계획(함수 목록, 재사용하는 기존 구현의 `파일:줄`)을 한꺼번에 보이고 승인을 기다린다. (2) 큐 끝. 승인 뒤에는 T25를 끝내고 멈추지 않고 T26으로 넘어간다.
- 게이트 실패: 그 단위는 보고서를 쓰고 커밋하되 태그를 달지 않는다. **T25와 T26은 서로의 결과를 쓰지 않으므로**, T25 게이트가 실패해도 T26은 진행한다. T26 게이트 중 "멈추고 보고"로 적힌 것이 실패하면 거기서 큐를 멈춘다.
- 학습 코드(`main.py`, `model.py`, `utils.py`, `embedder.py`, `argument.py`, `layers/`)와 기존 `tools/*.py`, `tools/gpn/`는 고치지 않고 import만 한다.
- 레이블은 지표 계산(probe, 순도)과 GPN 학습(base)·선택(valid)에만 쓴다. PCA는 레이블 없이 전체 노드에 적합한다(CLAUDE.md §3).
- 해석은 쓰지 않는다. 판정값과 사전 등록은 계산값만 적는다.

---

## T25 — Â²X를 PCA로 같은 차원까지 줄여도 학습한 표현보다 나은가 (학습 없음)

### 1. 목적

T21·T24에서 학습한 임베딩(TEG 64차원, GPN 16차원)과 비교한 D의 Â²X는 수천 차원이다. 학습 없이 Â²X를 같은 차원으로 줄인 표현과 비교해, 학습한 쪽의 손실이 "차원 축소" 때문인지 "base 지도 학습" 때문인지를 나눠 본다.

### 2. 표현 (세 데이터셋, seed 0–2)

| 기호 | 정의 | 출처 |
|---|---|---|
| S0 | 행 정규화 Â²X | `fusion_baseline.diffusion_features_hops(ds, 2)` |
| P64 | S0의 PCA 64차원 점수 | 새로 계산 |
| P16 | S0의 PCA 16차원 점수 | 새로 계산 |
| S3 | TEG 원본 GCN 임베딩(64) | `results/T21/analysis_<ds>.json`의 S3 값(재계산 없음) |
| G | GPN 임베딩(16) | `results/T24/analysis_<ds>.json`의 G 값(재계산 없음) |

- PCA: 전체 노드에 적합, 평균 중심화, `sklearn.decomposition.PCA(n_components=64, svd_solver="randomized", random_state=18500)`를 한 번 적합하고 P16은 그 앞 16개 성분을 쓴다. 분할과 무관하므로 데이터셋마다 한 번 계산한다. 설명 분산 비율(16, 64 누적)을 기록한다.
- probe는 T21·T24와 동일: `analyze_t06b.episodes` + `fewshot_acc`(코사인 프로토타입, 500 에피소드, `random.Random(1000 + seed)`, 그룹 base → valid → test 순, 모든 표현이 공유, 사전 정규화 없음). 분할은 T22 `base`의 `split.json`·`labels.npy`.
- 순도: `analyze_t14.knn_purity`(top-10 코사인), P16·P64, 그룹 base·valid·test.
- 유클리드 프로토타입 probe(`probe_acc(..., metric="euc")`)도 P16·P64에 대해 기록한다.

### 3. 산출 표 (`reports/T25_summary.md`)

| 표 | 내용 |
|---|---|
| T1 | (ds, k) × {S0, P64, S3, P16, G} × {acc_base, acc_valid, acc_test} (seed 평균 ± sd) |
| T2 | seed 단위 짝지은 차(평균 ± SE, 부호 일치/3): test G − P16, valid G − P16, base G − P16, test S3 − P64, valid S3 − P64, base S3 − P64, test P16 − S0, test P64 − S0 |
| T3 | 순도: ds × {S0, P64, P16} × 그룹 (S3·G는 T21·T24 값 전재) |
| T4 | 설명 분산 비율(16, 64), 유클리드 probe(기록) |

### 4. 판정 (값만, 단위 (ds, k) 6개, seed 평균)

| 판정 | 정의 |
|---|---|
| K1 | acc_test(G) < acc_test(P16)인 수 |
| K2 | acc_base(G) > acc_base(P16)인 수 |
| K3 | acc_test(S3) < acc_test(P64)인 수 |
| K4 | acc_base(S3) > acc_base(P64)인 수 |
| K5 (기록) | acc_valid(G) < acc_valid(P16)인 수; acc_valid(S3) < acc_valid(P64)인 수 |
| K6 (기록) | acc_test(P16) < acc_test(S0)인 수; acc_test(P64) < acc_test(S0)인 수 |

### 5. 사전 등록 (수정 금지)

| 예측 | 내용 | 근거(채팅 해석용) |
|---|---|---|
| P131 | K1 ≥ 4/6 | GPN test − S0가 −0.1 ~ −13.6. 16차원 PCA의 손실이 그보다 작다고 본다(확신 낮음) |
| P132 | K2 ≥ 5/6 | GPN base > S0 5/6. PCA는 S0보다 base가 높을 이유가 없다 |
| P133 | K3 ≤ 3/6 | T06b: 64차원 PCA(ÂX) 대비 test에서 분명히 진 곳 1/6 |
| P134 | K4 ≥ 5/6 | T06b: 학습 이득이 base에 쏠림 6/6 |

### 6. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ia 선행 | `t21`, `t24` 존재. `results/T21/analysis_<ds>.json`, `results/T24/analysis_<ds>.json` 각 3개, T22 `base` split 18개 | T25 시작하지 않음, T26로 |
| Ib probe 동일 | S0의 acc_base/valid/test가 T21 S0와 54개 모두 \|Δ\| ≤ 1e-6 | 보고서 쓰고 태그 없음, T26로 |
| Ic PCA | 성분 행렬의 직교성 \|CᵀC − I\|max ≤ 1e-4. 설명 분산 비율이 유한하고 증가 | 같음 |
| Id 변경 범위 | `git diff --stat <H0> <T25 보고서 커밋>` = `tools/analyze_t25.py`, `reports/T25_*`뿐 | 같음 |
| Ie 유한 | 모든 지표 유한 | 해당 칸 제외, 기록 |

---

## T26 — ogbn-arxiv(입력 128차원)에서 D vs GPN (TEG는 이번에 하지 않음)

### 1. 목적

입력 차원이 작아 압축 손실이 적은 데이터셋에서도 "학습한 GPN이 학습 없는 D보다 test에서 못하고 base에서는 낫다"가 성립하는지 본다. ogbn-arxiv 특징은 128차원 조밀 벡터(제목·초록 단어 임베딩 평균)다.

### 2. 사전 확인 (구현 전, 결과를 계획 보고에 포함)

1. `python -c "import ogb; print(ogb.__version__)"`, `ls -la dataset/ogbn_arxiv/`, `cls_split.pkl` 존재.
2. `utils.load_data("ogbn-arxiv")`가 끝까지 도는지(OGB 데이터가 없으면 OGB가 내려받는다; 실패하면 **멈추고 보고**).
3. 기록: 노드 수, 엣지 수, `edge_index`가 대칭인지(역방향 엣지 비율), 클래스 분할 크기(train/valid/test 클래스 수), 그룹별 최소 클래스 크기, `degree`(load_data가 돌려주는 값)가 0인 노드 수.
4. **그룹별 최소 클래스 크기가 25(= 5-shot + query 20) 미만이면 멈추고 보고**한다.

### 3. 표현과 그래프 연산자

- `load_data("ogbn-arxiv")`의 `adj`는 행 정규화 D⁻¹(A+I)이고(`utils.py`의 ogbn-arxiv 분기, `normalize`), 엣지는 원 방향 그대로다. GPN은 이 `adj`를 그대로 쓴다(TEG의 arxiv 설정과 같음).
- **S0(주)** = 행 L2 정규화( adj² · X ), 즉 GPN이 받는 것과 같은 연산자로 2-hop 전파한 Â²X. D = S0 위 support 평균 + 코사인.
- **S0_sym(기록)** = `fusion_baseline.diffusion_features_hops("ogbn-arxiv", 2)`(대칭 gcn_norm, Amazon·dblp의 D와 같은 정의). 이 함수가 arxiv에서 동작하지 않으면 "미계산"으로 적고 계속한다.
- **P16(기록)** = S0의 PCA 16차원(T25와 같은 설정, `random_state=18500`).
- **G** = GPN best-valid 임베딩(16차원).

### 4. GPN 학습 (`tools/run_t26.py`, `tools/run_t24.py`의 함수 import)

- 단위: {1, 5}-shot × seed 0–2 = **6 run**. GPU 3번 제외, 동시 ≤ 6, UUID 지정.
- 분할: `cls_split.pkl`의 train/valid/test 클래스(고정, seed와 무관). 레이블은 `load_data`의 `labels`.
- 모델·학습·선택: T24와 완전히 같다(`run_t24`의 `task`, `gpn_logp`, `train_step`, `evaluate`를 import; nhid 16, dropout 0.5, Adam lr 0.005·wd 5e-4 두 개, 에피소드 1000, query 20, 10 에피소드마다 valid 50 task로 best 선택, test 50 task는 기록만). 난수 오프셋도 T24와 같다(18000 학습 에피소드, 18100 valid task, 18200 test task, 18300 초기화; CLAUDE.md에 "T24·T26 GPN"으로 등록).
- **degree 0 노드**: GPN 수식은 log(degree)를 쓰므로 degree 0이면 −inf가 된다. degree 0 노드가 있으면 그 노드에 한해 degree를 1로 둔다(log 1 = 0 → support 가중치 0.5, T24의 degree 1 노드와 같은 처리). 해당 노드 수를 보고서에 적는다. 이 외의 수식 변경은 없다.
- **시간 점검**: 첫 run(5-shot seed 0)을 먼저 단독으로 돌려 시간과 GPU 메모리를 잰다. run당 20분을 넘을 것으로 보이거나 메모리가 부족하면 **멈추고 보고**한다. 문제가 없으면 나머지 5 run을 병렬로 돌린다.
- 출력: `results/T26/gpn/ogbn-arxiv_5w<k>s/seed<s>/{run.json, emb_best.npy, score_best.npy, ckpt_best.pt, curves.json, console.log}`.

### 5. 분석 (`tools/analyze_t26.py`, `tools/analyze_t24.py` 복사 후 수정)

- probe: T24와 같다(코사인 프로토타입, 500 에피소드, `random.Random(1000 + seed)`, 그룹 base → valid → test). 표현: S0, S0_sym, P16, G.
- 출력 수준: 같은 500 test 에피소드에서 D(S0 위 코사인 프로토타입), D_sym(기록), GPN 출력(`ckpt_best.pt` + 업스트림 `test()` 수식).
- 순도: top-10 코사인, 그룹별, S0와 G(5-shot run).
- 참고값: GPN 논문에는 arxiv 값이 없다. TENT 논문 표의 GPN(ogbn-arxiv 5w5s 53.06 ± 3.13)을 T2 표의 "참고: 문헌값" 열에 적는다(분할·프로토콜이 다를 수 있음을 함께 적는다).

### 6. 산출 표 (`reports/T26_summary.md`)

| 표 | 내용 |
|---|---|
| T0 | 사전 확인 사실(2절 3) |
| T1 | k × {S0, S0_sym, P16, G} × {acc_base, acc_valid, acc_test} (seed 평균 ± sd) |
| T2 | 출력 수준: k × {D, D_sym, GPN 출력, GPN 원 프로토콜 50-task test, 참고: 문헌값} |
| T3 | 짝지은 차: seed 단위 test/valid/base G − S0, test G − P16; 에피소드 단위 GPN출력 − D(1500개, 평균 ± SE, 양수 비율) |
| T4 | 순도: {S0, G} × 그룹 |
| T5 | 학습 곡선 요약(T24 T6과 같은 형식), run당 시간·GPU 메모리 |
| 그림 | `reports/T26_ladder.png`: k = 5에서 S0, P16, G의 acc_base·acc_test |

### 7. 판정 (값만, 단위 k ∈ {1, 5} 2개, seed 평균)

| 판정 | 정의 |
|---|---|
| L1 | acc_test(GPN 출력) < acc_test(D)인 수 (/2) |
| L2 | acc_test(G) < acc_test(S0)인 수 (/2) |
| L3 | acc_base(G) > acc_base(S0)인 수 (/2) |
| L4 | acc_valid(G) < acc_valid(S0)인 수 (/2) |
| L5 (기록) | acc_test(G) < acc_test(P16)인 수; pur_test(G) < pur_test(S0); D_sym 기준 L1 |

### 8. 사전 등록 (수정 금지)

| 예측 | 내용 | 근거(채팅 해석용) |
|---|---|---|
| P135 | L3 = 2/2 | 지금까지 모든 백본·데이터셋에서 base 이득이 나타났다 |
| P136 | L2 ≥ 1/2 | 압축 손실이 작아도 base 지도는 학습 미사용 클래스 정보를 잃는다고 본다(확신 낮음: 입력이 이미 학습된 저차원 임베딩이고 그래프 동질성이 낮다) |
| P137 | L4 ≥ 1/2 | 같음 |
| P138 | L1 ≤ 1/2 | 저차원·저동질성에서 D의 출력이 약해, GPN 출력이 적어도 한 shot에서 D 이상일 것으로 본다(확신 낮음) |

### 9. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ja 데이터 | 2절의 1·2·4 충족 | 멈추고 보고 |
| Jb 모델 동일 | `run_t24`의 함수를 import해 쓰고, T24의 Hc 합성 점검을 그대로 다시 실행해 \|Δ\| = 0 | 멈추고 보고 |
| Jc 학습 성립 | run마다 마지막 100 에피소드 학습 정확도 > 0.30, best valid > valid@10 | 미달 run 표시, 6개 중 3개 이상 미달이면 멈추고 보고 |
| Jd 시간·메모리 | 4절 시간 점검 | 멈추고 보고 |
| Je 변경 범위 | `git diff --stat <T25 마지막 커밋> <T26 보고서 커밋>` = `tools/run_t26.py`, `tools/analyze_t26.py`, `reports/T26_*`뿐 | 보고서 쓰고 태그 없음 |
| Jf 유한 | 지표 유한, `emb_best.npy`에 NaN 없음 | 해당 run 제외, 기록 |

---

## 10. 큐 끝 보고 (`reports/T25_T26_queue_summary.md`)

- 단위별: 태그, 커밋 해시, 실행 시간, 게이트, 판정값, 사전 등록 계산값(한 줄씩).
- T25 T1 표와 T26 T1·T2 표를 그대로 옮긴다.
- 디스크 사용량(`results/T25/`, `results/T26/`), push 결과(force 없음).
- `results/T25/`, `results/T26/`는 지우지 않는다.
