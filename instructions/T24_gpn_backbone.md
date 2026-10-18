# T24 — GPN 백본에서 지움 재현 (TEG 코드 밖의 두 번째 백본, 학습 18 run + 분석)

2026-10-16 · 작성: 채팅 Claude · 모드: CLAUDE.md §2(파일 단위 계획 → 승인 → 구현) · 브랜치 `gpn_backbone`(`hier_purity` HEAD `1d83ffc`에서 생성) · 태그 `t24` · 선행: `t21`, `t22` · 사전 등록 P126–P130

## 0. 목적과 검증 가능한 주장

**배경.** T06b·T21은 TEG 리포의 인코더(1층 GCN 64)와 그 변형(S1 base-CE MLP, S2 γ = 0)에서 "base 지도 학습이 학습 미사용 클래스(valid·test)의 구분력을 Â²X 아래로 떨어뜨린다"를 보였다. 교수님 의문(2026-10-15): "최초 GFSL인 GPN도 학습 없는 것보다 떨어지나?" 이 지시서는 **GPN 공식 구현(Ding et al., CIKM 2020, `kaize0409/GPN_Graph-Few-shot`, 커밋 `40ef635`)의 인코더·분류기를 그대로 학습**시키고, 그 임베딩을 T21과 **같은 probe·같은 에피소드**로 Â²X(S0)와 비교한다. TEG 코드 밖의 두 번째 백본이다.

**GPN 공식 구현의 사실(지시서 작성 시 확인).**
- 데이터 로더: `utils.load_data`가 TEG 리포 `utils.py:16–69`와 같은 파일(`<ds>_network`, `<ds>_train.mat`, `<ds>_test.mat`), 같은 Â = D^−½(A+I)D^−½, 같은 raw 특징, 같은 valid 클래스 추출(`random.sample(train_class, valid_num)`)을 쓴다. 따라서 **TEG 리포의 로더와 분할을 그대로 쓴다**(Hb로 확인).
- 인코더 `GPN_Encoder`: GCN(nfeat → 32) → ReLU → dropout 0.5 → GCN(32 → **16**). 출력 16차원.
- 분류기: `GPN_Valuator`(GCN 2층 + Linear → 노드 점수) · support 가중치 = normalize(sigmoid(log degree × score)) · prototype = 가중합 · **유클리드 거리** → log_softmax → NLL.
- 학습: Adam(lr 0.005, weight_decay 5e-4) 두 개(인코더, valuator), 에피소드 1000개, 5-way, shot k, query 20/클래스. 10 에피소드마다 valid 50 task·test 50 task 평가(출력만, 체크포인트 저장·선택 없음).

**주장.** 4절의 표와 5절의 판정값, 6절의 사전 등록 대조를 산출한다. 해석은 하지 않는다. 레이블 사용: base 레이블은 학습, valid 레이블은 best 에피소드 선택, test 레이블은 평가에만.

## 1. 착수

1. `git switch -c gpn_backbone` (`hier_purity` HEAD `1d83ffc`에서). `git status`에는 사용자가 수정한 `CLAUDE.md`(modified)와 이 지시서(untracked)만 보여야 한다. 다른 것이 보이면 멈추고 보고한다.
2. `CLAUDE.md`와 이 지시서를 H0 커밋으로 넣는다. Agent는 CLAUDE.md를 고치지 않는다.
3. GPN 공식 코드를 가져온다: `git clone --depth 1 https://github.com/kaize0409/GPN_Graph-Few-shot.git /tmp/gpn_upstream` (네트워크가 막히면 멈추고 보고). `models.py`, `layers.py`를 `tools/gpn/models.py`, `tools/gpn/layers.py`로 **바이트 그대로** 복사하고, `tools/gpn/UPSTREAM.md`에 업스트림 커밋 해시와 두 파일의 sha256을 적는다(지시서 작성 시 값: 커밋 `40ef635aa2a81f175e56c5588e1d50688f6aa85e`, `models.py` sha256 `7105c206…37cc`, `layers.py` `15806922…4ee2`; 다르면 멈추고 보고). `train_gpn.py`, `utils.py`는 복사하지 않고 참조만 한다.
4. 코드 작성 전 파일 단위 계획(함수 목록, 재사용하는 기존 구현의 `파일:줄`)을 보이고 승인을 기다린다(CLAUDE.md §2).

## 2. 학습 (`tools/run_t24.py`, 새 파일)

- 단위: 3 데이터셋 × 5-way {1, 5}-shot × seed 0–2 = **18 run**. 큐 순서는 T21과 같다(seed 우선, Clothing → dblp → Electronics, 1-shot 먼저). GPU 3번 제외, 동시 실행 ≤ 6, UUID 지정(`tools/run_t21.py`의 `GPUS` 재사용). 실패한 run은 한 번 재시도.
- **데이터·분할**: `utils.load_data(ds)`를 `seed_everything(seed)` 뒤에 호출하지 않는다. 대신 `results/T22/base/<ds>_5w<k>s/seed<s>/split.json`·`labels.npy`를 읽어 `class_list_train/valid/test`와 레이블로 쓴다(T21·T22와 같은 분할). 특징·Â·degree는 `utils.load_data`에서 받는다(분할 항목은 버린다).
- **모델**: `tools/gpn/models.py`의 `GPN_Encoder(nfeat, nhid=16, dropout=0.5)`, `GPN_Valuator(nfeat, nhid=16, dropout=0.5)`. 초기화 난수 `torch.manual_seed(18300 + seed)`.
- **학습 루프**: 업스트림 `train_gpn.py`의 `train()`·`test()`를 **수식 그대로** 옮긴다(가중 prototype, 유클리드, log_softmax, NLL; Adam 두 개, lr 0.005, wd 5e-4). 에피소드 1000개. 학습 에피소드 생성은 업스트림 `task_generator`와 같은 규칙(클래스 5개 무작위, 클래스마다 k + 20개 무작위)이되 전용 생성기 `random.Random(18000 + seed)`를 쓴다.
- **선택(새로 추가, 업스트림에 없음)**: 10 에피소드마다(에피소드 10, 20, …, 1000) eval 모드로 **GPN 원 프로토콜 valid task 50개**(`task_generator` 규칙, query 20, `random.Random(18100 + seed)`로 학습 전에 한 번 생성)의 정확도 평균을 잰다. 최고 에피소드 = best(동점은 나중). 그 시점의 인코더 출력(전체 노드, eval 모드, dropout 꺼짐)을 `emb_best.npy`([N, 16], float32), valuator 점수를 `score_best.npy`([N])로 저장한다. 같은 시점의 인코더·valuator state_dict를 `ckpt_best.pt`로 저장한다.
- **기록 전용**: GPN 원 프로토콜 test task 50개(`random.Random(18200 + seed)`)의 정확도·F1을 10 에피소드마다 함께 재서 곡선으로 남긴다(업스트림 출력과 같은 양). 학습 에피소드 정확도의 100-에피소드 이동 평균도 남긴다.
- 출력: `results/T24/gpn/<ds>_5w<k>s/seed<s>/{run.json, emb_best.npy, score_best.npy, ckpt_best.pt, curves.json, console.log}`. `run.json`에 best 에피소드, best valid 정확도, 그 시점 GPN test(50 task) 정확도, 학습 시간, split.json 경로와 sha256을 적는다.
- 코드 변경 없음: `main.py`, `model.py`, `utils.py`, `embedder.py`, `argument.py`, `layers/`는 건드리지 않는다. GPN 코드는 `tools/gpn/` 안에만 둔다.

## 3. 분석 (`tools/analyze_t24.py`, `tools/analyze_t21.py` 복사 후 수정)

**표현.** S0 = 행 정규화 Â²X(`fusion_baseline.diffusion_features_hops(ds, 2)`), **G** = `results/T24/gpn/.../emb_best.npy`(probe shot과 같은 shot의 run), 기록용 S3 = T21 분석 JSON의 S3(TEG 원본) 값과 S1 ep200·S2 값(재계산하지 않고 `results/T21/analysis_<ds>.json`에서 읽는다).

**probe(T21과 동일).** `analyze_t06b.episodes` + `fewshot_acc`(코사인 프로토타입, 500 에피소드, `random.Random(1000 + seed)`, 그룹 base → valid → test 순으로 한 번 뽑아 모든 표현이 공유, 벡터는 사전 정규화 없음). 추가로 **유클리드 프로토타입** probe(`analyze_t06b.probe_acc(..., metric="euc")`)를 G와 S0에 대해 기록한다(GPN의 거리가 유클리드이므로).

**출력 수준(새로 추가).** 같은 500 test 에피소드에서,
- D: `fusion_baseline.view_d_scores(S0, sup, qry, k)` argmax 정확도(= T06b·T22의 D와 같은 규칙).
- GPN 출력: `ckpt_best.pt`를 불러 업스트림 `test()`와 같은 수식(valuator 가중 prototype, 유클리드)으로 같은 에피소드를 분류한 정확도. eval 모드.
- 각 에피소드의 짝지은 차 GPN출력 − D(평균 ± SE, 양수 비율).

**순도.** `analyze_t14.knn_purity(V, nodes_g, y)`(top-10 코사인), 그룹 base·valid·test, V ∈ {S0, G(5-shot run)}.

**곡선.** `curves.json`에서 best 에피소드, valid 곡선 최고값, 에피소드 10에서의 valid 값(상승 확인용), GPN 원 프로토콜 test(50 task) 정확도.

## 4. 산출 표 (`reports/T24_summary.md`)

| 표 | 내용 |
|---|---|
| T1 | 사다리: (데이터셋, k) × {S0, **G**, S1 ep200, S2, S3} × {acc_base, acc_valid, acc_test} (seed 평균 ± sd). S1·S2·S3는 T21 값 전재(출처 표기) |
| T2 | 출력 수준: (데이터셋, k) × {D, GPN 출력, GPN 원 프로토콜 50-task test, 논문 보고값(희태가 채움, 비워 둠)} |
| T3 | 순도: 데이터셋 × {S0, G} × {pur_base, pur_valid, pur_test} |
| T4 | 짝지은 차(seed 단위, 평균 ± SE, 부호 일치/3): test G − S0, valid G − S0, base G − S0, test G − S3; 에피소드 단위: GPN출력 − D |
| T5 | 유클리드 probe: (데이터셋, k) × {S0, G} × 그룹 (기록) |
| T6 | 학습 곡선 요약: best 에피소드, valid 최고, valid@10, 마지막 100 에피소드 학습 정확도 평균 |

## 5. 판정 (값만. 단위 (데이터셋, k) 6개, seed 평균, 코사인 probe)

| 판정 | 정의 |
|---|---|
| J1 | 출력 수준: acc_test(GPN 출력) < acc_test(D)인 수 (/6) |
| J2 | acc_test(G) < acc_test(S0)인 수 (/6) |
| J3 | acc_base(G) > acc_base(S0)인 수 (/6) |
| J4 | acc_valid(G) < acc_valid(S0)인 수 (/6) |
| J5 | pur_test(G) < pur_test(S0)인 수 (/3) |
| J6 (기록) | acc_test(G) − acc_test(S3)의 짝지은 차 6칸; 유클리드 probe에서의 J2 |

## 6. 사전 등록 (수정 금지, 어긋나면 날짜 있는 부기)

| 예측 | 내용 | 근거(채팅 해석용) |
|---|---|---|
| P126 | J1 ≥ 4/6 | T06b·T22에서 D ≥ TEG 5/6. GPN 분류기는 TEG보다 단순(EGNN 없음)하고 16차원이라 D를 넘기 어렵다고 본다. 단 GPN 분류기는 유클리드·가중 prototype이라 D(코사인)와 규칙이 다르므로 6/6까지는 걸지 않는다 |
| P127 | J2 ≥ 5/6 | T21에서 S1(비에피소드 MLP) 5/6, S2(γ = 0) 6/6. dblp 1-shot이 예외일 수 있다 |
| P128 | J3 ≥ 5/6 | T21 J3 = 6/6 |
| P129 | J4 ≥ 5/6 | T06b: valid도 test처럼 손해(6/6) |
| P130 | J5 ≥ 2/3 | T21 J5 = S1 2/3, S2 3/3 |

## 7. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ha 선행 | `t21`, `t22` 존재. `results/T21/analysis_<ds>.json` 3개, `results/T22/base/<ds>_5w<k>s/seed<s>/{split.json, labels.npy}` 18개 존재. 같은 seed의 1-shot·5-shot run의 split.json이 같다 | 시작하지 않음 |
| Hb 로더 동일 | 업스트림 `utils.py`의 `load_data`와 TEG `utils.py:16–69`의 `load_data`(세 데이터셋 분기)를 `diff`해 경로 문자열·데이터셋 이름 철자(`eletronics`/`electronics`)·`edges` 반환 외에 차이가 없다. diff 전문을 보고서에 붙인다 | 멈추고 보고 |
| Hc 모델 동일 | `tools/gpn/models.py`, `layers.py`의 sha256이 업스트림과 같다. `run_t24.py`의 학습 스텝(prototype 가중, 거리, 손실)이 업스트림 `train()`과 같음을 합성 입력(노드 50, 특징 8, way 2, shot 1, query 2)에서 손실값 \|Δ\| ≤ 1e-6으로 확인 | 멈추고 보고 |
| Hd 학습 성립 | run마다 마지막 100 에피소드 학습 정확도 평균 > 0.30(5-way 우연 0.20). best valid 정확도 > valid@에피소드 10 | 미달 run은 표에 표시하고 계속. 미달이 6 run 이상이면 멈추고 보고 |
| He probe 동일 | S0의 acc_base/valid/test가 T21 분석 JSON의 S0 값과 (ds, k, seed)마다 \|Δ\| ≤ 1e-6 | 멈추고 보고 |
| Hf 변경 범위 | `git diff --stat <H0> <보고서 커밋>`의 파일이 `tools/gpn/`, `tools/run_t24.py`, `tools/analyze_t24.py`, `reports/`, `instructions/` 안에만 있다 | 멈추고 보고 |
| Hg 유한 | 모든 지표가 유한, `emb_best.npy`에 NaN 없음 | 해당 run 제외, 기록 |

## 8. 보고

- `reports/T24_gpn.md`(§6 형식: 한 것 / 게이트 / 판정값 / 사전 등록 대조 / 이상 징후 / 변경·생성 파일), `reports/T24_summary.md`, `reports/T24_ladder.png`(T21 그림과 같은 형식에 G 막대 추가).
- 커밋 → 태그 `t24` → push → 부기 커밋.
- 실행 시간(분), `results/T24/` 디스크 사용량을 적는다. `results/T24/`는 지우지 않는다.
- 해석은 쓰지 않는다(CLAUDE.md §6 금지어).
