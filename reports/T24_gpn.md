# T24 — GPN 백본에서 지움 재현 (GPN 공식 구현 학습 18 run + 분석)

2026-10-18 · 브랜치 `gpn_backbone`(`hier_purity` `1d83ffc`에서 생성) · H0 `1f2265d` · 지시서 `instructions/T24_gpn_backbone.md` · 요약표 `reports/T24_summary.md` · 그림 `reports/T24_ladder.png`

## 1. 변경 파일과 위치

| 파일 | 내용 |
|---|---|
| `tools/gpn/models.py`, `tools/gpn/layers.py` (신규) | 업스트림 `kaize0409/GPN_Graph-Few-shot` `40ef635aa2a81f175e56c5588e1d50688f6aa85e`에서 바이트 그대로 복사 |
| `tools/gpn/UPSTREAM.md` (신규) | 업스트림 커밋 해시, 두 파일 sha256 |
| `tools/run_t24.py` (신규) | GPN 학습. `load_inputs`(특징·Â·degree는 `utils.load_data`, 분할·레이블은 T22 `base` split.json·labels.npy), `task`(업스트림 `task_generator`, `random` → 전용 생성기), `gpn_logp`·`train_step`·`evaluate`(업스트림 `train()`·`test()` 수식), `train_one`, `hc_check`, 드라이버(`tools/run_t21.py` 형식, 재시도 1회, 10분 초과 표시) |
| `tools/analyze_t24.py` (신규) | `tools/analyze_t21.py` 복사 후 수정. probe(코사인·유클리드), D, GPN 출력, 순도, 게이트 Hd·He·Hg, 표 T1–T6, J1–J6, P126–P130 계산값, 그림 |
| `reports/T24_gpn.md`, `reports/T24_summary.md`, `reports/T24_ladder.png` | 보고서·요약표·그림 |

- 난수: 학습 에피소드 `random.Random(18000 + seed)`, valid task 50 `Random(18100 + seed)`, test task 50 `Random(18200 + seed)`(학습 전 생성), 모델 초기화 `torch.manual_seed(18300 + seed)`. 다른 새 오프셋 없음. probe 에피소드는 기존 1000 + seed.
- `main.py`, `model.py`, `utils.py`, `embedder.py`, `argument.py`, `layers/` 변경 없음. GPN 코드는 `tools/gpn/` 안에만 있다. `tools/gpn/`을 `sys.path` 맨 앞에 넣어 업스트림 `from layers import GraphConvolution`이 `tools/gpn/layers.py`를 불러오게 했고, 불러온 파일 경로를 assert로 확인한다.
- `utils.load_data`는 GPN 학습·분석 프로세스 안에서 전역 `random`을 한 번 소비한다(valid 클래스 추출, 결과는 버림). 이후 전역 `random`은 쓰지 않는다.

## 2. 실행 명령과 출력

| 단계 | 명령 | 시간 | 출력 |
|---|---|---|---|
| Hc 합성 | `python tools/run_t24.py hc` (CPU) | < 1분 | `results/T24/hc_check.json` |
| 학습 | `python tools/run_t24.py` (GPU 0, 1, 2, 4, 5, 6, UUID, 동시 6) | 07:17:15–07:37:55 (20.6분, `total_sec=1237.0`) | 18/18 성공, 재시도 0. run당 284–503 s(10분 초과 0) |
| 분석 | `python tools/analyze_t24.py run <ds>` × 3 (GPU 0, 1, 2) | 07:37:51–07:39:11 (1.3분) | `results/T24/analysis_<ds>.json` |
| 보고 | `python tools/analyze_t24.py report` | 07:39:16–07:39:19 | `results/T24/analysis.json`, 요약표, 그림 |

- 디스크: `results/T24/` 80M(`emb_best.npy` 18개 40M, `ckpt_best.pt` 18개 37M). 리포 디스크 여유 9.0G(실행 후).

### 게이트

| 게이트 | 결과 | 충족 |
|---|---|---|
| Ha | `t21`, `t22` 있음. `results/T21/analysis_<ds>.json` 3/3. T22 `base` `split.json`·`labels.npy` 18/18. 같은 seed의 1-shot·5-shot `split.json`·`labels.npy` 9/9 같음 | 예 |
| Hb | 아래 diff 전문. 경로 문자열, `edges` 반환 외 차이: `return`의 `num_nodes` 추가, 주석 `# unsorted` 삭제. 사용자 판정(2026-10-18): 통과 | 예(사용자 판정) |
| Hc | sha256 `models.py` `7105c206…11b37cc`, `layers.py` `15806922…c1c4ee2` = 지시서·업스트림. 합성(노드 50, 특징 8, 2-way 1-shot, query 2, 3 스텝): 업스트림 `train()`·`test()` 원문을 실행해 비교한 손실·1스텝 후 파라미터·test log-prob·정확도 \|Δ\| 최대 0.0 | 예 |
| Hd | 18 run 모두 마지막 100 에피소드 학습 정확도 > 0.30(최소 Amazon_electronics 1-shot), best valid > valid@10. 미달 0 | 예 |
| He | S0 acc_base/valid/test vs T21 S0, 54개 \|Δ\| 최대 0.0 | 예 |
| Hf | `git diff --cached --stat 1f2265d`(커밋 전): `reports/T24_gpn.md`, `reports/T24_ladder.png`, `reports/T24_summary.md`, `tools/analyze_t24.py`, `tools/gpn/UPSTREAM.md`, `tools/gpn/layers.py`, `tools/gpn/models.py`, `tools/run_t24.py`. 범위 안. 커밋 후 값은 부기 | 예 |
| Hg | 지표 전부 유한, `emb_best.npy` NaN 없음, 제외 run 없음 | 예 |

**Hb diff 전문.** 업스트림 `utils.py` `load_data` vs TEG `utils.py:16–69`(세 데이터셋 분기, 한 단계 내어 씀) + `utils.py:214`(`return`). 업스트림 파일은 CRLF 줄끝·작은따옴표라 CR을 지우고 양쪽을 `black -l 200`으로 정규화한 뒤 `diff -u -B --strip-trailing-cr`.

```diff
@@ -1,18 +1,20 @@
 def load_data(dataset_source):
     n1s = []
     n2s = []
-    for line in open("few_shot_data/{}_network".format(dataset_source)):
+    for line in open(f"./dataset/{dataset_source}/{dataset_source}_network"):
         n1, n2 = line.strip().split("\t")
         n1s.append(int(n1))
         n2s.append(int(n2))
 
+    edges = torch.LongTensor([n1s, n2s])
+
     num_nodes = max(max(n1s), max(n2s)) + 1
     adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)), shape=(num_nodes, num_nodes))
 
-    data_train = sio.loadmat("few_shot_data/{}_train.mat".format(dataset_source))
+    data_train = sio.loadmat(f"./dataset/{dataset_source}/{dataset_source}_train.mat")
     train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))
 
-    data_test = sio.loadmat("few_shot_data/{}_test.mat".format(dataset_source))
+    data_test = sio.loadmat(f"./dataset/{dataset_source}/{dataset_source}_test.mat")
     class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))
 
     labels = np.zeros((num_nodes, 1))
@@ -26,7 +28,7 @@
     class_list = []
     for cla in labels:
         if cla[0] not in class_list:
-            class_list.append(cla[0])  # unsorted
+            class_list.append(cla[0])
 
     id_by_class = {}
     for i in class_list:
@@ -49,5 +51,4 @@
     class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])
 
     class_list_train = list(set(train_class).difference(set(class_list_valid)))
-
-    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class
+    return (edges, adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class, num_nodes)
```

- `valid_num_dic`: 업스트림 `{'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}`, TEG `{"Amazon_clothing": 17, "Amazon_electronics": 36, "dblp": 27}`(철자만 다름).
- 보조 함수 `normalize_adj`, `sparse_mx_to_torch_sparse_tensor`: 같은 정규화 후 차이 없음.

**데이터 파일 sha256 (TEG `dataset/<ds>/` / 업스트림 `few_shot_data/`)**

| 파일 (TEG / 업스트림) | sha256 | 비교 |
|---|---|---|
| `Amazon_clothing_network` / `Amazon_clothing_network` | `6362cf42e3d2eda602a6e237e1be0dfaa35efe7ab8eca0a9027a133c23954b2c` | 같음 |
| `Amazon_clothing_train.mat` / `Amazon_clothing_train.mat` | `06130fff67e7f71ffe8b9fb62a39820ab8b56cf0cb899bcb837ed8074534163f` | 같음 |
| `Amazon_clothing_test.mat` / `Amazon_clothing_test.mat` | `7656d0831b4a41f7aad1a9fc7cfb1434b920fa5ed5275a1e74aaefaea68eae81` | 같음 |
| `dblp_network` / `dblp_network` | `304a6feddd1a13ac78c7f40fdc72353911f85450a448ce7cf83366a380ade22f` | 같음 |
| `dblp_train.mat` / `dblp_train.mat` | `609b310d733f772c95e6269808e421e9b9d6863f51c99c775228822cf0f1bbf7` | 같음 |
| `dblp_test.mat` / `dblp_test.mat` | `24dc54f855e9c1b5febb2ebd0d1aaf0e782592b3ad8855f76516bc7d07994b6c` | 같음 |
| `Amazon_electronics_network` / `Amazon_eletronics_network` | `4aa2042f22bf88204552d770b9b8e4d65c6e19b188d44d216525a466ab23f4e1` | 같음 |
| `Amazon_electronics_train.mat` / `Amazon_eletronics_train.mat` | `af9e1040af226ef2d7bbef4291479164b2e1bfb80798e4e75d51c72b7f5fa502` | 같음 |
| `Amazon_electronics_test.mat` / `Amazon_eletronics_test.mat` | `89ffee2c5ec76926a43d4cedc0bc56f6937cae570a62b6b1ebf8a80596913160` | 같음 |

### 판정 (값만; 표는 `reports/T24_summary.md`)

| 판정 | 정의 | 값 |
|---|---|---|
| J1 | acc_test(GPN 출력) < acc_test(D) | 6/6 |
| J2 | acc_test(G) < acc_test(S0) | 6/6 |
| J3 | acc_base(G) > acc_base(S0) | 5/6 |
| J4 | acc_valid(G) < acc_valid(S0) | 6/6 |
| J5 | pur_test(G) < pur_test(S0) | 3/3 |
| J6 (기록) | acc_test(G) − acc_test(S3), %p, 평균 ± SE | clothing 1s −2.98 ± 1.03, 5s −3.42 ± 0.92; dblp 1s −0.30 ± 0.65, 5s −0.30 ± 0.16; electronics 1s −9.11 ± 1.02, 5s −5.13 ± 0.38 |
| J6 (기록) | 유클리드 probe의 J2 | 6/6 |

- J3에서 acc_base(G) ≤ acc_base(S0)인 칸: Amazon_electronics 1-shot(73.95 vs 80.54).
- J2에서 G − S0 차가 가장 작은 칸: dblp 1-shot(72.45 vs 72.50, seed 단위 −0.06 ± 0.79, 부호 일치 1/3).

### 사전 등록 (조건과 계산값만)

| 예측 | 조건 | 계산값 |
|---|---|---|
| P126 | J1 ≥ 4/6 | J1 = 6/6 |
| P127 | J2 ≥ 5/6 | J2 = 6/6 |
| P128 | J3 ≥ 5/6 | J3 = 5/6 |
| P129 | J4 ≥ 5/6 | J4 = 6/6 |
| P130 | J5 ≥ 2/3 | J5 = 3/3 |

## 3. 완료 조건 대비

| 조건 | 결과 | 충족 |
|---|---|---|
| H0 커밋(브랜치 `gpn_backbone`, CLAUDE.md·지시서) | `1f2265d` | 예 |
| 업스트림 clone, `tools/gpn/` 바이트 복사, `UPSTREAM.md` | 커밋·sha256 지시서 값과 같음 | 예 |
| 학습 18 run, 산출 파일 6종 | 18/18(`run.json`, `emb_best.npy`, `score_best.npy`, `ckpt_best.pt`, `curves.json`, `console.log`) | 예 |
| 표 T1–T6, 그림 | `reports/T24_summary.md`, `reports/T24_ladder.png` | 예 |
| J1–J6, P126–P130 계산값 | 위 표(판정하지 않음) | 예 |
| 게이트 Ha–Hg | 전부 충족(Hb는 사용자 판정) | 예 |
| 실행 시간·디스크 기록 | 2절 | 예 |
| `results/T24/` 유지 | 지우지 않음 | 예 |

## 4. 이상 징후

1. 평가 시점: 업스트림은 0부터 센 에피소드 번호가 10, 20, …, 990일 때(학습 스텝 11, 21, …, 991 뒤) 평가하고 마지막 스텝 뒤에는 평가하지 않는다. T24는 지시서대로 학습 스텝 10, 20, …, 1000을 마친 뒤 평가했다(100회). best 에피소드가 1000인 run: dblp 5-shot seed 1.
2. degree 1인 노드는 log degree = 0이라 support 가중치가 valuator 점수와 무관하게 sigmoid(0) = 0.5로 고정된다(업스트림 수식 그대로). degree 1 노드 수: Amazon_clothing 6,647 / 24,919, Amazon_electronics 27,591 / 42,318, dblp 0 / 40,672. degree 0 노드는 세 데이터셋 모두 0개.
3. `evaluate`는 평가 시점마다 임베딩·점수를 한 번 계산해 task 50개에 쓴다(업스트림 `test()`는 task마다 다시 계산). Hc에서 업스트림 `test()`와 log-prob \|Δ\| 0.0.
4. `ckpt_best.pt`에서 다시 계산한 임베딩·점수와 저장한 `emb_best.npy`·`score_best.npy`의 \|Δ\| 최대 1.9e-6(학습 GPU와 분석 GPU가 다름). `ckpt_best.pt`의 episode는 18/18 `run.json` best 에피소드와 같다. GPN 출력 정확도는 다시 계산한 값으로 냈다.
5. 1-shot에서 GPN 출력 정확도와 G의 유클리드 probe acc_test가 같다(clothing 66.74, dblp 69.09, electronics 53.12). 1-shot에서는 support 가중치가 정규화 후 1이다.
6. S0는 행 정규화 벡터라 1-shot에서 S0의 유클리드 probe 값이 코사인 probe 값과 같다(T5).
7. `load_data` 레이블과 T22 `labels.npy`는 18/18 같다.
8. best 에피소드 범위 180–1000(T6). Amazon_clothing 5-shot seed 0 = 180.
9. 표 T2의 "논문 보고값" 열은 지시서대로 비워 두었다.
10. 그림은 한글 글꼴이 없어 축·눈금 문자를 영어로 적었다(G = "GPN", S3 = "TEG"). y축은 패널별 최소값 − 8에서 시작한다(T21 그림과 같은 형식).
11. Hc 합성 점검은 `/tmp/gpn_upstream`의 업스트림 원문을 읽어 실행한다. 이 경로는 리포 밖이다.
