# T28 → T29 큐 — 학습 없는 기준선 (T28: 순수 X 프로토타입 · T29: SGC 세 변형)

2026-10-21 · 작성: 채팅 Claude · 브랜치 `trainfree_baselines`(`dim_control` HEAD `d2f0f22`에서 생성) · 태그 `t28`, `t29` · 선행: `t21`, `t22` · 사전 등록 T28 P144–P146, T29 P147–P151 · 참조 스크립트 `instructions/ref/pilot_sgc_vs_d.py`(실행하지 않음, 함수 이식용)

> **개정(2026-10-21, 0단계 보고 반영, 커밋 전).** T22 `base`의 `split.json`·`labels.npy`가 seed 0–2에만 있다. 그래서 T28 게이트 Ga에서 split 요구를 뺐고, T29의 분할은 2.1의 방식으로 복원한다(게이트 Gs). 0단계는 완료된 것으로 본다.

## 0. 현황 파악 (가장 먼저, 코드·파일 수정 없음)

이 단계에서는 **어떤 파일도 수정하거나 커밋하지 않는다.** CLAUDE.md도 고치지 않는다(사용자가 이 보고를 본 뒤 직접 고친다).

1. `git status`, `git branch -a`, `git log --oneline -15 --decorate`, `git tag --list 't2*'`, `git ls-remote --tags origin | tail -5`. 현재 브랜치와 HEAD, 원격과의 차이(ahead/behind), 커밋되지 않은 파일.
2. `CLAUDE.md` 전체를 읽고, **실제 리포 상태와 어긋나는 서술**을 목록으로 적는다. 최소한 다음을 대조한다.
   - §1 브랜치 표의 `dim_control` 행 (T27 완료 여부, HEAD)
   - §1 태그 목록과 "다음 태그는 …" 문장 (실제 태그와 비교)
   - §1 번호 문단의 "다음 사전 등록 번호" (T27 지시서·보고서의 마지막 P 번호와 비교)
   - §8 오프셋 등록부의 미배정 구간 (이 큐가 쓸 18600–18900이 비어 있는지)
3. 선행 결과의 존재 확인 (개수만 센다).
   - `results/T22/base/<ds>_5w<k>s/seed<s>/` 의 `fixed_eval_logits.npz`, `split.json`, `labels.npy`: 3 데이터셋 × {1, 5} × seed 0–9 = 60개씩
   - `results/T22/infonce_all_l<λ>/…/fixed_eval_logits.npz`: 선택 λ(아래 표) 기준 60개

     | 데이터셋 | 1-shot | 5-shot |
     |---|---|---|
     | Amazon_clothing | 10 | 10 |
     | dblp | 10 | 10 |
     | Amazon_electronics | 100 | 30 |

     (`reports/T22_summary.md` 표 T1 "선택 λ (nb / all)"의 all 값. 일치하는지 확인)
   - `results/T21/analysis_<ds>.json` 3개
4. 환경: `conda` 환경 `python310`에서 `import sklearn, torch; print(sklearn.__version__, torch.__version__)`, 리포 디스크 여유(`df -h .`), 사용 가능한 GPU(`nvidia-smi`).
5. **보고하고 멈춘다.** 보고 형식: (a) git 상태 요약, (b) CLAUDE.md 불일치 목록(항목 / CLAUDE.md 서술 / 실제), (c) 선행 결과 개수, (d) 환경. 해석·제안은 쓰지 않는다.

## 1. 이 큐의 규칙 (0단계 이후)

- **T28과 T29는 별개의 작업 단위다.** 코드, 결과 디렉터리, 보고서, 커밋, 태그를 모두 따로 둔다.

  | 단위 | 새 코드 | 결과 | 보고서 | 태그 |
  |---|---|---|---|---|
  | T28 | `tools/analyze_t28.py` | `results/T28/` | `reports/T28_xproto.md`, `reports/T28_summary.md` | `t28` |
  | T29 | `tools/run_t29.py`, `tools/analyze_t29.py` | `results/T29/` | `reports/T29_sgc.md`, `reports/T29_summary.md` | `t29` |

- 순서: (사용자의 CLAUDE.md 수정 뒤) 브랜치 생성 → **H0 커밋**(사용자가 수정한 CLAUDE.md + 이 지시서 + `instructions/ref/pilot_sgc_vs_d.py`, 내용은 고치지 않음) → **멈춤 1**: 두 단위의 파일 단위 계획(함수 목록, 재사용하는 기존 구현의 `파일:줄`)을 한꺼번에 보이고 승인을 기다린다 → T28 구현·실행·보고서 커밋 → 태그 `t28` → push → T29 구현·실행·보고서 커밋 → 태그 `t29` → push → **멈춤 2**: 큐 요약 보고.
- 승인 뒤에는 T28을 끝내고 멈추지 않고 T29로 넘어간다.
- 게이트 실패: 그 단위는 보고서를 쓰고 커밋하되 태그를 달지 않는다. "멈추고 보고"로 적힌 게이트가 실패하면 큐를 멈춘다.
- 학습 코드(`main.py`, `model.py`, `utils.py`, `embedder.py`, `argument.py`, `layers/`)와 기존 `tools/*.py`는 고치지 않고 import만 한다.
- **TEG·GPN 학습은 없다.** T29의 선형층(SGC head) 학습만 있다. T22 결과는 재사용만 한다.
- 레이블은 지표 계산과 T29의 SGC head 학습(base 노드 레이블, support 레이블)에만 쓴다. valid·test 노드 레이블은 학습에 쓰지 않는다(CLAUDE.md §3).
- 해석은 쓰지 않는다. 판정과 사전 등록은 계산값만 적는다.

## 2. 공통 평가 설정 (T28·T29)

- **에피소드**: T22 `base` run의 `fixed_eval_logits.npz`에서 `mode == 1`(고정 test 200개/seed, `Random(9000 + seed)`), `ep_idx` 순. `support`, `query`, `query_y`를 그대로 쓴다. 에피소드를 다시 뽑지 않는다. 5-way, query 클래스당 5, {1, 5}-shot, seed 0–9 → (ds, k)마다 2,000 에피소드.
- **분할**: T28 본표(T1·T2)는 분할이 필요 없다(에피소드에 support·query·`query_y`가 있다). 분할은 T29의 SGC-B 사전학습(base 클래스)에만 쓴다. `split.json`·`labels.npy`는 seed 0–2에만 있으므로(0단계 보고) 2.1의 방식으로 복원한다.

### 2.1 분할 복원 (T29용, seed 0–9)

- **labels**: `results/T22/base/<ds>_5w1s/seed0/labels.npy`를 쓴다. 있는 18개 `labels.npy`가 데이터셋마다 서로 같은지 확인한다.
- **test 클래스**: seed 0의 `split.json` `class_list_test`. seed 0–2의 18개 `split.json`에서 같은지 확인한다.
- **valid 클래스** (ds, seed): 그 seed의 1-shot·5-shot `base` run 두 개의 `eval_logits.npz`와 `fixed_eval_logits.npz`에서 `mode == 0`인 에피소드의 `classes`를 모두 모은 합집합.
- **base 클래스** = 전체 클래스(labels의 고유값) − valid − test. **정렬한 리스트**로 쓴다.
- 이 복원 분할을 `results/T29/splits.json`에 저장한다.
- 근거: valid 클래스 추출은 `utils.py:67`(`random.sample(train_class, valid_num_dic[...])`)이다. valid 에피소드가 epoch마다 50개씩 반복되므로 모든 valid 클래스가 등장하는지를 게이트 Gs로 확인한다.
- **코사인 프로토타입**: `fusion_baseline.cosine_probe_acc(V, support, query, k)` (support 평균 → L2 정규화 → 코사인 최대).
- **재사용 열**: TEG = `base`의 고정 test 출력, C_all = 선택 λ(0단계 표) `infonce_all`의 고정 test 출력. 둘 다 `analyze_t22.fixed_teg(npz)`로 계산한다.
- **짝지은 차**: 에피소드 단위(2,000개) 평균 ± SE, 양수 비율, seed 부호 일치(/10). "분명" = CLAUDE.md §5.

---

## T28 — 순수 X로 프로토타입 분류하면 어떤가 (학습 없음)

### 3. 목적

10/19 피드백 F1. 학습 없는 기준선에서 그래프 전파(Â²)를 빼고 원 특징 X만으로 같은 코사인 프로토타입 분류를 하면 정확도가 어떤지, 전파 단계 0·1·2를 같은 에피소드에서 잰다.

### 4. 표현

| 기호 | 정의 | 출처 |
|---|---|---|
| X0 | 행 L2 정규화 X | `fusion_baseline.diffusion_features_hops(ds, 0)` |
| X1 | 행 L2 정규화 ÂX | `fusion_baseline.diffusion_features_hops(ds, 1)` |
| D | 행 L2 정규화 Â²X | `fusion_baseline.diffusion_features(ds)` |
| TEG, C_all | 재사용(2절) | T22 |

### 5. 산출 표 (`reports/T28_summary.md`)

| 표 | 내용 |
|---|---|
| T1 | (ds, k) × {X0, X1, D, TEG, C_all} 고정 test 정확도 (seed 평균 ± sd) |
| T2 | 짝지은 차: X0 − D, X0 − TEG, X0 − C_all, X1 − D |
| T3 (기록) | 표현 수준 probe: T21과 같은 probe(`analyze_t06b.episodes` + `fewshot_acc`, 500 에피소드, `random.Random(1000 + seed)`, 그룹 base → valid → test, seed 0–2, T21이 쓴 분할)로 {X0, X1, S0} × {acc_base, acc_valid, acc_test}. S3(TEG 임베딩)는 `results/T21/analysis_<ds>.json`에서 전재 |

### 6. 판정 (값만, 단위 (ds, k) 6개)

| 판정 | 정의 |
|---|---|
| M1 | acc(X0) > acc(D)인 수 (seed 평균) |
| M2 | X0 − D가 분명한 양수인 수 / 분명한 음수인 수 |
| M3 | acc(X0) > acc(TEG)인 수 (seed 평균) |
| M4 (기록) | acc(X0) > acc(C_all)인 수; acc(X1) > acc(D)인 수; T3에서 acc_test(X0) > acc_test(S3)인 수 |

### 7. 사전 등록 (수정 금지)

| 예측 | 내용 | 근거(채팅 해석용) |
|---|---|---|
| P144 | M1 = 2/6이고, 그 두 칸은 Amazon_electronics 1·5-shot이다 | T19a test top-10 순도: Electronics만 X(0.631) > Â²X(0.473), 나머지는 X < Â²X |
| P145 | dblp 1·5-shot 모두 X0 − D가 분명한 음수 | T19a dblp 순도 X 0.155 vs Â²X 0.294 |
| P146 | M3 ≥ 2/6이고 Amazon_electronics 2칸을 포함한다 (확신 낮음) | 순도 X 0.631 vs TEG 0.461(Electronics). 순도와 정확도의 대응은 확인되지 않음 |

### 8. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ga 선행 | `base`와 선택 λ `infonce_all`의 `fixed_eval_logits.npz` 각 60개, `results/T21/analysis_<ds>.json` 3개 | **멈추고 보고** |
| Gb 재현 | D, TEG, C_all의 (ds, k) seed 평균이 `reports/T22_summary.md` 표 T2의 D, TEG, C_all과 \|Δ\| ≤ 0.005 %p (18값) | **멈추고 보고** (T29도 같은 에피소드를 쓴다) |
| Gc probe 동일 | T3의 S0가 T21 S0와 54개 모두 \|Δ\| ≤ 1e-6 | T3만 제외하고 계속, 기록 |
| Gd 변경 범위 | `git diff --stat <H0> <T28 보고서 커밋>` = `tools/analyze_t28.py`, `reports/T28_*`뿐 | 보고서 쓰고 태그 없음, T29로 |
| Ge 유한 | 모든 지표 유한 | 해당 칸 제외, 기록 |

---

## T29 — 발표된 SGC 기준선의 세 변형 (선형 head 학습만)

### 9. 목적

10/19 피드백 F3. 발표된 SGC 기준선(Meta-GNN 공개 코드 절차)을 세 변형으로 나눠, 같은 분할·같은 고정 test 에피소드에서 학습 없는 기준선, TEG, 레이블 없는 대조를 더한 TEG와 함께 잰다. 예비 실험(`instructions/ref/pilot_sgc_vs_d.py`)을 정식 조건으로 다시 실행하는 것이다.

### 10. 특징과 arm

- **SGC 특징** `V_sgc` = Â²(행 L1 정규화 X), 사후 정규화 없음. Â는 `fusion_baseline.diffusion_features_hops`와 같은 `gcn_norm`(자기 루프 포함) 연산자다. 새 함수 `sgc_features(ds)`를 `tools/run_t29.py`에 두고, `diffusion_features_hops`의 행렬 구성 코드를 그대로 따른다(행 L1 정규화 한 줄과 마지막 정규화 생략만 다름). 행 합이 0인 노드는 그대로 둔다(pilot과 같음).

| arm | 이름 | 특징 | 분류 규칙 | base 레이블 |
|---|---|---|---|---|
| D | 학습 없는 기준선 | 행 L2 Â²X | 코사인 프로토타입 | 안 씀 |
| SGC-P (F3.1) | SGC 프로토타입 | V_sgc | 코사인 프로토타입 | 안 씀 |
| SGC-S (F3.3) | support만 학습 | V_sgc | 선형층 무작위 초기화 → support로 학습 | 안 씀 |
| SGC-B (F3.2) | base 사전학습 → support 재학습 | V_sgc | base 사전학습한 선형층 → support로 학습 | 씀 |
| LR (기록) | 규제 로지스틱 회귀 | 행 L2 Â²X | sklearn `LogisticRegression(C=1.0, max_iter=1000)`, support만 | 안 씀 |
| TEG, C_all | 재사용 | — | — | — |

### 11. 선형층 학습 (pilot의 `init_linear`, `fit_linear`를 그대로 이식)

- 층: W(5 × d), b(5). 초기화 U(−1/√d, 1/√d)(nn.Linear 기본).
- `fit_linear`: 매 호출마다 새 Adam(lr **0.2**, weight decay **5e-6**), cross-entropy, **100 에폭**, 전체 배치. Meta-GNN 공개 코드 설정이며 바꾸지 않는다.
- **SGC-S**: 에피소드마다 무작위 초기화 → support(5k개)로 `fit_linear` → query 예측.
- **SGC-B 사전학습** (Meta-GNN `sgc_fewshot/citation.py`의 5-way 일반화, pilot과 같음): 하나의 선형층을 무작위 초기화한 뒤 `ITER` 50회 반복한다. 각 ITER에서 base 클래스(`class_list_train`) 5개를 고르고 레이블 0–4를 붙인다. 그 5클래스로 `STEP` 50회 동안, 매번 클래스당 20개 노드를 새로 뽑아 같은 층에 `fit_linear`를 한다(총 2,500회). 최종 층을 저장한다.
- **SGC-B 재학습**: 저장한 층에서 시작해 에피소드마다 support로 `fit_linear` → query 예측.
- 사전학습은 (ds, seed)마다 한 번이다. base 클래스는 2.1에서 복원한 정렬 리스트다.
- 장치: GPU(UUID 지정, 3번 제외, 동시 ≤ 6) 또는 CPU. 한 프로세스가 (ds, seed) 하나의 두 shot을 처리한다. float32.

### 12. 난수 (새 오프셋, seed에 더함. 사용자가 CLAUDE.md §8에 등록)

| 오프셋 | 용도 |
|---|---|
| 18600 | SGC-B 사전학습의 클래스·노드 추출 (`random.Random(18600 + seed)`) |
| 18700 | SGC-B 사전학습 선형층 초기화 (`torch.Generator().manual_seed(18700 + seed)`) |
| 18800 | SGC-S 1-shot 초기화 (`torch.Generator().manual_seed(18800 + seed)` 하나에서 `ep_idx` 순으로 W → b) |
| 18900 | SGC-S 5-shot 초기화 (같은 방식, 18900 + seed) |

전역 `random`·`torch` 상태는 쓰지 않는다. LR은 난수가 없다(lbfgs).

### 13. 산출 표 (`reports/T29_summary.md`)

| 표 | 내용 |
|---|---|
| T1 | (ds, k) × {D, SGC-P, SGC-S, SGC-B, LR, TEG, C_all} 고정 test 정확도 (seed 평균 ± sd) |
| T2 | 짝지은 차: SGC-S − SGC-B, SGC-P − SGC-S, D − SGC-P, D − SGC-B, C_all − SGC-P, C_all − SGC-S, TEG − SGC-B, LR − D |
| T3 | 사전학습 기록: (ds, seed)마다 마지막 라운드 학습 손실·학습 정확도, 사전학습 시간, 프로세스 전체 시간 |
| T4 (기록) | 5-shot 비교 열: 예비 실험 값(SGC-P 92.0 / 84.3 / 88.3, SGC-S 88.6 / 84.2 / 84.7, SGC-B 85.3 / 80.9 / 81.1; Clothing / dblp / Electronics)과 GPN 논문 SGC 보고값(62.2 / 65.0 / 60.8). 예비 실험과 분할·query 수가 다르다는 사실을 함께 적는다 |

### 14. 판정 (값만, 단위 (ds, k) 6개)

| 판정 | 정의 |
|---|---|
| N1 | SGC-S − SGC-B가 분명한 양수인 수 |
| N2 | acc(SGC-S) < acc(SGC-P)인 수 (seed 평균) |
| N3 | acc(SGC-B) < acc(D)인 수 (seed 평균) |
| N4 | acc(C_all) > acc(SGC-P)인 수 (seed 평균) |
| N5 | acc(TEG) > acc(SGC-B)인 수 (seed 평균) |
| N6 (기록) | LR − D가 분명한 양수인 수; \|D − SGC-P\| 최대값 |

### 15. 사전 등록 (수정 금지)

| 예측 | 내용 | 근거(채팅 해석용) |
|---|---|---|
| P147 | N1 = 6/6 | 예비 실험에서 base 사전학습 몫이 9칸 모두 분명(−3.3 ~ −7.5 %p) |
| P148 | N2 ≥ 5/6 | 예비 실험 5-shot dblp에서 SGC-S와 SGC-P가 0.1 %p 차 |
| P149 | N3 = 6/6 | 예비 실험 9/9 |
| P150 | N4 ≥ 4/6 | C_all > D 5/6(T22), D와 SGC-P 차는 예비 실험에서 ≤ 2 %p |
| P151 | N5 = 6/6 | 예비 실험 SGC-B가 T22 TEG보다 모든 5-shot 칸에서 3 %p 이상 낮음 |

### 16. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Gs 분할 복원 | (1) seed 0–2: 복원한 base·valid·test가 `split.json`과 집합으로 같다(9개 (ds, seed) 모두). (2) seed 0–9: 복원 valid 클래스 수 = `valid_num_dic`(Clothing 17, dblp 27, Electronics 36). (3) base·valid·test 서로소. (4) 같은 seed의 1-shot run과 5-shot run에서 각각 모은 valid 집합이 서로 모순되지 않는다(합집합 크기 = `valid_num_dic`). (5) labels·test 클래스 일치(2.1) | 하나라도 실패하면 **멈추고 보고** |
| Hb 재현 | T28 Gb와 같은 조건(D, TEG, C_all 18값) | **멈추고 보고** |
| Hc 시간 점검 | Amazon_clothing seed 0 프로세스를 먼저 단독 실행해 시간을 잰다. 프로세스당 60분을 넘을 것으로 보이면 | **멈추고 보고** |
| Hd 사전학습 | 모든 라운드 손실 유한 | 해당 (ds, seed) 제외, 기록 |
| He 변경 범위 | `git diff --stat <t28 커밋> <T29 보고서 커밋>` = `tools/run_t29.py`, `tools/analyze_t29.py`, `reports/T29_*`뿐 | 보고서 쓰고 태그 없음 |
| Hf 유한 | 모든 지표 유한 | 해당 칸 제외, 기록 |

출력: `results/T29/splits.json`, `results/T29/<ds>/seed<s>/{run.json, per_episode.npz, pretrain.json, console.log}`. `per_episode.npz`에는 (k, arm)별 에피소드 정확도와 query 예측을 저장한다.

## 17. 범위 밖

- 레이블 없이 학습한 인코더만 쓰는 arm(TLP류, T17 지시서)은 이 큐에 넣지 않는다. 다음 미팅 결정 뒤 별도 지시서로 낸다.
- 3-shot, ogbn-arxiv는 하지 않는다.
- 해석, 원인 분석, 다음 실험 제안은 쓰지 않는다.