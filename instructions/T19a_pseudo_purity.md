# T19a — E1: 유사도 기반 pseudo-class의 순도 지도 (학습 없음, 분석 전용)

2026-10-11 · 작성: 채팅 Claude · 모드: 채팅 지시문을 따른다 · 브랜치 `m1_sensor_check`(이 지시서에서 생성) · 태그 `t19a`

## 0. 목적과 검증 가능한 주장

**배경.** 10.10 회의 이후 방향은 M1(task 생성기)이다. NaQ·VISION류 비지도 task는 "anchor + 유사도 top-K"로 pseudo-class를 만든다. 그러면 그 pseudo-class의 순도(top-K가 실제로 같은 클래스인 비율)가 task의 질을 정한다. 이 지시서는 그 순도를 관점·K·클래스 그룹별로 잰다. 특히 base 지도 학습 임베딩으로 task를 만들 때 학습 미사용 클래스 영역의 순도가 어떻게 되는지를 본다.

**주장.** 4절의 표와 5절의 판정값, 6절의 사전 등록 대조를 산출한다. 해석은 하지 않는다.

**학습 실행 없음.** 기존 `emb_best.npy`와 입력 특징만 쓴다.

## 1. 착수 (하나라도 어긋나면 멈추고 보고)

1. 브랜치가 `erasure_diagnosis`, HEAD가 `3b7be89`인지 확인한다. `git status`에서 수정된 파일은 `CLAUDE.md` 하나뿐이어야 한다(사용자가 직접 갱신함). 그 밖의 변경이나 추적되지 않은 파일이 있으면 멈춘다.
2. `git switch -c m1_sensor_check`. 커밋하지 않은 `CLAUDE.md` 변경은 새 브랜치로 그대로 옮겨진다.
3. `CLAUDE.md`는 고치지 않는다. `git diff --stat CLAUDE.md`를 5항 계획 보고에 붙인다.
4. 사용자가 고친 `CLAUDE.md`, 이 지시서, `instructions/T19b_sensor_check.md`를 함께 커밋한다. 메시지는 `docs: T19a/T19b 지시서 + CLAUDE.md 갱신`. 이 커밋이 **H0**이다(diff 기준).
5. **코드 작성 전 멈춤(CLAUDE.md §2 멈춤 1).** 아래를 파일 단위 계획으로 보이고 승인을 기다린다. 채팅 지시문이 auto mode를 명시했다면 이 멈춤은 면제한다.
   - 새 스크립트 `tools/analyze_t19.py`의 함수 목록
   - ÂX·Â²X를 만드는 기존 구현의 `파일:줄`(T07·T09의 "Â¹X(행 정규화)", D = 행 정규화 Â²X를 만든 코드). 새로 구현하지 말고 그것을 재사용한다.
   - rand1(T13: ÂX·R, R ~ N(0, 1/64), `default_rng(2000 + seed)`)을 만드는 기존 구현의 `파일:줄`
   - T14 `emb_best.npy` 60개(아래)의 존재 여부. overnight 큐 이후 지워진 것이 있는지 확인한다.

## 2. 입력

| 이름 | 내용 | 출처 |
|---|---|---|
| X | 입력 특징 | `utils.load_data` |
| ÂX, Â²X | 1-hop, 2-hop 확산 특징 | 기존 구현 재사용 |
| rand1 | ÂX의 64차원 무작위 투영. 학습 없는 대조군(차원·hop을 인코더와 맞춤) | T13 구현 재사용, seed 0–4 |
| 원본 | T14 `base`의 `emb_best.npy` | `results/T14/...` |
| 대조 | T14 `uni`의 `emb_best.npy` | `results/T14/...` |

- 단위: 3 데이터셋 × 5-way {1, 5}-shot × seed 0–4. 원본·대조는 run마다 임베딩이 있다(30 + 30).
- X, ÂX, Â²X는 shot·seed와 무관하다. 데이터셋마다 한 번만 계산한다. rand1은 seed마다 계산한다.
- 클래스 그룹(base, valid, test)과 레이블은 각 T14 run의 `split.json`, `labels.npy`를 쓴다.
- 모든 유사도는 행 L2 정규화 후의 코사인이다. 그래서 ÂX의 행 정규화 여부는 결과에 영향을 주지 않는다.

## 3. 측정

**K ∈ {1, 5, 10, 20, 50}.** 자기 자신은 이웃에서 뺀다.

| 기호 | 정의 |
|---|---|
| `pur_all(Z, g, K)` | **주 지표.** 그룹 g의 노드를 각각 anchor로 삼고, **전체 노드 N개** 중 코사인 top-K 이웃을 찾는다. 그중 anchor와 같은 레이블인 비율을 anchor마다 구해 평균한다. NaQ·VISION처럼 그룹을 모르는 상태의 task 구성과 같다 |
| `pur_grp(Z, g, K)` | 이웃 후보를 그룹 g의 노드로 제한한 같은 값. K = 10이면 `tools/analyze_t14.py`의 `knn_purity`와 같아야 한다(게이트 Gc) |
| `frac_other(Z, g, K)` | `pur_all`의 이웃 중 anchor와 다른 그룹에 속한 노드의 비율 |
| `frac_unassigned` | 이웃 중 base·valid·test 어느 클래스 목록에도 속하지 않는 노드의 비율(T15 J절의 레이블 0 노드 문제). 이런 노드는 "다른 클래스"로 센다 |

- **무작위성 없음.** 모든 노드를 anchor로 쓰고 표본을 뽑지 않는다. rand1만 등록된 오프셋 2000을 쓴다.
- kNN은 GPU에서 행 블록(예: 4,096)으로 계산한다. 중간 산출물(이웃 인덱스 등)은 디스크에 저장하지 않는다. 요약 csv/json만 `results/T19a/`에 쓴다.

## 4. 산출 표 (`reports/T19a_summary.md`)

| 표 | 내용 |
|---|---|
| T1 | `pur_all`. 행 = 관점(X, ÂX, Â²X, rand1, 원본, 대조), 열 = 그룹(base, valid, test) × K. (데이터셋, shot)별. 원본·대조·rand1은 seed 평균 ± sd |
| T2 | 짝지은 차(seed 단위, 평균 ± SE, 부호 일치 수/5), K = 10과 K = 50: 원본 − ÂX, 대조 − ÂX, 원본 − rand1, 대조 − 원본. 그룹별 |
| T3 | `pur_grp`(K = 10)의 같은 형식 표. T14 표 4의 kNN 순도와 비교할 수 있게 둔다 |
| T4 | `frac_other`, `frac_unassigned`(K = 10, 50) |

## 5. 판정 (값만. 단위 (데이터셋, shot) 6개, 학습 없는 관점은 데이터셋 3개)

| 판정 | 정의 |
|---|---|
| Q1 | test 그룹 `pur_all`(K = 10): 원본 < ÂX인 (데이터셋, shot) 수(seed 평균) |
| Q2 | base 그룹 `pur_all`(K = 10): 원본 > ÂX인 수 |
| Q3 | test 그룹 `pur_all`(K = 10): 대조 − 원본이 분명히 양수인 수(평균 > 0이고 > 2·SE) |
| Q4 | test 그룹 `pur_all`(K = 10): Â²X ≥ ÂX ≥ X인 데이터셋 수 |
| Q5 (기록) | valid 그룹에서 Q1·Q3과 같은 값 |

## 6. 사전 등록 (수정 금지, 어긋나면 날짜 있는 부기)

| 예측 | 내용 |
|---|---|
| P100 | Q1 ≥ 4/6 |
| P101 | Q2 ≥ 5/6 |
| P102 | Q3 ≥ 5/6 |
| P103 | Q4 ≥ 2/3 |

## 7. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ga 분할 일치 | 각 run의 `labels.npy`가 `load_data`의 레이블과 같고, `split.json`의 base·valid·test 클래스 목록이 같은 seed의 `utils.py` 분할과 같다. T16a에서 `utils.py`가 바뀌었으므로(`--relabel_base`) 반드시 확인한다 | 멈추고 보고 |
| Gb 입력 존재 | T14 `base`·`uni`의 `emb_best.npy` 60개가 있고 모양이 [N, 64]다 | (데이터셋, shot)마다 seed 4개 이상이면 진행하고 누락을 기록. 그보다 적으면 그 칸을 "불완전"으로 표기 |
| Gc 구현 일치 | `pur_grp`(K = 10)가 `analyze_t14.knn_purity`와 \|Δ\| ≤ 1e-6. 대상: T14 `uni` Amazon_clothing 1-shot seed 0, test 그룹. `pur_all`은 노드 200개짜리 무작위 행렬에서 전수 계산과 완전 일치 | 멈추고 보고 |
| Gd 변경 범위 | `git diff --stat H0 <보고서 커밋>`의 파일이 `tools/analyze_t19.py`, `reports/`, `instructions/` 안에만 있다. 학습 코드와 `tools/erasure_metrics.py`는 바꾸지 않는다 | 멈추고 보고 |
| Ge 유한 | 모든 지표가 유한 | 해당 run 제외, 기록 |

## 8. 보고

- `reports/T19a_purity.md`(CLAUDE.md §6 형식), `reports/T19a_summary.md`(표, 판정, 사전 등록 대비).
- 커밋 → 태그 `t19a` → push(force 없음) → 부기 커밋.
- 실행 시간(분)을 보고서에 적는다.
