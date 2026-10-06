# T06 — coverage gap 진단: best-valid 임베딩 덤프 + 오프라인 분석 (브랜치 `novel_like_class`)

정본 `CLAUDE.md`. 분기 기준: `HT_01` HEAD(343e9b1, T05 결과 포함). 멈춤 지점은 CLAUDE.md §2를 따른다(채팅 지시문이 auto mode를 명시하면 그에 따름).

## 0. 목표 (검증할 주장)
학습된 TEG 인코더가 base 클래스를 구분하는 능력에 비해 novel(test) 클래스를 구분하는 능력을 얼마나 덜 얻었는지, 그리고 그 차이가 "인코더가 novel 구분 정보를 버렸기 때문"인지를 **학습 경로 변경 없이** 숫자로 남긴다.

## 1. 코드 변경 (학습·정확도 경로 불변)
- `argument.py`: `--dump_emb` (store_true, 기본 False). `config2string` 제외 목록에 추가.
- `model.py`:
  - `--dump_emb`이고 `--out_dir`가 있을 때, **valid 평가 직후 원본의 모델 선택 규칙(`acc_valid == best_acc_valid`일 때 갱신)과 같은 조건**에서 전체 노드의 GCN 출력 임베딩(eval 모드, `no_grad`, EGNN·LayerNorm 이전)을 `out_dir/emb_best.npy`(float32, |V|×d)로 덮어쓴다. 결과적으로 마지막에 남는 파일 = `test_acc_at_best_valid`를 낸 에폭의 임베딩이다. 저장한 에폭 번호를 `run.json`의 `emb_epoch`에 기록한다.
  - 같은 out_dir에 `split.json`을 쓴다: `class_list_train`, `class_list_valid`, `class_list_test`(seed마다 다른 valid 분할 그대로), 그리고 `labels.npy`(전체 노드 라벨, 원본 인덱스).
  - 새 코드에서 전역 `random`·`np.random`·`torch` 난수를 호출하지 않는다.

## 2. 실행
- 3 데이터셋(`Amazon_clothing`, `dblp`, `Amazon_electronics`) × 5-way {1, 5}-shot × seed {0, 1, 2} = 18 run.
- 명령: `CUDA_VISIBLE_DEVICES=<UUID> python main.py --dataset <ds> --way 5 --shot <k> --seed <s> --num_seed 1 --device 0 --dump_emb --out_dir results/T06/<ds>_5w<k>s/seed<s>`
- `tools/run_t04.py`를 복사한 `tools/run_t06.py`로 GPU 6장(UUID)에 배분. `--mem` 사용 안 함.

## 3. 분석 (`tools/analyze_t06.py` → `reports/T06_summary.md`)
모든 분석은 저장된 파일만 읽는다. 라벨은 **분석에만** 쓴다. 분석용 난수는 `random.Random(1000 + seed)` 전용 인스턴스.

### 3.1 표현 공간 3종
- `raw`: 모델 입력 특징 X (`load_data`가 반환하는 것과 동일한 전처리).
- `diff`: Â²X. Â = self-loop를 더한 대칭 정규화 인접행렬(GCN과 같은 정규화). 희소 곱으로 계산.
- `emb`: `emb_best.npy`.

### 3.2 (a) 일반화 격차 — 기존 로그에서
run마다 `best_valid_epoch`(원본 규칙)의 `train_acc`, `valid_acc`, `test_acc`. 표: (데이터셋, shot)별 seed 평균 ± sd, 그리고 `train − test`, `valid − test`.

### 3.3 (b) 공간별 few-shot 분리도 — 핵심
- 클래스 그룹 G ∈ {base(`class_list_train`), valid, test}.
- 각 (공간, G, K ∈ {1, 5})에서 에피소드 500개: G에서 클래스 5개 무작위, 클래스마다 support K개 + query 5개. support 평균을 프로토타입으로, query를 **cosine 유사도** 최근접 프로토타입에 배정. 정확도 평균.
- 표 B1: (데이터셋, shot) × 공간 3종 × G 3종의 정확도(seed 평균 ± sd). K는 run의 shot과 같게 쓴다.
- 표 B2: Δ_G = acc(`emb`, G) − acc(`diff`, G). 열: Δ_base, Δ_valid, Δ_test, (Δ_base − Δ_test).

### 3.4 (c) 구분 축 coverage — 보조
- `emb` 공간에서 클래스 프로토타입(해당 클래스 전 노드 평균, L2 정규화).
- base 클래스 쌍의 프로토타입 차이 벡터 집합으로 PCA → 분산 90%를 설명하는 주성분 r개의 부분공간 P.
- 잔차 비율 = ‖(I − P)d‖² / ‖d‖². test 쌍, valid 쌍, 그리고 기준선으로 무작위 Gaussian 방향 1,000개의 잔차 평균.
- 표 C: (데이터셋, shot)별 r, 잔차(test), 잔차(valid), 잔차(무작위). seed 평균 ± sd.

## 4. 게이트 (실패 시 새 코드 안에서만 수정, 최대 2회 → 그래도 실패면 멈추고 보고)
- G1: 18 run 모두 `emb_best.npy`, `split.json`, `labels.npy`, `run.json` 존재. `emb_epoch` = `run.json`의 `best_epoch_valid`.
- G2: 새 코드가 학습·평가 계산에 개입하지 않음 — diff에서 덤프 코드가 `no_grad`·읽기 전용이고 난수 호출이 없음을 보고서에 명시. 참고로 각 run의 `test_acc_at_best_valid`와 T04 같은 seed(seed 0–2) 값을 나란히 기록(원본 비결정성 때문에 일치는 요구하지 않음).
- G3: 표 A, B1, B2, C 생성.

## 5. 사전 등록 (수정 금지, 2026-10-06 작성)
- **P40.** 표 B2: (Δ_base − Δ_test) > 0 인 (데이터셋, shot)이 6개 중 4개 이상 — 인코더 학습이 base 구분을 novel 구분보다 더 많이 개선한다.
- **P41.** 표 B1: acc(`emb`, test) < acc(`diff`, test)인 (데이터셋, shot)이 6개 중 2개 이상 — 일부 조건에서는 학습된 임베딩이 확산 특징보다 novel 구분을 오히려 못한다.

## 6. 보고 `reports/T06_coverage_gap_diag.md` (사실만, 해석 금지)
1. 변경 파일·위치·줄 수, 커밋 해시, 태그 `t06`
2. 게이트 G1–G3
3. 표 A, B1, B2, C 본문
4. P40–P41 "예측 / 관측 / 일치·어긋남"
5. 이상 징후(자율 판단 포함)

## 7. 하지 말 것
CLAUDE.md §7 전부. 학습·정확도 경로 변경. 지정 외 공간·지표·설정 추가. 결과 해석.
