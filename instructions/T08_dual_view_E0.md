# T08 — (ii) E0: 두 관점 결합 여지 확인 — TEG logit 덤프 + 오프라인 결합 분석 (브랜치 `dual_view_trust`)

정본 `CLAUDE.md`. 분기 기준: `t06b`(ee3ef3e). 이 파일은 `instructions/T08_dual_view_E0.md`에 둔다.
**auto mode: CLAUDE.md §2의 멈춤 지점 두 곳을 이 지시서 동안 해제한다.** §5 게이트에서 실패하면 멈춘다. CLAUDE.md §7은 유효.
설계 배경: `docs/dual_view_rehearsed_trust_methodology_v0.1.md`(있으면 참고, 없어도 이 지시서만으로 수행 가능).

## 0. 목표 (검증할 주장)
TEG의 학습 경로를 바꾸지 않고, 같은 에피소드에서 두 관점의 예측을 결합했을 때 각 관점 단독보다 나은지를 숫자로 남긴다.
- 관점 L: TEG 원본이 test 정확도 계산에 쓰는 query × 클래스 점수(logit).
- 관점 D: 확산 특징 Â²X(행 정규화)의 코사인 프로토타입 점수(학습 없음).
1. **(a) 결합 곡선** — 고정 가중치 α로 섞을 때 곡선 내부에 두 끝점(α = 0, 1)보다 높은 지점이 있는가.
2. **(b) valid로 고른 α** — valid 에피소드로 고른 하나의 α가 test에서 두 끝점 중 나은 쪽을 넘는가.
3. **(c) task별 선택의 여지** — 에피소드마다 α를 달리 고를 때 (b)보다 얼마나 더 얻을 수 있는가.

## 1. 코드 변경 (학습·정확도 경로 불변)
### 1.1 `--dump_logits` (store_true, 기본 False)
- valid·test 평가에서, **원본이 정확도 계산(argmax)에 쓰는 query × 클래스 점수 텐서**를 그대로 읽어 쌓는다. 어느 텐서인지 코드에서 찾아 보고서에 파일·줄로 적는다.
- 에피소드마다 함께 기록: `epoch`, `ep_idx`, `mode`(0 = valid, 1 = test), `support`(N·K, 전역 노드 ID, 클래스 순서대로 K개씩), `query`(N·Q, 같은 규칙), `classes`(N), `query_y`(N·Q, 에피소드 내 클래스 순번 0..N−1), `logits`(N·Q × N, float32). 열 순서는 `classes` 순서와 같아야 한다.
- `--out_dir`가 있으면 학습 종료 시 `out_dir/eval_logits.npz` 저장. 에폭 0을 포함한 모든 에폭의 valid·test 에피소드.
- `no_grad`·읽기 전용. 새 코드에서 전역 `random`·`np.random`·`torch` 난수를 호출하지 않는다. 계산 경로를 바꾸지 않는다.
- `config2string` 제외 목록에 추가.

## 2. 실행 (18 run)
- 3 데이터셋(`Amazon_clothing`, `dblp`, `Amazon_electronics`) × 5-way {1, 5}-shot × seed {0, 1, 2}. 1층(원본) GCN만.
- 명령: `CUDA_VISIBLE_DEVICES=<UUID> python main.py --dataset <ds> --way 5 --shot <k> --seed <s> --num_seed 1 --device 0 --dump_logits --out_dir results/T08/<ds>_5w<k>s/seed<s>`
- `tools/run_t08.py`(`run_t06b.py` 복사, 층 축 제거), GPU 6장 UUID, GPU당 1 run, seed 우선 순서. `--mem` 사용 안 함. 실패 시 1회 재시도(`_retry1`).

## 3. 분석 (`tools/analyze_t08.py` → `reports/T08_summary.md`)
라벨은 분석에만 쓴다. 분석 난수: query 반분할 `random.Random(5000 + seed)` 전용 인스턴스.

### 3.1 공통 정의
- **대상 에피소드**: run마다 `run.json`의 `best_epoch_valid`(원본 규칙)에 해당하는 valid 50개, test 50개.
- **관점 D 점수**: h̄ = Â²X 행 정규화(T06b `diff2`와 같은 Â, 같은 전처리). π^D_c = support h̄ 평균, ℓ^D_{q,c} = cos(h̄_q, π^D_c).
- **관점 L 점수**: ℓ^L = 덤프한 `logits`.
- **온도 보정(관점별, valid로만)**: run마다 valid 50 에피소드의 평균 NLL을 최소화하는 값을 격자에서 고른다.
  - T_L: log10 T_L ∈ {−2.0, −1.8, …, 4.0}(31개), 관점 L 확률 = softmax(ℓ^L / T_L)
  - s: log10 s ∈ {0.0, 0.1, …, 3.0}(31개), 관점 D 확률 = softmax(s · ℓ^D)
  - 최적값이 격자 끝이면 이상 징후에 기록.
- **결합**: log p̃_{q,c} = α · log softmax(ℓ^L/T_L)_c + (1 − α) · log softmax(s · ℓ^D)_c, p̂ = softmax_c(log p̃). 예측 = argmax.
- **α 격자**: A21 = {0, 0.05, …, 1}(21개), A5 = {0, 0.25, 0.5, 0.75, 1}.
- 값은 (데이터셋, shot)별 seed 3개 평균 ± sd. 짝지은 차는 에피소드 150개(50 × seed 3) 평균 ± SE, 양수 비율, seed 부호 일치 수.

### 3.2 표
- **표 A — 결합 곡선(test)**: (데이터셋, shot) × α ∈ A21 정확도. 열 끝에 acc(α=0), acc(α=1), max_{α∈A21} acc, argmax α, 그리고 [max_α acc − max(acc(α=0), acc(α=1))]. 이 표의 max는 test로 고른 값이다(상한 참고용으로만 표기).
- **표 B — valid로 고른 α**: run마다 α_val = argmin_{α∈A21} (valid 평균 NLL). 동률이면 1에 가까운 값. 열: α_val(seed별), T_L·s(seed별), test 정확도(α_val), 짝지은 차 vs TEG(α=1), 짝지은 차 vs D(α=0), 짝지은 차 vs **두 끝점 중 (데이터셋, shot) seed 평균 test 정확도가 높은 쪽**.
- **표 C — task별 선택의 여지(query 반분할)**: test 에피소드마다 query 25개를 무작위로 12 / 13으로 나눈다(분할 1회, 위 난수).
  - 적응: 한쪽 절반에서 정확도가 가장 높은 α ∈ A5를 고르고(동률이면 α_val에 가장 가까운 값) 다른 절반에서 평가. 양방향 평균.
  - 고정: α_val을 같은 두 절반에서 평가해 양방향 평균.
  - 열: acc(적응), acc(고정), 짝지은 차(적응 − 고정) 평균 ± SE, 양수 비율.
- **표 D — query 단위 상보성(test, α_val 결합 전 단독 예측)**: 비율: L만 정답, D만 정답, 둘 다 정답, 둘 다 오답. 그리고 둘 중 하나라도 정답인 비율(query 단위 상한).

### 3.3 판정 (사전 고정, 해석 아님 — 값만 계산해 표에 적는다)
- **J1 결합 가치**: 표 B의 "짝지은 차 vs 나은 끝점"이 평균 > 0 이고 평균 > 2·SE인 (데이터셋, shot) 수 / 6.
- **J2 task별 판단 가치**: 표 C의 (적응 − 고정)이 평균 ≥ 1.0%p 이고 평균 > 2·SE인 (데이터셋, shot) 수 / 6.

## 4. 사전 등록 (수정 금지, 2026-10-06 작성)
- **P46.** 표 A: [max_α acc − max(끝점)] ≥ 1.0%p 인 (데이터셋, shot)이 6개 중 4개 이상.
- **P47.** 표 B: 짝지은 차 vs 나은 끝점 > 0 (평균)인 (데이터셋, shot)이 6개 중 4개 이상.
- **P48.** J2 ≤ 2/6 (task별 선택으로 얻는 추가 이득은 대부분 조건에서 1%p 미만).
- **P49.** 표 B: α_val의 seed 평균이 dblp 두 shot에서 ≥ 0.5, Amazon_clothing 두 shot에서 ≤ 0.5.

## 5. 게이트 (실패 시 새 코드 안에서만 수정, 최대 2회 → 그래도 실패면 멈추고 보고)
- **G1 — 기본 경로 불변**: `git diff`의 추가 줄에 난수 호출이 없고 덤프 코드는 `no_grad`·읽기 전용임을 보고서에 명시. run별 `test_acc_at_best_valid`를 T06b L1 같은 seed 값과 나란히 기록(원본 비결정성 때문에 일치는 요구하지 않음).
- **G2 — 덤프 일관성**: 18 run 모두 `eval_logits.npz`·`run.json` 존재. 덤프된 모든 test 에피소드에서 argmax(`logits`)와 `query_y`로 계산한 정확도가 `episodes.jsonl`의 같은 epoch·ep_idx 정확도와 **완전 일치**. `classes`도 일치. valid는 에폭별 평균 정확도가 `run.json`의 valid 정확도와 일치(에피소드별 기록이 있으면 에피소드별로).
- **G3 — 분석 점검**: 대상 에피소드가 run마다 valid 50·test 50. 표 A의 α = 1 정확도가 에피소드별로 TEG 정확도와 완전 일치. α = 0 정확도가 T06b 분석 코드의 cosine probe 함수(`diff2`)를 같은 에피소드에 적용한 값과 완전 일치.
- **G4**: 표 A, B, C, D와 J1, J2 생성.

## 6. 보고 `reports/T08_dual_view_E0.md` (사실만, 해석 금지)
1. 변경 파일·위치·줄 수, 덤프한 텐서의 파일·줄, 커밋 해시, 태그 `t08`
2. 게이트 G1–G4
3. 표 A, B, C, D 본문과 J1, J2 값(`reports/T08_summary.md`에서 복사)
4. P46–P49 "예측 / 관측 / 일치·어긋남"
5. 이상 징후(자율 판단 포함, 격자 끝 최적값 포함)

## 7. 하지 말 것
CLAUDE.md §7 전부. 학습·정확도 경로 변경. 판단기·가짜 클래스 구현(다음 지시서). 지정 외 공간·지표·α 격자 추가. 판정 기준·문턱·사전 등록 변경. 결과 해석.
