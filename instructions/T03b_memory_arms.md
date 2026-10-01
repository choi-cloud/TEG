# T03b — 평가 시점 보정 arm A·B

정본 `CLAUDE.md`, 설계 `docs/relation_textbook_methodology_v0.md`. 선행: T03a. manual mode, 멈춤 지점 두 곳(§2).

## 목표 (검증할 주장)
`--mem`을 켜면 valid·test 에피소드마다 같은 에피소드 위에서 7개 설정(off, A×3, B×3)의 정확도가 계산되고, 설정마다 TEG 방식의 "valid 최고 에폭의 test 정확도"가 기록된다. off 설정은 T03a의 정확도와 같다.

## 설정
- **off**: 원본.
- **A_β** (값 = 관계 자체), β ∈ {0.1, 0.3, 0.5}: 첫 층 메시지 m을 `(1−β)·m + β·v̄`로 바꾼다. v̄ = 조회된 k개 메모리 메시지의 가중 평균(T03a의 `idx`, `w`, 보관된 원본 m 사용). 바뀐 메시지는 첫 층의 **좌표 갱신과 구조 특징 갱신 모두**에 들어간다.
- **B_β** (값 = 같은 클래스 여부), β ∈ {0.1, 0.3, 0.5}: 첫 층 좌표 갱신 가중치 w = `trans_mlp(m)`을 `w − β·σ·(2p̂ − 1)`로 바꾼다. σ = 그 에피소드 첫 층 w의 표준편차. p̂가 높을수록 w가 작아져 r이 c 쪽으로 당겨진다. 구조 특징 갱신은 바꾸지 않는다. σ < 1e-8이면 보정하지 않고 횟수를 센다.
- 조회는 에피소드마다 1회(보정 전 메시지로) 하고 6개 설정이 공유한다. 2층은 어느 설정에서도 직접 바꾸지 않는다.

## 바꿀 것 (**멈춤 지점 1: 수정 전 계획 승인**)
- `layers/EGNN.py`
  - `EGCL.forward(edge_index, str_feature, coord_feature, msg_fn=None, w_fn=None)`: `msg_fn`이 있으면 `msg = msg_fn(msg)`.
  - `EGCL.coord_model(..., w_fn=None)`: `w = self.trans_mlp(msg)`; `w_fn`이 있으면 `w = w_fn(w)`; `trans = coord_diff * w`.
  - `EGNN.forward(str_feature, coord_feature, edge_index, msg_fn=None, w_fn=None)`: 두 인자는 **첫 층(gcl_0)에만** 전달.
  - 인자가 `None`이면 원본과 같은 연산 순서. (`trans_mlp` 결과를 변수에 담았다 곱하는 것은 원본과 같은 연산이다.)
- `model.py`
  - "EGNN → 프로토타입 → 예측 → 정확도" 구간을 함수로 묶어 off와 6개 설정이 함께 쓴다. off 경로의 연산은 원본과 같아야 한다.
  - `--mem`이 켜진 valid·test 에피소드(에폭 ≥ 1)에서 6개 설정을 추가 평가한다. 에피소드 추출·GCN 계산은 재사용하고, 난수를 소비하지 않는다.
  - 에피소드 기록에 설정별 `acc` 추가. 에폭 기록에 설정별 valid/test 정확도 추가.
  - 학습 종료 시 설정마다 `best_valid_epoch`, `test_acc_at_best_valid`(원본의 동점 규칙 그대로)를 계산해 `run.json`의 `mem_configs`에 저장. 에폭 0은 메모리가 비어 있으므로 6개 설정의 에폭 0 값은 off 값으로 둔다.
  - B의 σ 보정 생략 횟수를 `run.json`에 기록.

## 확인 실행
- A: `... --dataset Amazon_clothing --way 5 --shot 1 --seed 0 --mem --out_dir results/T03b/amac_5w1s_s0`
- B: `... --dataset Amazon_clothing --way 5 --shot 5 --seed 0 --mem --out_dir results/T03b/amac_5w5s_s0`

## 완료 조건
1. off 설정의 에폭별 valid/test 정확도 = T03a 같은 run (T01이 비결정적이었다면 차이 보고)
2. `--mem` 없이 실행한 run의 콘솔 출력 = T02 (EGNN 인자 추가가 기본 동작을 바꾸지 않음)
3. 7개 설정 모두 `test_acc_at_best_valid` 기록
4. 점검: β = 0에 해당하는 계산을 한 번 수동으로 돌려 off와 같은 정확도가 나오는지(코드 경로 점검용, 기록만)

## 보고 (**멈춤 지점 2**) — `reports/T03b_memory_arms.md`
- diff 요약
- 표: run A·B 각각, 설정 7개 × {best_valid_epoch, test_acc_at_best_valid}
- 완료 조건 1~4 표, σ 생략 횟수, 소요 시간
- 이상 징후, 커밋 해시, 태그 `t03b`

## 하지 말 것
CLAUDE.md §7 전부. 2층 변경. β·k·τ 외 값 탐색. 결과 해석.
