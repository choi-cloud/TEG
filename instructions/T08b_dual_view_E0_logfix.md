# T08b — (ii) E0 재분석: 관점 L 점수를 log 확률로 교정 + 결합 기준선 함수 분리 (브랜치 `dual_view_trust`, (ii) 마지막 지시서)

정본 `CLAUDE.md`. 선행: T08(`t08`, 8cedc2a). 이 파일은 `instructions/T08b_dual_view_E0_logfix.md`에 둔다.
**auto mode: CLAUDE.md §2의 멈춤 지점 두 곳을 이 지시서 동안 해제한다.** §5 게이트에서 실패하면 멈춘다. CLAUDE.md §7은 유효.
**학습 run 없음. `model.py`·`argument.py` 등 학습 코드 변경 없음.** T08의 `results/T08/**/eval_logits.npz`와 `run.json`만 읽는다.

## 0. 목표
1. T08 지시서의 정의 오류를 고친다. T08은 관점 L 점수로 덤프 텐서(softmax 확률)를 그대로 쓰고 그 위에 다시 softmax를 적용했다. 이번에는 **ℓ^L = log(덤프 확률)**(= −거리 + 에피소드·query별 상수, 즉 원본 logit)로 바꿔 T08의 표 A–D와 J1, J2를 다시 계산한다.
2. α = 0.5 고정 결합(α 선택 단계 없음)의 성능을 추가로 기록한다.
3. 결합 기준선 계산을 다른 브랜치에서도 쓸 수 있는 독립 함수 파일로 분리한다.

## 1. 코드 (tools/만)
### 1.1 `tools/fusion_baseline.py` (신규, 재사용용)
- 의존: 리포에 `t06b` 시점부터 있던 데이터 로더와 `gcn_norm`만 사용. `tools/analyze_t06b.py`·`analyze_t08.py`를 import하지 않는다(다른 브랜치로 옮길 수 있게).
- 함수(이름은 자유, 보고서에 시그니처 기재):
  - 확산 특징 h̄ = 행 정규화 Â²X (T08과 같은 정의)
  - 관점 D 점수: π^D_c = support h̄ 평균, ℓ^D_{q,c} = cos(h̄_q, π^D_c)
  - 관점 L 점수 변환 옵션 `l_transform ∈ {"log", "prob"}`: `"log"` = log(p), `"prob"` = p 그대로(T08 재현용)
  - 온도 보정: valid 에피소드 평균 NLL 최소화, 격자 T08과 같음(log10 T_L ∈ {−2.0, −1.8, …, 4.0}, log10 s ∈ {0.0, 0.1, …, 3.0})
  - 결합: log p̃ = α · log softmax(ℓ^L/T_L) + (1 − α) · log softmax(s · ℓ^D), p̂ = softmax(log p̃)
  - α 선택: `alpha="val"`(A21에서 valid NLL 최소, 동률이면 1에 가까운 값) 또는 실수 고정값
  - 입력 = run 디렉터리(`eval_logits.npz`, `run.json`)와 데이터셋 이름, 출력 = 대상 에피소드(best_epoch_valid의 test 50)별 정확도와 선택된 (α, T_L, s)
- log(0) 처리: 덤프 확률에 정확히 0인 원소가 있으면 float32 최소 양수(`np.finfo(np.float32).tiny`)로 바꾼 뒤 log. 바뀐 원소 수를 반환·기록.
- 새 코드에서 전역 난수 호출 금지(분석 난수는 아래 전용 인스턴스만).

### 1.2 `tools/analyze_t08b.py` (신규) → `reports/T08b_summary.md`
- `fusion_baseline.py`만 사용해 계산한다. 분석 난수: query 반분할 `random.Random(5000 + seed)`(T08과 같은 분할이 나와야 함).
- 동률 규칙은 T08 보고서에 기록된 것과 같게 한다(표 A argmax: 1에 가까운 값 / 표 C: α_val에 가장 가까운 A5 값, 그래도 동률이면 큰 α).

## 2. 분석 (`l_transform="log"` 기준, 대상·집계 방식은 T08 §3.1과 동일)
- **표 A** — 결합 곡선(test, α ∈ A21). T08 표 A와 같은 열.
- **표 B** — valid로 고른 α. T08 표 B와 같은 열.
- **표 B2** — α = 0.5 고정(T_L, s는 valid 보정): test 정확도, 짝지은 차 vs TEG(α=1), vs D(α=0), vs 나은 끝점, vs 표 B의 α_val 결합.
- **표 C** — task별 선택의 여지. T08 표 C와 같은 절차·같은 분할.
- **표 D** — query 단위 상보성(단독 예측이므로 T08과 같아야 함, §4 G2).
- **표 E — T08 대비**: (데이터셋, shot)별 T08 값과 T08b 값을 나란히: α_val 평균, T_L 평균, test acc(α_val), 차 vs 나은 끝점, [표 A max − max(끝점)], 표 C 적응 − 고정. 그리고 J1, J2 두 값.
- **판정 J1, J2** — T08 §3.3과 같은 정의.

## 3. 사전 등록 (수정 금지, 2026-10-06 작성)
- **P50.** J2 ≤ 1/6.
- **P51.** 표 B: 결합(α_val) − TEG가 평균 > 2·SE인 (데이터셋, shot)이 6/6.
- **P52.** J1 ≥ 3/6.
- **P53.** 표 B2: |acc(α = 0.5) − acc(α_val)| ≤ 0.5%p인 (데이터셋, shot)이 6개 중 5개 이상.

## 4. 게이트 (실패 시 새 코드 안에서만 수정, 최대 2회 → 그래도 실패면 멈추고 보고)
- **G1 — 학습 코드 불변**: 이 지시서의 커밋에서 `git diff --stat`이 `tools/`, `reports/`, `instructions/`만 포함.
- **G2 — 변환 점검**: 대상 에피소드 모든 query에서 argmax(log p) = argmax(p). α = 1과 α = 0의 에피소드별 정확도가 T08과 완전 일치. 표 D가 T08 표 D와 모든 셀 일치. log(0) 대체 원소 수 기록.
- **G3 — 함수 회귀**: `fusion_baseline.py`에 `l_transform="prob"`를 주면 T08 표 B의 α_val·T_L·s·test acc(α_val)를 18 run 모두 정확히 재현.
- **G4**: 표 A, B, B2, C, D, E와 J1, J2 생성.

## 5. 보고 `reports/T08b_dual_view_E0_logfix.md` (사실만, 해석 금지)
1. 변경 파일·줄 수, `fusion_baseline.py` 함수 시그니처, 커밋 해시, 태그 `t08b`
2. 게이트 G1–G4 (log(0) 대체 원소 수 포함)
3. 표 A, B, B2, C, D, E 본문과 J1, J2
4. P50–P53 "예측 / 관측 / 일치·어긋남"
5. 이상 징후(자율 판단, 격자 끝 최적값 포함)
6. 마지막 줄에 사실로만 기록: "이 지시서가 브랜치 `dual_view_trust`의 마지막 지시서다."

## 6. 하지 말 것
CLAUDE.md §7 전부. 학습 run 실행. 학습 코드 변경. 판단기·가짜 클래스·query 단위 판단기 구현. 지정 외 표·지표·격자 추가. T08 보고서·요약 파일 수정. 판정 기준·사전 등록 변경. 결과 해석.
