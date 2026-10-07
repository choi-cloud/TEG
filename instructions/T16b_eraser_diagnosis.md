# T16b — 주체 진단: 무엇이 학습 미사용 클래스의 표현을 지우는가

2026-10-09 · 작성: 채팅 Claude · 모드: auto(overnight 큐) · 브랜치 `erasure_diagnosis` · 태그 `t16b` · 선행: `t16a`(게이트 전부 통과)

## 0. 목적과 검증 가능한 주장

**주장**: 12개 설정 각각에서 다음을 측정해, 4절의 판정값과 5절의 사전 등록 대조를 산출한다.
- 학습 미사용 클래스(valid, test) 표현 지표가 초기화 시점에서 얼마나 변했는가(이하 "지움")
- base 학습이 얼마나 진행됐는가

해석은 하지 않는다.

## 1. 설정

**공통 조건**
- 3 데이터셋 × 5-way {1, 5}-shot × seed 0–4
- `--fixed_eval --traj_every 10`. `--traj_dump_ends 1`은 seed 0에만 쓴다.
- `--dump_emb`은 끈다. 나머지는 TEG 기본값이다.

| # | 설정 | 플래그 |
|---|---|---|
| A0 | 원본 | (기본) |
| A1 | L_N만 | `--gamma 1.0` |
| A2 | L_G만(= EGNN을 끈 TEG의 학습 경로) | `--gamma 0.0` |
| A3 | γ 0.25 | `--gamma 0.25` |
| A4 | γ 0.75 | `--gamma 0.75` |
| A5 | 레이블 재분할 | `--relabel_base 1` |
| A6 | 지도 손실 0(wd만) | `--sup_coef 0` |
| A7 | wd 0 | `--weight_decay 0` |
| A8 | wd 5e-5 | `--weight_decay 5e-5` |
| A9 | wd 2e-3 | `--weight_decay 2e-3` |
| A10 | 대조 | T14 `uni`와 같은 인자. λ = T10r 표 1 선택값(Clothing 1 = 10, Clothing 5 = 30, dblp 1 = 30, dblp 5 = 10, Electronics 1 = 100, Electronics 5 = 30) |
| A11 | 대조 + wd 0 | A10 + `--weight_decay 0` |

- 합계 360 run.
- **실행 순서(핵심 설정 먼저)**: A0, A2, A1, A7, A6, A10 → A5, A11 → A3, A4, A8, A9. 같은 설정 안에서는 (데이터셋, shot, seed) 순서다. 큐의 마감 시각에 걸리면 핵심 설정부터 남는다.

## 2. 정의

- run r, 지표 m, 그룹 g, 측정점 t에 대해 **E_g(t) = m_g(t) − m_g(0)**.
- **주 지표는 M3(probe)**, 부 지표는 M1(kNN 순도). 둘 다 보고한다. 아래 판정과 사전 등록은 따로 적지 않으면 M3 기준이다.
- **"지움 있음"** (설정, 데이터셋, shot 단위): 다음 셋을 모두 만족할 때다.
  - seed 평균 E_test(500) < 0
  - |평균| > 2·SE(seed 5개)
  - 부호 일치 ≥ 4/5
- **base 학습량**: B(t) = M3_base(t) − M3_base(0).
- **학습량을 맞춘 지움 ME**
  - b_ref = 0.5 × (해당 (데이터셋, shot)에서 A0의 seed 평균 B(500))
  - 각 run에서 B(t) ≥ b_ref가 처음 성립하는 t*를 측정점 사이 선형 보간으로 구한다.
  - ME = E_test(t*). 도달하지 못하면 "미도달"로 적는다.
  - A0와 같은 seed끼리 짝지어 비교한다.
- **t_half**: E_test(500) < 0인 run에서, E_test(t) ≤ 0.5·E_test(500)이 처음 성립하는 측정점.

## 3. 산출 표 (`reports/T16b_summary.md`)

| 표 | 내용 |
|---|---|
| T1 | 설정 × (데이터셋, shot)별 E_test(500), E_valid(500). M3와 M1 각각. seed 평균 ± sd, A0 대비 seed 짝지은 차(평균 ± SE, 부호 일치 수/5) |
| T2 | B(500). 고정 test TEG 정확도(원 보고 방식 `test_acc_at_best_valid` 병기). M4 `proto_euc_test`를 step 500에서와 M4 valid 최고 측정점에서 각각 |
| T3 | ME와 A0 대비 짝지은 차, 미도달 수 |
| T4 | 궤적: E_test(M3, M1)의 seed 평균. 측정점 0, 10, 20, 30, 50, 100, 200, 300, 500. 1-shot 3조건은 본문, 5-shot은 부록 |
| T5 | 기전: run별 r_j = colnorm_j(500) / colnorm_j(0). 다음을 seed 평균으로. 대상 설정은 A0, A2, A6, A7, A10, A11 |
| T6 | t_half 분포(설정별 중앙값) |
| 그림 1 | Clothing 1-shot, 설정별 E_test(M3) 궤적(`reports/T16b_traj.png`) |

T5 세부:
- Spearman ρ(r, log(1 + touch_count))
- ρ(r, freq_base − freq_unused)
- r < 0.1인 열의 비율을 `touch_count == 0` 열과 `> 0` 열로 나눠서

## 4. 판정 (값만, 단위는 (데이터셋, shot) 6개)

| 판정 | 정의 |
|---|---|
| K0 | A0 지움 있음 |
| S_N | A1 지움 있음 |
| S_G | A2 지움 있음 |
| S_W | A6 지움 있음 |
| N_W | A7 지움 **없음** |
| W_red | A7 − A0의 E_test 짝지은 차가 분명히 양수(덜 지움) |
| D_W | wd 단조: seed 평균 E_test가 wd 0 ≥ 5e-5 ≥ 5e-4 ≥ 2e-3 (A7, A8, A0, A9) |
| D_γ | γ 단조: seed 평균 E_test가 γ 0, 0.25, 0.5, 0.75, 1에서 단조. 증가·감소 방향별 수 |
| R | A5 지움 있음 |
| C1 | A10 지움 있음 |
| C2 | A11 − A10의 E_test 짝지은 차가 분명(양수 수, 음수 수) |
| M_A0 | A0의 seed 평균 ρ(r, log(1 + touch_count)) > 0 |
| M_A7 | A7의 같은 값 > 0 |

M1(순도) 기준의 K0, S_N, S_G, S_W, N_W도 병기한다.

## 5. 사전 등록 (수정 금지, 어긋나면 부기)

| 예측 | 내용 |
|---|---|
| P85 | K0 ≥ 4/6 |
| P86 | S_G ≥ 4/6 (EGNN 없는 L_G 경로만으로도 지운다) |
| P87 | S_N ≥ 4/6 |
| P88 | S_W ≤ 1/6 (지도 손실 없이 wd만으로는 지우지 않는다) |
| P89 | W_red ≥ 4/6 (wd를 끄면 덜 지운다) |
| P90 | N_W ≤ 3/6 (wd를 꺼도 지움이 남는 조건이 절반 이상) |
| P91 | R ≥ 4/6 |
| P92 | C1 ≤ 2/6 |
| P93 | M_A0 = 6/6 |
| P94 | D_W ≥ 4/6 |

## 6. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ga | 같은 (데이터셋, shot, seed)의 12개 설정에서 에폭별 전역 `random` 상태 해시가 일치하고, valid·test 에피소드(전 에폭)와 고정 에피소드가 일치. A5의 학습 에피소드는 노드가 다를 수 있다(설계상) | 이 지시서 정지 |
| Gb | 손실·지표가 모두 유한 | 해당 run 제외, 이상 징후에 기록 |
| Gc | 설정별 완료 run ≥ 95%(30개 중 29개 이상) | 미달 설정은 판정에서 "불완전"으로 표기 |
| Gd | 변경 범위: `git diff --stat <T16b 착수 HEAD> <보고서 커밋>`이 `tools/run_t16b.py`, `tools/analyze_t16b.py`, `reports/`, `instructions/`만 포함(학습 코드 변경 없음) | 이 지시서 정지 |
| Ge | 실행 묶음 시작 전 디스크 여유 확인(CLAUDE.md §7) | §7 절차 |

## 7. 보고

- `reports/T16b_eraser.md`(§6 형식), `reports/T16b_summary.md`(표와 판정, 사전 등록 대비), `reports/T16b_traj.png`.
- 커밋 → 태그 `t16b` → push → 부기 커밋.
- 마감 때문에 부분 완료라면 태그를 달지 않고, 보고서 첫 줄에 "부분 완료"와 완료 설정 목록을 적는다.
