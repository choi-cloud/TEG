# T30 — 클래스별 분해: 어느 test 클래스가 지워지고, 그 클래스는 base와 얼마나 다른 축에 있나 (학습 없음)

2026-10-21 · 작성: 채팅 Claude · 브랜치 `classwise_decomp`(`trainfree_baselines`의 메모 삭제 커밋에서 생성, §1) · 태그 `t30` · 선행: `t22`, `t28`, `t29` · 사전 등록 P152–P154 · 모드: 채팅 지시문이 정한다

## 0. 목적과 검증 가능한 주장

지금까지의 비교는 (데이터셋, shot) 평균이다. 이 지시서는 같은 고정 test 에피소드를 **test 클래스 단위**로 나눠 다음을 잰다.

- (i) 학습 없는 입력이 그 클래스를 가르는가: acc_D(c)
- (ii) base 지도가 그 클래스에서 얼마나 잃었나: TEG − D
- (iii) 레이블 없는 대조가 얼마나 되찾았나: C_all − TEG
- (iv) 그 클래스를 가르는 방향이 base 클래스가 지도하는 방향과 얼마나 겹치나: e_c, r_c (4절)

주장: 위 네 값과 그 사이의 순위 상관을 (데이터셋, shot) 6칸마다 산출한다. 해석은 하지 않는다.

## 1. 순서와 멈춤

1. `git status`와 `git log --oneline -5`로 상태를 확인한다. 기대 상태: 브랜치 `trainfree_baselines`, HEAD `e675be0`, `results/T28_T29_Halt2.txt` 삭제가 stage됨, `CLAUDE.md` 수정됨(stage 안 됨), 이 지시서 untracked. 다르면 **멈추고 보고**한다. 사용자가 수정한 `CLAUDE.md`를 읽는다.
2. **메모 삭제 커밋** (`trainfree_baselines`): `results/T28_T29_Halt2.txt` 삭제 하나만 커밋한다(`CLAUDE.md`와 이 지시서는 넣지 않는다). 메시지: `chore: remove user memo results/T28_T29_Halt2.txt`. push한다.
3. 그 커밋에서 브랜치 `classwise_decomp`를 만든다.
4. **H0 커밋** (`classwise_decomp`): `CLAUDE.md`와 이 지시서(`instructions/T30_classwise.md`) 두 파일만. 내용은 고치지 않는다.
5. **멈춤 1**: 메모 삭제 커밋·H0 커밋의 해시와 파일 단위 계획(함수 목록, 재사용하는 기존 구현의 `파일:줄`)을 보이고 승인을 기다린다.
6. 구현 → 실행 → 보고서 커밋 → 태그 `t30` → push → **멈춤 2**: CLAUDE.md §6 형식으로 보고.

- 학습 코드와 기존 `tools/*.py`는 고치지 않고 import만 한다. 새 파일은 `tools/analyze_t30.py` 하나다.
- 학습은 없다. 새 난수도 쓰지 않는다(새 오프셋 없음).

## 2. 입력 (재사용만)

| 입력 | 위치 |
|---|---|
| 에피소드, TEG 출력 | `results/T22/base/<ds>_5w<k>s/seed<s>/fixed_eval_logits.npz`, mode 1, `ep_idx` 순 (seed 0–9) |
| C_all 출력 | 선택 λ `infonce_all`의 같은 파일 (λ: Clothing·dblp 10, Electronics 1-shot 100, 5-shot 30) |
| 분할 (seed 0–9) | `results/T29/splits.json` (T29 §2.1 복원, 게이트 Gs 통과) |
| labels | `results/T22/base/<ds>_5w1s/seed0/labels.npy` |
| 특징 | X0 = `fusion_baseline.diffusion_features_hops(ds, 0)`, D = `fusion_baseline.diffusion_features(ds)` (T28과 같음) |

레이블 사용: test·base 노드 레이블은 클래스별 집계와 클래스 평균(4절) 계산에만 쓴다. 모델 계산은 없다(CLAUDE.md §3의 "평가 지표 계산").

## 3. 클래스별 정확도

- arm: X0, D (코사인 프로토타입 예측 = `fusion_baseline.view_d_scores`의 argmax), TEG, C_all (`logits` argmax).
- (ds, k)마다 test 클래스 c에 대해: c가 포함된 모든 고정 에피소드(seed 0–9)에서, 참 레이블이 c인 query 5개의 정확도를 에피소드 단위 값으로 둔다.
- acc_arm(c) = 그 에피소드 값들의 평균 ± SE, n_ep(c) = 에피소드 수.
- 짝지은 차(같은 에피소드끼리): TEG − D, C_all − TEG, C_all − D, D − X0. "분명" = CLAUDE.md §5 기준을 그 클래스의 에피소드들에 적용.

## 4. 클래스별 "base 축과의 겹침" (seed마다 계산 후 seed 평균)

**e_c (D 공간, 주 지표)**
1. D 공간에서 클래스 평균 μ_c(모든 노드 중 레이블 c).
2. base 부분공간: 그 seed의 base 클래스 평균들을 base 평균으로 중심화한 행렬(n_base × d)의 SVD에서 0이 아닌 특이값의 오른쪽 특이벡터 U (rank를 기록, 기대값 n_base − 1).
3. test 클래스 c의 구분 방향 δ_c = μ_c − (c를 뺀 test 클래스 평균들의 평균).
4. e_c = ‖Uᵀδ_c‖² / ‖δ_c‖². 기준값 e_rand = rank / d(무작위 방향의 기대값)를 함께 적는다.
5. 참고(기록): valid 클래스에도 같은 방식(δ는 다른 valid 클래스 평균 기준)으로 e를 계산해 분포(최소·중앙·최대)를 적는다.

**r_c (X 공간, 보조 지표: "c를 가르는 특징 열이 base 노드에 얼마나 나타나나")**
1. X0 공간에서 δ_c^X = μ_c^X − (c를 뺀 test 클래스 평균들의 평균).
2. T_c = δ_c^X가 가장 큰 특징 열 50개.
3. f_j = 그 seed의 base 노드 중 X_{ij} ≠ 0인 비율.
4. r_c = T_c 위 f_j의 평균. 기준값 f̄ = 모든 열 f_j의 평균을 함께 적는다.

## 5. 산출

- `results/T30/classwise_<ds>.json`: (k, c)마다 n_ep, 네 arm 정확도 ± SE, 네 짝지은 차, e_c, r_c, seed별 e·r.
- `reports/T30_per_class.md`: (ds, k)마다 클래스별 전체 표(TEG − D 오름차순 정렬).
- `reports/T30_summary.md`:

| 표 | 내용 |
|---|---|
| T1 | (ds, k) × {test 클래스 수, n_ep 최소·중앙, TEG − D 분명한 음수 / 분명한 양수 / 그 외 클래스 수, C_all − TEG 분명한 양수 클래스 수} |
| T2 | (ds, k) × Spearman ρ(값, p): ρ(e_c, TEG − D), ρ(TEG − D, C_all − TEG), ρ(TEG − D, C_all − D), ρ(r_c, TEG − D), ρ(e_c, acc_D), ρ(r_c, acc_D) |
| T3 | (ds, k)마다 3분위 교차표: e_c 하위 3분위 클래스 중 (a) acc_D도 하위 3분위인 수, (b) acc_D는 하위가 아니고 TEG − D가 분명한 음수인 수. 그 외 칸도 함께 |
| T4 | (ds, k)마다 TEG − D가 가장 음수인 클래스 3개와 가장 양수인 클래스 3개: 클래스 id, acc_X0·D·TEG·C_all, e_c, r_c |
| T5 | 기준값: 데이터셋·seed별 base 부분공간 rank, e_rand, f̄, valid 클래스 e 분포 |

## 6. 판정 (값만, 단위 (ds, k) 6개)

| 판정 | 정의 |
|---|---|
| Q1 | TEG − D 분명한 음수 클래스 수 > 분명한 양수 클래스 수인 칸의 수 |
| Q2 | ρ(e_c, TEG − D) > 0인 칸의 수 |
| Q3 | ρ(TEG − D, C_all − TEG) < 0인 칸의 수 |
| Q4 (기록) | ρ(TEG − D, C_all − D) < 0인 칸의 수; ρ(r_c, TEG − D) > 0인 칸의 수; T3 (a)와 (b)의 칸별 수 |

## 7. 사전 등록 (수정 금지)

| 예측 | 내용 | 근거(채팅 해석용) |
|---|---|---|
| P152 | Q1 = 5/6이고, 아닌 칸은 dblp 1-shot이다 | 칸 평균에서 TEG < D가 dblp 1-shot을 뺀 5/6(T22). 지움이 소수 클래스가 아니라 다수 클래스에 걸쳐 있다고 본다 |
| P153 | Q2 ≥ 4/6 (확신 낮음) | 교수님 "드문 축" 가설의 방향: base 부분공간과 덜 겹치는 클래스가 더 잃는다 |
| P154 | Q3 = 6/6 | 칸 단위에서 TEG가 많이 잃은 곳에서 대조가 많이 되찾음(T22). 주의: 두 차에 TEG가 반대 부호로 들어가 측정 잡음만으로도 음의 상관이 생길 수 있다. 그래서 Q4에 ρ(TEG − D, C_all − D)를 함께 적는다 |

## 8. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ga 선행 | `base`·선택 λ `infonce_all` npz 각 60개, `results/T29/splits.json`, 데이터셋별 `labels.npy` | **멈추고 보고** |
| Gb 재현 | 클래스별 값을 query 수로 가중 합산한 (ds, k) 정확도가 `reports/T28_summary.md` 표 T1의 X0, D, TEG, C_all과 \|Δ\| ≤ 0.005 %p (24값) | **멈추고 보고** |
| Gc 표본 | 모든 test 클래스의 n_ep ≥ 100 | 미달 클래스를 표에 표시하고 계속 |
| Gd 변경 범위 | `git diff --stat <H0> <T30 보고서 커밋>` = `tools/analyze_t30.py`, `reports/T30_*`뿐 | 보고서 쓰고 태그 없음 |
| Ge 유한·rank | 모든 지표 유한. base 부분공간 rank를 seed별로 기록 | 해당 값 제외, 기록 |

## 9. 범위 밖

- 학습, 새 arm, 클래스 이름 매핑(데이터 파일에 없음), 해석.
