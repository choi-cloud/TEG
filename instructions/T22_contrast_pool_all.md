# T22 — 대조(InfoNCE)를 전체 노드 풀(`--pres_pool all`)로 재실행

2026-10-15 · 작성: 채팅 Claude · 모드: 채팅 지시문을 따른다 · 브랜치 `pool_all`(`erasure_ladder` HEAD에서 생성) · 태그 `t22` · 선행: `t21`

## 0. 목적과 검증 가능한 주장

**배경.** T09·T10r·T14의 대조(InfoNCE)와 보존(KL)은 모두 `--pres_pool nb`(base가 아닌 노드만)로 돌렸다. 2026-10-14 결정(CLAUDE.md §3 "unlabeled 노드 사용 범위")에 따라 새 실험은 전체 노드 풀을 쓴다. T07에서 보존은 nb와 all의 차이가 0.7 %p 이하였지만 대조는 확인하지 않았다. 이 지시서는 T10r의 대조 실험을 풀만 `all`로 바꿔 다시 돌리고, 같은 코드·같은 실행 묶음에서 `nb`도 함께 돌려 짝지은 비교를 만든다.

| 설정 | 풀 | λ 격자 | 비고 |
|---|---|---|---|
| `base` | — | 0 | 원본 TEG. 같은 묶음 안의 기준 |
| `infonce_nb_l{λ}` | nb | {10, 30, 100} | T10r 대조와 같은 명령. 같은 코드에서의 재실행 |
| `infonce_all_l{λ}` | all | {10, 30, 100} | **이번의 주 설정** |

보존(KL)은 이번에 돌리지 않는다.

**주장.** 4절의 표와 5절의 판정값, 6절의 사전 등록 대조를 산출한다. 해석은 하지 않는다. 레이블 사용은 T10r과 같다(base 레이블은 학습, valid는 선택, test는 평가만).

## 1. 착수

1. 현재 브랜치가 `erasure_ladder`이고 `t21` 태그가 있는지 확인한다. `git status`에는 사용자가 수정한 `CLAUDE.md`(modified)와 이 지시서 `instructions/T22_contrast_pool_all.md`(untracked)만 보여야 한다. 다른 것이 보이면 멈추고 보고한다.
2. `git switch -c pool_all`.
3. `CLAUDE.md`와 이 지시서를 H0 커밋으로 넣는다. Agent는 CLAUDE.md를 고치지 않는다. H0 해시가 Ge 게이트의 기준이다.
4. 코드 작성 전 파일 단위 계획을 보이고 승인을 기다린다(CLAUDE.md §2).

## 2. 실행 (`tools/run_t22.py`, `tools/run_t10r.py` 복사)

- 설정 7개 × 3 데이터셋 × 5-way {1, 5}-shot × seed 0–9 = **420 run**. T10r과 같은 규모다.
- 명령은 T10r과 같다. `--pres_pool`만 설정에 따라 `nb` 또는 `all`을 **명시**한다(`base`는 `nb`를 명시, λ = 0이라 풀을 만들지 않는다):
  `python main.py --dataset <ds> --way 5 --shot <k> --seed <s> --num_seed 1 --device 0 --pres_lambda <λ> --pres_pool <nb|all> --pres_loss infonce --dump_logits --fixed_eval --out_dir results/T22/<cfg>/<ds>_5w<k>s/seed<s>`
  (`base`는 `--pres_lambda 0`이고 `--pres_loss`를 주지 않는다. T10r의 `base` 명령과 같다.)
- seed 0–2에 한해 `--dump_emb`를 더한다(이후 순도 분석용). seed 3–9는 넣지 않는다.
- 큐 순서: T10r과 같이 seed 우선(seed 0의 설정 7개 → seed 1 → …). 데이터셋 순서 Clothing → dblp → Electronics, 1-shot 먼저.
- T10r의 마감 시각 로직은 뺀다. 디스크 여유 1.5 GB 점검은 남긴다.
- GPU: 3번 제외, 동시 실행 ≤ 6, UUID 지정. 실패한 run은 `console_retry1.log`로 한 번 재시도한다.
- 새 난수 없음. 풀 표본은 기존 전용 생성기(7000 + seed), 증강은 기존 생성기(8000 + seed)를 그대로 쓴다.
- 코드 변경 없음(`model.py`, `utils.py`, `argument.py`는 건드리지 않는다). `--pres_pool all`은 기존 경로다(`model.py:81–85`).

## 3. 분석 (`tools/analyze_t22.py`, `tools/analyze_t10r.py` 복사)

- 고정 에피소드(`fixed_eval_logits.npz`, seed당 test 200)에서 T10r과 같은 방식으로 TEG 정확도, D, F(cfg)(`fusion_baseline.evaluate_npz`, α = 0.5, log)를 구한다.
- λ 선택: 풀별로 따로, (풀, 데이터셋, shot)마다 `best_acc_valid` seed 평균 최고, 동률이면 작은 λ. CLAUDE.md §5와 같다.
- 아래 기호를 쓴다. C_nb = 선택 λ의 `infonce_nb`, C_all = 선택 λ의 `infonce_all`, TEG = `base`.
- 재현 대조(기록 + 게이트 Gf): T22 `infonce_nb_l{λ}`와 T10r `infonce_l{λ}`를 같은 λ·같은 seed끼리 비교한다. λ는 T10r 선택 λ를 쓴다.

## 4. 산출 표 (`reports/T22_summary.md`)

| 표 | 내용 |
|---|---|
| T1 | 원 보고 방식(`test_acc_at_best_valid`): 셀 × {TEG, C_nb, C_all}, 선택 λ(nb / all), seed 평균 ± sd, 95 % CI |
| T2 | 고정 에피소드: 셀 × {TEG, D, F(base), C_nb, C_all, F(C_nb), F(C_all)} (seed 평균 ± sd) |
| T3 | 고정 에피소드 짝지은 차(평균 ± SE, 양수 비율, seed 부호 일치/10): C_all − TEG, C_all − C_nb, C_all − D, C_all − F(base), F(C_all) − F(base), F(C_all) − F(C_nb) |
| T4 | λ 곡선: 셀 × {nb, all} × λ ∈ {10, 30, 100} → `best_acc_valid`, 고정 test 정확도 (seed 평균 ± sd) |
| T5 | 재현 대조: 셀 × λ(T10r 선택) → T22 nb seed 평균, T10r seed 평균, 차, seed 단위 차의 평균 ± SE |

## 5. 판정 (값만. 단위 (데이터셋, shot) 6개. "분명" = 평균 > 0이고 > 2·SE, 고정 에피소드)

| 판정 | 정의 |
|---|---|
| K1 | C_all − TEG가 분명히 양수인 수 |
| K2 | C_all − D가 분명히 양수인 수; 평균 > 0인 수(기록) |
| K3 | \|C_all − C_nb\| 평균 ≤ 1.0 %p인 수; C_all − C_nb가 분명히 음수인 수(기록) |
| K4 | F(C_all) − F(base)가 분명히 양수인 수 |
| K5 (기록) | C_all − F(base)가 분명히 양수인 수 |
| K6 (기록) | 선택 λ가 nb와 all에서 같은 셀 수 |

## 6. 사전 등록 (수정 금지, 어긋나면 날짜 있는 부기)

| 예측 | 내용 | 근거(채팅 해석용) |
|---|---|---|
| P112 | K1 = 6/6 | T10r N1(nb) = 6/6 |
| P113 | K2(분명) ≥ 4/6 | T10r에서 nb 대조 − D의 평균이 5/6에서 양수 |
| P114 | K3 ≥ 5/6 | T07 보존 nb vs all 차이 ≤ 0.7 %p |
| P115 | K4 = 6/6 | T10r F(InfoNCE) − F(base) 6/6 |

## 7. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ha 선행 | `t21` 존재. T10r의 `infonce_l{10,30,100}` seed 0–9 `run.json`·`fixed_eval_logits.npz` 존재(Gf용) | 시작하지 않음 |
| Gb 에피소드 동일 | (셀, seed)마다 설정 7개의 `eval_logits.npz`·`fixed_eval_logits.npz` 에피소드 키가 같다(T10r Ga와 같은 점검) | 멈추고 보고 |
| Gc ckpt | `ckpt_recheck`가 T10r Gb와 같은 조건을 만족 | 멈추고 보고 |
| Gd 풀 기록 | `run.json` `config.pres_pool`이 설정 이름과 일치(`nb`/`all`). λ > 0인 run의 에폭별 `pres_loss_mean`이 유한 | 멈추고 보고 |
| Ge 변경 범위 | `git diff --stat <H0> <보고서 커밋>`의 파일이 `tools/run_t22.py`, `tools/analyze_t22.py`, `reports/`, `instructions/` 안에만 있다 | 멈추고 보고 |
| Gf 재현 | T5에서 6셀 모두 \|T22 nb seed 평균 − T10r seed 평균\| ≤ 2.0 %p(고정 에피소드, T10r 선택 λ) | 분석은 끝내되 판정·사전 등록 대조 전에 멈추고 보고 |

## 8. 보고

- `reports/T22_pool_all.md`(§6 형식), `reports/T22_summary.md`.
- 커밋 → 태그 `t22` → push → 부기 커밋.
- 실행 시간(분), `results/T22/` 디스크 사용량을 적는다.
- `results/T22/`는 지우지 않는다.
