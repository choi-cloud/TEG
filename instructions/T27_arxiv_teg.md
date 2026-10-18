# T27 — ogbn-arxiv에서 TEG vs D (T26과 같은 에피소드로 공정 비교)

2026-10-18 · 작성: 채팅 Claude · 모드: CLAUDE.md §2(파일 단위 계획 → 승인 → 구현) · 브랜치 `dim_control`(현재 HEAD에서 이어서) · 태그 `t27` · 선행: `t26` · 사전 등록 P139–P143

## 0. 목적과 검증 가능한 주장

T26에서 ogbn-arxiv(입력 128차원)의 GPN이 D에 졌다. 같은 데이터·같은 에피소드에서 TEG 원본을 학습해 D·GPN과 비교한다. 비교는 두 수준이다.

| 수준 | 에피소드 | 비교 |
|---|---|---|
| ② 표현 | T26 probe 에피소드(`analyze_t06b.episodes`, 500/seed, `random.Random(1000 + seed)`, 그룹 base → valid → test) | TEG GCN 임베딩(64) vs S0, P64, P16, G. 모두 코사인 프로토타입 |
| ① 출력 | TEG 고정 test 에피소드(`fixed_eval_logits.npz`의 mode 1, 200/seed, `Random(9000 + seed)`) | TEG 출력 vs D vs GPN 출력. **세 방법을 같은 200개 에피소드에서 잰다** |

S0·D·P16·G·GPN 출력의 정의는 T26과 같다(S0 = 행 L2 정규화(adj²X), adj = `load_data("ogbn-arxiv")`의 행 정규화 D⁻¹(A+I); D_sym·S0_sym = `analyze_t26.s0_sym`). P64 = S0의 PCA 64차원(`analyze_t25.pca`, `random_state=18500`).

**주장.** 4절 표, 5절 판정값, 6절 사전 등록 계산값을 낸다. 해석은 쓰지 않는다. 레이블 사용은 T22·T26과 같다.

## 1. 착수

1. 현재 브랜치가 `dim_control`이고 태그 `t26`이 있는지 확인한다. `git status`에는 사용자가 수정한 `CLAUDE.md`(modified)와 이 지시서(untracked)만 보여야 한다. 아니면 멈추고 보고한다.
2. `CLAUDE.md`와 이 지시서를 H0 커밋으로 넣는다. Agent는 CLAUDE.md를 고치지 않는다.
3. **사실 확인(계획 보고에 포함):** TEG의 GCN 층(`layers/GCN.py`, `embedder.py`에서 edge_index를 넘기는 곳)이 ogbn-arxiv의 방향 엣지를 어떤 방향으로 집계하는지(source → target인지, 정규화가 in-degree 기준인지), 그리고 그것이 `load_data`의 `adj`(행 i가 out-neighbor를 평균)와 같은 방향인지 다른 방향인지를 코드 줄과 함께 적는다. 해석하지 말고 사실만 적는다.
4. 코드 작성 전 파일 단위 계획을 보이고 승인을 기다린다.

## 2. TEG 학습 (`tools/run_t27.py`, `tools/run_t22.py` 형식)

- 6 run: 5-way {1, 5}-shot × seed 0–2. 명령은 T22 `base`와 같다.
  `python main.py --dataset ogbn-arxiv --way 5 --shot <k> --seed <s> --num_seed 1 --device 0 --pres_lambda 0 --pres_pool nb --dump_logits --dump_emb --fixed_eval --out_dir results/T27/teg/ogbn-arxiv_5w<k>s/seed<s>`
- 코드 변경 없음(`main.py`, `model.py`, `utils.py`, `embedder.py`, `argument.py`, `layers/`).
- **시간 점검:** 첫 run(5-shot seed 0)을 GPU 하나에서 단독으로 돌린다. 구조 특징 계산(앵커 SPD)에 걸린 시간, 전체 run 시간, GPU 최대 메모리를 기록한다. **run당 20분을 넘을 것으로 보이거나 메모리가 부족하면 멈추고 보고한다.** 통과하면 나머지 5 run을 병렬로 돌린다(GPU 3번 제외, UUID, 동시 ≤ 6). 실패 run은 한 번 재시도.
- 새 난수 없음. TEG의 고정 에피소드는 기존 9000/9100 + seed.

## 3. 분석 (`tools/analyze_t27.py`, `tools/analyze_t26.py`의 함수 import)

- **② 표현 수준:** T26과 같은 probe 에피소드에서 표현 S0, P64, P16, TEGe(= `results/T27/teg/.../emb_best.npy`, probe shot과 같은 shot의 run)의 코사인 acc를 base·valid·test 그룹별로 잰다. G는 `results/T26/analysis_arxiv.json` 값을 옮긴다(재계산 없음). 순도(top-10 코사인, 그룹별): TEGe(5-shot run).
- **① 출력 수준:** (k, seed)마다 `results/T27/teg/.../fixed_eval_logits.npz`에서 mode 1(test) 200 에피소드의 support·query·classes를 읽는다. 같은 에피소드에서
  - TEG 출력 = npz의 logits argmax 정확도,
  - D = S0 위 support 평균 + 코사인(`fusion_baseline.view_d_scores`와 같은 규칙),
  - D_sym = S0_sym 위 같은 규칙(기록),
  - GPN 출력 = `results/T26/gpn/ogbn-arxiv_5w<k>s/seed<s>/ckpt_best.pt` + `analyze_t24.gpn_acc`(degree 0 → 1, T26과 같음).
  에피소드 단위 짝지은 차(200 × 3 = 600개, 평균 ± SE, 양수 비율): TEG − D, GPN − D, TEG − GPN.
- 분할 점검: TEG run의 `split.json` 클래스 집합이 `cls_split.pkl`과 같은지(Kc).

## 4. 산출 표 (`reports/T27_summary.md`)

| 표 | 내용 |
|---|---|
| T0 | 1절 3의 사실(GCN 집계 방향), 첫 run의 SPD 시간·전체 시간·GPU 메모리, run별 시간 |
| T1 | k × {S0, P64, P16, TEGe, G} × {acc_base, acc_valid, acc_test} (seed 평균 ± sd; G는 T26 전재) |
| T2 | 출력 수준(TEG 고정 test 200 에피소드): k × {D, D_sym, TEG 출력, GPN 출력} (seed 평균 ± sd) + TEG 원 보고 방식 `test_acc_at_best_valid` |
| T3 | 짝지은 차: seed 단위 test/valid/base TEGe − S0, test TEGe − P64; 에피소드 단위 TEG − D, GPN − D, TEG − GPN |
| T4 | 순도: {S0, P64, TEGe} × 그룹 |
| 그림 | `reports/T27_ladder.png`: k = 5에서 S0, P64, TEGe, P16, G의 acc_base·acc_test |

참고값: VISION 논문 표의 TEG(ogbn-arxiv 5w5s 62.07 ± 1.72)를 T2 아래 주석으로 적고, 분할·프로토콜이 다를 수 있다고 적는다.

## 5. 판정 (값만, 단위 k ∈ {1, 5} 2개, seed 평균)

| 판정 | 정의 |
|---|---|
| M1 | acc_test(TEG 출력) < acc_test(D)인 수 (/2, 출력 수준) |
| M2 | acc_test(TEGe) < acc_test(S0)인 수 (/2) |
| M3 | acc_base(TEGe) > acc_base(S0)인 수 (/2) |
| M4 | acc_valid(TEGe) < acc_valid(S0)인 수 (/2) |
| M5 | acc_test(TEGe) < acc_test(P64)인 수 (/2) |
| M6 (기록) | acc_test(TEG 출력) − acc_test(GPN 출력); D_sym 기준 M1; pur_test(TEGe) < pur_test(S0) |

## 6. 사전 등록 (수정 금지)

| 예측 | 내용 | 근거(채팅 해석용) |
|---|---|---|
| P139 | M1 = 2/2 | T26에서 D가 GPN을 2/2 이겼고, 문헌의 TEG(5w5s 62.1)가 T26의 D(70.8)보다 낮다 |
| P140 | M2 = 2/2 | T21·T25·T26 패턴 |
| P141 | M3 ≥ 1/2 | T26에서 1-shot base 이득이 불분명했다 |
| P142 | M4 ≥ 1/2 | T26 L4 = 2/2 |
| P143 | M5 = 2/2 | T25 K3 = 6/6 |

## 7. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ka 선행 | `t26` 존재. `results/T26/analysis_arxiv.json`, `results/T26/gpn/.../ckpt_best.pt` 6개 존재 | 시작하지 않음 |
| Kb 시간·메모리 | 2절 시간 점검 | 멈추고 보고 |
| Kc 분할 | TEG run의 `split.json` 클래스 = `cls_split.pkl`(6 run 모두). 레이블 = `load_data` 레이블 | 멈추고 보고 |
| Kd 재현 | S0·P16의 probe acc가 T26 값과 (k, seed, 그룹)마다 \|Δ\| ≤ 1e-6. GPN 출력을 T26 probe 에피소드에서 다시 재면 T26 값과 \|Δ\| ≤ 1e-6 | 멈추고 보고 |
| Ke ckpt | T22와 같은 `ckpt_recheck` 조건(덤프된 best 에폭 test 에피소드 재평가 일치) | 멈추고 보고 |
| Kf 학습 성립 | run마다 `best_acc_valid`가 첫 에폭 valid보다 높다 | 미달 run 표시, 3개 이상이면 멈추고 보고 |
| Kg 변경 범위 | `git diff --stat <H0> <보고서 커밋>` = `tools/run_t27.py`, `tools/analyze_t27.py`, `reports/T27_*`뿐 | 보고서 쓰고 태그 없음 |
| Kh 유한 | 지표 유한, `emb_best.npy`에 NaN 없음 | 해당 run 제외, 기록 |

2026-10-18 부기: Kd2의 1e-5 절대 기준은 float32에서 |값| 최대 70인 출력에 맞지 않았다. Kd2를 'float64로 계산한 GCNConv(단위 가중치) X와 Â_in X의 |Δ|max ≤ 1e-8'로 바꾼다. float32 |Δ|max와 상대 차이는 기록만 한다. 이에 따라 Kg 허용 목록에 이 지시서 파일(부기만)을 추가한다.

## 8. 보고

- `reports/T27_arxiv_teg.md`(§6 형식), `reports/T27_summary.md`, `reports/T27_ladder.png`.
- 커밋 → 태그 `t27` → push → 부기 커밋. 실행 시간, `results/T27/` 디스크 사용량. `results/T27/`는 지우지 않는다.
- 해석은 쓰지 않는다.
