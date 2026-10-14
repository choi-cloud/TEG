# T21 — 지움의 사다리: 학습 0 → 에피소드 없는 base 지도 → 에피소드(EGNN 없음) → TEG

2026-10-14 · 브랜치 `erasure_ladder` · H0 `377d5f3` · 지시서 `instructions/T21_erasure_ladder.md` · 요약표 `reports/T21_summary.md` · 그림 `reports/T21_ladder.png`

## 1. 변경 파일과 위치

| 파일 | 내용 |
|---|---|
| `tools/run_t21.py` (신규) | `tools/run_t09.py` 복사. config `gamma0` 하나(T09 `base` 명령 + `--gamma 0`, `--pres_pool nb` 없음), 마감 시각 로직 제거. 18 run |
| `tools/ladder_t21.py` (신규) | S1. Â²X(`fusion_baseline.diffusion_features_hops(ds, 2)`) → `Linear(d, 64)` → ReLU → `Linear(64, \|base\|)`, base 노드 전부 full-batch CE, Adam lr 5e-3, wd 5e-4, 1000 에폭(부기 후), 스냅샷 {0, 25, 50, 100, 200, 500, 1000} |
| `tools/analyze_t21.py` (신규) | probe(`analyze_t06b.episodes`·`fewshot_acc`), LR head, 순도(`analyze_t14.knn_purity`), 게이트 Hb–Hf, 표 T1–T5, J1–J6·J1′·J3′·J5′, P107–P111, 그림 |
| `tools/analyze_t21.py` 수정(실행 전) | `knn_purity`에 넘기는 레이블을 `labels` 전체에서 `labels[nodes]`로 고침(노드 부분집합과 정렬, `analyze_t13.groups_of`와 같은 방식) |
| `instructions/T21_erasure_ladder.md` | 날짜 부기 2건만 추가(§2 정규화, §7 Hc(1)). 원문 유지 |
| `reports/T21_ladder.md`, `reports/T21_summary.md`, `reports/T21_ladder.png` | 보고서·요약표·그림 |

- S1 초기화: 파라미터를 `nn.Linear`로 만들지 않고(생성자가 torch 기본 생성기를 소비) CPU `torch.Generator().manual_seed(16000 + seed)`에서 W1, b1, W2, b2 순으로 각각 uniform(−1/√fan_in, +1/√fan_in)으로 직접 뽑음. 새 오프셋은 16000뿐.
- `model.py`, `utils.py`, `argument.py` 변경 없음. S2는 기존 `--gamma` 플래그(`argument.py:30`)만 사용.
- `fusion_baseline.diffusion_features_hops`는 내부에서 `utils.load_data`를 호출하므로 분석·S1 프로세스 안에서 전역 `random`을 소비한다(T12와 같은 기존 경로, TEG 학습 프로세스와 별개).

## 2. 실행 명령과 출력

| 단계 | 명령 | 시간 | 출력 |
|---|---|---|---|
| S2 | `python tools/run_t21.py` (GPU 0, 1, 2, 4, 5, 6, UUID) | 05:50:30–05:51:59 (1.5분) | 18/18 성공, 재시도 0. run당 25–30 s |
| S1 (첫 실행, 200 에폭) | `CUDA_VISIBLE_DEVICES=<GPU 0 UUID> python tools/ladder_t21.py` | 05:52–05:53 | Hc(1) 원 기준 실패 → 멈추고 보고 |
| S1 (재실행, 1000 에폭) | 같은 명령 | 05:58:38–05:59:36 (seed당 2.9–4.5 s, 합 33.8 s) | 9/9. ep0–200 재현 |
| 분석 | `python tools/analyze_t21.py run <ds>` × 3 병렬(CPU) | 06:00:08–06:47:59 (48분) | `results/T21/analysis_<ds>.json` |
| 보고 | `python tools/analyze_t21.py report` | < 1분 | `results/T21/analysis.json`, 요약표, 그림 |

- 디스크: `results/T21/` 734M(`gamma0` 180M, 그중 `emb_best.npy` 159M; `mlp` 554M). 리포 디스크 여유 12G.

### 게이트

| 게이트 | 결과 |
|---|---|
| Ha | `t19b` 있음. T09 `base`·`infonce_l10` seed 0–2 `emb_best.npy` 36개 있음 |
| Hb | 3 데이터셋 × seed 0–2 × k ∈ {1, 5}: T21 `gamma0`·T09 `infonce_l10`의 `split.json`·`labels.npy`가 T09 `base`와 같음. S1이 쓴 1-shot 분할 = 5-shot 분할 |
| Hc(1) 원 기준 (ep200 ≥ 80 %) | 실패: dblp 56.17 / 52.04 / 55.75, Amazon_electronics 79.84 / 80.35 / 76.61 (seed 0 / 1 / 2). Amazon_clothing 84.26 / 84.77 / 84.67 |
| Hc(1) 부기 기준 | 9/9 충족. 최종(ep1000) − ep0 = 77.67–87.49 %p(최소 Amazon_electronics seed 2), 손실이 스냅샷 7개에서 단조 감소 |
| Hc(1) 재현 | ep200 base 학습 정확도 \|Δ\| = 0.000 %p (9/9). ep0–200 정확도 최대 \|Δ\| 0, 손실 최대 \|Δ\| 0 |
| Hc(2) | acc_test(k = 5) T21 − T12 표 2: S0 +0.004 / −0.001 / −0.004, S3 +0.003 / +0.005 / −0.003 (clothing / dblp / electronics) |
| Hd | 18 run `config.gamma` = 0.0, `final.test_acc_at_best_valid` 기록됨 |
| He | `git diff --stat 377d5f3 6fadbbc`: 7개 파일, 전부 `tools/run_t21.py`, `tools/ladder_t21.py`, `tools/analyze_t21.py`, `reports/`, `instructions/` 안 (701줄 추가, 삭제 0). 충족 |
| Hf | 지표 909개 유한, 표현 행렬 전부 유한. 제외 run 없음 |

### 판정과 사전 등록 (값만; 표는 `reports/T21_summary.md`)

| 판정 | 값 |
|---|---|
| J1 | 5/6 |
| J2 | 6/6 |
| J3 | 6/6 |
| J4 | 0/6; k = 1에서 0/3 |
| J5 | S1 ep200 2/3; S2 3/3 |
| J1′ / J3′ / J5′ (기록, ep1000) | 5/6 / 6/6 / 2/3 |
| J6 (S2 − S3, %p, 평균 ± SE) | clothing 1s −1.46 ± 1.25, 5s −1.25 ± 1.03; dblp 1s −1.60 ± 0.38, 5s −1.18 ± 0.35; electronics 1s −4.25 ± 0.70, 5s −2.04 ± 0.63 |

| 예측 | 조건 | 관측 | 일치 |
|---|---|---|---|
| P107 | J1 ≥ 5/6 | 5/6 | 일치 |
| P108 | J2 ≥ 5/6 | 6/6 | 일치 |
| P109 | J3 ≥ 5/6 | 6/6 | 일치 |
| P110 | J4 ≥ 4/6, k = 1에서 3/3 | 0/6, 0/3 | 어긋남 |
| P111 | J5 둘 다 3/3 | 2/3, 3/3 | 어긋남 |

- J1에서 acc_test(S1 ep200) ≥ acc_test(S0)인 칸: dblp 1-shot(76.65 vs 72.50). J5에서 pur_test(S1 ep200) ≥ pur_test(S0)인 데이터셋: dblp(0.5903 vs 0.5836).

## 3. 완료 조건 대비

| 조건 | 결과 | 충족 |
|---|---|---|
| H0 커밋(브랜치 `erasure_ladder`, CLAUDE.md·지시서) | `377d5f3` | 예 |
| S2 18 run | 18/18 | 예 |
| S1 학습·스냅샷 | 9/9, 스냅샷 7개 | 예 |
| 표 T1–T5, 그림 1 | `reports/T21_summary.md`, `reports/T21_ladder.png` | 예 |
| J1–J6, P107–P111 대조 | 위 표 | 예 |
| 게이트 Ha–Hf | Hc(1)은 원 기준 실패, 사용자 부기 기준 충족. 나머지 충족 | 예(부기 기준) |
| 실행 시간·디스크 기록 | 2절 | 예 |
| `results/T21/` `emb_best.npy` 유지 | 지우지 않음 | 예 |

## 4. 이상 징후

1. Hc(1) 원 기준(S1 ep200 base 학습 정확도 ≥ 80 %)이 dblp(3 seed)와 Amazon_electronics(seed 0, 2)에서 실패해 멈추고 보고했다. 사용자 지시로 §7에 날짜 부기(대체 기준)를 달고 S1을 1000 에폭으로 다시 학습했다(lr·wd·초기화 동일). 첫 실행의 `run.json`은 `results/T21/mlp/<ds>/seed<s>/run_ep200first.json`, 로그는 `_driver_ep200first.log`로 남겼고 `h_ep{0,25,50,100,200}.npy`는 재실행 값으로 덮어썼다. 재실행 ep0–200 정확도·손실은 첫 실행과 완전히 같다. P107–P111과 J1–J6은 ep200 기준이다.
2. S2 명령에 `--pres_pool nb`를 넣지 않았다(지시서 §3.2 명령에 없음). 기본값이 `nb`이고 λ = 0이면 pool을 만들지 않는다(`model.py:76`). `run.json` `config.pres_pool`은 `"nb"`로 기록됐다.
3. 지시서 §3.3은 대조의 선택 λ 출처를 `reports/T09_summary.md` 표 2로 적었다. `reports/T12_summary.md` 표 2의 선택 λ(여섯 칸 모두 10)를 썼다. 지시서는 고치지 않았다.
4. `diffusion_features_hops`가 내부의 `utils.load_data`로 분석·S1 프로세스의 전역 `random`을 소비한다(1절).
5. 지시서 §2 "probe 전 행 L2 정규화"는 사용자 결정으로 적용하지 않았다(날짜 부기). 모든 표현을 T12와 같이 `fewshot_acc`에 그대로 넣었다. 미리 정규화한 값은 내지 않았다.
6. Hc(2)의 비교 대상은 T12 표 2의 소수 둘째 자리 반올림 값이다. 차이 최대 \|0.005\| %p.
7. S0의 pur_test seed 간 sd가 0.0000이다(3 데이터셋). S0는 seed와 무관하고 test 클래스 집합이 seed 0–2에서 같다.
8. 그림 1은 한글 글꼴이 없어(`DejaVu Sans`만 있음) 축·눈금 문자를 영어로 적었다(S3 = "TEG", 대조 = "InfoNCE λ=10"). y축은 0에서 시작하지 않는다(패널별 최소값 − 8).
9. 분석은 CPU 3 프로세스로 48분 걸렸다(S0 고차원 표현의 probe·LR·순도).

## 부기 (2026-10-14, push)

- 보고서 커밋 `6fadbbc`, 태그 `t21`. `git push origin erasure_ladder` → `* [new branch] erasure_ladder -> erasure_ladder`, `git push origin t21` → `* [new tag] t21 -> t21`. `git ls-remote`: `refs/heads/erasure_ladder` = `refs/tags/t21` = `6fadbbc`. 강제 푸시 없음.
- He: `git diff --stat 377d5f3 6fadbbc` = `instructions/T21_erasure_ladder.md`(+3), `reports/T21_ladder.md`, `reports/T21_ladder.png`, `reports/T21_summary.md`, `tools/analyze_t21.py`, `tools/ladder_t21.py`, `tools/run_t21.py`. 범위 안.
