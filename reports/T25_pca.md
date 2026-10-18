# T25 — Â²X의 PCA(16·64) vs 학습한 임베딩 (학습 없음)

2026-10-18 · 브랜치 `dim_control`(`gpn_backbone` `95e3647`에서 생성) · H0 `59708da` · 지시서 `instructions/T25_T26_QUEUE.md` T25 · 요약표 `reports/T25_summary.md`

## 1. 변경 파일과 위치

| 파일 | 내용 |
|---|---|
| `tools/analyze_t25.py` (신규) | `pca`(S0에 `PCA(n_components=64, svd_solver="randomized", random_state=18500)` 한 번 적합, P64 점수, P16 = 앞 16열, 설명 분산 비율, \|CᵀC − I\|max), `run(ds)`(probe 코사인·유클리드, 순도), `report()`(게이트 Ib·Ic·Ie, 표 T1–T4, K1–K6, P131–P134 계산값) |
| `reports/T25_pca.md`, `reports/T25_summary.md` | 보고서·요약표 |

- 재사용(import만): `fusion_baseline.diffusion_features_hops`(`tools/fusion_baseline.py:141`), `analyze_t06b.episodes`·`fewshot_acc`(`tools/analyze_t06b.py:90`, `:116`), `analyze_t24.fewshot_acc_euc`(`tools/analyze_t24.py:50`), `analyze_t14.knn_purity`(`tools/analyze_t14.py:56`). S3는 `results/T21/analysis_<ds>.json`, G는 `results/T24/analysis_<ds>.json`에서 읽음(재계산 없음).
- 새 난수 오프셋은 18500(PCA `random_state`)뿐. probe 에피소드는 기존 1000 + seed. PCA는 레이블 없이 전체 노드에 적합.
- 학습 코드, 기존 `tools/*.py`, `tools/gpn/` 변경 없음.

## 2. 실행 명령과 출력

| 단계 | 명령 | 시간 | 출력 |
|---|---|---|---|
| 분석 | `CUDA_VISIBLE_DEVICES="" python tools/analyze_t25.py run <ds>` × 3 병렬(CPU) | 12:44:11–12:45:18 (1.1분) | `results/T25/analysis_<ds>.json` |
| 보고 | `python tools/analyze_t25.py report` | < 1분 | `results/T25/analysis.json`, `reports/T25_summary.md` |

- 디스크: `results/T25/` 44K. 리포 디스크 여유 9.0G.

### 게이트

| 게이트 | 결과 | 충족 |
|---|---|---|
| Ia | `t21`, `t24` 있음. `results/T21/analysis_<ds>.json` 3개, `results/T24/analysis_<ds>.json` 3개, T22 `base` split 18개 | 예 |
| Ib | S0 acc_base/valid/test vs T21 S0, 54개 \|Δ\| 최대 0.0 | 예 |
| Ic | \|CᵀC − I\|max = 1.02e-06 / 1.07e-06 / 1.11e-06 (clothing / dblp / electronics, C = `components_.T`, d × 64). 설명 분산 비율 유한, 누적 증가 | 예 |
| Id | `git diff --cached --stat 59708da`(커밋 전): `tools/analyze_t25.py`, `reports/T25_pca.md`, `reports/T25_summary.md`. 커밋 후 값은 부기 | 예 |
| Ie | 지표 전부 유한, 표현 행렬 전부 유한. 제외 칸 없음 | 예 |

### 판정 (값만; 표는 `reports/T25_summary.md`)

| 판정 | 정의 | 값 |
|---|---|---|
| K1 | acc_test(G) < acc_test(P16) | 6/6 |
| K2 | acc_base(G) > acc_base(P16) | 5/6 |
| K3 | acc_test(S3) < acc_test(P64) | 6/6 |
| K4 | acc_base(S3) > acc_base(P64) | 6/6 |
| K5 (기록) | acc_valid(G) < acc_valid(P16); acc_valid(S3) < acc_valid(P64) | 6/6; 6/6 |
| K6 (기록) | acc_test(P16) < acc_test(S0); acc_test(P64) < acc_test(S0) | 4/6; 2/6 |

- K2에서 acc_base(G) ≤ acc_base(P16)인 칸: Amazon_electronics 1-shot(73.95 vs 77.81).
- K6에서 acc_test(P16) ≥ acc_test(S0)인 칸: clothing 1-shot, dblp 1-shot. acc_test(P64) < acc_test(S0)인 칸: dblp 5-shot, electronics 5-shot.

### 사전 등록 (조건과 계산값만)

| 예측 | 조건 | 계산값 |
|---|---|---|
| P131 | K1 ≥ 4/6 | K1 = 6/6 |
| P132 | K2 ≥ 5/6 | K2 = 5/6 |
| P133 | K3 ≤ 3/6 | K3 = 6/6 |
| P134 | K4 ≥ 5/6 | K4 = 6/6 |

## 3. 완료 조건 대비

| 조건 | 결과 | 충족 |
|---|---|---|
| H0 커밋(브랜치 `dim_control`, CLAUDE.md·지시서) | `59708da` | 예 |
| P16·P64 계산, 설명 분산 비율 기록 | 3 데이터셋, T4 | 예 |
| 표 T1–T4 | `reports/T25_summary.md` | 예 |
| K1–K6, P131–P134 계산값 | 위 표(판정하지 않음) | 예 |
| 게이트 Ia–Ie | 전부 충족 | 예 |
| 실행 시간·디스크 기록 | 2절 | 예 |

## 4. 이상 징후

1. P16은 64성분 적합의 앞 16개 성분 점수다(16성분을 따로 적합하지 않음, 지시서 §2대로).
2. PCA 점수는 평균 중심화된 S0의 좌표이며, probe 전에 다시 정규화하지 않았다(코사인 probe는 중심화된 좌표 위에서 계산).
3. 누적 설명 분산(16 / 64): clothing 0.3209 / 0.5344, dblp 0.4354 / 0.6501, electronics 0.2697 / 0.4753.
4. S0·P64·P16의 pur_test seed 간 sd가 0.0000이다. 세 표현이 분할과 무관하고 test 클래스 집합이 seed 0–2에서 같다(T21 이상 징후 7과 같은 사정).
5. `diffusion_features_hops`가 내부의 `utils.load_data`로 분석 프로세스의 전역 `random`을 소비한다(T21과 같은 기존 경로).
6. `analyze_t24.fewshot_acc_euc`를 import하면서 `run_t24`(→ `tools/gpn/models.py`)도 불러온다. GPN 코드는 실행하지 않는다.

## 부기 (2026-10-18, push)

- 보고서 커밋 `540bc3b`, 태그 `t25`. `git push origin dim_control` → `* [new branch] dim_control -> dim_control`, `git push origin t25` → `* [new tag] t25 -> t25`. `git ls-remote`: `refs/heads/dim_control` = `refs/tags/t25` = `540bc3b`. 강제 푸시 없음.
- Id: `git diff --stat 59708da 540bc3b` = `reports/T25_pca.md`, `reports/T25_summary.md`, `tools/analyze_t25.py`(3개 파일, 392줄 추가, 삭제 0). 범위 안.
