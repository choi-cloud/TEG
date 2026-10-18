# T26 — ogbn-arxiv(입력 128차원)에서 D vs GPN

2026-10-18 · 브랜치 `dim_control` · H0 `59708da` · T25 마지막 커밋 `e350ce7` · 지시서 `instructions/T25_T26_QUEUE.md` T26 · 요약표 `reports/T26_summary.md` · 그림 `reports/T26_ladder.png`

## 1. 변경 파일과 위치

| 파일 | 내용 |
|---|---|
| `tools/run_t26.py` (신규) | `load_arxiv`(`utils.load_data("ogbn-arxiv")`, ROOT에서), `fix_degree`(degree 0 → 1), `load_inputs_arxiv`(`run_t24.load_inputs`와 같은 반환 형태, 클래스는 `cls_split.pkl`, 레이블은 `load_data`), `train_one`(`run_t24.load_inputs`·`run_t24.RESULTS`를 바꿔 끼우고 `run_t24.train_one`을 그대로 호출, 끝난 뒤 `run.json`에 GPU 최대 메모리 추가), `hc`(Jb, `run_t24.hc_check` 재실행), 드라이버 `phase1`(5-shot seed 0 단독, GPU 0) / `phase2`(나머지 5 run, GPU 0, 1, 2, 4, 5) |
| `tools/analyze_t26.py` (신규, `tools/analyze_t24.py` 복사 후 수정) | `s0_arxiv`(load_data adj로 2-hop, 행 L2 정규화), `s0_sym`(A ∪ Aᵀ 중복 제거 + I, 대칭 정규화 2-hop, 행 L2 정규화), `load_gpn_t26`, `run`, `report`(게이트 Jb·Jc·Jf, 표 T0–T5, 그림, L1–L5, P135–P138 계산값) |
| `reports/T26_arxiv.md`, `reports/T26_summary.md`, `reports/T26_ladder.png` | 보고서·요약표·그림 |

- 재사용(import만): `run_t24`의 `task`·`gpn_logp`·`train_step`·`evaluate`·`build_models`·`train_one`·`hc_check`(`tools/run_t24.py:66–258`), `analyze_t24.d_acc`·`gpn_acc`(`tools/analyze_t24.py:60`, `:85`), `analyze_t25.pca`, `analyze_t06b.episodes`·`fewshot_acc`, `analyze_t14.knn_purity`, `run_t21.GPUS`.
- S0_sym은 사용자 조정(2026-10-18)에 따라 `fusion_baseline.diffusion_features_hops`를 부르지 않고 `analyze_t26.s0_sym`에서 직접 계산했다. D_sym도 이 S0_sym으로 계산했다.
- 난수: GPN 학습은 T24와 같은 18000(학습 에피소드)·18100(valid task)·18200(test task)·18300(초기화). P16은 18500. probe 에피소드는 기존 1000 + seed.
- 학습 코드, 기존 `tools/*.py`, `tools/gpn/`, `fusion_baseline.py` 변경 없음. `hc`는 실행 중에만 `run_t24.ROOT`를 `results/T26/_hc`로 바꿔, `results/T24/hc_check.json`을 덮어쓰지 않았다(실행 전후 md5 `a998000c…` 같음).

## 2. 실행 명령과 출력

| 단계 | 명령 | 시간 | 출력 |
|---|---|---|---|
| Jb | `CUDA_VISIBLE_DEVICES="" python tools/run_t26.py hc` | < 1분 | `results/T26/hc_check.json` |
| 학습 1단계 | `python tools/run_t26.py phase1` (GPU 0, UUID) | 12:47:07–12:47:43 (33.5 s) | 5-shot seed 0 성공. 학습 30.6 s, GPU 최대 할당 364 MB / 예약 398 MB |
| 학습 2단계 | `python tools/run_t26.py phase2` (GPU 0, 1, 2, 4, 5, UUID, 동시 5) | 12:47:52–12:53:45 (5.9분) | 5/5 성공, 재시도 0. run당 331–351 s |
| 분석 | `python tools/analyze_t26.py run` (GPU 0) | 12:53:52–12:55:03 (1.2분) | `results/T26/analysis_arxiv.json` |
| 보고 | `python tools/analyze_t26.py report` | < 1분 | `results/T26/analysis.json`, 요약표, 그림 |

- 디스크: `results/T26/` 67M. 리포 디스크 여유 9.0G.

### 게이트

| 게이트 | 결과 | 충족 |
|---|---|---|
| Ja | ogb 1.3.6, `dataset/ogbn_arxiv/`(`cls_split.pkl` 있음), `load_data("ogbn-arxiv")` 성공(내려받기 없음), 그룹별 최소 클래스 크기 29 / 411 / 127 ≥ 25 | 예 |
| Jb | `run_t24` 함수 import 사용. `run_t24.hc_check` 재실행: 손실·파라미터·test log-prob·정확도 \|Δ\| 최대 0.0 | 예 |
| Jc | 6 run 모두 마지막 100 에피소드 학습 정확도 > 0.30(최소 1-shot seed 1 0.3784), best valid > valid@10. 미달 0 | 예 |
| Jd | 첫 run 33.5 s, GPU 메모리 364 MB(24 GB 장치) | 예 |
| Je | `git diff --cached --stat e350ce7`(커밋 전): `tools/run_t26.py`, `tools/analyze_t26.py`, `reports/T26_arxiv.md`, `reports/T26_summary.md`, `reports/T26_ladder.png`. 커밋 후 값은 부기 | 예 |
| Jf | 지표 전부 유한, `emb_best.npy` NaN 없음, 제외 run 없음 | 예 |

### 판정 (값만; 표는 `reports/T26_summary.md`)

| 판정 | 정의 | 값 |
|---|---|---|
| L1 | acc_test(GPN 출력) < acc_test(D) | 2/2 |
| L2 | acc_test(G) < acc_test(S0) | 2/2 |
| L3 | acc_base(G) > acc_base(S0) | 2/2 |
| L4 | acc_valid(G) < acc_valid(S0) | 2/2 |
| L5 (기록) | acc_test(G) < acc_test(P16); pur_test(G) < pur_test(S0); D_sym 기준 L1 | 2/2; 1/1; 2/2 |

- L3의 1-shot: acc_base G 41.33 vs S0 40.47, seed 단위 차 +0.86 ± 2.07, 부호 일치 1/3.

### 사전 등록 (조건과 계산값만)

| 예측 | 조건 | 계산값 |
|---|---|---|
| P135 | L3 = 2/2 | L3 = 2/2 |
| P136 | L2 ≥ 1/2 | L2 = 2/2 |
| P137 | L4 ≥ 1/2 | L4 = 2/2 |
| P138 | L1 ≤ 1/2 | L1 = 2/2 |

## 3. 완료 조건 대비

| 조건 | 결과 | 충족 |
|---|---|---|
| 사전 확인(§2) 결과 기록 | 표 T0 | 예 |
| GPN 6 run(첫 run 단독 → 나머지 병렬), 산출 파일 6종 | 6/6 | 예 |
| 표 T0–T5, 그림 | `reports/T26_summary.md`, `reports/T26_ladder.png` | 예 |
| L1–L5, P135–P138 계산값 | 위 표(판정하지 않음) | 예 |
| 게이트 Ja–Jf | 전부 충족 | 예 |
| 실행 시간·GPU 메모리·디스크 기록 | 2절, 표 T5 | 예 |
| `results/T26/` 유지 | 지우지 않음 | 예 |

## 4. 이상 징후

1. ogbn-arxiv `edge_index`는 방향 그래프다(역방향 엣지가 있는 비율 0.0145). `load_data`의 `adj`(행 정규화 D⁻¹(A+I))와 `degree`(A 행합)는 out-neighbor 기준이다. out-degree 0 노드 17,440개는 adj 행이 자기 루프뿐이라 전파 후에도 자기 특징만 남는다(S0 행 = 행 정규화 X, 확인함).
2. GPN에서 degree 0 노드 17,440개는 지시서 §4대로 degree 1로 두었다(support 가중치 0.5 고정). degree 1 노드 27,853개도 같은 가중치 0.5다. 분석의 GPN 출력도 같은 degree를 썼다.
3. S0_sym: A ∪ Aᵀ 비대칭 항목 2,315,598개(무방향 쌍 1,157,799개), 대칭화 후 degree 0 노드 0개.
4. 1단계 단독 run은 33.5 s, 2단계 병렬 5 run은 run당 331–351 s였다. 2단계 중 load average 최대 약 73(`uptime` 1분·5분 값 30.58 / 72.92). GPU 최대 메모리는 6 run 모두 할당 364 MB / 예약 398 MB.
5. 1-shot seed 1의 best 에피소드는 20이다(valid 최고 40.34, valid@10 39.64).
6. S0의 순도 sd 0.0000: 클래스 분할이 seed와 무관하고 S0도 seed와 무관하다.
7. PCA(S0) 누적 설명 분산 16 / 64 = 0.6134 / 0.8784.
8. T2 "참고: 문헌값"은 TENT 논문 표의 GPN(5w5s 53.06 ± 3.13)이며 클래스 분할·에피소드 프로토콜이 다를 수 있다. 1-shot 문헌값은 지시서에 없어 비워 두었다.
9. `ckpt_best.pt`에서 다시 계산한 임베딩·점수와 저장본의 \|Δ\| 최대 2.4e-7. `ckpt_best.pt`의 episode는 6/6 `run.json` best 에피소드와 같다.
10. 그림은 한글 글꼴이 없어 축·눈금 문자를 영어로 적었다. y축은 최소값 − 8에서 시작한다.
11. Jb(합성 점검)는 리포 밖 `/tmp/gpn_upstream`의 업스트림 원문을 읽어 실행한다(T24와 같음).
