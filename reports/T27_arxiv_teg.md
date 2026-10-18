# T27 — ogbn-arxiv에서 TEG 원본 vs D·GPN (T26과 같은 에피소드)

2026-10-18 · 브랜치 `dim_control` · H0 `a3d4804` · 지시서 `instructions/T27_arxiv_teg.md` · 요약표 `reports/T27_summary.md` · 그림 `reports/T27_ladder.png`

## 1. 변경 파일과 위치

| 파일 | 내용 |
|---|---|
| `tools/run_t27.py` (신규, `tools/run_t22.py` 형식) | 지시서 §2 명령(T22 `base`, `--dataset ogbn-arxiv`)으로 6 run. `phase1`(5-shot seed 0, GPU 0 단독) / `phase2`(나머지 5 run, GPU 0, 1, 2, 4, 5). stdout을 줄 단위로 `console.log`(원문)와 `console_times.tsv`(경과 초)에 기록(`PYTHONUNBUFFERED=1`). GPU memory.used를 1초마다 `nvidia-smi`로 읽어 `gpu_mem.json`에 최대값 기록. 재시도 1회 |
| `tools/analyze_t27.py` (신규, T26 함수 import) | `a_in`(PyG `gcn_norm(add_self_loops=True, flow="source_to_target")` 희소 행렬), `kd2_check`, `fixed_eps`, `run`(② probe, Kc·Kd, ① 고정 test 출력, 순도), `run_times`, `report`(게이트, 표 T0–T4, 그림, M1–M6, P139–P143 계산값) |
| `instructions/T27_arxiv_teg.md` | §7 표 아래 날짜 부기 1건(사용자 지시, Kd2 기준 변경과 Kg 허용 목록 추가). 표·원문은 그대로 |
| `reports/T27_arxiv_teg.md`, `reports/T27_summary.md`, `reports/T27_ladder.png` | 보고서·요약표·그림 |

- 재사용(import만): `run_t26.load_arxiv`·`fix_degree`·`SPLIT_PKL`, `analyze_t26.s0_arxiv`·`s0_sym`·`l2rows`·`load_gpn_t26`(`tools/analyze_t26.py:48–90`), `analyze_t25.pca`, `analyze_t24.d_acc`·`gpn_acc`(`tools/analyze_t24.py:60`, `:85`), `analyze_t06b.episodes`·`fewshot_acc`, `analyze_t14.knn_purity`, `run_t21.GPUS`.
- 사용자 추가(2026-10-18 승인 시): 기록용 S0_in·D_in(Â_in = PyG `gcn_norm`), 게이트 Kd2, T0의 in-/out-degree 0 노드 수.
- `main.py`, `model.py`, `utils.py`, `embedder.py`, `argument.py`, `layers/`, 기존 `tools/*.py`, `tools/gpn/` 변경 없음. 새 난수 오프셋 없음(TEG 고정 에피소드 9000/9100 + seed, probe 1000 + seed, P16·P64는 T25의 18500). GPN은 재학습하지 않고 T26 `ckpt_best.pt`를 썼다.

## 2. 실행 명령과 출력

| 단계 | 명령 | 시간 | 출력 |
|---|---|---|---|
| 학습 1단계(첫 실행) | `python tools/run_t27.py phase1` | 13:55:51–13:56:22 (31.1 s) | 성공. GPU 메모리 조회 실패(이상 징후 1). 결과는 `results/T27/phase1_nomem/`로 옮겨 보존 |
| 학습 1단계(재실행) | `python tools/run_t27.py phase1` (GPU 0, UUID) | 13:57:58–13:58:28 (30.0 s) | 5-shot seed 0 성공. SPD 구간 9.3 s, GPU 최대 1792 MiB |
| 학습 2단계 | `python tools/run_t27.py phase2` (GPU 0, 1, 2, 4, 5, UUID, 동시 5) | 13:58:37–13:59:09 (32.5 s) | 5/5 성공, 재시도 0. run당 31.3–32.4 s |
| 분석(첫 실행) | `python tools/analyze_t27.py run` (GPU 0) | 13:59:14–14:00:55 (1.7분) | Kd2 원 기준 실패 → 멈추고 보고 |
| 분석(부기 후 재실행) | 같은 명령 | 14:17:28–14:19:13 (1.8분) | `results/T27/analysis_arxiv.json`. acc·pur·kd·kc·fixed 값이 첫 실행과 같음 |
| 보고 | `python tools/analyze_t27.py report` | < 1분 | `results/T27/analysis.json`, 요약표, 그림 |

- 디스크: `results/T27/` 306M(`teg/` 262M, `phase1_nomem/` 44M). 리포 디스크 여유 8.7G.

**run별 GPU (CUDA는 UUID로 지정, `nvidia-smi` 조회는 물리 번호)**

| k | seed | 물리 번호 | UUID | 전체 (s) | GPU 최대 (MiB) |
|---|---|---|---|---|---|
| 5 | 0 | 0 | `GPU-4b1a9f61-18d5-93a0-790e-87ba97de881e` | 30.0 | 1792 |
| 1 | 0 | 0 | `GPU-4b1a9f61-18d5-93a0-790e-87ba97de881e` | 32.4 | 1782 |
| 1 | 1 | 1 | `GPU-510575ba-3107-755b-c130-9eb114d07060` | 31.8 | 1782 |
| 5 | 1 | 2 | `GPU-2b6846d8-a8a0-c83b-dbf5-8bb183352d29` | 32.0 | 1792 |
| 1 | 2 | 4 | `GPU-e2a9ef79-7e90-3f82-2543-f8c2de140545` | 31.3 | 1782 |
| 5 | 2 | 5 | `GPU-b428f48d-97a3-a021-5db8-6abc37367a97` | 32.3 | 1792 |
| 5 | 0 (첫 실행, `phase1_nomem`) | 0 | `GPU-4b1a9f61-18d5-93a0-790e-87ba97de881e` | 31.0 | 조회 실패(0) |

### 게이트

| 게이트 | 결과 | 충족 |
|---|---|---|
| Ka | `t26` 있음. `results/T26/analysis_arxiv.json`, T26 `ckpt_best.pt` 6개 | 예 |
| Kb | 첫 run 30.0 s(SPD 구간 9.3 s), GPU 최대 1792 MiB(24 GB 장치) | 예 |
| Kc | 6 run `split.json` 클래스 = `cls_split.pkl`, `labels.npy` = `load_data` 레이블 | 예 |
| Kd | S0·P16 probe acc vs T26, GPN 출력(T26 probe test 에피소드) vs T26: \|Δ\| 최대 0.0 | 예 |
| Kd2 원 기준 | float32 \|Δ\|max 1.98e-04 > 1e-5 | 아니오 → 멈추고 보고 |
| Kd2 부기 기준 | float64 GCNConv(단위 가중치) X vs Â_in X \|Δ\|max 6.39e-13 ≤ 1e-8. 기록: float32 \|Δ\|max 1.98e-04, Â_in X \|값\| 최대 70.46, 상대 차이 2.82e-06 | 예 |
| Ke | 6 run `ckpt_recheck` epoch = `ckpt_epoch`, n = n_equal = 50 | 예 |
| Kf | 6 run `best_acc_valid` > 첫 에폭 valid. 미달 0 | 예 |
| Kg | `git diff --cached --stat a3d4804`(커밋 전): `instructions/T27_arxiv_teg.md`(부기), `reports/T27_arxiv_teg.md`, `reports/T27_ladder.png`, `reports/T27_summary.md`, `tools/analyze_t27.py`, `tools/run_t27.py`. 부기 기준 허용 목록 안. 커밋 후 값은 부기 | 예 |
| Kh | 지표 전부 유한, `emb_best.npy` NaN 없음. 고정 test 에피소드 6 × 200, query 순서 = 클래스 순서 | 예 |

### 판정 (값만; 표는 `reports/T27_summary.md`)

| 판정 | 정의 | 값 |
|---|---|---|
| M1 | acc_test(TEG 출력) < acc_test(D) (① 출력) | 1/2 |
| M2 | acc_test(TEGe) < acc_test(S0) | 1/2 |
| M3 | acc_base(TEGe) > acc_base(S0) | 1/2 |
| M4 | acc_valid(TEGe) < acc_valid(S0) | 1/2 |
| M5 | acc_test(TEGe) < acc_test(P64) | 2/2 |
| M6 (기록) | acc_test(TEG 출력) − acc_test(GPN 출력) | k = 1: +11.13, k = 5: +7.02 %p |
| M6 (기록) | D_sym 기준 M1 | 2/2 |
| M6 (기록) | pur_test(TEGe) < pur_test(S0) | 1/1 |
| M6 (기록) | D_in 기준 M1 | 1/2 |
| M6 (기록) | acc_test(TEGe) < acc_test(S0_in) | 1/2 |

- M1–M4와 D_in 기준 M1, TEGe < S0_in에서 성립한 칸은 모두 k = 5이고 k = 1은 성립하지 않는다(M3은 k = 1에서 성립, k = 5에서 불성립).

### 사전 등록 (조건과 계산값만)

| 예측 | 조건 | 계산값 |
|---|---|---|
| P139 | M1 = 2/2 | M1 = 1/2 |
| P140 | M2 = 2/2 | M2 = 1/2 |
| P141 | M3 ≥ 1/2 | M3 = 1/2 |
| P142 | M4 ≥ 1/2 | M4 = 1/2 |
| P143 | M5 = 2/2 | M5 = 2/2 |

## 3. 완료 조건 대비

| 조건 | 결과 | 충족 |
|---|---|---|
| H0 커밋(CLAUDE.md·지시서) | `a3d4804` | 예 |
| §1-3 사실 확인(집계 방향) | 표 T0 | 예 |
| TEG 6 run(첫 run 단독 → 나머지 병렬) | 6/6 | 예 |
| ② 표현·① 출력·순도, 표 T0–T4, 그림 | `reports/T27_summary.md`, `reports/T27_ladder.png` | 예 |
| M1–M6, P139–P143 계산값 | 위 표(판정하지 않음) | 예 |
| 게이트 Ka–Kh | Kd2는 원 기준 실패, 부기 기준 충족. 나머지 충족 | 예(부기 기준) |
| 실행 시간·디스크 기록, `results/T27/` 유지 | 2절, 지우지 않음 | 예 |

## 4. 이상 징후

1. GPU 3번 고장("Unknown Error")으로 `nvidia-smi -i <UUID> --query-gpu=memory.used`가 rc=255로 끝나, 첫 phase1 실행의 GPU 메모리 조회가 31회 모두 실패했다(`max_mib` 0). `run_t27.py`의 조회만 물리 번호(`-i <index>`)로 바꿨다. CUDA 장치 지정은 그대로 UUID다. run별 UUID와 물리 번호 대응은 2절 표.
2. 첫 phase1 실행 결과(`ogbn-arxiv_5w5s/seed0`, `_runs.jsonl`, `_driver_phase1.log`)를 `results/T27/phase1_nomem/`로 옮겨 보존하고 phase1을 다시 실행했다. 표·판정은 재실행 결과를 쓴다.
3. 두 phase1 실행의 `fixed_eval_logits.npz`는 바이트가 다르다(`cmp`: byte 48729에서 처음 다름). `run.json` `final.test_acc_at_best_valid`는 두 실행 모두 0.6744다.
4. Kd2: 원 기준(float32 \|Δ\|max ≤ 1e-5)에서 1.98e-04로 실패해 멈추고 보고했다. 사용자 지시로 지시서 §7 아래에 날짜 부기를 달고(float64 \|Δ\|max ≤ 1e-8), `analyze_t27.py`의 Kd2를 float64 계산으로 바꿔 분석을 다시 실행했다. 다시 실행한 acc·pur·kd·kc·fixed 값은 첫 실행과 같고, 요약표에서 바뀐 줄은 Kd2 줄뿐이다.
5. TEG GCN(PyG GCNConv, `source_to_target`)은 in-neighbor를 in-degree 기반 대칭 정규화로 모으고, `load_data` adj(S0·D·GPN이 쓰는 연산자)는 out-neighbor를 행 정규화로 평균한다(T0). in-degree 0 노드 62,006개, out-degree 0 노드 17,440개.
6. SPD 구간은 `console_times.tsv`에서 "Done." 줄부터 그 뒤 두 번째 줄(“Done.\n”의 빈 줄 다음)까지의 시간이다. `embedder.py:61–79`의 그래프 구성과 SPD가 이 구간에 들어 있다.
7. TEG의 `run.json` `wall_time_sec`(16.5–17.5 s)는 드라이버가 잰 전체 시간(30.0–32.4 s)보다 짧다.
8. S0·P64의 순도 sd 0.0000: 클래스 분할과 두 표현이 seed와 무관하다.
9. T2 아래 참고값은 VISION 논문 표의 TEG(5w5s 62.07 ± 1.72)이며, 클래스 분할·에피소드 프로토콜이 다를 수 있다.
10. 그림은 한글 글꼴이 없어 축·눈금 문자를 영어로 적었다. y축은 최소값 − 8에서 시작한다.

## 부기 (2026-10-18, push)

- 보고서 커밋 `3e806cf`, 태그 `t27`. `git push origin dim_control` → `e698d0b..3e806cf  dim_control -> dim_control`, `git push origin t27` → `* [new tag] t27 -> t27`. `git ls-remote`: `refs/heads/dim_control` = `refs/tags/t27` = `3e806cf`. 강제 푸시 없음.
- Kg: `git diff --stat a3d4804 3e806cf` = `instructions/T27_arxiv_teg.md`(+2, 부기), `reports/T27_arxiv_teg.md`, `reports/T27_ladder.png`, `reports/T27_summary.md`, `tools/analyze_t27.py`, `tools/run_t27.py`(6개 파일, 663줄 추가, 삭제 0). 부기 기준 허용 목록 안.
