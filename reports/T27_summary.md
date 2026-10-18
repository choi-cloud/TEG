# T27 요약 — ogbn-arxiv에서 TEG 원본 vs D·GPN

- 표현: S0 = 행 L2 정규화(adj²X), adj = `load_data("ogbn-arxiv")` 행 정규화 D⁻¹(A+I)(out-neighbor). S0_in(기록) = 행 L2 정규화(Â_in²X), Â_in = PyG `gcn_norm(edge_index, add_self_loops=True, flow="source_to_target")`(TEG GCNConv와 같은 연산자, in-neighbor). P64 / P16 = S0의 PCA(`random_state=18500`). TEGe = TEG 원본 GCN 임베딩(64, `results/T27/teg/.../emb_best.npy`, probe shot과 같은 shot의 run). G = GPN 임베딩(T26 `analysis_arxiv.json` 전재). seed 0–2.
- ② probe: `analyze_t06b.episodes` + `fewshot_acc`(코사인 프로토타입, 500 에피소드, `random.Random(1000 + seed)`, 그룹 base → valid → test, T26과 같은 에피소드).
- ① 출력: TEG 고정 test 에피소드(`fixed_eval_logits.npz` mode 1, 200 / seed, `Random(9000 + seed)`). TEG 출력 = 덤프된 logits argmax, D / D_sym / D_in = S0 / S0_sym / S0_in 위 support 평균 + 코사인, GPN 출력 = T26 `ckpt_best.pt` + 업스트림 `test()` 수식(degree 0 → 1). 정확도 %.

## T0 — 사실과 실행
| 항목 | 값 |
|---|---|
| TEG GCN에 넘기는 엣지 | `load_data`의 `edges`(OGB `edge_index` 원 방향) 그대로 (`embedder.py:25–37`, `model.py:354`, `utils.py:155`) |
| TEG GCN 층 | `GCNConv(..., cached=True, normalize=True)`, `flow` 기본값 `source_to_target` (`layers/GCN.py:10`) |
| TEG GCN 집계 방향 | 노드 t가 엣지 (s → t)의 s(edge_index[0])를 모은다 = in-neighbor (PyG `message_passing.py:343`, `:408`; `gcn_conv.py:270–271`) |
| TEG GCN 정규화 | 자기 루프 추가 후 deg = edge_index[1] 기준(in-degree + 1), 가중치 deg^-½[row]·deg^-½[col] (PyG `gcn_conv.py:98–111`) |
| `load_data` adj | A[edge_index[0], edge_index[1]] = 1, 행 정규화 D⁻¹(A+I): 행 i = out-neighbor와 자신의 평균 (`utils.py:159–162`, `:227–234`) |
| 두 연산자의 방향 | 서로 반대(TEG GCN = in-neighbor, load_data adj = out-neighbor) |
| 앵커 SPD | 무방향 `nx.Graph`, 앵커 16개 BFS (`embedder.py:44–79`) |
| in-degree(edge_index[1] 기준) 0 노드 수 | 62,006 |
| out-degree(edge_index[0] 기준) 0 노드 수 | 17,440 |
| Kd2(부기 기준): float64 GCNConv(128→128, 단위 가중치, bias 없음) X vs Â_in X, \|Δ\|max | 6.39e-13 |
| Kd2(기록): float32 \|Δ\|max / Â_in X \|값\| 최대 / 상대 차이 | 1.98e-04 / 70.46 / 2.82e-06 |
| 첫 run(5-shot seed 0, 단독): SPD 구간 / 전체 / GPU 최대 | 9.3 s / 30.0 s / 1792 MiB |

| k | seed | 실행 단계 | SPD 구간 (s) | 전체 (s) | `wall_time_sec` (run.json) | GPU 최대 (MiB) |
|---|---|---|---|---|---|---|
| 1 | 0 | 2(병렬) | 10.4 | 32.4 | 17.1 | 1782 |
| 5 | 0 | 1(단독) | 9.3 | 30.0 | 16.5 | 1792 |
| 1 | 1 | 2(병렬) | 10.3 | 31.8 | 17.3 | 1782 |
| 5 | 1 | 2(병렬) | 9.9 | 32.0 | 17.5 | 1792 |
| 1 | 2 | 2(병렬) | 10.5 | 31.3 | 16.5 | 1782 |
| 5 | 2 | 2(병렬) | 10.5 | 32.3 | 17.5 | 1792 |

- SPD 구간 = `console_times.tsv`에서 "Done." 줄부터 다음 출력 줄까지(`embedder.py:61–79`의 그래프 구성 + SPD). GPU 최대 = 1초 간격 `nvidia-smi` memory.used 최대.

## T1 — ② 표현 (seed 평균 ± sd; G는 T26 전재)
| k | 표현 | acc_base | acc_valid | acc_test |
|---|---|---|---|---|
| 1 | S0 | 40.47 ± 0.26 | 41.95 ± 0.84 | 47.13 ± 0.32 |
| 1 | S0_in | 37.35 ± 0.18 | 37.63 ± 0.34 | 42.56 ± 0.27 |
| 1 | P64 | 49.73 ± 0.25 | 50.25 ± 0.29 | 55.95 ± 0.52 |
| 1 | P16 | 48.78 ± 0.21 | 49.39 ± 0.46 | 54.13 ± 0.54 |
| 1 | TEGe | 47.07 ± 0.97 | 42.50 ± 0.85 | 47.51 ± 0.60 |
| 1 | G | 41.33 ± 3.38 | 39.65 ± 2.51 | 38.42 ± 2.25 |
| 5 | S0 | 62.76 ± 0.27 | 63.02 ± 0.48 | 70.77 ± 0.73 |
| 5 | S0_in | 59.87 ± 0.16 | 60.30 ± 0.24 | 68.19 ± 0.44 |
| 5 | P64 | 64.17 ± 0.22 | 63.49 ± 0.51 | 71.75 ± 0.51 |
| 5 | P16 | 62.52 ± 0.40 | 61.79 ± 0.32 | 69.06 ± 0.26 |
| 5 | TEGe | 62.12 ± 3.00 | 56.64 ± 0.73 | 63.32 ± 0.56 |
| 5 | G | 69.84 ± 0.80 | 55.41 ± 0.82 | 56.54 ± 0.27 |

## T2 — ① 출력 (TEG 고정 test 200 에피소드, seed 평균 ± sd)
| k | D | D_sym | D_in | TEG 출력 | GPN 출력 | TEG 원 보고 방식 `test_acc_at_best_valid` |
|---|---|---|---|---|---|---|
| 1 | 45.73 ± 0.68 | 55.17 ± 1.53 | 42.65 ± 0.69 | 49.57 ± 1.55 | 38.43 ± 0.70 | 50.11 ± 1.29 |
| 5 | 71.27 ± 0.59 | 79.91 ± 0.45 | 68.71 ± 1.43 | 66.43 ± 0.93 | 59.41 ± 1.25 | 66.91 ± 0.86 |

- 참고: VISION 논문 표의 TEG(ogbn-arxiv 5-way 5-shot) 62.07 ± 1.72. 클래스 분할·에피소드 프로토콜이 이 표와 다를 수 있다.

## T3 — 짝지은 차 (%p)
- seed 단위: 평균 ± SE (부호 일치/3), ② probe. 에피소드 단위: ① 고정 test 3 seed × 200 = 600개의 평균 ± SE (양수 비율).

| k | test: TEGe − S0 | valid: TEGe − S0 | base: TEGe − S0 | test: TEGe − P64 | 에피소드: TEG − D | 에피소드: GPN − D | 에피소드: TEG − GPN | 에피소드: TEG − D_in (기록) |
|---|---|---|---|---|---|---|---|---|
| 1 | +0.38 ± 0.32 (2/3) | +0.55 ± 0.76 (1/3) | +6.60 ± 0.71 (3/3) | -8.44 ± 0.60 (3/3) | +3.83 ± 0.59 (0.56) | -7.30 ± 0.55 (0.24) | +11.13 ± 0.63 (0.73) | +6.91 ± 0.49 (0.67) |
| 5 | -7.45 ± 0.10 (3/3) | -6.38 ± 0.52 (3/3) | -0.64 ± 1.82 (1/3) | -8.43 ± 0.05 (3/3) | -4.85 ± 0.48 (0.29) | -11.87 ± 0.45 (0.12) | +7.02 ± 0.53 (0.66) | -2.28 ± 0.36 (0.32) |

## T4 — 순도 (top-10 코사인 kNN, seed 평균 ± sd; TEGe는 5-shot run)
| 표현 | pur_base | pur_valid | pur_test |
|---|---|---|---|
| S0 | 0.7518 ± 0.0000 | 0.7376 ± 0.0000 | 0.8213 ± 0.0000 |
| P64 | 0.7522 ± 0.0000 | 0.7356 ± 0.0000 | 0.8190 ± 0.0000 |
| TEGe | 0.6812 ± 0.0071 | 0.6525 ± 0.0004 | 0.7347 ± 0.0031 |

## 판정 (값만, seed 평균)
| 판정 | 정의 | 값 |
|---|---|---|
| M1 | acc_test(TEG 출력) < acc_test(D) (① 출력) | 1/2 |
| M2 | acc_test(TEGe) < acc_test(S0) | 1/2 |
| M3 | acc_base(TEGe) > acc_base(S0) | 1/2 |
| M4 | acc_valid(TEGe) < acc_valid(S0) | 1/2 |
| M5 | acc_test(TEGe) < acc_test(P64) | 2/2 |
| M6 (기록) | acc_test(TEG 출력) − acc_test(GPN 출력), %p | k = 1: +11.13; k = 5: +7.02 |
| M6 (기록) | D_sym 기준 M1 | 2/2 |
| M6 (기록) | pur_test(TEGe) < pur_test(S0) | 1/1 |
| M6 (기록) | D_in 기준 M1 | 1/2 |
| M6 (기록) | acc_test(TEGe) < acc_test(S0_in) | 1/2 |

## 사전 등록 (조건과 계산값만)
| 예측 | 조건 | 계산값 |
|---|---|---|
| P139 | M1 = 2/2 | M1 = 1/2 |
| P140 | M2 = 2/2 | M2 = 1/2 |
| P141 | M3 ≥ 1/2 | M3 = 1/2 |
| P142 | M4 ≥ 1/2 | M4 = 1/2 |
| P143 | M5 = 2/2 | M5 = 2/2 |
