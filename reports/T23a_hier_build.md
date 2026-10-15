# T23a — 다중 해상도 클러스터 구축 (학습 없음)

2026-10-15 · 브랜치 `hier_purity` · H0 `14d9dfe` · 큐 T0 = 04:55:38 · 요약표 `reports/T23a_summary.md`

## 1. 변경 파일과 위치 (파일 단위 계획)

| 파일 | 내용 |
|---|---|
| `tools/t23.py` (신규, 하위 명령 방식) | T23a: `l2n_t`, `kmeanspp`, `assign_of`, `fix_empty`, `skmeans`, `hier_from`, `cluster_view`, `gate_split_t22`, `a_run`, `synth_blobs`, `a_gates`, `nested_ok`, `size_stats`, `a_report`. T23b·T23c 하위 명령도 같은 파일(이 커밋 시점에 함께 들어감, 실행은 각 지시서에서) |

- 재사용(import만): `tools/analyze_t19.py:86` `fixed_views`(X, AX = `diffusion_features_hops(ds, 1)`, A2X = `diffusion_features(ds)`), `tools/analyze_t19.py:99` `rand1_views`(offset 2000), Ga는 `tools/analyze_t19.py:125` `gate_split` 방식을 T22 경로로 옮긴 `gate_split_t22`(`utils.seed_everything`, `utils.load_data`). `scipy.cluster.hierarchy.linkage(method="ward")`, `fcluster(criterion="maxclust")`, `sklearn.metrics.adjusted_rand_score`(Gc).
- 구면 k-means: 행 L2 정규화 float32 GPU. k-means++ = 첫 중심 `torch.randint(N, generator=g)`, 이후 중심마다 `torch.rand(1, generator=g)` 하나를 누적 D²(D = 1 − cos)에 역CDF(`searchsorted`, right)로 대응. g = `torch.Generator().manual_seed(17000 + s)`(학습 없는 뷰 s = 0), 해상도 16 → 4096 순으로 이어 씀. 반복 = 최대 코사인 배정 → 빈 클러스터 재심기 → 중심(멤버 합 L2 정규화), 최대 50회, 바뀐 비율 < 0.1 %에서 멈춤, 재시작 없음.
- 빈 클러스터 재심기: 자기 중심과의 코사인이 가장 낮은 노드(동점 시 인덱스 작은 것)부터, 그 노드의 현재 클러스터가 크기 2 이상일 때 빈 클러스터로 옮기고 중심을 그 노드로 둔다.
- H 계열: M의 4096 중심(float64)에 Ward → `fcluster` maxclust k ∈ {1024, 256, 64, 16}; 노드 수준 ℓ = 4096 클러스터의 상위 클러스터. H ℓ5 = M ℓ5.
- 저장: `results/T23/clusters/<view>/<ds>[_5w<k>s]/seed<s>/{assign_M.npy, assign_H.npy, meta.json}` 54개.

## 2. 실행 명령과 출력

| 명령 | 시간 | 출력 |
|---|---|---|
| `python tools/t23.py a-gates` (GPU 0) | < 1분 | Gc ARI = 1.0 |
| `python tools/t23.py a-run <ds> <k>` × 6 (GPU 0, 1, 2, 4, 5, 6, UUID) | 05:01:15–05:04:33 (3.3분; 프로세스 164–196 s) | 배정 54개 |
| `python tools/t23.py a-report` | < 1분 | `reports/T23a_summary.md`, `results/T23/gate_checks_a.json` |

- 클러스터링 1회(M 5개 + H) 시간: X·AX·A2X 36–53 s, rand1·TEGe·CONe 0.8–1.9 s(표 A3). 착수(T0)부터 완료까지 약 9분.
- `results/T23/` 75M.

### 게이트

| 게이트 | 결과 | 충족 |
|---|---|---|
| Ga 분할 일치 | 36개 run(TEGe 18, CONe 18): `labels.npy` = `load_data` 레이블, `split.json` 세 목록 = `seed_everything(s)` 뒤 `load_data` 목록. 36/36 | 예 |
| Gb 입력 존재 | TEGe·CONe 36/36, 모양 [N, 64] | 예 |
| Gc k-means 정확성 | 합성 3군집 × 200점 × 16차원(생성기 17000), k = 3 ARI = 1.0 | 예 |
| Gd 결정성 | AX, Amazon_clothing, M, k = 256 재실행 배정 완전 일치 | 예 |
| Ge 중첩 | H 54/54 모든 ℓ < ℓ′ 쌍 중첩 | 예 |
| Gf 클러스터 수 (기록) | M 실제 수 = k: 270/270. H: fcluster가 k보다 적게 낸 경우 0 | 기록 |
| Gg 변경 범위 | 부기에 기록 | — |

## 3. 완료 조건 대비

| 조건 | 결과 | 충족 |
|---|---|---|
| 착수 확인(브랜치 `pool_all`, HEAD `199efae`, `t22`, git status, scipy·sklearn import, 디스크 ≥ 1.5 GB) | 모두 확인(디스크 여유 8.8G) | 예 |
| H0 커밋 | `14d9dfe` | 예 |
| 배정 54개(M, H) | 54/54 | 예 |
| 표 A1–A3 | `reports/T23a_summary.md` | 예 |
| 게이트 Ga–Gf | 모두 충족 | 예 |

## 4. 이상 징후

1. 첫 실행(04:57:4x 시작)은 k-means++ 표본 추출을 CPU `torch.multinomial`로 했고 6개 프로세스의 CPU 경합으로 클러스터링 1회가 54–67 s(64차원) 걸렸다. 중단하고 표본 추출을 GPU 역CDF 방식(위 1절)으로 바꾸고 `torch.set_num_threads(4)`를 넣어 처음부터 다시 돌렸다(05:01:15). 첫 실행의 배정(2개 완료)은 지웠고 이후 계산에 쓰지 않았다. Gc는 바꾼 코드로 다시 확인했다.
2. 첫 실행을 멈출 때 쓴 `pkill -f` 명령이 그 명령을 실행한 셸까지 종료했다(exit 144). 파일 편집은 다음 명령에서 다시 했다.
3. 지시서 §3의 k-means++는 거리 1 − cos의 제곱(D²) 가중으로 구현했다(지시서에 가중 지수가 적혀 있지 않음).
4. Gd는 Amazon_clothing AX만 대상이다(지시서대로). k-means++의 역CDF가 GPU `cumsum`(float64)을 쓴다.
5. H 계열 ℓ1(k = 16)의 가장 큰 클러스터 비율이 M보다 큰 칸이 있다(예: Amazon_clothing X, H 0.330, M 0.102; 표 A1).
6. 클러스터링·분할 점검(Ga) 과정에서 `utils.load_data`·`seed_everything`이 분석 프로세스의 전역 `random`·torch 기본 생성기를 소비한다(분석 프로세스 안의 일, 학습 없음).
