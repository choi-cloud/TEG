# T23a — 다중 해상도 클러스터 구축 (학습 없음)

2026-10-15 · 작성: 채팅 Claude · 모드: auto mode(`T23_QUEUE.md`) · 브랜치 `hier_purity`(이 지시서에서 생성) · 태그 `t23a` · 선행: `t22`

## 0. 목적과 검증 가능한 주장

T23b·T23c가 쓸 클러스터 배정을 만든다. 뷰 6종 × 해상도 5개 × 계열 2개(M, H). 이 지시서의 주장은 "배정이 게이트를 통과하고, 크기 분포가 4절 표와 같다"이다. 해석은 하지 않는다.

## 1. 착수 (하나라도 어긋나면 멈추고 보고)

1. 브랜치가 `pool_all`, HEAD가 `199efae`이고 태그 `t22`가 있는지 확인한다.
2. `git status`에 사용자가 수정한 `CLAUDE.md`(modified)와 이 큐의 지시서 4개(`instructions/T23_QUEUE.md`, `T23a_hier_build.md`, `T23b_cluster_support_purity.md`, `T23c_hier_signal.md`, untracked)만 보여야 한다. 그 밖의 변경이 있으면 멈춘다.
3. `git switch -c hier_purity`.
4. `CLAUDE.md`와 지시서 4개를 커밋한다. 메시지 `docs: T23 큐 지시서 + CLAUDE.md 2차 개정`. 이 커밋이 **H0**이다(T23a–c 공통 diff 기준).
5. `scipy.cluster.hierarchy`, `sklearn`이 import되는지 확인한다(설치 금지). scipy가 없으면 H 계열을 빼고 진행하며 이상 징후에 적는다.
6. 디스크 여유가 1.5 GB 이상인지 확인한다(CLAUDE.md §7). 이 큐가 쓰는 디스크는 100 MB 이하로 예상한다.

## 2. 뷰

| 뷰 | 내용 | 단위 |
|---|---|---|
| X | 입력 특징 | 데이터셋당 1개 |
| AX | 행 정규화 ÂX (`fusion_baseline.diffusion_features_hops(ds, 1)`) | 데이터셋당 1개 |
| A2X | 행 정규화 Â²X (`fusion_baseline.diffusion_features(ds)`) | 데이터셋당 1개 (기록용) |
| rand1 | ÂX의 64차원 무작위 투영 (`analyze_t19.rand1_views`) | 데이터셋 × seed 0–2 |
| TEGe | T22 `base`의 `emb_best.npy` | (데이터셋, shot) × seed 0–2 |
| CONe | T22 선택 λ `infonce_all`의 `emb_best.npy` | (데이터셋, shot) × seed 0–2 |

- 클러스터링 수 = 9 + 9 + 36 = 54.
- 모든 뷰는 행 L2 정규화 후 코사인을 쓴다. X는 고차원(수천 차원)이므로 float32 GPU 행렬로 둔다.

## 3. 클러스터링

**해상도** R = (16, 64, 256, 1024, 4096). 수준 번호 ℓ = 1(가장 거침) … 5(가장 세밀함).

**M 계열(다중 해상도, 중첩 아님).** 해상도마다 독립된 구면 k-means.
- 초기화: k-means++(코사인 거리 1 − cos). 전용 생성기 `torch.Generator().manual_seed(17000 + s)`. 학습 없는 뷰(X, AX, A2X)는 s = 0. rand1·TEGe·CONe는 그 run의 seed. 해상도 순서(16 → 4096)대로 같은 생성기를 이어 쓴다.
- 반복: 배정(최대 코사인) → 중심(멤버 합의 L2 정규화). 최대 50회, 배정이 바뀐 노드 비율 < 0.1 %면 멈춘다.
- 빈 클러스터: 자기 중심과의 코사인이 가장 낮은 노드(동점이면 인덱스 작은 것)로 다시 심는다. 결정적이다.
- 재시작 없음(1회).

**H 계열(중첩 계층).** M 계열의 4096 배정에서 시작한다.
- 4096개 중심(L2 정규화)에 `scipy.cluster.hierarchy.linkage(method="ward")`.
- `fcluster(criterion="maxclust", t=k)`로 k ∈ {1024, 256, 64, 16}에서 자른다. 노드의 수준 ℓ 클러스터 = 그 노드의 4096 클러스터가 속한 상위 클러스터.
- 따라서 H의 ℓ = 5는 M의 ℓ = 5와 같다.

**저장.** `results/T23/clusters/<view>/<ds>[_5w<k>s]/seed<s>/`에 `assign_M.npy`, `assign_H.npy`([N, 5] int32, 열 = ℓ 1–5), `meta.json`(해상도별 반복 수, 바뀐 비율, 비어 있던 클러스터 수, 실제 클러스터 수, 시간). 학습 없는 뷰는 `<ds>/seed0`.

**병렬.** 프로세스 6개 = (데이터셋, shot). 각 프로세스는 자기 셀의 TEGe·CONe(seed 0–2)를 맡고, shot = 1 프로세스가 그 데이터셋의 X, AX, A2X, rand1까지 맡는다.

## 4. 산출 표 (`reports/T23a_summary.md`)

| 표 | 내용 |
|---|---|
| A1 | 뷰 × 계열 × 해상도: 실제 클러스터 수, 크기 최소·중앙값·최대, 크기 1인 클러스터 수, 가장 큰 클러스터가 차지하는 노드 비율. 데이터셋별. TEGe·CONe·rand1은 seed 평균(셀은 shot 1, 5를 따로) |
| A2 | 뷰별 k-means 반복 수와 마지막 바뀐 비율(해상도별 최댓값) |
| A3 | 실행 시간(뷰 종류별, 데이터셋별) |

## 5. 게이트

| 게이트 | 조건 | 실패 시 |
|---|---|---|
| Ga 분할 일치 | 36개 run 디렉터리의 `labels.npy`가 `load_data` 레이블과 같고, `split.json` 세 클래스 목록이 `seed_everything(s)` 뒤 `load_data` 목록과 같다(`analyze_t19.gate_split`의 방식을 T22 경로로) | 멈추고 보고 |
| Gb 입력 존재 | TEGe·CONe 36개 파일이 있고 모양이 [N, 64]다 | 셀마다 seed 2개 이상이면 진행하고 누락 기록, 그보다 적으면 그 셀을 "불완전" 표기 |
| Gc k-means 정확성 | 합성 자료(3개의 잘 떨어진 구면 군집, 각 200점, 16차원, 생성기 17000)에서 k = 3 배정의 ARI = 1.0 | 멈추고 보고 |
| Gd 결정성 | AX, Amazon_clothing, M 계열, k = 256을 한 번 더 돌린 배정이 처음 것과 완전히 같다 | 멈추고 보고 |
| Ge 중첩 | H 계열에서 모든 ℓ < ℓ′ 쌍에 대해, 수준 ℓ′의 각 클러스터가 수준 ℓ의 클러스터 하나에만 속한다. 54개 모두 | 멈추고 보고 |
| Gf 클러스터 수 | M 계열 실제 클러스터 수 = k(빈 클러스터 재심기 후). H 계열은 fcluster가 k보다 적게 내면 기록 | M 실패는 기록, H는 기록 |
| Gg 변경 범위 | `git diff --stat <H0> <보고서 커밋>`의 파일이 `tools/t23.py`, `reports/`, `instructions/` 안에만 있다 | 멈추고 보고 |

## 6. 보고

- `reports/T23a_hier_build.md`(CLAUDE.md §6 형식), `reports/T23a_summary.md`.
- 보고서 1절에 `tools/t23.py`의 함수 목록과 재사용한 기존 구현의 `파일:줄`을 적는다.
- 커밋 → 태그 `t23a` → push(force 없음). 그다음 T23b로 넘어간다.
- `results/T23/`는 지우지 않는다.
