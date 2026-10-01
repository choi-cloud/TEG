# T03a — 관계 메모리 기록과 전제 진단 (모델 계산 불변)

정본 `CLAUDE.md`, 설계 `docs/relation_textbook_methodology_v0.md`. manual mode, 멈춤 지점 두 곳(§2).

## 목표 (검증할 주장)
`--mem`을 켜면 학습 중 support↔query 쌍과 그 쌍의 같은 클래스 여부(base 정답 라벨)가 기록되고, 평가 시점에 test 쌍마다 메모리 조회 결과가 계산·기록된다. 이때 학습과 정확도는 `--mem`을 끈 경우와 같다.

## 용어
- 메시지: EGNN 첫 층(`gcl_0`)의 `msg_model` 출력, 64차원. 간선 (r ← c)마다 하나.
- p̂: test 간선 하나에 대해, 메시지가 가장 비슷한 메모리 항목 k개 중 "같은 클래스" 비율(유사도 가중).

## 바꿀 것 (**멈춤 지점 1: 수정 전 계획 승인**)

### 플래그 (`argument.py`)
`--mem` (store_true, 기본 False), `--mem_k` (int, 10), `--mem_tau` (float, 0.1), `--mem_chunk` (int, 256)

### 새 파일 `memory.py` — `RelationMemory`
- `add(r_gid, c_gid, same)`: 방향 있는 간선 (r ← c)의 전역 노드 ID와 같은 클래스 여부(0/1)를 추가. 같은 (r, c)는 한 번만 보관.
- `build(conv, egnn, features, edges, str_feat)`: `torch.no_grad()`, 모델 eval 상태에서
  1. `emb = conv(features, edges)` (전체 그래프), `x = egnn.LayerNorm(emb)`
  2. 메모리 간선마다 `d = ||x[r] − x[c]||²`, `m = egnn.gcl_0.msg_model(str_feat[r], str_feat[c], d)`
  3. 키 = L2 정규화한 `m`, 값 = `same` (그리고 `m` 원본도 보관 — T03b에서 사용)
- `query(m_edges)`: 정규화한 `m_edges`와 키의 코사인 유사도로 상위 `mem_k`개 → 가중치 `softmax(sim / mem_tau)` → 반환: `p_hat`(가중 평균 same), `top1_sim`, `idx`, `w`. 간선을 `mem_chunk`개씩 나눠 계산한다.

### `model.py`
- `--mem`이 켜져 있으면:
  - **기록**: mode `train`, `epoch ≥ 1`인 에피소드에서, 모든 (query q, support s) 쌍에 대해 (q ← s)와 (s ← q) 두 방향을 `add`. `same = (labels[q] == labels[s])`. 전역 ID는 `id_query`, `id_support`에서 가져온다. 기록은 계산·난수에 관여하지 않는다.
  - **구축**: 에폭 `≥ 1`의 valid 평가 직전에 `build` 1회. valid와 test가 같은 메모리를 쓴다.
  - **조회와 기록**(valid·test 에피소드, 에폭 ≥ 1): 원본 forward와 별도로, 에피소드의 첫 층 메시지를 원본과 같은 방식으로 계산한다(`egnn.LayerNorm(embeds_epi)` → `coord2dist` → `msg_model`). 그리고 `query`를 호출한다. 간선의 정답 `same`(평가 지표 전용)도 계산한다.
  - 에피소드 기록(T02 `episodes.jsonl`에 필드 추가): `n_edges`, `auc_phat`(p̂ vs 정답 same, ROC AUC), `auc_dist`(−d vs 정답 same), `mean_phat_same`, `mean_phat_diff`, `mean_top1_sim`. 한쪽 클래스만 있으면 AUC는 null.
  - 에폭 기록(`run.json`): `mem_size`, 그 에폭 test 간선 전체를 모은 `auc_phat_pooled`, `auc_dist_pooled`.
- **모델의 정확도 계산 경로는 건드리지 않는다.** 조회는 정확도에 들어가지 않는다.

### 점검 하나
에폭 1 구축 직후, 그 에폭의 학습 에피소드 1개(메모리에 들어 있는 쌍)에 대해 `query`를 실행해 `auc_phat`와 `top1_sim` 평균을 한 번 출력·기록한다(`run.json`의 `sanity`).

## 확인 실행
- A: `... --dataset Amazon_clothing --way 5 --shot 1 --seed 0 --mem --out_dir results/T03a/amac_5w1s_s0`
- B: `... --dataset Amazon_clothing --way 5 --shot 5 --seed 0 --mem --out_dir results/T03a/amac_5w5s_s0`

## 완료 조건
1. A의 에폭별 train/valid/test 정확도 = T02 A (같은 seed, `--mem` 없음). T01이 비결정적이었다면 차이를 표로 보고.
2. `mem_size`가 에폭마다 증가하고, 에폭 10의 값이 이론 상한 이하: 1-shot 50·2·5·25 = 12,500/에폭, 5-shot 50·2·25·25 = 62,500/에폭의 누적(중복 제거 전).
3. `sanity`의 `top1_sim` 평균 ≈ 1.0
4. 에피소드·에폭 기록 필드가 모두 채워짐
5. run 소요 시간(T02 대비 증가분)

## 보고 (**멈춤 지점 2**) — `reports/T03a_memory_record_diag.md`
- diff 요약, 완료 조건 1~5 표
- 에폭별 표(A, B 각각): `mem_size`, `auc_phat_pooled`, `auc_dist_pooled`
- 이상 징후, 커밋 해시, 태그 `t03a`

## 하지 말 것
CLAUDE.md §7 전부. 정확도 계산 경로 변경. test 라벨을 모델 계산에 사용(지표 계산만 허용). 난수 호출 추가.
