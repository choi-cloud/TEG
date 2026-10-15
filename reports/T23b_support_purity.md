# T23b — 클러스터에서 고른 support의 레이블 일치 (학습 없음)

2026-10-15 · 브랜치 `hier_purity` · H0 `14d9dfe` · 선행 `t23a`(`9d8cf08`) · 요약표 `reports/T23b_summary.md`

## 1. 변경 파일과 위치 (파일 단위 계획)

| 파일 | 내용 |
|---|---|
| `tools/t23.py` | T23b 함수: `labels_eff`, `restricted_knn`, `pur_rows`, `b1_purity`, `load_assign`, `shuf_of`, `b_run`, `synth_tie`, `b_gates`, `seed_stats`, `clear_pos`, `clear_neg`, `fmt_ms`, `load_b`, `b_index`, `lstar`, `b_report`. 이 보고서 커밋에서 바뀐 부분: T23c 하위 명령(`c_report`) 추가와 `c_gates`의 쌍 목록을 난수 없는 결정적 목록으로 바꾼 것 |

- 재사용: `tools/analyze_t19.py:41` `knn_sorted`의 동점 규칙(GPU float64 유사도, (유사도 내림차순, 노드 인덱스 오름차순), 가져온 후보 밖까지 동점이 이어지면 그 행 전체 재정렬)을 `restricted_knn`에 같은 방식으로 구현(후보 여유 16, 클러스터가 다른 열·자기 자신은 −∞). `analyze_t19.py:80` `normed`, `:112` `split_of`, `:118` `group_ids`, `:172` `measure_view`(Hb 기준값). B2의 X·AX 기준 순도는 `results/T19a/<ds>.json`의 `fixed` `pur_all`(같은 seed 분할)을 읽었다.
- u_K는 같은 `restricted_knn`(클러스터 없음, K = 50)에서 K = 5, 10, 20, 50으로 자른 값. r_K는 K = 10으로 구한 정렬 목록에서 K = 5, 10을 자름. 멤버 < K이면 있는 멤버로 계산(`frac_short`), 멤버 0이면 제외.
- SHUF: `np.random.default_rng(17300 + s)`에서 수준 1–5 순으로 `permutation(N)` 1회씩, H·M 같은 순열. 레이블 없는 노드는 고유 음수 레이블(다른 클래스)로 센다.

## 2. 실행 명령과 출력

| 명령 | 시간 | 출력 |
|---|---|---|
| `python tools/t23.py b-gates` (GPU 0) | 05:05:24–05:05:32 | `results/T23/gates_b.json` |
| `python tools/t23.py b-run <ds> <k>` × 6 (GPU 6개, UUID) | 05:05:38–05:08:34 (2.9분; 프로세스 74–170 s) | `results/T23/T23b/<cell>/seed<s>.csv` 18개 |
| `python tools/t23.py b-report` | < 1분 | `reports/T23b_summary.md`, `results/T23/judgements_b.json` |

### 게이트

| 게이트 | 결과 | 충족 |
|---|---|---|
| Ha 선행 | `t23a` 있음, 배정 54개 | 예 |
| Hb 기준 일치 | u_10(TEGe, Amazon_clothing 1-shot seed 0, test) 0.6868910256 = `analyze_t19.measure_view` `pur_all`, \|Δ\| = 0 | 예 |
| Hc 제한 kNN | (a) 한 클러스터 마스크 r_10 = u_10, \|Δ\| = 0, 이웃 목록 동일. (b) 합성 200노드(생성기 17000, 중복 행 20개) × 수준 5 × K ∈ {5, 10}: 2000행 전수 정렬과 불일치 0(멤버 < K 400행 포함) | 예 |
| Hd B-1 | 합성 1000행 전수 계산과 불일치 0 | 예 |
| He 유한 | 모든 값 유한 | 예 |
| Hf 변경 범위 | 부기에 기록 | — |

### 판정과 사전 등록 (값만)

| 판정 | 값 |
|---|---|
| R1 gain(TEGe, AX) 분명히 양수 | 6/6 |
| R2 gain(TEGe, X) | 2/6 |
| R3 gain(TEGe, SELF) | 4/6 |
| R4 gain(TEGe, SHUF) 분명히 음수 | 6/6 |
| R5 gain(TEGe, AX) − gain(CONe, AX) | 6/6 |
| R6 gain(TEGe, rand1) | 2/6 |
| R7 (기록) gain_H − gain_M (TEGe, AX) | 0/6; ℓ* 분포 {ℓ1: 40, ℓ5: 104} |

| 예측 | 조건 | 관측 | 일치 |
|---|---|---|---|
| P116 | R1 ≥ 4/6 | 6/6 | 일치 |
| P117 | R3 ≤ 1/6 | 4/6 | 어긋남 |
| P118 | R4 = 6/6 | 6/6 | 일치 |
| P119 | R5 ≥ 4/6 | 6/6 | 일치 |
| P120 | R6 ≤ 2/6 | 2/6 | 일치 |

- B3 gain(TEGe, AX), ℓ* = 5 전 셀: +0.0361, +0.0376, +0.0084, +0.0071, +0.0422, +0.0344 (clothing 1, 5 / dblp 1, 5 / electronics 1, 5).

## 3. 완료 조건 대비

| 조건 | 결과 | 충족 |
|---|---|---|
| run 18개 결과 | 18/18 | 예 |
| 표 B1–B6 | `reports/T23b_summary.md` | 예 |
| R1–R7, P116–P120 | 위 표 | 예 |
| 게이트 Ha–He | 충족 | 예 |

## 4. 이상 징후

1. ℓ* = 5(4096 클러스터)인 칸에서 test anchor의 `frac_short`(클러스터 멤버 < 10)가 0.30–0.56이다(B3). 이 anchor의 r_10은 10개보다 적은 이웃으로 계산된다.
2. R7 = 0/6: H ℓ5 = M ℓ5(같은 배정)이고, (TEGe, AX)의 ℓ*가 H·M 모두 5인 셀에서 gain_H − gain_M = 0이다.
3. ℓ* 분포가 ℓ1과 ℓ5 둘뿐이다(ℓ2–ℓ4 0회).
4. SHUF의 ℓ*는 모든 셀에서 1이다.
5. 지시서 §2 "어느 클래스 목록에도 없는 노드"는 이 세 데이터셋 run에서 0개다(`group_ids` −1 없음; T19a frac_unassigned = 0과 같음).
6. B2의 X·AX 행은 T19a 결과를 읽은 값이다(T19a 분할 = T14 run 분할, T23a Ga·T19a Ga에서 둘 다 `load_data(seed)`와 같음).
