# T05 — 메모리 보완성 진단: 간선 덤프, 키 교체, TEG 최종 쌍 판별력 (auto mode, 2026-10-02)

정본 `CLAUDE.md`. 선행: T04(`t04`). **auto mode: CLAUDE.md §2의 멈춤 지점 두 곳을 이 지시서 동안 해제한다.** 단 §4의 게이트에서 실패하면 멈춘다. CLAUDE.md §7은 유효.

## 0. 목표 (검증할 주장)
`--mem` run에서 test 간선마다 (첫 층 거리, TEG 최종 좌표 거리, 키 3종의 p̂, 같은 클래스 정답)이 파일로 남고, 이를 이용해 다음을 수치로 낼 수 있다.
1. 메모리 p̂가 거리에 **더해서** 판별력을 올리는가(보완성)
2. 키를 바꾸면 결과가 달라지는가
3. TEG 자신의 최종 좌표는 쌍을 얼마나 잘 가르는가

**정확도 계산 경로와 학습은 바꾸지 않는다.** 이번 run에서는 보정 arm(A·B)을 평가하지 않는다.

## 1. 바꿀 것

### 1.1 플래그 (`argument.py`)
- `--mem_keys` (str, 기본 `raw`): 쉼표로 구분한 키 목록. 값: `raw`, `center`, `str`.
- `--mem_arms` (str, 기본 `all`): `all` = T03b/T04와 같은 6개 arm 평가, `none` = arm 평가 생략.
- `--dump_edges` (store_true): test 간선 덤프.
- 새 플래그 4개를 `config2string` 제외 목록에 추가(T02·T03a와 같은 방식).
- 기본값이면 T04와 같은 동작이어야 한다.

### 1.2 키 3종 (`memory.py`)
`build` 시점에 키 종류마다 키 행렬을 따로 만든다. 조회 규칙(top-k = `mem_k`, `softmax(sim / mem_tau)`, 가중 평균 same)은 T03a와 같다.

| 키 | 메모리 쪽 | 조회(test 간선) 쪽 |
|---|---|---|
| `raw` | T03a와 동일: 첫 층 메시지 m을 L2 정규화 | 같은 방식 |
| `center` | m에서 **메모리 메시지 평균 μ_m**을 빼고 L2 정규화 | test m에서 같은 μ_m을 빼고 L2 정규화 |
| `str` | 구조 특징 이어 붙임 `[s_r, s_c]`(32차원, 방향 순서 유지, 거리 미포함)에서 **메모리 평균 μ_s**를 빼고 L2 정규화 | 같은 μ_s를 빼고 L2 정규화 |

- μ_m, μ_s는 `build`마다(에폭마다) 메모리 전체로 다시 계산한다.
- `str` 키의 구조 특징은 EGNN 입력 전의 원래 구조 특징(`self.structural_features` 등 첫 층에 들어가는 값)이다.

### 1.3 TEG 최종 좌표 거리 (`model.py`)
off 경로의 EGNN 출력(2층 이후, 프로토타입 계산 직전의 support·query 좌표)에서, task 그래프의 각 간선 (r ← c)에 대해 `d_final = ||z_r − z_c||²`를 계산해 둔다. off 경로의 연산은 바꾸지 않고, 출력을 읽기만 한다.

### 1.4 간선 덤프 (`--dump_edges`)
- 대상: 에폭 ≥ 1의 **test** 에피소드 전 간선(양방향).
- 파일: `out_dir/edges_test.npz` (압축). 배열(길이 = 간선 수):
  `epoch`, `ep_idx`, `dir`(0 = query←support, 1 = support←query), `same`(정답, 지표 전용), `d0`(첫 층 `sqr_dist`), `d_final`, 그리고 키마다 `phat_<key>`, `top1_<key>`, `top10_<key>`(10번째 유사도).
- `run.json` 에폭 기록에 키마다 `auc_phat_pooled_<key>`, 그리고 `auc_dfinal_pooled`를 추가한다. 기존 `auc_phat_pooled`, `auc_dist_pooled`는 그대로 둔다(`raw` 키 기준).

## 2. 실행
- `tools/run_t05.py`: T04 스크립트를 복사해 명령만 바꾼다. 결과 `results/T05/<ds>_5w<k>s/seed<s>/`.
- 명령: `CUDA_VISIBLE_DEVICES=<UUID> python main.py --dataset <ds> --way 5 --shot <k> --seed <s> --num_seed 1 --device 0 --mem --mem_keys raw,center,str --mem_arms none --dump_edges --out_dir results/T05/<ds>_5w<k>s/seed<s>`
- 범위: 3 데이터셋 × 5-way {1,5}-shot × seed 0–4 = 30 run. seed 우선 순서, GPU 6장 UUID, GPU당 1 run. 실패 시 1회 재시도(`_retry1`).

## 3. 분석 (`tools/analyze_t05.py` → `reports/T05_summary.md`)
모든 표는 run마다 **off의 `best_valid_epoch`**에 해당하는 test 간선만 쓴다(T04 표 3과 같은 기준). 값은 (데이터셋, shot)별 seed 5개의 평균 ± sd. 점수 방향: 거리는 부호를 뒤집어 "클수록 같은 클래스"로 쓴다.

- **표 A — 단일 점수 AUC**: `−d0`, `−d_final`, `phat_raw`, `phat_center`, `phat_str`.
- **표 B — 보완성(교차검증)**: run마다 로지스틱 회귀(표준화한 특징, sklearn 기본 L2)를 `ep_idx` 기준 5겹 그룹 교차검증으로 학습하고, 겹 밖(out-of-fold) 예측 AUC를 계산한다.
  - 기준선 1: `[−d0]`. 비교: `[−d0, phat_<key>]` (키 3종). 열 = ΔAUC = 비교 − 기준선, 그리고 seed 부호 일치 수.
  - 기준선 2: `[−d_final]`. 비교: `[−d_final, phat_<key>]` (키 3종). 같은 형식.
- **표 C — 거리가 헷갈리는 쌍**: run마다 해당 에폭 간선 전체에서 `d0`의 25·75 백분위를 구한다. 부분집합 = (same = 1이고 d0 ≥ 75 백분위) ∪ (same = 0이고 d0 ≤ 25 백분위). 이 부분집합에서 `−d0`, `−d_final`, `phat_<key>` 각각의 AUC와 부분집합 크기(same 1/0 개수).
- **표 D — 키별 유사도 분포**: `top1_<key>` 평균, `top1_<key> − top10_<key>` 평균.
- **표 E — 회귀 확인**: 1-shot run의 off 에폭별 정확도와 T04 같은 seed의 일치 여부. `auc_phat_pooled_raw`와 T04 `auc_phat_pooled`의 최대 차.

## 4. 게이트 (실패 시 새 코드 안에서만 수정, 최대 2회 → 그래도 실패면 멈추고 보고)
- G1: 1-shot 15 run의 off 에폭별 train/valid/test 정확도와 `test_acc_at_best_valid`가 T04 같은 seed와 **완전 일치**. 5-shot은 T03b에서 재정의한 기준(최대 차 ≤ 0.0024) 안.
- G2: `auc_phat_pooled_raw`와 T04 `auc_phat_pooled`의 차가 전 에폭에서 ≤ 0.002(메모리 값의 실행 간 미세 차이 허용).
- G3: 30 run 모두 `edges_test.npz`와 `run.json` 존재(실패 목록 제외), 표 A–E 생성.

## 5. 사전 등록 (수정 금지, 2026-10-02 작성)
- **P37.** 표 B 기준선 1: `phat_raw`를 더한 ΔAUC < 0.01, 6개 (데이터셋, shot) 모두.
- **P38.** 표 B 기준선 1: 세 키 중 어느 것이든 ΔAUC ≥ 0.01인 (데이터셋, shot)이 2개 이하.
- **P39.** 표 A: AUC(`−d_final`) > AUC(`−d0`), 6개 모두.

## 6. 시간
- 목표 완료 12:30 KST. **13:30 KST 이후에는 새 run을 시작하지 않는다.** 완료된 run만으로 분석하고, 표 머리에 run 수를 적는다.

## 7. 보고 `reports/T05_complementarity_diag.md` (사실만, 해석 금지)
1. 변경 파일·위치·줄 수, 커밋 해시, 태그 `t05`
2. 게이트 G1–G3 결과
3. 표 A–E 본문(`reports/T05_summary.md`에서 복사)
4. P37–P39 "예측 / 관측 / 일치·어긋남"
5. 이상 징후(자율 판단 포함)

## 8. 하지 말 것
CLAUDE.md §7 전부. 정확도 계산 경로 변경. arm 평가 규칙 변경. 지정 외 키·특징·모델 추가. 결과 해석.
