# T03b 보고 — 평가 시점 보정 arm A·B (2026-10-02, TN1 체인 2단계) — **게이트 실패, 중단**

## 1. 변경 파일과 위치
커밋: 이 보고서와 같은 커밋 (태그 없음 — TN1 §2 실패 처리). TN1 §0에 따라 멈춤 지점 1 없이 진행.

| 파일 | 위치 | 내용 |
|---|---|---|
| `layers/EGNN.py` | `EGCL.coord_model` | `w_fn=None` 인자. `w = self.trans_mlp(msg)`; `w_fn`이 있으면 `w = w_fn(w)`; `trans = coord_diff * w` |
| `layers/EGNN.py` | `EGCL.forward` | `msg_fn=None, w_fn=None`. `msg_fn`이 있으면 `msg = msg_fn(msg)` (좌표·구조 특징 갱신 모두에 사용), `w_fn`은 `coord_model`로 전달 |
| `layers/EGNN.py` | `EGNN.forward` | `msg_fn=None, w_fn=None`, `gcl_0`에만 전달. 2층 호출은 원본 인자 그대로 |
| `model.py` | 상수 | `MEM_ARMS` 6개 (`A_0.1/0.3/0.5`, `B_0.1/0.3/0.5`) |
| `model.py` | 새 메서드 `episode_forward` | 원본 "EGNN → 프로토타입 → 예측" 구간을 그대로 옮김(`msg_fn`, `w_fn` 전달만 추가). 반환 `output`, `output_softmax` |
| `model.py` | 새 메서드 `arm_fns`, `arm_beta0_check`, `mem_configs` | A: `v̄ = Σ w·msgs[idx]`, `msg_fn = (1−β)m + βv̄`. B: `σ = w.std()`(전체 원소, 불편 추정), `σ < 1e-8`이면 보정 생략·카운트, 아니면 `w − βσ(2p̂−1)`. `mem_configs`: 설정별 원본 동점 규칙 |
| `model.py` | `train_epoch` | 원본 EGNN~예측 18줄 → `episode_forward` 호출 3줄. 정확도 계산 뒤 `--mem`·valid/test에서 6개 설정 평가(에폭 0은 off 값), 에피소드 기록 `arm_acc`, 에폭별 설정 평균 |
| `model.py` | `train` | `arm_valid_acc`·`arm_test_acc` 에폭 기록, `run.json`에 `mem_configs`, `sigma_skip`, `beta0_check` |

합계: `layers/EGNN.py` +15 −6, `model.py` +104 −19.

## 2. 실행 명령과 출력
`CUDA_VISIBLE_DEVICES=<UUID> python main.py --dataset Amazon_clothing --way 5 --shot {1,5} --seed 0 --num_seed 1 --device 0 --mem --out_dir results/T03b/amac_5w{1,5}s_s0` (1-shot 물리 0, 5-shot 물리 1 — T03a와 같은 GPU). `--mem` 없는 run: 물리 0, `results/T03b/amac_5w1s_s0_nomem.log`.

설정 7개 × {best_valid_epoch, test_acc_at_best_valid}:

| 설정 | run A 5w1s epoch | run A 5w1s test_acc_at_best_valid | run B 5w5s epoch | run B 5w5s test_acc_at_best_valid |
|---|---|---|---|---|
| off | 7 | 0.8048 | 7 | 0.9008 |
| A_0.1 | 7 | 0.8048 | 7 | 0.9008 |
| A_0.3 | 7 | 0.8040 | 7 | 0.9008 |
| A_0.5 | 7 | 0.8048 | 7 | 0.9008 |
| B_0.1 | 7 | 0.8056 | 7 | 0.9008 |
| B_0.3 | 7 | 0.8056 | 7 | 0.8992 |
| B_0.5 | 7 | 0.8056 | 7 | 0.8992 |

| run | 프로세스 전체 (s) | `wall_time_sec` (s) | T03a 대비 | σ 생략 횟수 (B_0.1/0.3/0.5) |
|---|---|---|---|---|
| A 5w1s | 30.18 | 23.88 | +5.78 | 0 / 0 / 0 |
| B 5w5s | 45.74 | 39.87 | +7.80 | 0 / 0 / 0 |
| nomem 5w1s | 20.94 | — | — | — |

## 3. 완료 조건 대비 (TN1 게이트, 완전 일치 기준)
| 조건 | 결과 | 충족 |
|---|---|---|
| 1. off 에폭별 valid/test = T03a 같은 run (게이트 a) | A 5w1s: 22개 완전 일치. **B 5w5s: 에폭 8 test 0.8872 vs T03a 0.8864 불일치** (그 외 train 에폭 10: 0.9448 vs 0.9440) | **미충족** |
| 2. `--mem` 없는 run 콘솔 = T02 B (게이트 b) | 33개 완전 일치, `==>` 줄 같음 | 충족 |
| 3. 7개 설정 `test_acc_at_best_valid` 기록 (게이트 c) | A·B 모두 7/7 | 충족 |
| 4. β = 0 점검 (기록만) | 에폭 1 test ep 0: A 5w1s off 0.88 / A_0.0 0.88 / B_0.0 0.88, softmax 최대 차 2.53e-7 / 2.38e-7. B 5w5s off 0.96 / 0.96 / 0.96, 최대 차 2.61e-8 / 1.68e-8 | 기록 |

게이트 (a) 실패. 수정 시도 0회(아래 진단). TN1 §2에 따라 중단, T04 미시작.

### 게이트 실패 진단 (원본 코드 재실행)
원본 코드(태그 `t01`, 별도 worktree)로 Amazon_clothing 5w5s seed 0을 `--mem` 없이 물리 GPU 1·2에서 각 2회 실행 (`results/T03b/diag_orig/*_retry1.log`; 첫 시도 4개는 worktree의 dataset 링크 오류로 시작 직후 종료, `*_run{1,2}.log`로 남아 있음).

기준: 원본 gpu1 run1. 콘솔 정확도 33개 비교.

| run | 코드 | Acc_Test_At_Best_Valid (에폭) | 다른 값 수 | 첫 차이 (에폭, 종류, 값, 기준값) |
|---|---|---|---|---|
| 원본 gpu1 run1 | t01 | 0.9008 (7) | 0 | — |
| 원본 gpu1 run2 | t01 | 0.9008 (7) | 5 | 8, valid, 0.8760, 0.8768 |
| 원본 gpu2 run1 | t01 | **0.9016** (7) | 9 | 5, valid, 0.8784, 0.8792 |
| 원본 gpu2 run2 | t01 | 0.9008 (7) | 6 | 6, valid, 0.8888, 0.8896 |
| T03a B | t03a, `--mem` | 0.9008 (7) | 4 | 6, valid, 0.8888, 0.8896 |
| T03b B | 현재, `--mem` | 0.9008 (7) | 2 | 6, valid, 0.8888, 0.8896 |

- 6개 run의 정확도 33개 해시가 모두 다르다.
- 5w1s seed 0은 7회(T01 ×2, T02 ×2, T03a, T03b, T03b nomem) 정확도 33개가 모두 같다.

## 4. 이상 징후
1. 게이트 (a) 5w5s 불일치. 원본 코드(`t01`) 5w5s seed 0 반복 실행도 run마다 정확도가 다르다(위 표). T01의 결정성 확인은 5w1s만 대상이었다.
2. β = 0 점검에서 정확도는 off와 같고 softmax 값은 비트 단위로 같지 않다(최대 차 1.7e-8 ~ 2.5e-7).
3. B의 σ는 그 에피소드 첫 층 `w` 전체 원소의 `torch.std`(불편 추정, 보정 전 메시지 기준)로 계산했다(지시서에 추정 방식 미기재 — 가장 단순한 쪽).
4. 진단용 원본 worktree를 scratchpad(`…/scratchpad/wt_t01`, detached `19e54f6`)에 만들었다. `git worktree list`에 남아 있다. 첫 진단 시도 4 run은 dataset 링크 오류(`FileNotFoundError`)로 실패했고, 재실행 로그는 `_retry1` 이름으로 저장했다(덮어쓰기 없음).
5. A 설정의 에폭별 test 정확도가 off와 같은 에폭이 다수다(표는 `results/T03b/*/run.json` `arm_test_acc`).

## 부기 (2026-10-02 10:21 KST) — 게이트 (a) 재정의와 통과 처리
사람 지시로 게이트 (a)를 재정의했다. 위 본문은 수정하지 않았다.

- **재정의**
  - 1-shot: T03a 같은 run과 완전 일치를 유지한다.
  - 5-shot: 원본 코드 반복 실행 간 차이 범위 안이면 통과한다.
- **5-shot 판정 방식**
  - 비교 대상은 off 설정의 에폭별 valid/test 정확도 22개다.
  - max|T03b − T03a|가 원본 코드(`t01`) 반복 4 run(`results/T03b/diag_orig/*_retry1.log`)의 쌍별(6쌍) 최대 |차|보다 크지 않으면 통과다.

| 항목 | 값 | 결과 |
|---|---|---|
| 1-shot: off valid/test 22개 vs T03a | 완전 일치 | 통과 |
| 5-shot: 원본 4 run 쌍별 최대 \|차\| (22개 값) | 0.0024 | — |
| 5-shot: T03b off vs T03a 최대 \|차\| | 0.0008 | 통과 (≤ 0.0024) |
| 참고: 값별 원본 4 run 최소~최대 구간 안 | T03b 아님, T03a도 아님 | 판정에 사용하지 않음 |

- 게이트 (a)·(b)·(c) 모두 통과로 처리한다. 태그 `t03b`.
