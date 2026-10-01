# T03a 보고 — 관계 메모리 기록과 전제 진단 (2026-10-02, TN1 체인 1단계)

## 1. 변경 파일과 위치
커밋: 이 보고서와 같은 커밋 (태그 `t03a`). TN1 §0에 따라 멈춤 지점 1 없이 진행.

| 파일 | 위치 | 줄 | 내용 |
|---|---|---|---|
| `memory.py` (신규) | `RelationMemory` | +74 | `add`(중복 (r,c) 1회 보관), `edge_messages`, `build`(eval·no_grad, 전체 그래프 GCN → `egnn.LayerNorm` → `gcl_0.msg_model`, 키 = L2 정규화, 원본 `msgs` 보관, 65,536 간선 단위), `query`(코사인 top-k, `softmax(sim/τ)`, `mem_chunk` 단위, 반환 `p_hat`/`top1_sim`/`idx`/`w`) |
| `argument.py` | `parse_args` | +4 | `--mem`, `--mem_k` 10, `--mem_tau` 0.1, `--mem_chunk` 256 |
| `argument.py` | `config2string` 제외 목록 | +4 | `mem`, `mem_k`, `mem_tau`, `mem_chunk` (이상 징후 1) |
| `model.py` | import·상수 | +4 | `roc_auc_score`, `RelationMemory`, `MEM_EP_KEYS` |
| `model.py` | `__init__` 끝, 새 메서드 | +53 | 메모리 생성, `labels_np`; `mem_record`, `mem_diag`, `mem_stats`, `mem_sanity_check` |
| `model.py` | `train_epoch` (task 그래프 생성 뒤, EGNN 앞) | +19 | `--mem`·에폭 ≥ 1: train이면 (q←s),(s←q) 기록, valid·test면 첫 층 메시지 계산·조회·지표 |
| `model.py` | `train_epoch` 에피소드 기록 | +2 | 에피소드 필드 6개 추가 (에폭 0은 null) |
| `model.py` | `train` | +16 | 에폭 ≥ 1 valid 직전 `build`, 에폭 1 `sanity`, 에폭 기록 `mem_size`·`auc_phat_pooled`·`auc_dist_pooled`, `run.json`에 `sanity` |

원본 함수의 연산 줄 변경 없음(삭제 1줄은 T02에서 추가한 `json.dump` 인자 줄).

## 2. 실행 명령과 출력
`CUDA_VISIBLE_DEVICES=<UUID> python main.py --dataset Amazon_clothing --way 5 --shot {1,5} --seed 0 --num_seed 1 --device 0 --mem --out_dir results/T03a/amac_5w{1,5}s_s0 > results/T03a/amac_5w{1,5}s_s0.log 2>&1`
A(1-shot) 물리 GPU 0, B(5-shot) 물리 GPU 1, 동시 실행.

| run | exit | Acc_Test_At_Best_Valid (에폭) | 프로세스 전체 (s) | `wall_time_sec` (s) | T02 A `wall_time_sec` 대비 |
|---|---|---|---|---|---|
| A 5w1s | 0 | 0.8048 (7) | 24.09 | 18.10 | +4.59 s (13.51 → 18.10) |
| B 5w5s | 0 | 0.9008 (7) | 38.57 | 32.07 | — (T02 대응 run 없음) |

에폭별 표:

| 에폭 | A `mem_size` | A `auc_phat_pooled` | A `auc_dist_pooled` | B `mem_size` | B `auc_phat_pooled` | B `auc_dist_pooled` |
|---|---|---|---|---|---|---|
| 0 | 0 | null | null | 0 | null | null |
| 1 | 12,496 | 0.8537 | 0.8803 | 62,414 | 0.8813 | 0.9145 |
| 2 | 24,982 | 0.8514 | 0.8885 | 124,680 | 0.8590 | 0.8972 |
| 3 | 37,456 | 0.8775 | 0.9181 | 186,804 | 0.8731 | 0.9066 |
| 4 | 49,922 | 0.8523 | 0.8990 | 248,812 | 0.8666 | 0.9029 |
| 5 | 62,396 | 0.8506 | 0.8863 | 310,626 | 0.8752 | 0.9188 |
| 6 | 74,872 | 0.8731 | 0.9095 | 372,252 | 0.8695 | 0.9113 |
| 7 | 87,326 | 0.8484 | 0.8878 | 433,738 | 0.8674 | 0.9105 |
| 8 | 99,776 | 0.8527 | 0.8869 | 494,980 | 0.8763 | 0.9156 |
| 9 | 112,230 | 0.8600 | 0.8853 | 556,082 | 0.8732 | 0.9174 |
| 10 | 124,654 | 0.8782 | 0.9094 | 617,064 | 0.8883 | 0.9249 |

`sanity` (에폭 1, 학습 에피소드 ep_idx 49): A `n_edges` 250, `auc_phat` 0.99425, `top1_sim_mean` 1.0 / B `n_edges` 1250, `auc_phat` 0.976854, `top1_sim_mean` 1.0.

## 3. 완료 조건 대비 (TN1 게이트 포함, 완전 일치 기준)
| 조건 | 결과 | 충족 |
|---|---|---|
| 1. A 에폭별 train/valid/test 정확도 = T02 A (게이트 a) | 33개 float 완전 일치, `final` 11값 완전 일치 | 충족 |
| 2. `mem_size` 에폭마다 증가, 상한 이하 (게이트 c) | A·B 모두 단조 증가. 에폭 e 값 ≤ 12,500·e (A), 62,500·e (B). 에폭 10: 124,654 ≤ 125,000, 617,064 ≤ 625,000 | 충족 |
| 3. `sanity` `top1_sim` 평균 ≈ 1.0 (게이트 b: ≥ 0.999) | A 1.0, B 1.0 | 충족 |
| 4. 에피소드·에폭 기록 필드 | 에폭 ≥ 1의 valid·test 에피소드 1000줄 × 6필드 null 0 (A·B). 에폭 ≥ 1 pooled 값 null 0 | 충족 |
| 5. 소요 시간 | 표(2절) | 보고 |

게이트: 통과, 재시도 0회.

## 4. 이상 징후
1. `config2string` 제외 목록에 `mem`, `mem_k`, `mem_tau`, `mem_chunk`를 추가했다(지시서 변경 목록 외). 미추가 시 `[Config]`·`# Current Settings` 줄에 `mem_k_10_mem_tau_0.1_mem_chunk_256`이 붙는다. 출력 문자열에만 관여한다.
2. 에폭 0 에피소드 기록의 메모리 필드 6개와 에폭 0의 pooled 값은 null이다(에폭 0은 조회하지 않음).
3. test 에피소드의 `mean_top1_sim`이 0.99999 수준이다(예: A 에폭 1 test ep 0: 0.9999940, B: 0.9999971).
4. `sanity`의 `top1_sim_mean`은 1.0이고 `auc_phat`은 1.0이 아니다(A 0.99425, B 0.976854).
5. `mem_sanity_check`는 지시서의 task 그래프 간선 생성 규칙을 별도 함수 안에 다시 작성해 쓴다(원본 `train_epoch` 코드는 그대로).
6. `mem_size`가 상한보다 작다(중복 (r,c) 제거). 에폭 1: A 12,496 / 12,500, B 62,414 / 62,500.
7. 콘솔에 `# mem sanity : {...}` 한 줄이 추가된다(`--mem` 켰을 때만).
