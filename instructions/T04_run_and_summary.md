# T04 — 본 실행 30 run과 요약표

정본 `CLAUDE.md`. 선행: T03b. manual mode, 멈춤 지점 두 곳(§2).

## 목표 (검증할 주장)
세 데이터셋 × 5-way {1, 5}-shot × seed 0–4 = 30 run이 `--mem`으로 끝까지 돌고, 설정별 짝지은 비교표와 전제 진단표가 자동 생성된다.

## 바꿀 것 (**멈춤 지점 1: 수정 전 계획 승인**)
- `tools/run_t04.sh` (또는 `.py`): 30개 명령을 물리 GPU 0,1,2,4,5,6에 GPU당 하나씩 순차 배분. 명령 형식:
  `CUDA_VISIBLE_DEVICES=<g> python main.py --dataset <ds> --way 5 --shot <k> --seed <s> --num_seed 1 --device 0 --mem --out_dir results/T04/<ds>_5w<k>s/seed<s>`
  - 순서: Amazon_clothing → dblp → Amazon_electronics. 각 run의 콘솔 출력은 같은 폴더의 `console.log`.
  - 실패한 run은 한 번 재시도하고, 그래도 실패하면 목록에 남기고 나머지를 계속한다.
- `tools/summarize_t04.py`: `results/T04/**/run.json`을 읽어 `reports/T04_summary.md`를 쓴다.

## 요약표 (`reports/T04_summary.md`)
- **표 1 — 정확도**: 행 = (데이터셋, shot) 6개, 열 = 설정 7개. 셀 = "seed 평균 `test_acc_at_best_valid` (off 대비 짝지은 차 ± SE, 부호 일치 수)". 검출·의미 표시(CLAUDE.md §5). 단위 %p.
- **표 2 — valid로 고른 β**: 설정 A, B 각각, seed마다 valid 최고값을 낸 β를 골라 그 β의 `test_acc_at_best_valid`를 쓴 "A_sel", "B_sel"의 표 1 형식.
- **표 3 — 전제 진단**: 행 = (데이터셋, shot). off 설정의 `best_valid_epoch`에서 `auc_phat_pooled`, `auc_dist_pooled`, 그 차, `mean_phat_same`, `mean_phat_diff`의 seed 평균 ± sd.
- **표 4 — seed별 원값**: 30 run × 설정 7개.
- 부록: 실패·재시도 목록, run별 소요 시간.

## 사전 등록 (수정 금지, 2026-10-01 작성)
- **P34.** A_sel − off: 6개 (데이터셋, shot) 모두에서 |mean d| < 1.0 %p.
- **P35.** 전제 진단: 6개 모두에서 `auc_phat_pooled` < `auc_dist_pooled` (메모리 조회가 거리만으로 판별하는 것보다 못하다).
- **P36.** B_sel − off: 6개 중 4개 이상에서 |mean d| < 1.0 %p.

## 완료 조건
1. 30 run의 `run.json` 존재(실패 목록 제외)
2. 표 1~4 생성
3. P34~P36 "예측 / 관측 / 일치·어긋남" 표

## 보고 (**멈춤 지점 2**) — `reports/T04_run_and_summary.md`
- 실행 스크립트 요약, 총 소요 시간, 실패 목록
- `reports/T04_summary.md` 경로와 표 1·3 본문 복사
- P34~P36 대비 표
- 이상 징후, 커밋 해시, 태그 `t04`

## 하지 말 것
CLAUDE.md §7 전부. 설정 추가, β·k·τ 변경, 결과 해석.
