# H2 — BSC 착수 전 정리 (코드 변경 없음, 학습 run 없음, 태그 없음)

정본 `CLAUDE.md`. 선행: `t13`(a3c4212), push 부기 커밋 421a430. 이 파일은 `instructions/H2_housekeeping_pre_bsc.md`에 둔다.
**auto mode: CLAUDE.md §2의 멈춤 지점을 이 지시서 동안 해제한다.** 아래 "멈춤 조건"에 해당하면 멈춘다.
**원칙: 먼저 상태를 확인하고, 이미 반영된 항목은 건너뛴 뒤 그 사실을 기록한다.** 앞선 정리 커밋(47cc88d, c340a4c)과 겹치는 항목이 있을 수 있다.

---

## 0. 시작 상태 기록 (아무것도 바꾸기 전)
- `git branch --show-current`, `git rev-parse HEAD`, `git status --short`, `git remote -v`, `git log --oneline -8`
- 리포 디스크 여유(`df -h .`), `du -sh results/*`
- `reports/housekeeping_20261007.md`에 적힌 처리 항목 요약(한 줄씩)
- 현재 브랜치가 `preserve_rehearse`가 아니면 **멈추고 보고한다.**

## 1. CLAUDE.md §1 원격 표기
- §1이 이미 `origin` = choi-cloud/TEG를 가리키면 건너뛰고 "이미 반영(커밋 해시)"으로 적는다.
- 아니면 원격 줄을 `origin`: choi-cloud/TEG로 고친다. `Dragor0123/TEG`는 "과거 fork, 사용하지 않음"으로 한 줄만 남긴다.

## 2. `results/T10/` 삭제 (사용자 결정에 따른 예외)
- 근거: T10은 디스크 부족으로 실패했고 태그가 없다. 사양은 T10r로 대체됐고, 부분 결과는 어느 분석에도 쓰이지 않았다.
- CLAUDE.md §7(H1)의 "태그가 달린 실험만 삭제" 규칙에 대한 **이번 1회 예외**다(사용자 결정, 2026-10-07).
- 절차:
  1. `results/T10/`이 없으면 건너뛰고 "이미 없음"으로 적는다.
  2. 있으면 `grep -rn "results/T10/" tools/ reports/ *.py`로 참조를 찾는다(`results/T10r`는 해당하지 않는다). `tools/run_t10.py`, `tools/analyze_t10.py` 외의 파일이 참조하면 **멈추고 보고한다.**
  3. 파일 수와 용량(`du -sh`)을 기록한 뒤 `results/T10/`만 삭제한다. 다른 경로는 지우지 않는다.

## 3. CLAUDE.md 정비 (문서만)
1. §0 끝에 다음 한 줄을 추가한다: "T07 이후 주제: TEG에 레이블 없는 노드 보조 손실(보존·대조)을 더하는 실험, 다음은 Blind-Spot Contrast(BSC). 관계 메모리는 T01–T05 범위."
2. §5 제목 바로 아래에 다음 한 줄을 추가한다: "이 절은 T01–T06 기준이다. T07 이후의 표준은 §8을 따른다."
3. 새 절 **§8 현행 표준 (T07 이후)**를 아래 내용 그대로 추가한다.
   - **별칭**: 원본 = TEG(λ = 0), 보존 = `kl_h2`(선택 λ), 대조 = `infonce`(선택 λ), D = 행 정규화 Â²X 코사인 프로토타입, +D = α = 0.5 사후 결합(`tools/fusion_baseline.py`).
   - **평가**: `--fixed_eval`로 seed당 고정 test 에피소드 200개(`Random(9000 + seed)`)와 valid 100개(`Random(9100 + seed)`)를 쓴다. 비교는 같은 에피소드끼리의 짝지은 차로 하고, "분명"은 평균 > 0이고 > 2·SE인 경우다. 원 보고 방식(`test_acc_at_best_valid`)은 병기한다.
   - **λ 선택**: (변형, 데이터셋, shot)마다 원본 `best_acc_valid`의 seed 평균이 최고인 λ를 고른다. 동률이면 작은 λ를 고른다.
   - **게이트 diff 기준**: 지시서의 "변경 범위" 게이트는 **그 지시서 착수 시점의 HEAD** 기준으로 계산한다.
   - **전용 난수 오프셋 등록부**(seed에 더함):

     | 오프셋 | 용도 |
     |---|---|
     | 1000 | probe 에피소드 |
     | 2000 | rand·pca 분석 공간 |
     | 3000 | base 분할(T06b) |
     | 4000 | 순열(T06b) |
     | 7000 | 보존 표본 |
     | 8000 | 대조 증강 |
     | 9000 / 9100 | 고정 test / valid |
     | 11000 / 12000 | T11 |
     | 13000 | T13 게이트 |
     | **14000–14999** | BSC 예약 |

     새 코드는 등록부에 없는 오프셋을 쓰지 않는다. 새로 쓸 오프셋은 지시서에 명시한 뒤 여기에 추가한다.
4. 등록부 점검(읽기 전용): 코드에서 `random.Random(`, `default_rng(`, `torch.Generator`, `manual_seed(`, `random_state=`를 검색하고, 쓰인 오프셋을 등록부와 대조한 표를 보고서에 적는다. 등록부에 없는 값이 나와도 고치지 않고 적기만 한다.

## 4. T11 미커밋 파일 보관 (커밋보다 먼저)
1. 현재 HEAD에서 새 브랜치 `t11_wip`를 만든다.
2. `tools/run_t11.py`, `tools/analyze_t11.py`만 커밋한다. 메시지: `wip: T11 rehearse scripts (not run)`
3. `origin`에 push한 뒤 `preserve_rehearse`로 돌아온다.
4. T11 파일 2개가 `preserve_rehearse` 작업 트리에 없음을 확인한다. 1–3에서 바꾼 파일은 커밋하지 않은 채로 따라오며, 5에서 커밋한다.

## 5. 커밋·push (`preserve_rehearse`)
- 보고서에는 이 시점까지(0–4) 결과를 적는다.
- 커밋 대상은 `CLAUDE.md`, `instructions/H2_housekeeping_pre_bsc.md`, `reports/housekeeping_H2.md`다.
- 메시지: `docs: H2 housekeeping before BSC (CLAUDE.md §8, results/T10 removal record)`
- `origin`에 push한다. **force push 금지.**

## 6. BSC 작업 브랜치
- `preserve_rehearse` HEAD(5의 커밋)에서 브랜치 `blind_spot_contrast`를 만들고 `origin`에 push한다.
- 체크아웃한 상태로 둔다. 이 브랜치에는 코드 변경을 하지 않는다.
- G3·G4를 확인한 뒤 결과를 `reports/housekeeping_H2.md` 끝에 "부기"로 추가하고, 이 브랜치에 별도 커밋(`docs: H2 addendum (gates G3–G4)`)으로 push한다.

## 7. 게이트
- **G1**: 5의 커밋이 바꾼 파일이 `CLAUDE.md`, `instructions/`, `reports/`뿐이다(기준: 착수 시점 HEAD).
- **G2**: 이 지시서 전체에서 `*.py` 변경은 `t11_wip` 브랜치의 T11 파일 2개뿐이다.
- **G3**: 부기 커밋 전, 브랜치가 `blind_spot_contrast`이고 작업 트리가 깨끗하다.
- **G4**: 부기 push 후 `git ls-remote origin`에서 `preserve_rehearse`, `t11_wip`, `blind_spot_contrast`가 로컬 해시와 같다(부기에는 G4 확인 직전 해시를 적는다).

## 8. 멈춤 조건
- 0에서 브랜치가 다르다.
- 2에서 예상 밖의 참조가 있다.
- push가 거부된다. 이때는 거부 내용을 기록하고 이후 단계를 진행하지 않는다.

## 9. 보고 `reports/housekeeping_H2.md` (사실만)
1. 시작 상태(0), 단계별 처리 결과를 "항목 / 이미 반영·처리·건너뜀 / 근거(해시·경로)"로 적는다.
2. 삭제 기록: 경로, 파일 수, 확보 용량, 디스크 여유 전·후
3. `du -sh results/*`(삭제 후)
4. 난수 오프셋 점검 표
5. 커밋 해시, push 결과, 게이트 G1–G4
6. 이상 징후

## 10. 하지 말 것
- CLAUDE.md §7 전부.
- `results/T10/` 외 삭제.
- 코드 변경.
- 학습 run.
- force push, 히스토리 재작성.
- 태그 생성.
- 결과 해석.
