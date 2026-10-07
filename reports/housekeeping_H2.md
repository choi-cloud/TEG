# H2 — BSC 착수 전 정리 (2026-10-08, 브랜치 `preserve_rehearse`, auto mode)

## 1. 시작 상태와 단계별 처리
시작 상태(2026-10-08 00:01 KST, 변경 전):
- 브랜치 `preserve_rehearse`, HEAD `421a430c05e80be74fe06d6d9dbb481f33c7b8e3`.
- `git status --short`: `?? instructions/H2_housekeeping_pre_bsc.md`, `?? tools/analyze_t11.py`, `?? tools/run_t11.py`.
- `git remote -v`: `origin https://github.com/choi-cloud/TEG.git` (fetch/push).
- `git log --oneline -8`: `421a430`, `a3c4212`, `c340a4c`, `47cc88d`, `3f2ad51`, `fe34c7a`, `5134f90`, `fd33314`.
- `df -h .`: `/dev/mapper/ubuntu--vg-ubuntu--lv 98G 80G 14G 86% /`.
- `reports/housekeeping_20261007.md` 처리 항목:
  - `results/T10/` 전체 삭제(파일 590개, `emb_best.npy` 74개, 971M).
  - 근거 기록: T10r 대체, 태그 `t10r` 원격 존재.
  - 다른 `results/` 변경 없음.
  - `df -h /` 정리 전 81G 사용 / 12G 여유(88%), 정리 후 80G 사용 / 13G 여유(87%).

| 항목 | 처리 | 근거 |
|---|---|---|
| 0. 시작 상태 | 기록 | 위 |
| 1. CLAUDE.md §1 원격 표기 | 이미 반영 — 건너뜀 | `47cc88d`(§1 = `origin` `choi-cloud/TEG`, `Dragor0123/TEG` 표기 0건) |
| 2. `results/T10/` 삭제 | 이미 없음 — 건너뜀(참조 검색·삭제 안 함) | `c340a4c`, `reports/housekeeping_20261007.md` |
| 3.1 §0 끝 한 줄 | 처리 | `CLAUDE.md` 7행 |
| 3.2 §5 제목 아래 한 줄 | 처리 | `CLAUDE.md` 46행 |
| 3.3 §8 현행 표준 | 처리(지시서 문구 그대로) + 사용자 추가 지시 반영: 등록부에 `5000 | T08·T08b 분석` 행, 11000 / 12000 행에 "(리포 밖 패치에서만 사용, 미실행)", 표 아래 "등록부 밖 기존 값" 한 줄 | `CLAUDE.md` §8 |
| 3.4 난수 오프셋 점검 | 처리(읽기 전용) | 4절 |
| 4. T11 파일 `t11_wip` 보관 | 처리 — **사용자 직접 실행**(Claude의 같은 명령은 auto mode 권한 분류기에 거부됨, 6절). `git switch -c t11_wip && git add tools/run_t11.py tools/analyze_t11.py && git commit -m "wip: T11 rehearse scripts (not run)" && git push -u origin t11_wip && git switch preserve_rehearse` | `t11_wip` `06b7d30`(파일 2개, +265), 원격 `refs/heads/t11_wip` = `06b7d30`. `preserve_rehearse` 작업 트리에 T11 파일 2개 없음 확인 |
| 5. 커밋·push | 처리 | 5절 |
| 6. `blind_spot_contrast` | 처리 | 부기 |

## 2. 삭제 기록
이번 지시서에서 삭제한 것 없음(`results/T10/`은 `c340a4c`에서 이미 삭제).

## 3. `du -sh results/*` (시작 시점 = 현재)
| 경로 | 크기 |
|---|---|
| results/_diag.txt | 0 |
| results/T01 | 152K |
| results/T02 | 180K |
| results/T03a | 708K |
| results/T03b | 1.1M |
| results/T04 | 15M |
| results/T05 | 184M |
| results/T06 | 166M |
| results/T06b | 335M |
| results/T07 | 1.3G |
| results/T08 | 14M |
| results/T08b | 8.0K |
| results/T09 | 2.9G |
| results/T10r | 398M |
| results/T12_gate_checks.txt | 4.0K |
| results/T13_gate_checks.txt | 4.0K |

## 4. 난수 오프셋 점검 (`random.Random(`, `default_rng(`, `torch.Generator`, `manual_seed(`, `random_state=`; 리포 `*.py`, `dataset/` 제외)
| 위치 | 호출 | 오프셋 | 등록부 |
|---|---|---|---|
| `utils.py:261–262` | `torch.manual_seed(seed)`, `torch.cuda.manual_seed(seed)` | 0 (원본 TEG) | 없음 |
| `model.py:65` | `torch.Generator().manual_seed(7000 + set_seed)` | 7000 | 있음(보존 표본) |
| `model.py:68` | `torch.Generator().manual_seed(8000 + set_seed)` | 8000 | 있음(대조 증강) |
| `model.py:294–295` | `random.Random(seed_off + self.set_seed)`, seed_off ∈ {9000, 9100} | 9000 / 9100 | 있음 |
| `tools/analyze_t06.py:116` | `random.Random(1000 + s)` | 1000 | 있음 |
| `tools/analyze_t06b.py:79–80` | `PCA(..., random_state=0)` | 0 (상수) | 없음 |
| `tools/analyze_t06b.py:85` | `np.random.default_rng(2000 + seed)` | 2000 | 있음 |
| `tools/analyze_t06b.py:172` | `np.random.default_rng(4000 + seed)` | 4000 | 있음 |
| `tools/analyze_t06b.py:261` | `random.Random(1000 + s)` | 1000 | 있음 |
| `tools/analyze_t06b.py:273` | `random.Random(3000 + s)` | 3000 | 있음 |
| `tools/analyze_t08.py:160` | `random.Random(5000 + s)` | 5000 | 없음 |
| `tools/analyze_t08b.py:62` | `random.Random(5000 + s)` | 5000 | 없음 |
| `tools/analyze_t12.py:133` | `random.Random(1000 + s)` | 1000 | 있음 |
| `tools/analyze_t13.py:89` | `np.random.default_rng(13000)` | 13000 (seed 무관 상수) | 있음(T13 게이트) |
| `tools/analyze_t13.py:104` | `PCA(..., random_state=2000 + s)` | 2000 | 있음 |
| scratchpad `apply_t11.py:30`(리포 밖, 미커밋 T11 패치) | `KMeans(..., random_state=11000 + set_seed)` | 11000 | 있음(T11) |
| scratchpad `apply_t11.py:38`(리포 밖) | `random.Random(12000 + set_seed)` | 12000 | 있음(T11) |

- `tools/run_t11.py`, `tools/analyze_t11.py`에는 검색 대상 호출이 없다.
- 등록부에 없는 값: 0(`utils.py` 원본 seed, `analyze_t06b.py` PCA `random_state=0`), 5000(`analyze_t08.py`, `analyze_t08b.py`). 수정하지 않았다.

## 5. 커밋·push·게이트
- 5단계 커밋(`preserve_rehearse`): `docs: H2 housekeeping before BSC (CLAUDE.md §8, results/T10 removal record)` — 대상 `CLAUDE.md`, `instructions/H2_housekeeping_pre_bsc.md`, `reports/housekeeping_H2.md`. 해시와 push 결과는 부기에 적는다(이 파일이 커밋 대상).
- G1·G2·G3·G4 결과는 부기에 적는다.

## 6. 이상 징후
1. 4단계 명령(`git switch -c t11_wip`, T11 파일 2개 커밋, `git push -u origin t11_wip`, `git switch preserve_rehearse`)을 Claude가 실행하려 했을 때 Claude Code auto mode 권한 분류기가 거부했다(사유 표기: "Out-of-Place Publication", 실행 전 차단). 같은 명령을 사용자가 직접 실행했다.
2. T13 보고서 이상 징후 1(T06b pca1 `random_state=0`)과 같은 값이 등록부 밖 값으로 4절에 나온다. 사용자 지시로 §8 등록부 아래에 기존 값으로 기록했다.
3. 4절의 5000(`analyze_t08.py`, `analyze_t08b.py`)은 점검 당시 등록부 밖이었고, 사용자 지시로 등록부에 추가했다.

## 부기 (2026-10-08, 5·6단계 결과와 게이트)
- 5단계 커밋: `cecc750`(`preserve_rehearse`). 변경 파일은 `CLAUDE.md`, `instructions/H2_housekeeping_pre_bsc.md`, `reports/housekeeping_H2.md`다.
- 5단계 push: `git push origin preserve_rehearse` → `421a430..cecc750`(force 없음).
- 6단계: `cecc750`에서 `blind_spot_contrast`를 만들었다. `git push -u origin blind_spot_contrast` → 새 브랜치. 코드 변경 없음.

| 게이트 | 조건 | 관측 | 결과 |
|---|---|---|---|
| G1 | 5의 커밋 변경 파일이 `CLAUDE.md`, `instructions/`, `reports/`뿐(기준 착수 시점 HEAD `421a430`) | `git diff --name-only 421a430 cecc750`: `CLAUDE.md`, `instructions/H2_housekeeping_pre_bsc.md`, `reports/housekeeping_H2.md` | 통과 |
| G2 | 전체 `*.py` 변경이 `t11_wip`의 T11 파일 2개뿐 | `421a430..cecc750` `*.py` 변경 0개. `421a430..t11_wip(06b7d30)` `*.py` 변경은 `tools/analyze_t11.py`, `tools/run_t11.py` | 통과 |
| G3 | 부기 커밋 전 브랜치 `blind_spot_contrast`, 작업 트리 깨끗 | 브랜치 `blind_spot_contrast`, `git status --short` 출력 없음 | 통과 |
| G4 | 부기 push 후 `git ls-remote origin`의 세 브랜치 = 로컬 해시 | 확인 직전 로컬 해시: `preserve_rehearse` `cecc750`, `t11_wip` `06b7d30`, `blind_spot_contrast` = 이 부기 커밋(부모 `cecc750`). 확인 결과는 채팅 보고에 적는다(이 파일은 부기 커밋에 포함되므로 자기 해시를 담을 수 없다) | 부기 push 후 확인 |
