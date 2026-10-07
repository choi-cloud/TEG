# T16a 보고 — 게이트 실패(G3(c)), 태그 없음. 진단용 플래그 구현 + 동적 사실 확인 (2026-10-08, 브랜치 `erasure_diagnosis`, auto mode, overnight 큐)

- 시작 2026-10-08 02:33 KST, 종료 02:44 KST.
- H1 = `a8a67bf`(`docs: overnight 큐 지시서 + CLAUDE.md 정정`, 부모 `e1bec02`).
- 결과: G3(c) 실패. 큐 규칙 S1에 따라 큐 전체를 정지했다. T16b, T17, T18은 시작하지 않았다.

## 1. 변경 파일과 위치
보고서 커밋: 이 파일과 같은 커밋(태그 없음).

**착수(H1)**: `CLAUDE.md` 세 곳만 고쳤다(§1 태그 줄, §1 번호 줄, §8 오프셋 표에 `15000 | 레이블 재분할(T16, --relabel_base)` 추가 및 "16000 이후 미배정"). `instructions/` 신규 5개를 추가했다.

| 파일 | 위치 | 내용 |
|---|---|---|
| `argument.py` | `--referee_k` 다음 | `--sup_coef`(1.0), `--weight_decay`(5e-4), `--optim`(adam), `--gcn_out`(0), `--relabel_base`(0), `--traj_every`(0), `--traj_lr`(0), `--traj_dump_ends`(0), `--grad_probe_steps`("") 추가. 모두 `config2string` 제외 목록에 넣음 |
| `model.py` | `__init__` | `--gcn_out`: 0이면 `conf["gcn_out"]`, `conf["egnn_in"]` 그대로 사용. `pres_head` 입력 차원도 같은 값 |
| `model.py` | `__init__`(optimizer) | `optim.AdamW if --optim adamw else optim.Adam`, `weight_decay=args.weight_decay`. `pres_head` 그룹에 `"weight_decay": args.weight_decay` 명시 |
| `model.py` | `__init__`(`--relabel_base 1`) | base 노드 목록을 클래스 순서대로 이어 `random.Random(15000 + seed).shuffle` 후 원래 순서·크기로 다시 잘라 `id_by_class[c]`(c ∈ base) 대체. 학습 모드 `label_list`용 `relabel_label` 배열(`labels`는 불변) |
| `model.py` | `traj_setup`, `traj_measure`, `grad_probe`(`train_epoch` 앞) | 측정 훅(M1–M9), gradient probe |
| `model.py` | `train_epoch` | `t16_on`이면 각 pass 시작의 `random.getstate()` 해시 기록. 첫 업데이트 직전 step 0 측정 |
| `model.py` | `train_epoch` | `--relabel_base 1`이면 학습 모드 `label_list`를 `relabel_label`로 계산(기본 경로는 원래 줄) |
| `model.py` | `train_epoch` | `--sup_coef != 1.0`일 때만 곱셈(기본은 곱셈 없음). `pres_term = λ·pres`로 분리(산술 순서 동일) |
| `model.py` | `train_epoch` | backward 직전 `grad_probe`, step 후 `upd_step` 증가, M7·M8 누적, K step마다 측정 |
| `model.py` | `train`, run 출력 | 상태 초기화, `t16_info.json`, `grad_probe.json`, `traj.json`, `traj_w.npz`, `traj_z_ends.npz` 저장 |
| `tools/erasure_metrics.py` (신규) | | `knn_purity`, `episodes`, `fewshot_acc`, `proto_euc_acc`, `proto_cos_acc`, `lr_acc`. import 부작용 없음(chdir·sys.path·난수 없음) |
| `tools/t16a_gate_run.py` (신규) | | 게이트용 래퍼. 옛/새 코드를 같은 방식으로 실행하며 읽기 전용 탭으로 `random` 상태 해시, 에피소드, NLL 값, `dump_embedding` 호출, 첫 학습 모드 dropout 마스크·RNG 상태를 기록 |
| `tools/t16a_run_gates.py`, `tools/t16a_facts.py`, `tools/t16a_relabel_check.py`, `tools/analyze_t16a.py` (신규) | | 게이트·사실 확인 실행과 집계 |

- **측정 훅 격리:** `traj_measure`는 `torch.random.fork_rng(devices=[device])` + `no_grad` 안에서 계산한다. `conv`·`egnn`의 `training` 값을 저장했다가 복원한다.
- **에피소드 구성:** probe 에피소드는 `random.Random(1000 + seed)` 인스턴스 하나로 base → valid → test 순서로 200개씩 만든다(T12 방식). 고정 에피소드는 `fixed_episodes(Random(9000/9100 + seed))`다.
- **M8·M9:** ÂX(gcn_norm, self-loop) 희소 행렬의 0 아닌 패턴으로 계산한다.

`id_by_class` 사용 위치(`grep`):

| 위치 | 용도 |
|---|---|
| `utils.py:49–53`(그 밖의 데이터셋 `88–92`, `124–128`, `173–177`, `196–200`) | 생성 |
| `utils.py:277` | `task_generator_in_class`(학습·valid·test 에피소드 노드) |
| `embedder.py:34` | 수신 |
| `model.py:66–73` | `--relabel_base` |
| `model.py:251` | BSC base 프로토타입(`bsc_rwr`) |
| `model.py:373` | `fixed_episodes` |
| `model.py:609` | 학습·valid·test 에피소드 생성 호출 |

## 2. 실행 명령과 출력
- `git worktree add <scratchpad>/t16/old_wt a8a67bf`(옛 코드). 게이트 후 `git worktree remove`.
- `python -u tools/t16a_run_gates.py <old_wt>`: 34 run, 전부 exit 0, `results/T16a/`. seed 0, `--fixed_eval --dump_logits` 공통. 각 run은 `tools/t16a_gate_run.py`로 실행.
- `python tools/t16a_facts.py`(J1·J2·J3·J5), `python tools/t16a_relabel_check.py`(G4(b)(c)), `python tools/analyze_t16a.py` → `results/T16a/gates.json`.
- G3(c) 진단: 기록값 vs CPU·GPU 오프라인 재계산 → `results/T16a/g3c_diagnosis.json`.

### D — gradient 경로 (`grad_probe.json`, 노름)
D1 기본(γ 0.5), W = `conv1.lin.weight`. "0열" = 지도 gradient 열 노름이 정확히 0인 특징 열 수 / F.

| run | step | ‖∂L_N/∂W‖ | ‖∂L_G/∂W‖ | ‖∂sup/∂W‖ | ‖wd·W‖ | ‖∂L_N/∂b‖ | ‖∂L_G/∂b‖ | EGNN ‖∂L_N‖ | EGNN ‖∂L_G‖ | 0열 |
|---|---|---|---|---|---|---|---|---|---|---|
| Clothing 1 | 1 | 352 | 3.39 | 177 | 0.00564 | 71.1 | 3.8e-08 | 71.5 | 0 | 7484/9034 |
| Clothing 1 | 50 | 12.4 | 3.36 | 7.19 | 0.00405 | 1.11 | 4.3e-08 | 7.4 | 0 | 7998/9034 |
| Clothing 1 | 500 | 6.38 | 2.56 | 3.75 | 0.00684 | 0.655 | 3.4e-08 | 6.33 | 0 | 7556/9034 |
| Clothing 5 | 1 | 142 | 1.64 | 71.0 | 0.00564 | 26.8 | 2.5e-08 | 50.0 | 0 | 7741/9034 |
| Clothing 5 | 500 | 0.0832 | 1.21 | 0.620 | 0.00659 | 0.0142 | 3.4e-08 | 0.305 | 0 | 7420/9034 |
| dblp 1 | 1 | 289 | 3.45 | 145 | 0.00563 | 42.9 | 3.8e-08 | 202 | 0 | 5126/7202 |
| dblp 1 | 500 | 4.72 | 2.81 | 3.44 | 0.00657 | 0.405 | 4.2e-08 | 2.63 | 0 | 5207/7202 |
| dblp 5 | 1 | 90.9 | 2.20 | 45.8 | 0.00563 | 5.06 | 2.4e-08 | 25.8 | 0 | 5031/7202 |
| dblp 5 | 500 | 1.44 | 1.21 | 1.22 | 0.00634 | 0.0348 | 2.4e-08 | 0.614 | 0 | 4915/7202 |
| Electronics 1 | 1 | 409 | 7.28 | 206 | 0.00564 | 37.8 | 6.1e-08 | 235 | 0 | 7818/8669 |
| Electronics 1 | 500 | 9.74 | 14.0 | 10.7 | 0.00685 | 0.450 | 8.7e-08 | 1.88 | 0 | 7656/8669 |
| Electronics 5 | 1 | 130 | 3.95 | 66.0 | 0.00564 | 15.3 | 4.1e-08 | 48.3 | 0 | 7528/8669 |
| Electronics 5 | 500 | 7.01 | 10.5 | 7.11 | 0.00648 | 1.37 | 7.1e-08 | 7.45 | 0 | 7365/8669 |

D1 지도 gradient 열 노름 분위수(10/50/90%)와 ‖wd·W[:, j]‖ 분위수(step 1 → step 500):
- Clothing 1: 열 노름 [0, 0, 0.362] → [0, 0, 0.00683], wd [5.5e-05, 5.9e-05, 6.4e-05] → [3.6e-05, 6.9e-05, 9.5e-05].
- 전체 값(모든 run·step 1/10/50/250/500)은 `results/T16a/D*/grad_probe.json`.

D2–D4 (W, step 1 / 500):

| run | ‖∂(γL_N)/∂W‖ | ‖∂((1−γ)L_G)/∂W‖ | ‖∂sup/∂W‖ | ‖∂(λ·pres)/∂W‖ | EGNN ‖∂sup‖ |
|---|---|---|---|---|---|
| D2 γ = 1 Clothing 1 | 352 / 8.96 | 0 / 0 | 352 / 8.96 | — | 71.5 / 10.2 |
| D2 γ = 1 dblp 1 | 289 / 5.69 | 0 / 0 | 289 / 5.69 | — | — |
| D2 γ = 1 Electronics 1 | 409 / 2.73 | 0 / 0 | 409 / 2.73 | — | — |
| D2 γ = 0 Clothing 1 | 0 / 0 | 3.39 / 1.36 | 3.39 / 1.36 | — | 0 / 0 |
| D2 γ = 0 dblp 1 | 0 / 0 | 3.45 / 3.42 | 3.45 / 3.42 | — | 0 / 0 |
| D2 γ = 0 Electronics 1 | 0 / 0 | 7.28 / 12.9 | 7.28 / 12.9 | — | 0 / 0 |
| D3 sup_coef 0 Clothing 1 | 0 / 0 | 0 / 0 | 0 / 0 | — | 0 / 0 |
| D4 대조 Clothing 1 (λ 10) | 176 / 3.61 | 1.70 / 1.29 | 177 / 4.36 | 22.2 / 2.62 | — |
| D4 대조 dblp 1 (λ 30) | 144 / 2.73 | 1.73 / 1.38 | 145 / 3.89 | 87.7 / 12.0 | — |
| D4 대조 Electronics 1 (λ 100) | 204 / 4.17 | 3.64 / 4.09 | 206 / 7.43 | 156 / 28.1 | — |

- D3(`--sup_coef 0`)의 ‖wd·W‖: step 1 5.64e-03, 50 3.36e-04, 250 9.19e-09, 500 1.99e-14.
- ‖∂L_G/∂b‖(bias)는 모든 D run·step에서 최대 1.55e-07이고 0은 아니다.

### J — 동적 사실
| # | 결과 |
|---|---|
| J1 | F: Amazon_clothing 9034(N 24919), dblp 7202(N 40672), Amazon_electronics 8669(N 42318). `conv1.lin.weight` = (64, F) |
| J2 | `.mat`에 없는 노드 수: 세 데이터셋 모두 0. `id_by_class[0]` 크기 134 / 193 / 107. 클래스 0은 Clothing·Electronics에서 base(`class_list_train`), dblp에서 test 목록에 있음(seed 0). `nb` 풀에 드는 미등재 노드 0 |
| J3 | `structural_features` 0 행: Clothing 66, dblp 0, Electronics 528 |
| J4 | Clothing 1-shot seed 0 `--dump_emb`: `dump_embedding` 호출 에폭 [0, 1, 2, 3, 7]. 에폭 0 호출 있음. `emb_epoch` = 7 |
| J5 | 학습 모드 GCN forward 1회: torch CPU 상태 불변, CUDA 상태 변경(세 데이터셋). `fork_rng(devices=[device])` 구간 후 CPU·CUDA 상태 모두 구간 전과 같음(세 데이터셋). 학습 run의 첫 학습 모드 dropout(에폭 1, `cuda:0`)도 CPU 불변·CUDA 변경 |
| J6 | `--gcn_out 128`(Clothing 1-shot seed 0) vs 기본: 에폭별 전역 `random` 상태 해시 일치, 전 에피소드 일치, 고정 에피소드 일치. 첫 학습 모드 dropout 마스크 일치. 그 직전 CUDA RNG 상태 해시는 같고 CPU RNG 상태 해시는 다름 |

## 3. 완료 조건(게이트) 대비
τ: 각 지표 max(|old2 − old1|, 0.1%p), 손실 max(|old2 − old1|, 1e-6). 두 데이터셋 모두 old1·old2의 `random` 해시·에피소드가 일치했고, 정확도 τ = 0.1%p, 손실 τ = 1e-6이었다.

| 게이트 | 결과 | 충족 |
|---|---|---|
| G1 회귀 | Clothing 1 / dblp 5: (a) `random` 해시·학습·valid·test 에피소드·고정 에피소드 old1과 완전 일치. (b) 첫 업데이트 L_N, L_G: Clothing 6.255581855773926 / 1.5608608722686768(old1) vs 6.255580902099609 / 1.5608611106872559(new), 차 9.5e-07 / 2.4e-07. dblp 1.8954353332519531 / 1.4366939067840576 vs 1.8954354524612427 / 1.4366939067840576. (c) `best_acc_valid`, `test_acc_at_best_valid`, 고정 test: Clothing 79.60 / 80.48 / 81.72 동일. dblp 87.84 / 84.96 동일, 고정 test 83.96 vs 83.92(차 0.04%p) | 충족 |
| G2 측정 훅 무영향 | 새 기본 vs 훅 전부: (a) 일치. (b) 첫 업데이트 L_N, L_G 동일 값. (c) Clothing 동일. dblp 고정 test 83.92 vs 83.96(0.04%p), 나머지 동일 | 충족 |
| G3 측정 구현 일치 | (a) kNN 순도 0.8414316239316239 vs 0.8414316239316239, \|Δ\| = 0. (b) `Random(1000)` 에피소드 500개 일치, `fewshot_acc` 84.984 = 84.984(200개: 84.86 = 84.86). (c) G2 run step 500의 M1·M3 vs `traj_z_ends.npz` CPU 오프라인 재계산: dblp 5 최대 \|Δ\| 2.2e-16. **Clothing 1: `purity_base` 2.56e-05, `purity_test` 1.07e-05**(> 1e-6). M3는 \|Δ\| = 0 | **불충족** |
| G4 재분할 | Clothing 1 / dblp 1: (a) `random` 해시 일치, valid·test 에피소드·eval logits 에피소드·고정 에피소드 일치. 학습 클래스 목록 일치, 학습 노드는 다름(설계상). (b) 클래스 목록, base 크기 목록, base 노드 집합, valid·test `id_by_class`, `labels`, 구조 특징 모두 불변. base 목록 40/40, 80/80 변경. (c) 가짜 클래스별 최다 참 클래스 비율 평균 0.0925(최소 0.0715, 최대 0.1287) / 0.0546(0.0377, 0.1077) | 충족 |
| G5 스위치 정확성 | (a) γ = 1: ∂sup/∂GCN vs ∂(γL_N)/∂GCN 상대 차 최대 7.2e-09 / 3.8e-09 / 6.5e-09. (b) γ = 0: GCN의 γL_N 기여 최대 0, EGNN 손실 gradient 최대 0(세 데이터셋). (c) `sup_coef 0`: GCN·EGNN 손실 gradient 최대 0 | 충족 |
| G6 유한 | 34 run의 NLL 값, `run.json` final, `traj.json`, `grad_probe.json` 모두 유한 | 충족 |
| G7 변경 범위 | `git diff --stat a8a67bf <보고서 커밋>`: 부기에 기록 | 부기 |
| G8 시간(기록) | 아래 | 기록 |

G3(c) 실패 상세(`results/T16a/g3c_diagnosis.json`): Clothing 1-shot G2 run.

| step | 그룹 | 기록값 | CPU 재계산 | GPU 재계산 | 그룹 노드 수 | 서로 다른 행 수(소수 7자리) |
|---|---|---|---|---|---|---|
| 0 | base | 0.6332849829351537 | 0.6332935153583619 | 0.6332849829351537 | 11720 | 9986 |
| 0 | valid | 0.7284449075280021 | 같음 | 같음 | 3839 | 3113 |
| 0 | test | 0.8105235042735043 | 같음 | 같음 | 9360 | 7816 |
| 500 | base | 0.7116808873720137 | 0.7117064846416383 | 0.7116808873720137 | 11720 | 10070 |
| 500 | valid | 0.7689762959103934 | 같음 | 같음 | 3839 | 3122 |
| 500 | test | 0.8277243589743589 | 0.8277350427350427 | 0.8277243589743589 | 9360 | 7848 |

- 기록값(훅 안 GPU 계산)은 GPU 오프라인 재계산과 \|Δ\| = 0이다. CPU 재계산과의 차가 1e-6을 넘는다.
- 수정·재실행은 하지 않았다(큐 규칙 S1).

G8 (run당 시간, s, `_runs.jsonl`):

| 설정 | Clothing 1 | dblp 5 | Electronics 5 |
|---|---|---|---|
| 훅 끔(새 기본) | 24.9 | 27.7 | 27.6 |
| 옛 코드 old1 / old2 | 24.7 / 23.3 | 28.9 / 28.7 | — |
| `--traj_every 10`(T16b 설정) | 29.8 | — | 40.1 |
| 훅 전부(`--traj_every 10 --traj_lr 1 --traj_dump_ends 1 --grad_probe_steps 1,50`) | 51.3 | 68.3 | — |
| T17 G1 설정(`--sup_coef 0`, 대조 λ 10, `--traj_every 50 --traj_lr 1`) | 42.5 | — | — |
| T17 G2 설정(위 + `--gcn_layers 2`) | 42.7 | — | — |

예상 소요(run 수 × 평균 시간 ÷ 6 + 분석):

| 작업 | 계산 | 예상(분) |
|---|---|---|
| T16b | 360 × 35.0 s(`--traj_every 10` 두 값 평균) ÷ 6 = 35 + 분석 10 | 약 45 |
| T17 | (90 × 42.5 + 30 × 42.7) s ÷ 6 = 14 + 분석 5 | 약 19 |
| T18 | 90 × 35.0 s ÷ 6 = 9 + 분석 5 | 약 14 |

## 4. 이상 징후
1. **G3(c) 실패.** Clothing 1-shot의 base·test 그룹 kNN 순도에서 훅 기록값(GPU)과 CPU 재계산이 다르다. 기록값은 GPU 재계산과 같다. 해당 그룹에는 같은 값을 가진 행(서로 다른 행 수 < 노드 수)이 있다.
2. **큐 정지.** G3(c) 실패로 큐 규칙 S1에 해당하여 T16b, T17, T18은 시작하지 않았다.
3. **bias gradient.** D 표에서 ‖∂L_G/∂b‖는 최대 1.55e-07의 0 아닌 값이다.
4. **`--sup_coef 0`(D3).** ‖wd·W‖가 step 500에서 1.99e-14다. 같은 run의 ‖∂L_N/∂W‖(곱하기 전 원값)는 step 50에서 2.82e+03이다.
5. **J6.** `--gcn_out 128`에서 첫 dropout 직전의 CPU RNG 상태 해시는 기본과 다르고, 첫 dropout 마스크는 기본과 같다.
6. **게이트 run의 추가 플래그.** 게이트 run은 지시서 설정에 `--dump_logits`를 더해 실행했다(에피소드 비교용 덤프).
7. **probe 에피소드 생성 방식.** M3 probe 에피소드는 `Random(1000 + seed)` 한 인스턴스로 base → valid → test 순서로 만들었다(지시서 "그룹별로 한 번만"의 구현).
8. **시스템 시각 표기.** 드라이버 로그 시각은 UTC로 찍혀 있다(예: 17:39 = 02:39 KST).

## 부기 (2026-10-08, 커밋·push·G7)
- 보고서 커밋 `e851a9f`(태그 없음, 게이트 G3(c) 실패).
- push(force 없음): `git push origin erasure_diagnosis` → `e1bec02..e851a9f`.
- G7: `git diff --stat a8a67bf e851a9f`의 파일은 `argument.py`, `model.py`, `reports/T16a_impl_facts.md`, `tools/analyze_t16a.py`, `tools/erasure_metrics.py`, `tools/t16a_facts.py`, `tools/t16a_gate_run.py`, `tools/t16a_relabel_check.py`, `tools/t16a_run_gates.py`(9개)다. `configuration.yaml`, `layers/`, `embedder.py`는 변경 없음 — 충족.
