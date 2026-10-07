# T16a — 진단용 플래그 구현 + 동적 사실 확인

2026-10-09 · 작성: 채팅 Claude · 모드: auto(overnight 큐, `instructions/QUEUE_overnight_20261009.md`) · 브랜치 `erasure_diagnosis` · 태그 `t16a` · 선행: `t15`

## 0. 목적과 검증 가능한 주장

**주장**: 다음 세 가지가 성립하고, 아래 D·J 항목의 사실을 확보한다.
1. 새 플래그를 모두 기본값으로 둔 코드는 원본과 같은 경로로 학습한다.
2. 각 플래그는 지시한 부분만 바꾼다.
3. 측정 훅은 학습을 바꾸지 않는다.

**이 지시서의 게이트가 하나라도 실패하면 큐 전체를 멈추고 대기한다.** T16b, T17, T18이 모두 이 지시서에 의존하기 때문이다.

## 1. 착수

1. 다음을 확인한다. 어긋나면 큐 전체 정지.
   - `git status`가 깨끗한가. 단 사용자가 첨부한 지시서 파일(`instructions/` 아래 신규 5개)은 예외로 허용한다.
   - 브랜치가 `erasure_diagnosis`이고 HEAD가 T15 부기 커밋 `e1bec02`인가.
2. `CLAUDE.md`를 아래 세 곳만 고친다.
   - §1 태그 줄: "`t01`, `t02`, `t03a`, `t03b`, `t04`, `t05`, `t06`, `t06b`, `t07`, `t08`, `t08b`, `t09`, `t10r`, `t12`, `t13`, `t14`, `t15`. 다음 태그는 `t16a`."
   - §1 번호 줄: "T16a(구현·동적 사실), T16b(주체 진단), T17(TLP식 기준선), T18(조건, 선택). 사전 등록 P85–P99는 이 지시서들에 있다. 다음 사전 등록 번호는 **P100**."
   - §8 오프셋 표: "15000 | 레이블 재분할(T16, `--relabel_base`)" 행을 추가하고, "15000 이후 미배정"을 "16000 이후 미배정"으로 바꾼다.
3. 커밋한다. 메시지: `docs: overnight 큐 지시서 + CLAUDE.md 정정`. 포함 파일: `CLAUDE.md`, `instructions/` 신규 5개.
   - 이 커밋을 **H1**이라 부르고 해시를 기록한다. T16a의 diff 기준은 H1이다.

## 2. 구현 (모든 플래그의 기본값 = 원본 동작)

### 2.1 플래그

| 플래그 | 기본 | 동작 | 위치(T15 기준) |
|---|---|---|---|
| `--sup_coef` | 1.0 | 지도 손실 `γ·L_N + (1−γ)·L_G`에 곱한다. **기본값일 때는 곱셈 자체를 하지 않는 분기로 둔다**(산술 동일). | `model.py:565` |
| `--weight_decay` | 5e-4 | `model.py:50`의 상수를 대체한다. `pres_head` 그룹에도 같은 값을 명시한다. | `model.py:50`, `78` |
| `--optim` | `adam` | `adamw`이면 같은 param group으로 `torch.optim.AdamW(lr, weight_decay=args.weight_decay)`를 쓴다. | `model.py:50` |
| `--gcn_out` | 0 | 0이면 `configuration.yaml` 값을 쓴다. 0보다 크면 `gcn_out`과 `egnn_in`을 이 값으로 덮어쓴다. `pres_head` 입력 차원도 따라간다. | `model.py:45–48`, `71` |
| `--relabel_base` | 0 | 2.2 참고 | `utils.py:67–69` 이후 |
| `--traj_every` | 0 | K > 0이면 측정 훅을 켠다(2.3). K는 500의 약수다. | `train_epoch` 업데이트 루프 |
| `--traj_lr` | 0 | 1이면 측정점마다 M5를 추가한다. T17 전용. | 측정 훅 |
| `--traj_dump_ends` | 0 | 1이면 측정점 step 0과 step 500의 z(float32, [N, d])를 저장한다. | 측정 훅 |
| `--grad_probe_steps` | `""` | 쉼표로 구분한 업데이트 step 번호(1부터 셈). 2.4 참고 | `model.py:573–576` 직전 |

### 2.2 `--relabel_base 1` (레이블 재분할)

- **시점**: valid 분할(`utils.py:67–69`) 이후, 첫 에피소드 추출 전.
- **절차**: base 클래스(`class_list_train`)의 노드 목록을 클래스 순서대로 이어 붙인다. 이것을 `random.Random(15000 + seed)`로 섞은 뒤, **원래 클래스 순서와 크기 그대로** 다시 잘라 `id_by_class[c]`(c ∈ base)를 대체한다.
- **바꾸지 않는 것**: valid·test 클래스의 `id_by_class`, `labels` 배열, 보조 손실 풀 구성.
- **요건**: 전역 `random`의 호출 순서와 횟수가 바뀌지 않아야 한다(G4).
- `id_by_class`를 읽는 모든 위치를 `grep`으로 열거해 보고서에 적는다.

### 2.3 측정 훅 (`--traj_every K`)

- **측정점**: 첫 업데이트 직전(step 0), 그리고 업데이트 K번째마다. 업데이트는 에폭 1–10에 걸쳐 모두 500회이므로 마지막 측정점은 step 500이다.
- **격리 규칙**
  - `torch.random.fork_rng(devices=[현재 장치])`와 `torch.no_grad()` 안에서 측정한다.
  - `conv`와 `egnn`의 `training` 플래그를 저장해 두고, eval로 바꿔 측정한 뒤 **원래 값으로 복원**한다.
  - 전역 `random`, numpy 전역 난수, 학습 경로의 torch 난수를 소비하지 않는다.
- **측정 함수의 위치**: `tools/erasure_metrics.py`(신규)에 둔다. 모듈을 import할 때 부작용(`os.chdir` 등)이 없어야 한다.
  - kNN 순도는 `tools/analyze_t14.py`의 `knn_purity`와, probe는 `tools/analyze_t06b.py`의 `fewshot_acc`와 계산이 같아야 한다(G3).
- **표현 z**: `conv(features, edges)`의 eval 출력(LayerNorm 전).
- **그룹**: base(`class_list_train` 노드, 참 레이블), valid, test.

| 기호 | 항목 | 정의 |
|---|---|---|
| M1 | `purity_g` | 그룹 g 안에서 k = 10 kNN 순도. 행 L2 정규화 후 코사인 |
| M2 | `purity_ln_g` | valid·test만. `egnn.LayerNorm(z)`에 대한 M1 |
| M3 | `probe_g` | 코사인 프로토타입 5-way, run의 k_shot, query 5, 에피소드 200개. 에피소드는 `random.Random(1000 + seed)`로 **그룹별로 한 번만** 만들고 모든 측정점에서 재사용한다 |
| M4 | `proto_euc_{valid,test}` | 고정 valid 100 / test 200 에피소드(`--fixed_eval`과 같은 에피소드, `Random(9100/9000 + seed)`)에서 z 위 유클리드 프로토타입(support 평균) 정확도. L_G와 같은 판정 규칙 |
| M5 | `cos_{valid,test}`, `lr_{valid,test}` | `--traj_lr 1`일 때만. M4와 같은 고정 에피소드에서 행 L2 정규화 z 위의 코사인 프로토타입 정확도, 로지스틱 회귀 probe 정확도 |
| M6 | `colnorm` | `conv1.lin.weight`(d × F)의 특징 열별 L2 노름 [F]. bias 노름도 기록 |
| M7 | 손실 | 직전 K step의 L_N, L_G, pres 평균 |

- **로지스틱 회귀(M5)**: 다항, L2 penalty, C = 1.0, lbfgs, max_iter 100. sklearn `LogisticRegression` 기본값과 같다.
  - sklearn이 이미 설치돼 있으면 그것을 쓴다. 설치돼 있지 않으면 같은 목적 함수를 torch LBFGS로 구현한다. **설치는 금지**다.
  - 무작위성은 없어야 한다.
- **측정점이 아닌 1회 계산**
  - **M8 `touch_count[F]`**: ÂX(GCNConv와 같은 정규화, self-loop 포함)를 학습 시작 전에 `no_grad`로 한 번 계산한다. 업데이트 step마다 support ∪ query 노드의 ÂX 행에서 0이 아닌 특징 j의 등장 횟수를 누적한다. dropout은 무시한다.
  - **M9 `freq_base[F]`, `freq_unused[F]`**: base 클래스 노드 / valid ∪ test 클래스 노드 각각의 ÂX 행에서 특징 j가 0이 아닌 비율.
- **저장**
  - `results/<지시서>/<run>/traj.json`: M1–M5, M7, 측정 step 목록
  - `traj_w.npz`: M6 [측정점 수, F] float32, M8, M9
  - `--traj_dump_ends 1`이면 `traj_z_ends.npz`(step 0, step 500의 z, float32)

### 2.4 `--grad_probe_steps`

지정 step에서 backward 전에 아래 값을 계산해 `grad_probe.json`에 기록한다. 계산은 `torch.autograd.grad(..., retain_graph=True)`로 하고, 기록 후 원래의 backward와 step을 그대로 진행한다.

- GCN 파라미터에 대해, weight와 bias를 따로:
  - ‖∂L_N/∂θ‖, ‖∂L_G/∂θ‖, ‖∂(λ·pres)/∂θ‖(보조 손실이 켜진 경우)
  - ‖wd·θ‖
- `W = conv1.lin.weight`에 대한 지도 손실 합 `∂(γL_N + (1−γ)L_G)/∂W`(`sup_coef` 반영):
  - 특징 열 노름이 **정확히 0**인 열의 수와 비율
  - 열 노름의 분위수(10/50/90%)와, 같은 열들의 ‖wd·W[:, j]‖ 분위수
- EGNN 파라미터에 대해: ‖∂L_N/∂θ_EGNN‖, ‖∂L_G/∂θ_EGNN‖.

## 3. 동적 사실 확인

### 3.1 gradient 경로 (D)

| # | 설정 | run |
|---|---|---|
| D1 | 기본(γ 0.5), `--grad_probe_steps 1,10,50,250,500` | 3 데이터셋 × {1, 5}-shot, seed 0 → 6 |
| D2 | `--gamma 1.0`, `--gamma 0.0`(같은 step) | 3 데이터셋 1-shot → 6 |
| D3 | `--sup_coef 0` | Clothing 1-shot → 1 |
| D4 | 대조(T14 `uni`와 같은 인자, λ는 아래 표) | 3 데이터셋 1-shot → 3 |

대조의 λ(T10r 표 1 선택값): Clothing 1 = 10, Clothing 5 = 30, dblp 1 = 30, dblp 5 = 10, Electronics 1 = 100, Electronics 5 = 30.

### 3.2 T15 J절의 동적 확인

| # | 확인 내용 |
|---|---|
| J1 | 데이터셋별 F |
| J2 | `.mat`에 없는 노드(레이블 0으로 남은 노드)의 수, 그중 `nb` 풀에 드는 수, `id_by_class[0]`의 크기(레이블 0이 실제 클래스인가) |
| J3 | `structural_features`가 0인 행의 수 |
| J4 | 에폭 0 끝에 `dump_embedding(0)`이 호출되는가(Clothing 1-shot seed 0, `--dump_emb`) |
| J5 | 학습 forward 전후 `torch.get_rng_state()`·`torch.cuda.get_rng_state()`의 변화(dropout이 쓰는 생성기). `fork_rng(devices=[device])` 구간 전후 두 상태가 복원되는가 |
| J6 | `--gcn_out 128`에서 에폭별 전역 `random` 상태 해시와 에피소드 동일성. 첫 step의 dropout 마스크가 기본과 같은가 |

## 4. 게이트 (하나라도 실패하면 큐 전체 정지)

**옛 코드**: `git worktree add <리포 밖 경로> H1`로 만든다. H1의 코드는 T14 코드와 같다. 결과는 `results/T16a/old/`에 쓴다. 게이트 종료 후 worktree는 `git worktree remove`로 정리한다.

**허용 오차 τ**: 각 지표에서 max(|old2 − old1|, 0.1%p). 손실값은 max(|old2 − old1|, 1e-6).

| 게이트 | 조건 |
|---|---|
| G1 회귀(기본 경로) | Clothing 1-shot, dblp 5-shot, seed 0. 옛 코드 2회(old1, old2), 새 코드(새 플래그 모두 기본) 1회. (a) 에폭별 전역 `random` 상태 해시, 학습·valid·test 에피소드(전 에폭), 고정 에피소드가 old1과 완전 일치. (b) 첫 업데이트 step의 L_N, L_G가 old1과 τ 이내. (c) `best_acc_valid`, `test_acc_at_best_valid`, 고정 test 정확도가 old1과 τ 이내 |
| G2 측정 훅 무영향 | 새 코드 기본 vs `--traj_every 10 --traj_lr 1 --traj_dump_ends 1 --grad_probe_steps 1,50`. G1의 (a)(b)(c)와 같은 기준 |
| G3 측정 구현 일치 | (a) `erasure_metrics`의 kNN 순도가 `analyze_t14.knn_purity`와 \|Δ\| ≤ 1e-6. 대상: T14 `uni` Clothing 1-shot seed 0의 `emb_best.npy`, test 그룹. (b) probe가 `analyze_t06b.fewshot_acc`와 같은 `Random(1000 + seed)` 에피소드에서 완전 일치. (c) G2 run의 step 500에서 기록한 M1·M3가 `traj_z_ends.npz`로 오프라인 재계산한 값과 \|Δ\| ≤ 1e-6 |
| G4 재분할 | Clothing 1-shot, dblp 1-shot, seed 0, `--relabel_base 1` vs 기본. (a) 에폭별 전역 `random` 상태 해시가 일치하고 valid·test·고정 에피소드가 일치. (b) base 클래스 크기 목록, base 노드 집합, valid·test의 `id_by_class`, `labels` 배열이 불변. (c) 기록만: 가짜 클래스마다 가장 많은 참 클래스의 비율, 그 평균 |
| G5 스위치 정확성 | D2·D3 run에서: (a) γ = 1이면 GCN에 대한 총 지도 gradient가 ∂L_N 기여와 같다(상대 차 ≤ 1e-6). 즉 L_G 기여가 0. (b) γ = 0이면 ∂L_N의 GCN 기여가 0이고, EGNN 파라미터의 손실 gradient도 정확히 0. (c) `sup_coef 0`이면 GCN·EGNN의 손실 gradient 노름이 정확히 0 |
| G6 유한 | D·G run의 손실과 지표가 모두 유한 |
| G7 변경 범위 | `git diff --stat H1 <보고서 커밋>`에 나오는 파일이 `model.py`, `argument.py`, `utils.py`, `main.py`, `tools/`, `reports/`, `instructions/` 안에만 있다. `configuration.yaml`, `layers/`, `embedder.py`는 변경하지 않는다 |
| G8 시간(기록만, 게이트 아님) | 같은 설정에서 측정 훅을 켠 run과 끈 run의 run당 시간. 이를 바탕으로 T16b, T17, T18 각각의 예상 소요(분) = run 수 × 평균 시간 ÷ 6 + 분석 예상치. 큐의 SJF 순서에 쓴다 |

## 5. 보고

- 파일: `reports/T16a_impl_facts.md`. 형식은 CLAUDE.md §6.
- 표: 구현 위치, `id_by_class` 사용처, D1–D4의 gradient 표, J1–J6, G1–G8, 예상 소요.
- 커밋 → 태그 `t16a` → push(force 없음) → 부기 커밋.
- 사전 등록 없음.
