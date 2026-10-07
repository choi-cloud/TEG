# T12 — 학습 정확도 vs 안 본 클래스 구분력 (분석 전용) + H1 운영 규칙 (브랜치 `preserve_rehearse`)

정본 `CLAUDE.md`. 선행: `t10r`(6a16717). 이 파일은 `instructions/T12_train_vs_unseen.md`에 둔다.
**auto mode: CLAUDE.md §2의 멈춤 지점을 이 지시서 동안 해제한다.** §4의 게이트에서 실패하면 멈춘다.
**학습 run 없음. 학습 코드 변경 없음.** 기존 결과 파일만 읽는다.

---

## H1 — 운영 규칙 변경 (사용자 결정, 별도 커밋, 태그 없음)

1. CLAUDE.md §7의 결과·로그 삭제 금지 규칙을 아래로 바꾼다.
   - 코드, `reports/`, `instructions/`, `docs/`, `tools/`는 지우지 않는다.
   - `results/` 아래 파일은 **해당 실험의 보고서가 커밋·push되고 태그가 달린 뒤에만** 지울 수 있다. `emb_best.npy`부터 지우고, 지운 경로와 확보 용량을 그 시점 보고서에 기록한다.
   - 진행 중인 실험의 결과는 지우지 않는다.
   - 실행 묶음 시작 전 리포 디스크 여유가 1.5 GB 미만이면, 위 조건을 만족하는 가장 오래된 실험의 `emb_best.npy`부터 지워 3 GB 이상을 확보한 뒤 진행한다. 지울 대상이 없으면 멈추고 보고한다.
2. `.gitignore`에 `results/`가 없으면 추가한다.
3. 커밋 메시지: `docs: CLAUDE.md deletion rule (user decision), gitignore results/`.
4. 원격(`origin`)에 브랜치 `preserve_rehearse`, `dual_view_trust`, `novel_like_class`와 모든 태그를 push한다. **force push 금지.** push가 거부되면 멈추지 말고 거부 내용을 보고서에 기록한 뒤 T12로 진행한다.
5. 커밋하지 않은 T11 파일(`tools/run_t11.py`, `tools/analyze_t11.py`)은 그대로 둔다.

---

## T12 — 보조 손실이 바꾼 것은 "base 정확도"인가 "안 본 클래스 구분력"인가

### 0. 목표
보조 손실(보존 = KL·Â²X, 대조 = InfoNCE)을 더했을 때, (a) base 학습 정확도는 얼마나 변했고 (b) 안 본 클래스(valid·test)의 구분력은 얼마나 변했는지를 같은 표에 나란히 남긴다.

### 1. 이름
- `원본` = `base`(λ = 0). `보존` = `kl_h2`(선택 λ). `대조` = `infonce`(선택 λ).
- 선택 λ는 각 실험 요약의 선택값을 그대로 쓴다. 표 1은 `reports/T10r_summary.md` 표 1의 선택 λ, 표 2는 `reports/T09_summary.md` 표 2의 선택 λ(`kl_h2`, `infonce`).

### 2. 분석 (`tools/analyze_t12.py` → `reports/T12_summary.md`)

**표 1 — 정확도 세 가지 (T10r, seed 0–9)**
- run마다 `ckpt_epoch`에서의 세 값을 쓴다.
  - **train**: 원본 코드가 기록하는 학습(base) 에피소드 정확도. 어느 값인지(파일·줄, train/eval 모드 여부)를 보고서에 명시한다. 원본이 해당 에폭의 train 정확도를 기록하지 않으면, 기록된 값 중 가장 가까운 정의를 쓰고 그 사실을 적는다.
  - **valid**: `best_acc_valid`.
  - **test**: `test_acc_at_best_valid`.
- 열: (데이터셋, shot) × {원본, 보존, 대조}의 train, valid, test(seed 평균 ± sd). 그리고 train − test 격차.
- 짝지은 차(seed 단위, 평균 ± SE, 부호 일치 수/10): 보존 − 원본, 대조 − 원본을 train, valid, test 각각에 대해.

**표 2 — 클래스 그룹별 표현 구분력 (T09 `emb_best.npy`, seed 0–2)**
- T06b 표 B1과 같은 probe: 5-way, K = shot, query 5, 에피소드 500개, support 평균 프로토타입, 코사인 최근접. 에피소드는 `random.Random(1000 + seed)`로 그룹마다 한 번 뽑아 모든 공간에 공유.
- 그룹: base, valid, test. 공간: 원본·보존·대조의 `emb_best.npy`, 그리고 학습 없는 Â¹X·Â²X(행 정규화).
- 열: 공간별 정확도(seed 평균 ± sd), 그리고 Δ = 보조 손실 모델 − 원본을 그룹별로(Δ_base, Δ_valid, Δ_test).

**판정 (값만 계산)**
- R_tr: 표 1에서 |train 짝지은 차| < 1.0%p 인 (데이터셋, shot) 수 / 6 — 보존, 대조 각각.
- R_te: 표 1에서 test 짝지은 차가 평균 > 0 이고 > 2·SE 인 (데이터셋, shot) 수 / 6 — 보존, 대조 각각.
- R_gap: 표 2에서 Δ_test > Δ_base 인 (데이터셋, shot) 수 / 6 — 보존, 대조 각각.

### 3. 사전 등록 (수정 금지, 2026-10-07 작성)
- **P71.** R_te = 6/6 (보존, 대조 모두).
- **P72.** R_tr ≥ 4/6 (보존, 대조 모두) — 보조 손실은 base 학습 정확도를 거의 바꾸지 않는다.
- **P73.** R_gap ≥ 5/6 (보존, 대조 모두) — 보조 손실의 이득은 base보다 안 본 클래스에서 크다.

### 4. 게이트 (실패 시 새 코드 안에서만 수정, 최대 2회 → 그래도 실패면 멈추고 보고)
- G1: `git diff --stat`(기준 H1 커밋)이 `tools/`, `reports/`, `instructions/`만 포함.
- G2: 표 1의 test 열이 `reports/T10r_summary.md` 표 1의 해당 값과 일치(선택 λ 기준).
- G3: 표 2의 원본·Â²X 열의 test 그룹 값이 T06b 표 B1의 `emb1`·`diff2` test 값과 같은 방식으로 계산됐음을 확인(같은 probe 함수 사용 명시. 값 일치는 요구하지 않음 — 원본 run이 다름).
- G4: 표 1, 표 2, 판정 생성.

### 5. 보고 `reports/T12_train_vs_unseen.md` (사실만, 해석 금지)
1. H1 변경 내용, push 결과(원격 브랜치·태그 목록 또는 거부 내용), 커밋 해시
2. T12 변경 파일, train 정확도의 정의(파일·줄), 커밋 해시, 태그 `t12`
3. 게이트 G1–G4
4. 표 1, 표 2, 판정
5. P71–P73 "예측 / 관측 / 일치·어긋남"
6. 이상 징후

### 6. 하지 말 것
CLAUDE.md §7(H1로 바뀐 규칙 포함) 전부. 학습 run 실행. 학습 코드 변경. force push. T11 파일 커밋. 지정 외 표·지표 추가. 판정·사전 등록 변경. 결과 해석.
