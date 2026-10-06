# T07 — (i) E1: 이웃 분포 보존 손실(Preserve)만 추가한 TEG + 공통 에피소드 비교 (브랜치 `preserve_rehearse`)

정본 `CLAUDE.md`. 선행: `dual_view_trust` 끝(closing note 커밋, `t08b` 포함)으로 fast-forward된 `preserve_rehearse`. 이 파일은 `instructions/T07_preserve_E1.md`에 둔다.
**auto mode: CLAUDE.md §2의 멈춤 지점 두 곳을 이 지시서 동안 해제한다.** §6 게이트에서 실패하면 멈춘다. CLAUDE.md §7은 유효.
설계 배경: `docs/preserve_rehearse_methodology_v0.1.md`(있으면 참고). 이번에는 Phase 1-3(보존 손실)만 구현한다. 가짜 클래스(Rehearse)는 구현하지 않는다.

## 0. 목표 (검증할 주장)
1. TEG 학습에 보존 손실 λ·L_pres를 더하면 TEG보다 나은가.
2. 그 결과가 학습 없는 기준선 두 가지 — 확산 관점 단독(D), TEG + 확산 결합(α = 0.5, `t08b`) — 보다 나은가.
3. 보존한 모델에 확산 결합을 얹으면 원본 TEG + 결합보다 나은가.
비교의 공정성을 위해 모든 설정을 **seed마다 같은 고정 에피소드**에서 짝지어 비교한다.

## 1. 코드 변경
### 1.1 플래그 (`argument.py`, 모두 `config2string` 제외 목록에 추가)
- `--pres_lambda` (float, 기본 0.0): 0이면 원본과 같은 동작. 교사 계산·샘플링·손실 계산을 하지 않는다.
- `--pres_tau` (float, 기본 0.1)
- `--pres_m` (int, 기본 1024)
- `--pres_pool` (`nb` | `all`, 기본 `nb`): `nb` = base 클래스(`class_list_train`)에 속하지 않는 노드 전체, `all` = 전체 노드.
- `--fixed_eval` (store_true, 기본 False): §1.3.

### 1.2 보존 손실 (`--pres_lambda > 0`일 때만)
- **교사**: 학습 시작 전 1회, h̄ = 행 정규화 Â²X (Â = `gcn_norm(edge_index, add_self_loops=True)`, `tools/fusion_baseline.py`의 `diffusion_features`와 같은 정의). gradient 없음. 모델 코드가 `tools/`를 import하지 않도록 같은 계산을 모델 쪽에 둔다.
- **학생 표현 z**: 학습 에피소드 forward에서 **EGNN이 받는 노드 좌표 입력과 같은 텐서**(LayerNorm 등 EGNN 직전 처리 이후)를 전체 노드에 대해 쓴다. 어느 텐서인지 보고서에 파일·줄로 적는다.
  - **기존 forward가 이미 계산한 전체 노드 출력을 재사용한다. 추가 forward를 하지 않는다**(dropout 난수 소비가 달라지는 것을 막기 위함). 기존 forward에서 전체 노드 출력을 얻을 수 없으면 구현하지 말고 멈춰서 보고한다.
- **샘플링**: 학습 step마다 풀 U에서 m개 비복원 추출. **전용 생성기**(`torch.Generator` 또는 `np.random.default_rng`, 시드 `7000 + seed`)만 사용. 전역 `random`·`np.random`·`torch` 기본 생성기를 호출하지 않는다.
- **손실**: 표본 B에서 S^T_ij = cos(h̄_i, h̄_j), S^S_ij = cos(z_i, z_j), 대각 제외.
  p^T_i = softmax_{j≠i}(S^T_ij / τ), p^S_i = softmax_{j≠i}(S^S_ij / τ), L_pres = (1/m) Σ_i KL(p^T_i ‖ p^S_i). 교사 쪽 gradient 없음.
- **총 손실**: 원본 학습 step 손실 + λ · L_pres. 원본 optimizer·학습률·에폭 그대로. valid·test 평가 경로 불변.
- `run.json` 에폭 기록에 `pres_loss_mean`(그 에폭 학습 step 평균) 추가. λ = 0이면 기록하지 않는다.

### 1.3 고정 에피소드 평가 (`--fixed_eval`)
- 원본 모델 선택 규칙(`acc_valid == best_acc_valid`일 때 갱신)과 같은 조건에서 모델 `state_dict`를 메모리에 복사해 둔다(난수 없음).
- 학습 종료 후 마지막에 1회: 저장한 체크포인트를 불러 eval 모드·`no_grad`로 고정 에피소드를 평가한다.
  - 고정 test 에피소드 200개: `random.Random(9000 + seed)` 전용 인스턴스로 test 클래스에서 5-way, K = shot, 클래스당 query 5개.
  - 고정 valid 에피소드 100개: `random.Random(9100 + seed)`로 valid 클래스에서 같은 규칙.
  - 에피소드 forward는 원본 평가 경로 함수를 그대로 쓴다. **평가 forward 안에 난수 호출이 있으면 구현하지 말고 멈춰서 보고한다.**
- 저장: `out_dir/fixed_eval_logits.npz`. 키는 `eval_logits.npz`와 같다(`epoch` = −1, `ep_idx`, `mode`, `support`, `query`, `classes`, `query_y`, `logits`). `logits`는 원본이 정확도 argmax에 쓰는 텐서(확률)다.
- 기존 `--dump_logits`, `--dump_emb`는 그대로 함께 쓴다.

### 1.4 `tools/fusion_baseline.py` 확장 (기본 동작 불변)
- 임의의 npz 경로와 "파일 안 전체 에피소드"를 대상으로 같은 계산(관점 D 점수, log 변환, 온도 보정, α 결합)을 하는 함수를 추가한다(예: `evaluate_npz(npz_path, H, alpha=0.5, l_transform="log")` → test 에피소드별 정확도, T_L, s). 온도 보정은 같은 파일의 valid 에피소드로 한다.
- 기존 함수의 시그니처·기본값·결과를 바꾸지 않는다(§6 G4).

## 2. 실행 (126 run)
- 설정 7개: `base`(λ = 0), 그리고 λ ∈ {0.1, 1, 10} × pool ∈ {`nb`, `all`}.
- 각 설정 × 3 데이터셋(`Amazon_clothing`, `dblp`, `Amazon_electronics`) × 5-way {1, 5}-shot × seed {0, 1, 2}.
- 공통 플래그: `--dump_logits --dump_emb --fixed_eval`. τ = 0.1, m = 1024 고정.
- 명령 예: `CUDA_VISIBLE_DEVICES=<UUID> python main.py --dataset <ds> --way 5 --shot <k> --seed <s> --num_seed 1 --device 0 --pres_lambda <λ> --pres_pool <pool> --dump_logits --dump_emb --fixed_eval --out_dir results/T07/<cfg>/<ds>_5w<k>s/seed<s>` (`<cfg>` = `base`, `l0.1_nb`, `l1_nb`, `l10_nb`, `l0.1_all`, `l1_all`, `l10_all`)
- `tools/run_t07.py`(`run_t08.py` 복사, 설정 축 추가, 순서 seed → 설정 → 데이터셋 → shot), GPU 6장 UUID, GPU당 1 run. 실패 시 1회 재시도(`_retry1`).

## 3. 분석 (`tools/analyze_t07.py` → `reports/T07_summary.md`)
라벨은 분석에만 쓴다. 정확도 %, (데이터셋, shot)별 seed 3개 평균 ± sd. 짝지은 차는 **고정 test 에피소드 600개(200 × seed 3)** 평균 ± SE, 양수 비율, seed 부호 일치 수.

### 3.1 정의
- **TEG+P(cfg)**: 설정 cfg 모델의 고정 test 정확도(덤프 확률의 argmax).
- **설정 선택**: (데이터셋, shot)마다 λ > 0인 6개 설정 중 원본 `best_acc_valid`의 seed 평균이 가장 높은 설정 = `sel`. 동률이면 작은 λ, 그다음 `nb`.
- **D**: 고정 test 에피소드에서 관점 D 단독(h̄ 코사인 프로토타입, `fusion_baseline`).
- **F(cfg)**: cfg 모델의 고정 에피소드 덤프 + 관점 D, α = 0.5 결합, 온도는 같은 파일의 고정 valid로 보정(`fusion_baseline` 확장 함수, l_transform = `"log"`).

### 3.2 표
- **표 1 — 전체 격자**: (데이터셋, shot) × 설정 7개: 원본 `best_acc_valid`, 원본 `test_acc_at_best_valid`, 고정 test 정확도, F(cfg) 고정 test 정확도, 마지막 에폭 `pres_loss_mean`.
- **표 2 — 선택 설정 비교**: `sel`의 이름과, TEG+P(sel)에 대한 짝지은 차 vs TEG+P(base), vs D, vs F(base).
- **표 3 — 결합 위에서의 비교**: F(sel) − F(base), F(sel) − TEG+P(sel).
- **표 4 — 표현 진단(코사인 probe, 고정 test 에피소드)**: 설정별 `emb_best.npy` 위 코사인 프로토타입 정확도와, 같은 에피소드의 Â¹X 행 정규화 코사인 정확도(학습 없는 1-hop 기준).

### 3.3 판정 (사전 고정, 값만 계산해 적는다)
- **K1 단독 가치**: 표 2의 TEG+P(sel) − F(base)가 평균 > 0 이고 평균 > 2·SE인 (데이터셋, shot) 수 / 6.
- **K2 보완 가치**: 표 3의 F(sel) − F(base)가 평균 > 0 이고 평균 > 2·SE인 (데이터셋, shot) 수 / 6.
- **K0 TEG 대비**: 표 2의 TEG+P(sel) − TEG+P(base)가 평균 > 0 이고 평균 > 2·SE인 (데이터셋, shot) 수 / 6.

## 4. 사전 등록 (수정 금지, 2026-10-06 작성)
- **P54.** K0 ≥ 4/6.
- **P55.** 표 2: TEG+P(sel) − D 평균 > 0 인 (데이터셋, shot)이 6개 중 3개 이상.
- **P56.** K1 ≤ 2/6 (보존 손실만으로는 학습 없는 결합 기준선을 대부분 넘지 못한다).
- **P57.** K2 ≥ 3/6.

## 5. 시간
목표 완료: 실행 약 15분, 분석 수 분. 실행이 60분을 넘기면 새 run을 시작하지 말고 완료된 run만으로 분석하며 표 머리에 run 수를 적는다.

## 6. 게이트 (실패 시 새 코드 안에서만 수정, 최대 2회 → 그래도 실패면 멈추고 보고)
- **G1 — 기본 경로·난수 격리**:
  - `git diff`의 추가 줄에서 전역 난수 호출 없음(전용 생성기·전용 `random.Random` 인스턴스만).
  - λ = 0이면 교사 계산·샘플링·손실 계산 코드가 실행되지 않음을 보고서에 명시.
  - **같은 (데이터셋, shot, seed)의 7개 설정에서, 덤프된 모든 valid·test 에피소드(전 에폭)의 `support`·`query`·`classes`가 완전히 같다.** (보존 손실이 원본의 전역 난수 순서를 건드리지 않았다는 확인)
  - `base` run의 `test_acc_at_best_valid`를 T08 같은 seed 값과 나란히 기록(원본 비결정성 때문에 일치는 요구하지 않음).
- **G2 — 구현 점검**:
  - 모델 쪽 교사 h̄와 `fusion_baseline.diffusion_features`의 최대 절대 차 ≤ 1e-6 (데이터셋마다).
  - 고정 에피소드(test 200·valid 100)가 같은 (데이터셋, shot, seed)의 7개 설정에서 완전히 같다.
  - **체크포인트 재평가**: run마다 저장한 체크포인트로 `best_epoch_valid`의 test 에피소드 50개(`eval_logits.npz`)를 다시 평가한 정확도가 `episodes.jsonl`과 에피소드 단위로 완전히 같다(분석 단계 또는 학습 종료 시 점검 코드로 수행).
  - λ > 0인 모든 run에서 `pres_loss_mean`이 전 에폭 유한값.
- **G3 — 파일**: 126 run 모두 `run.json`, `eval_logits.npz`, `fixed_eval_logits.npz`, `emb_best.npy` 존재(60분 제한으로 생략된 run 제외 목록 기재).
- **G4 — 함수 회귀**: 확장한 `fusion_baseline.py`의 기존 함수가 `results/T08`에 대해 `t08b`의 표 B(α_val, T_L, s, test acc(α_val))를 18 run 모두 정확히 재현.
- **G5**: 표 1–4와 K0, K1, K2 생성.

## 7. 보고 `reports/T07_preserve_E1.md` (사실만, 해석 금지)
1. 변경 파일·위치·줄 수, z 텐서의 파일·줄, 커밋 해시, 태그 `t07`
2. 게이트 G1–G5
3. 표 1–4 본문과 K0, K1, K2 (`reports/T07_summary.md`에서 복사)
4. P54–P57 "예측 / 관측 / 일치·어긋남"
5. 이상 징후(자율 판단 포함)

## 8. 하지 말 것
CLAUDE.md §7 전부. λ = 0 경로 변경. 가짜 클래스·Rehearse 구현. τ·m 조정, 지정 외 λ·pool 추가. 이어 붙이기 등 다른 기준선 추가. 판정 기준·사전 등록 변경. 결과 해석.
