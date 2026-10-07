# T10r — seed 10개: TEG vs 보존(KL, Â²X) vs 대조 학습(InfoNCE) + λ 격자 끝 확장 (브랜치 `preserve_rehearse`)

정본 `CLAUDE.md`. 선행: `t09`(3d1ce14). 이 파일은 `instructions/T10r_seeds_kl_vs_infonce.md`에 둔다.
**auto mode: CLAUDE.md §2의 멈춤 지점을 이 지시서 동안 해제한다.** CLAUDE.md §7은 유효.
**TN2의 T10 사양을 이 지시서로 대체한다(사용자 결정, 2026-10-07).** TN2의 T11은 연기한다(실행하지 않음). 기존 `results/T10/`의 부분 결과는 지우지 않고 사용하지 않는다.

## 0. 목표
T09에서 대조 학습(InfoNCE)이 보존 손실(KL, Â²X)보다 같거나 나았고(L1 = 0/6), 두 변형 모두 선택 λ가 격자 끝(10)에 몰렸다. 다음 두 가지를 seed 10개로 확정한다.
1. λ 격자를 넓혀도 InfoNCE ≥ KL인가.
2. 두 변형 모두 TEG보다 나은가(원 보고 방식과 고정 에피소드 모두).

## 1. 코드
없음. `model.py` 등 학습 코드 변경 금지(`t09` 코드 그대로). `tools/run_t10r.py`, `tools/analyze_t10r.py`만 새로 쓴다(이미 쓴 `run_t10.py`·`analyze_t10.py`를 고쳐 써도 됨).

## 2. 실행 (420 run)
- 설정 7개: `base`(λ = 0) / `kl_h2` λ ∈ {1, 10, 30} / `infonce` λ ∈ {10, 30, 100}. 풀 `nb`, τ = 0.1, m = 1,024.
- 3 데이터셋(`Amazon_clothing`, `dblp`, `Amazon_electronics`) × 5-way {1, 5}-shot × **seed 0–9**.
- 플래그: `--dump_logits --fixed_eval`. **`--dump_emb`는 쓰지 않는다**(사용자 결정, 디스크 절약).
- 결과 경로 `results/T10r/<cfg>/<ds>_5w<k>s/seed<s>`. 순서 seed → 설정 → 데이터셋 → shot. GPU 6장 UUID, GPU당 1 run. 실패 시 1회 재시도(`_retry1`).
- 실행 묶음(seed 하나 단위)을 시작하기 전마다 리포 디스크 여유 공간을 확인한다. **1.5 GB 미만이면 새 run을 시작하지 말고** 완료된 run으로 분석한다.

## 3. 분석 (`tools/analyze_t10r.py` → `reports/T10r_summary.md`)
- **λ 선택**: (변형, 데이터셋, shot)마다 원본 `best_acc_valid`의 seed 10개 평균 최고. 동률이면 작은 λ. 선택 λ가 격자 끝인지 표에 표시.
- **표 1 — 원 보고 방식**: `test_acc_at_best_valid`의 seed 평균 ± 표준편차, 95% 신뢰구간(t 분포). TEG, KL(sel), InfoNCE(sel). seed 단위 짝지은 차(평균 ± SE, 부호 일치 수/10): KL − TEG, InfoNCE − TEG, KL − InfoNCE.
- **표 2 — 고정 에피소드**(seed당 test 200, 짝지은 차는 2,000개 기준): TEG, D, F(base), KL(sel), InfoNCE(sel), F(KL sel), F(InfoNCE sel)의 정확도. 짝지은 차: KL − TEG, InfoNCE − TEG, KL − InfoNCE, KL − F(base), InfoNCE − F(base), F(KL) − F(base), F(InfoNCE) − F(base).
- **표 3 — λ 곡선**: 7개 설정 전체의 원본 `best_acc_valid`, `test_acc_at_best_valid`, 고정 test 정확도(seed 평균 ± sd).
- **판정**(평균 > 0 이고 > 2·SE인 (데이터셋, shot) 수 / 6, 고정 에피소드 기준):
  - N0 = KL − TEG, N1 = InfoNCE − TEG, N2 = KL − InfoNCE, N3 = InfoNCE − KL.

## 4. 사전 등록 (수정 금지, 2026-10-07 작성)
TN2의 P62–P64(구 T10)는 이 지시서로 대체되어 판정하지 않는다.
- **P67.** N0 = 6/6. **P68.** N1 = 6/6. **P69.** N2 ≤ 1/6. **P70.** N3 ≥ 2/6.

## 5. 게이트 (실패 시 새 코드 안에서만 수정, 최대 2회 → 그래도 실패면 멈추고 보고)
- Ga: 같은 (데이터셋, shot, seed)의 7개 설정에서 덤프된 valid·test 에피소드(전 에폭)와 고정 에피소드가 완전히 같다.
- Gb: `ckpt_epoch` 기준 체크포인트 재평가 50/50(전 run).
- Gc: λ > 0 run의 `pres_loss_mean` 전 에폭 유한.
- Gd: seed 0–2의 `base`·`kl_h2` λ ∈ {1, 10}·`infonce` λ = 10 결과를 T09 같은 설정과 나란히 기록(일치 요구 없음).
- Ge: 표 1–3, N0–N3 생성.

## 6. 시간
- **16:00 KST 이후 새 run 시작 금지.** 그 시점에 완료된 run만으로 분석하고 표 머리에 run 수를 적는다.
- **17:00 KST까지** 보고서·커밋·태그 `t10r`, 그리고 `reports/TN2_overnight_summary.md` 작성.

## 7. 보고 `reports/T10r_seeds_kl_vs_infonce.md` (사실만, 해석 금지)
1. 변경 파일, 커밋 해시, 태그 `t10r`
2. 게이트 Ga–Ge
3. 표 1–3, N0–N3
4. P67–P70 "예측 / 관측 / 일치·어긋남"
5. 이상 징후(선택 λ 격자 끝 여부 포함)

## 8. 하지 말 것
CLAUDE.md §7 전부. 학습 코드 변경. `--dump_emb` 사용. T11 실행. 지정 외 설정·λ 추가. 판정·사전 등록 변경. 결과 해석.
