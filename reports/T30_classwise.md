# T30 보고 — 클래스별 분해 (학습 없음)

지시서 `instructions/T30_classwise.md` · 브랜치 `classwise_decomp` · 메모 삭제 커밋 `c45163e`(`trainfree_baselines`) · H0 `9fb9611` · 표 전체는 `reports/T30_summary.md`, 클래스별 전체 표는 `reports/T30_per_class.md`

## 1. 변경 파일

| 파일 | 내용 |
|---|---|
| `tools/analyze_t30.py` (새 파일) | `labels_path`, `load_eps`, `per_class_correct`, `class_means`, `class_overlap`(e_c·r_c, seed 하나), `run(ds)`, `t28_table1`, `report()`. 재사용: `fusion_baseline.diffusion_features`, `diffusion_features_hops`, `view_d_scores`, `cosine_probe_acc`; `analyze_t22.paired`; `analyze_t28.npz_path`, `call_cfg`, `LAM`, `CELLS` |
| `reports/T30_classwise.md`, `reports/T30_summary.md`, `reports/T30_per_class.md` | 보고서 |

- 학습 코드와 기존 `tools/*.py`는 고치지 않았다. 학습과 새 난수는 없다. GPU는 쓰지 않았다.

## 2. 실행 명령과 출력

```
/home/h2tae/anaconda3/envs/python310/bin/python tools/analyze_t30.py run Amazon_clothing      # 3개 데이터셋 동시 실행, CPU
/home/h2tae/anaconda3/envs/python310/bin/python tools/analyze_t30.py run dblp
/home/h2tae/anaconda3/envs/python310/bin/python tools/analyze_t30.py run Amazon_electronics
/home/h2tae/anaconda3/envs/python310/bin/python tools/analyze_t30.py report
```

- 실행 시간(`/usr/bin/time`): 26.7 s(Clothing), 30.5 s(dblp), 32.5 s(Electronics). 로그 `results/T30/run_<ds>_retry1.log`.
- `results/T30/` 디스크 사용량 276K.

T1 — 클래스 수와 짝지은 차의 부호 (클래스별, 분명 = CLAUDE.md §5 기준)

| 데이터셋 | shot | test 클래스 수 | n_ep 최소 | n_ep 중앙 | TEG − D 분명한 음수 | 분명한 양수 | 그 외 | C_all − TEG 분명한 양수 |
|---|---|---|---|---|---|---|---|---|
| Amazon_clothing | 1 | 20 | 452 | 498 | 11 | 2 | 7 | 19 |
| Amazon_clothing | 5 | 20 | 462 | 507 | 18 | 1 | 1 | 14 |
| dblp | 1 | 30 | 312 | 334 | 5 | 8 | 17 | 24 |
| dblp | 5 | 30 | 299 | 336 | 18 | 5 | 7 | 16 |
| Amazon_electronics | 1 | 40 | 220 | 250.5 | 18 | 10 | 12 | 39 |
| Amazon_electronics | 5 | 40 | 219 | 250 | 20 | 7 | 13 | 32 |

T2 — Spearman ρ (p), 단위 = test 클래스

| 데이터셋 | shot | ρ(e_c, TEG − D) | ρ(TEG − D, C_all − TEG) | ρ(TEG − D, C_all − D) | ρ(r_c, TEG − D) | ρ(e_c, acc_D) | ρ(r_c, acc_D) |
|---|---|---|---|---|---|---|---|
| Amazon_clothing | 1 | +0.654 (0.00176) | -0.376 (0.102) | +0.696 (0.00065) | +0.024 (0.92) | -0.398 (0.0818) | -0.107 (0.654) |
| Amazon_clothing | 5 | +0.377 (0.101) | -0.086 (0.719) | +0.695 (0.000675) | -0.272 (0.246) | -0.299 (0.2) | -0.215 (0.363) |
| dblp | 1 | +0.236 (0.208) | -0.020 (0.917) | +0.881 (1.3e-10) | +0.081 (0.671) | -0.270 (0.149) | -0.675 (4.25e-05) |
| dblp | 5 | +0.333 (0.0725) | -0.488 (0.00624) | +0.635 (0.000163) | -0.210 (0.266) | -0.273 (0.144) | -0.764 (8.9e-07) |
| Amazon_electronics | 1 | +0.460 (0.00279) | -0.570 (0.000123) | +0.794 (9.76e-10) | +0.222 (0.168) | -0.043 (0.793) | -0.169 (0.298) |
| Amazon_electronics | 5 | +0.445 (0.00399) | -0.346 (0.0288) | +0.729 (9.76e-08) | +0.127 (0.433) | -0.026 (0.871) | -0.210 (0.194) |

T3(3분위 교차표), T4(TEG − D 최저·최고 클래스), T5(기준값)는 `reports/T30_summary.md`에 있다.

판정값 (계산값만)

| 판정 | 정의 | 값 |
|---|---|---|
| Q1 | TEG − D 분명한 음수 클래스 수 > 분명한 양수 클래스 수 | 5/6 (dblp 1-shot 제외) |
| Q2 | ρ(e_c, TEG − D) > 0 | 6/6 |
| Q3 | ρ(TEG − D, C_all − TEG) < 0 | 6/6 |
| Q4 (기록) | ρ(TEG − D, C_all − D) < 0; ρ(r_c, TEG − D) > 0; T3 (a), (b) | 0/6; 4/6 (Amazon_clothing 5, dblp 5 제외); Clothing 1 (1, 5), Clothing 5 (1, 6), dblp 1 (3, 4), dblp 5 (4, 4), Electronics 1 (5, 8), Electronics 5 (6, 8) |

사전 등록 계산값: P152 Q1 = 5/6, 아닌 칸 dblp 1-shot · P153 Q2 = 6/6 · P154 Q3 = 6/6.

## 3. 완료 조건

| 조건 | 결과 | 충족 |
|---|---|---|
| Ga 선행 | `base` npz 60/60, 선택 λ `infonce_all` npz 60/60, `results/T29/splits.json` 존재(Gs 통과), `labels.npy` 3/3 | 예 |
| Gb 재현 (24값, \|Δ\| ≤ 0.005 %p) | 최대 \|Δ\| 0.0040 (T28 표 T1은 소수 둘째 자리). 기록: `results/T28/analysis_<ds>.json` 에피소드 값과의 \|Δ\| 최대 0 | 예 |
| Gc 표본 (n_ep ≥ 100) | n_ep 최소 219, 미달 0개 | 예 |
| Gd 변경 범위 | `git diff --stat 9fb9611 <보고서 커밋>` = `tools/analyze_t30.py`, `reports/T30_*` | 예 (부기에 확인값 기록) |
| Ge 유한·rank | 모든 지표 유한. base 부분공간 rank: Clothing 39, dblp 79, Electronics 90 (seed 0–9 모두 n_base − 1) | 예 |
| 산출 `classwise_<ds>.json`, 표 T1–T5, 판정 Q1–Q4, P152–P154 계산값 | `results/T30/`, `reports/T30_summary.md`, `reports/T30_per_class.md` | 예 |

## 4. 이상 징후

1. **첫 실행 실패:** 첫 실행은 `python`이 conda base(`/home/h2tae/anaconda3/bin/python`)로 잡혀 `ModuleNotFoundError: No module named 'torch_geometric'`로 0.75 s 만에 끝났다(로그 `results/T30/run_<ds>.log`, 지우지 않음). `python310` 환경 인터프리터로 다시 실행했고 로그는 `_retry1`이다.
2. **scipy 버전:** 멈춤 1 계획에 적은 1.11.4는 conda base의 버전이다. 실행 환경 `python310`의 scipy는 1.15.3이다.
3. **Gb 비교 방식:** T28 표 T1이 소수 둘째 자리라서 비교 대상은 반올림 값이다. D·X0 예측은 `view_d_scores` argmax이고, 에피소드 정확도가 `cosine_probe_acc`와 같은지 assert로 확인했다(통과).
4. **에피소드 클래스 순서:** 모든 고정 test 에피소드에서 `labels[support]`, `labels[query]`가 클래스 블록 순서대로 `classes`와 같은지, `base`와 `infonce_all`의 `query_y`·`classes`가 같은지 assert로 확인했다(통과).
5. **승인된 판단 1–4:** 3분위는 `np.quantile(·, 1/3)` 이하(경계 포함). Spearman은 `scipy.stats.spearmanr`(양측 p)로 그 칸의 test 클래스 전체에 계산했다. T1의 분명한 음수·양수는 클래스별 `paired` 결과로 셌다. `classwise_decomp`는 보고서 커밋과 함께 push했다.
6. **e_c·r_c와 shot:** e_c·r_c는 분할과 특징에만 의존하므로 같은 데이터셋의 1·5-shot에서 같다.
7. **착수 전 작업 트리:** §1-1 확인 때 `docs/dual_view_closing_note.md`, `docs/dual_view_rehearsed_trust_methodology_v0.1.md`의 삭제가 stage되어 있었다. 사용자가 파일을 복구했고(blob이 HEAD와 같음), 사용자 지시로 `git restore --staged`를 실행했다. 두 파일은 수정·삭제·커밋하지 않았다.

## 부기 (2026-10-21, push)

- 보고서 커밋 `4a99367`, 태그 `t30`(→ `4a99367`), `origin/classwise_decomp`(새 브랜치)와 태그 push 완료. force 없음.
- Gd 확인값: `git diff --stat 9fb9611 4a99367` = `reports/T30_classwise.md`(80+), `reports/T30_per_class.md`(209+), `reports/T30_summary.md`(174+), `tools/analyze_t30.py`(381+), 4 files.
