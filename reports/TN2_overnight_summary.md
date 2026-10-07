# TN2 종합 — 야간 연쇄 실행 H0 → T09 → T10(→ T10r) → T11(연기) (브랜치 `preserve_rehearse`, 2026-10-06 23:34 → 10-07 14:14 KST)

사실만 적는다. 판정·사전 등록은 각 지시서 원문 그대로다.

## 1. 단계별 상태
| 단계 | 시작 → 종료 (KST) | run 수 (완료/계획) | 게이트 | 태그 | 커밋 |
|---|---|---|---|---|---|
| H0 (CLAUDE.md 메모 2줄) | 10-06 23:35 | — | — | 없음(지시서 규정) | `945402b` |
| T09 보존 손실 성분 분해 + 자기지도 기준선 | 10-06 23:38 → 10-07 00:08 | 288/288 | Ga–Gd·교사 일치 통과, 재시도 0 | `t09` | `3d1ce14` |
| T10 seed 10개 확대(TN2 원 사양) | 10-07 00:07 → 00:13 (중단) | 기록 80개(성공 70, 실패 10) / 270 | 미판정 — 디스크 부족으로 중단(3절), T10r로 대체 | 없음 | 커밋 없음(스크립트는 `6a16717`에 기록용으로 포함) |
| T10r seed 10개: TEG vs KL vs InfoNCE(T10 대체, 사용자 결정) | 10-07 13:30 → 14:14 | 420/420 | Ga–Ge 통과, 재시도 0 | `t10r` | `6a16717` (지시서 `41c901a`) |
| T11 Rehearse 2×2 | — | 0/72 | — | 없음 | 연기(사용자 결정, 미실행) |

- 지시서 커밋: `d371be8`(TN2 지시서, `docs/T07_preserve_E1_explainer.md`), `41c901a`(T10r 지시서).
- 리포 디스크 `df -h /` (T10r 완료 시점, 14:11 KST): `/dev/mapper/ubuntu--vg-ubuntu--lv  98G  85G  8.4G  92% /`

## 2. 판정과 사전 등록
### T09 (`reports/T09_preserve_ablation.md`)
| 판정 | 비교(선택 λ 모델끼리, 고정 에피소드) | 값 |
|---|---|---|
| L1 | kl_h2 − infonce | 0/6 |
| L2 | kl_h2 − kl_h0 | 4/6 |
| L3 | kl_h2 − mse_h2 | 3/6 |
| L4 | infonce − base | 6/6 |

### T10r (`reports/T10r_seeds_kl_vs_infonce.md`)
| 판정 | 비교(선택 λ 모델, 고정 에피소드, seed 10개) | 값 |
|---|---|---|
| N0 | KL − TEG | 6/6 |
| N1 | InfoNCE − TEG | 6/6 |
| N2 | KL − InfoNCE | 1/6 |
| N3 | InfoNCE − KL | 3/6 |

### 사전 등록 대조
| 예측 | 내용 | 관측 | 일치·어긋남 |
|---|---|---|---|
| P58 | L1 ≥ 4/6 | 0/6 | 어긋남 |
| P59 | L2 ≥ 3/6 | 4/6 | 일치 |
| P60 | L3 ≥ 2/6 | 3/6 | 일치 |
| P61 | L4 ≥ 3/6 | 6/6 | 일치 |
| P62–P64 | (구 T10) | — | 판정 안 함(T10r 지시서로 대체) |
| P65–P66 | (T11) | — | 판정 안 함(T11 연기) |
| P67 | N0 = 6/6 | 6/6 | 일치 |
| P68 | N1 = 6/6 | 6/6 | 일치 |
| P69 | N2 ≤ 1/6 | 1/6 | 일치 |
| P70 | N3 ≥ 2/6 | 3/6 | 일치 |

## 3. 디스크 사고 경과
| 시각 (KST) | 사실 |
|---|---|
| 10-07 00:07 | T10(`tools/run_t10.py`, 270 run 계획) 시작 |
| 00:13:10 | 첫 run 실패(`OSError: [Errno 28] No space left on device`, `results/T10/...` 디렉터리 생성 단계) |
| 00:13:30 | T10 드라이버가 같은 오류로 중단(총 364 s). 이후 Bash 출력이 /tmp ENOSPC로 전달되지 않아 작업 중단·보고 |
| 13:29 이전 | 사용자 조치: `dataset/`에서 Amazon_clothing·Amazon_electronics·dblp를 제외한 CiteSeer(49M)·cora(1.4G)·CS(982M)·ogbn_arxiv(183M) 삭제, 약 2.6G 확보(삭제 직후 `df -h /`: 89G 사용 / 3.9G 여유) |
| 13:29 | 사용자 지시로 `git restore dataset/ogbn_arxiv/mapping/README.md`(추적 파일만 복원). 남은 프로세스 없음, GPU 0·1·2·4·5·6 비어 있음 확인 |
| 13:29 | scratchpad 점검용 run(`t07_smoke` 15M, `t09_smoke` 29M) 삭제. `df -h /`: 84G 사용 / 9.1G 여유(/tmp 같은 파일시스템) |
| 13:30 | T10r에서 `--dump_emb` 미사용(사용자 결정, 디스크 절약), run 시작 전 1.5 GB 여유 확인 |

- **영향 범위:** T10 기록 80개 중 실패 10개.
  - 실패 run(설정, 데이터셋, shot, seed): base Amazon_clothing 1·5 s2, base dblp 1·5 s2, base Amazon_electronics 1·5 s2, base corafull 1 s2, base ogbn-arxiv 1 s2, kl_h2_l10 ogbn-arxiv 1·5 s1
  - 드라이버 로그의 `ok=True` 줄은 75개로, `_runs.jsonl`의 성공 기록(70개)과 다르다.
  - `results/T10/`(971M)은 보존하고 사용하지 않는다.
- **13:29 기준 `du -sh results/*`:** T09 2.9G, T07 1.3G, T10 971M, T06b 335M, T05 184M, T06 166M, T04 15M, T08 14M, T03b 1.1M, T03a 708K, T02 180K, T01 152K, T08b 8.0K.
- **삭제한 데이터셋의 영향:** corafull·coauthorCS·ogbn-arxiv의 원본 파일이 없어져 T10 사양의 추가 데이터셋 run은 실행할 수 없다.

## 4. T10 대체, T11 연기
- **T10 → T10r(사용자 결정, 2026-10-07):** 설정 7개(`base`, `kl_h2` λ ∈ {1, 10, 30}, `infonce` λ ∈ {10, 30, 100}), 3 데이터셋 × {1, 5}-shot × seed 0–9, `--dump_emb` 없음. TN2의 P62–P64는 판정하지 않는다.
- **T11 연기(사용자 결정):** 실행하지 않았다.
  - `tools/run_t11.py`, `tools/analyze_t11.py`와 scratchpad의 `apply_t11.py`(모델 패치)는 커밋하지 않고 그대로 두었다.
  - `model.py`에는 T11 코드가 적용되지 않았다.

## 5. 단계별 이상 징후 (상세는 각 보고서 "이상 징후" 절)
- **T09** — `reports/T09_preserve_ablation.md` §6
  - 선택 λ 격자 끝 집중: `infonce`·`mse_h2` 6셀 모두 λ = 10.
  - InfoNCE 손실 크기(약 6).
  - GCNConv 캐시 우회 방식.
  - 투영 헤드 초기화 방식.
  - 에폭 0 손실 기록.
  - `base` 값이 T07과 일부 다름.
- **T10r** — `reports/T10r_seeds_kl_vs_infonce.md` §6
  - 선택 λ 격자 끝: KL 4/6(모두 λ = 1), InfoNCE 3/6.
  - KL 선택 λ가 T09와 다른 셀 있음.
  - seed 0–2 T09 대비 다른 값 6/72.

## 6. 실패·생략 항목
1. T10(원 사양): 디스크 부족으로 중단, 미판정, T10r로 대체.
2. T10 추가 데이터셋(corafull, coauthorCS, ogbn-arxiv): 데이터 삭제로 T10r에서 실행하지 않음(T10r 사양에도 없음).
3. T11: 연기, 미실행. P65–P66 미판정.
