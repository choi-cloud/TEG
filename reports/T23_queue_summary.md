# T23 큐 요약 — 다중 해상도 클러스터의 레이블 일치 (auto mode)

2026-10-15 · 브랜치 `hier_purity` · H0 `14d9dfe` · T0 = 04:55:38 · 마감 T0 + 60분 = 05:55:38 · 큐 종료 약 05:17(T0 + 22분)

## 지시서별

| 지시서 | 태그 | 커밋 | 실행 시간 | 게이트 | 판정값 | 사전 등록 |
|---|---|---|---|---|---|---|
| T23a | `t23a` | `9d8cf08` | 착수부터 9분(재실행 클러스터링 3.3분) | Ga 36/36, Gb 36/36, Gc ARI 1.0, Gd 일치, Ge 54/54, Gf(기록) M·H 모두 k와 같음 | — | — |
| T23b | `t23b` | `cb3fed1` | 2.9분(+ gate·보고) | Ha, Hb \|Δ\| 0, Hc (a) \|Δ\| 0 (b) 불일치 0/2000, Hd 0/1000, He 유한 | R1 6, R2 2, R3 4, R4 6, R5 6, R6 2, R7 0 (/6); ℓ* {1: 40, 5: 104} | P116 일치, P117 어긋남, P118 일치, P119 일치, P120 일치 |
| T23c | `t23c` | `dbd8eec` | 3.4분(+ gate·보고) | Ia, Ib \|Δ\| 0 (36), Ic 0, Id 충족, Ie 유한·미수렴 0 | S1 6, S2 6, S3 6, S4 6, S5 6 (/6); S6 X 6, rand1 6, SHUF 0; β* {0: 17, 0.05: 5, 0.1: 12, 0.2: 28, 0.5: 82} | P121 일치, P122 일치, P123 어긋남, P124 일치, P125 일치 |

- 변경 범위(Gg·Hf·If): `git diff --stat 14d9dfe dbd8eec` = `tools/t23.py`, `reports/T23a_hier_build.md`, `reports/T23a_summary.md`, `reports/T23b_summary.md`, `reports/T23b_support_purity.md`, `reports/T23c_hier_signal.md`, `reports/T23c_summary.md`(7개, 2157줄 추가, 삭제 0). 세 지시서 모두 범위 안.
- 시간 때문에 뺀 항목: 없음.
- `results/T23/` 78M. 지우지 않음. 리포 디스크 여유 8.7G.

## push (force 없음)

| 대상 | 결과 |
|---|---|
| `hier_purity` | `* [new branch]` (9d8cf08) → `9d8cf08..cb3fed1` → `cb3fed1..dbd8eec` |
| `t23a`, `t23b`, `t23c` | 각각 `* [new tag]`. `git ls-remote`: t23a = `9d8cf08`, t23b = `cb3fed1`, t23c = `dbd8eec`, `hier_purity` = `dbd8eec` |

## 큐 전체 이상 징후

1. T23a 첫 실행은 k-means++ 표본 추출(CPU)이 느려 중단하고, 표본 추출 방식을 바꿔 처음부터 다시 돌렸다(T23a 보고서 이상 징후 1). 첫 실행 결과는 쓰지 않았다.
2. `tools/t23.py`의 T23b·T23c 하위 명령은 T23a 보고서 커밋(`9d8cf08`)에 이미 들어 있었다(T23b 커밋에서 `c_report` 추가와 `c_gates` 쌍 목록 수정).
3. 각 지시서는 보고서 커밋 1개로 끝났고, Gg·Hf·If 값은 이 요약에 적었다(지시서별 부기 커밋 없음).
