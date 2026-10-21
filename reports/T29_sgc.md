# T29 보고 — 발표된 SGC 기준선의 세 변형 (선형 head 학습만)

지시서 `instructions/T28_T29_QUEUE.md` (T29 단위) · 브랜치 `trainfree_baselines` · 기준 `t28` (`4100884`) · 표 전체는 `reports/T29_summary.md`

## 1. 변경 파일

| 파일 | 내용 |
|---|---|
| `tools/run_t29.py` (새 파일) | 함수: `restore_splits`(§2.1, 게이트 Gs), `sgc_features`, `init_linear`·`fit_linear`(pilot:82–97 이식), `pretrain`(pilot:120–134 이식, 오프셋 18600/18700), `load_eps`, `one(ds, seed)`. 드라이버: `poll_mem`, `run_once`, `worker`, `main`(`tools/run_t27.py` 형식) |
| `tools/analyze_t29.py` (새 파일) | `load`, `report`. 재사용: `analyze_t22.fixed_teg`, `mean_sd`, `paired`; `analyze_t28.npz_path`, `call_cfg`, `t22_table2` |
| `reports/T29_sgc.md`, `reports/T29_summary.md` | 보고서 |

- 학습 코드와 기존 `tools/*.py`는 고치지 않았다.
- 새 난수 오프셋은 18600, 18700, 18800, 18900 네 개이며 CLAUDE.md §8에 등록된 값이다.
- 전역 `random`·`torch` 난수 상태는 쓰지 않았다.

## 2. 실행 명령과 출력

```
python tools/run_t29.py splits     # results/T29/splits.json, Gs
python tools/run_t29.py phase1     # Amazon_clothing seed 0 단독, 물리 GPU 0 (Hc)
python tools/run_t29.py phase2     # 나머지 29 프로세스, 물리 GPU 0, 1, 2, 4, 5, 6 (동시 ≤ 6)
python tools/analyze_t29.py
```

- **Hc 단독 run (Amazon_clothing seed 0):** 프로세스 전체 252.2 s, GPU 최대 1356 MiB.
  - 구간별: 특징 10.1 s, 사전학습 67.4 s, 1-shot 평가 53 s, 5-shot 평가 118 s.
- **phase2:** 29개 프로세스 전체 5070.6 s. 프로세스당 695.8–1308.2 s이고 재시도는 0회다.
  - GPU 최대 메모리: Clothing 1356 MiB, dblp 1614 MiB, Electronics 1896 MiB.
- **디스크:** `results/T29/` 15M.

T1 — 고정 test 정확도 (seed 0–9, 각 200 에피소드, seed 평균 ± sd)

| 데이터셋 | shot | D | SGC-P | SGC-S | SGC-B | LR | TEG | C_all |
|---|---|---|---|---|---|---|---|---|
| Amazon_clothing | 1 | 85.22 ± 0.49 | 85.25 ± 0.59 | 65.14 ± 0.85 | 61.25 ± 2.77 | 86.39 ± 0.48 | 81.78 ± 1.34 | 87.79 ± 0.48 |
| Amazon_clothing | 5 | 93.13 ± 0.50 | 92.93 ± 0.37 | 89.28 ± 0.59 | 85.87 ± 1.93 | 93.32 ± 0.55 | 90.37 ± 0.99 | 92.78 ± 0.46 |
| dblp | 1 | 72.43 ± 1.23 | 72.49 ± 1.30 | 67.01 ± 1.09 | 60.93 ± 2.04 | 74.26 ± 1.10 | 74.44 ± 1.20 | 78.19 ± 1.04 |
| dblp | 5 | 85.60 ± 0.75 | 85.01 ± 0.60 | 85.09 ± 0.74 | 82.71 ± 1.40 | 85.75 ± 0.63 | 84.23 ± 1.02 | 86.24 ± 0.58 |
| Amazon_electronics | 1 | 75.30 ± 0.88 | 77.16 ± 0.84 | 55.99 ± 0.98 | 53.51 ± 1.37 | 76.19 ± 0.89 | 73.59 ± 1.20 | 81.82 ± 1.81 |
| Amazon_electronics | 5 | 88.50 ± 0.74 | 88.24 ± 0.53 | 84.68 ± 0.84 | 81.62 ± 1.14 | 89.17 ± 0.85 | 86.92 ± 1.17 | 90.84 ± 0.81 |

짝지은 차 8종(T2), 사전학습 기록(T3), 5-shot 비교 열(T4)은 `reports/T29_summary.md`에 있다.

판정값 (계산값만)

| 판정 | 정의 | 값 |
|---|---|---|
| N1 | SGC-S − SGC-B 분명한 양수 | 6/6 |
| N2 | acc(SGC-S) < acc(SGC-P) | 5/6 (dblp 5-shot 제외) |
| N3 | acc(SGC-B) < acc(D) | 6/6 |
| N4 | acc(C_all) > acc(SGC-P) | 5/6 (Amazon_clothing 5-shot 제외) |
| N5 | acc(TEG) > acc(SGC-B) | 6/6 |
| N6 (기록) | LR − D 분명한 양수; \|D − SGC-P\| 최대 | 5/6 (dblp 5-shot 제외); 1.87 %p (Amazon_electronics 1-shot) |

사전 등록 계산값: P147 N1 = 6/6, P148 N2 = 5/6, P149 N3 = 6/6, P150 N4 = 5/6, P151 N5 = 6/6.

## 3. 완료 조건

| 조건 | 결과 | 충족 |
|---|---|---|
| Gs 분할 복원 (1)–(5) | 실패 0건. seed 0–2 복원 분할 = `split.json` (18개), valid 수 17 / 27 / 36, 서로소, 1·5-shot 합집합 크기 = `valid_num_dic`, labels·test 일치 | 예 |
| Hb 재현 (D·TEG·C_all 18값, \|Δ\| ≤ 0.005 %p) | 최대 \|Δ\| 0.0040 | 예 |
| Hc 시간 점검 (프로세스당 60분) | 단독 252.2 s, phase2 최대 1308.2 s | 예 |
| Hd 사전학습 손실 유한 | 30/30 (ds, seed) 모두 유한 | 예 |
| He 변경 범위 | `git diff --stat 4100884 <보고서 커밋>` = `tools/run_t29.py`, `tools/analyze_t29.py`, `reports/T29_*` | 예 (부기에 확인값 기록) |
| Hf 유한 | 에피소드 정확도 모두 유한 | 예 |
| 표 T1–T4, 판정 N1–N6, P147–P151 계산값 | `reports/T29_summary.md` | 예 |

## 4. 이상 징후

1. **실행 시간 차이:** phase2 프로세스(695.8–1308.2 s)는 단독 실행(252.2 s)보다 길었다. 같은 Clothing에서 5-shot 평가가 단독 118 s, phase2 677–763 s였다. 6개 프로세스가 CPU(LR, 특징 계산)를 함께 썼다. 프로세스당 `torch.set_num_threads(4)`.
2. **T3 정의 (승인된 판단 2):**
   - 마지막 라운드 손실·정확도는 마지막 `fit_linear` 호출 뒤 그 배치(5 클래스 × 20개)에서 잰 값이다.
   - "라운드 마지막 에폭 손실 최대"는 2,500회 `fit_linear` 각각의 100번째 에폭 손실 중 최댓값이다.
3. **시간 기록 (승인된 판단 3):** 프로세스 전체 시간은 드라이버가 잰 값이다. 사전학습·특징 시간은 프로세스 안에서 잰 값이다.
4. **에피소드 처리 (승인된 판단 4):** SGC-S·SGC-B는 에피소드마다 `fit_linear`를 순서대로 호출했다(벡터화 없음). SGC-S 초기화는 shot마다 생성기 하나에서 `ep_idx` 순으로 W → b를 뽑았다.
5. **D 계산 확인:** D와 SGC-P의 예측은 코드 안에서 코사인 argmax로 계산했다. 에피소드 정확도가 `fusion_baseline.cosine_probe_acc`와 같은지 assert로 확인했다(모든 프로세스 통과).
6. **T4 비교 조건:** 예비 실험은 분할(`random.seed(seed)` valid 추출), query 수(클래스당 20), 에피소드(`random.Random(1000 + seed)`), 난수 오프셋(사전학습 18000/18300, SGC-S 18400 + 1000·seed + e)이 이 실행과 다르다.
7. **복원 분할의 base 클래스 수:** Clothing 40, dblp 80, Electronics 91이고 seed 0–9에서 같다.
