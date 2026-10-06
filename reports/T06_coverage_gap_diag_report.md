# T06 보고 — coverage gap 진단: best-valid 임베딩 덤프 + 오프라인 분석 (2026-10-06, 브랜치 `novel_like_class`, auto mode)

## 1. 변경 파일과 위치
커밋: 이 보고서와 같은 커밋, 태그 `t06`. 선행 커밋 `b74467e`(지시서 위치 이동, 이상 징후 1).

| 파일 | 위치 | 줄 | 내용 |
|---|---|---|---|
| `argument.py` | `parse_args`, `config2string` 제외 목록 | +2 | `--dump_emb` (store_true, 기본 False) |
| `model.py` | 새 메서드 `dump_embedding` | +21 | `conv.eval()` + `torch.no_grad()`로 전체 그래프 GCN 출력(EGNN·LayerNorm 이전)을 계산해 `emb_best.npy`(float32)로 덮어쓰기, `self.emb_epoch` 기록. 처음 한 번 `split.json`(train/valid/test 클래스 목록), `labels.npy` 저장 |
| `model.py` | `train` 초기화 | +1 | `self.emb_epoch = None` |
| `model.py` | `train` 에폭 루프, 원본 `if acc_valid == best_acc_valid:` 블록 뒤 | +4 | `--dump_emb`·`--out_dir`이고 `acc_valid == best_acc_valid`이면 `dump_embedding(epoch)` |
| `model.py` | `run.json` 작성 | +1 | `--dump_emb`이면 `emb_epoch` |
| `tools/run_t06.py` (신규) | | | `run_t04.py` 복사. seed 0–2, `--dump_emb`(no `--mem`), 결과 폴더 `results/T06`, 완료 판정에 파일 4개 |
| `tools/analyze_t06.py` (신규) | | | 표 A, B1, B2, C, G1·G2 기록, P40–P41 → `reports/T06_summary.md` |

- **G2 명시**: 새 모델 코드는 원본 연산 줄을 바꾸지 않는다.
  - 덤프는 `torch.no_grad()` 안에서 GCN 출력을 읽어 파일로 쓰기만 한다.
  - 추가된 줄에 `random`·`np.random`·`torch` 난수 호출이 없다(`git diff`의 추가 줄 grep 결과 없음).
  - 덤프 시점은 그 에폭 test 평가 직후다. 모델은 test 평가로 이미 eval 상태이고, 다음 에폭 학습 시작 시 원본 코드가 train 모드로 되돌린다.

## 2. 실행
- `python -u tools/run_t06.py` — 2026-10-06 16:45–16:47 KST, 총 85.2 s. 18/18 성공, 재시도 0.
- 명령: `CUDA_VISIBLE_DEVICES=<UUID> python main.py --dataset <ds> --way 5 --shot <k> --seed <s> --num_seed 1 --device 0 --dump_emb --out_dir results/T06/<ds>_5w<k>s/seed<s>`
- 분석: `python tools/analyze_t06.py` (CPU, 약 77 s).

## 3. 게이트
| 게이트 | 조건 | 관측 | 결과 |
|---|---|---|---|
| G1 | 18 run 모두 `emb_best.npy`·`split.json`·`labels.npy`·`run.json` 존재, `emb_epoch` = `best_epoch_valid` | 18/18 존재, 18/18 일치 | 통과 |
| G2 | 덤프 코드가 `no_grad`·읽기 전용·난수 호출 없음 명시. T04 같은 seed `test_acc_at_best_valid` 병기(일치 요구 없음) | 1절에 명시. T04와 같은 값 16/18, 다른 값 2개(dblp 1-shot s2 0.7792 vs 0.7808, dblp 5-shot s0 0.8496 vs 0.8504) | 통과 |
| G3 | 표 A, B1, B2, C 생성 | 생성 | 통과 |

재시도 0회.

## 4. 표 (`reports/T06_summary.md`에서 복사)
- run 수: **18 / 18** (3 데이터셋 × 5-way {1,5}-shot × seed 0–2). 값 = (데이터셋, shot)별 seed 3개 평균 ± sd.

### 표 A — 일반화 격차 (off `best_valid_epoch`의 정확도, %)
| 데이터셋 | shot | best_valid_epoch (seed순) | train | valid | test | train − test | valid − test |
|---|---|---|---|---|---|---|---|
| Amazon_clothing | 1 | 7, 3, 2 | 82.32 ± 5.80 | 80.03 ± 1.96 | 79.76 ± 1.32 | +2.56 ± 4.72 | +0.27 ± 1.25 |
| Amazon_clothing | 5 | 7, 1, 10 | 90.29 ± 7.27 | 88.91 ± 0.86 | 90.27 ± 0.39 | +0.03 ± 7.66 | -1.36 ± 1.26 |
| dblp | 1 | 7, 7, 10 | 80.16 ± 5.75 | 77.81 ± 4.79 | 75.55 ± 2.29 | +4.61 ± 3.80 | +2.27 ± 6.83 |
| dblp | 5 | 10, 9, 9 | 88.61 ± 0.17 | 86.21 ± 3.61 | 83.87 ± 1.21 | +4.75 ± 1.24 | +2.35 ± 2.60 |
| Amazon_electronics | 1 | 9, 5, 6 | 81.17 ± 1.94 | 83.04 ± 1.69 | 73.09 ± 1.38 | +8.08 ± 3.26 | +9.95 ± 2.29 |
| Amazon_electronics | 5 | 4, 9, 8 | 93.20 ± 1.13 | 92.27 ± 2.06 | 86.96 ± 1.44 | +6.24 ± 1.44 | +5.31 ± 0.97 |

### 표 B1 — 공간별 few-shot 분리도 (cosine 최근접 프로토타입, 5-way, K = shot, query 5, 에피소드 500개, 정확도 %)
| 데이터셋 | shot | raw·base | raw·valid | raw·test | diff·base | diff·valid | diff·test | emb·base | emb·valid | emb·test |
|---|---|---|---|---|---|---|---|---|---|---|
| Amazon_clothing | 1 | 71.29 ± 1.18 | 69.65 ± 2.07 | 72.57 ± 0.37 | 79.69 ± 1.57 | 79.18 ± 2.73 | 85.24 ± 0.21 | 85.40 ± 3.08 | 73.11 ± 2.42 | 76.99 ± 1.73 |
| Amazon_clothing | 5 | 84.73 ± 0.96 | 84.95 ± 0.97 | 86.55 ± 0.09 | 89.15 ± 0.94 | 89.58 ± 0.94 | 92.54 ± 0.12 | 92.75 ± 3.00 | 84.12 ± 0.77 | 87.85 ± 0.79 |
| dblp | 1 | 48.13 ± 0.74 | 46.81 ± 3.43 | 48.46 ± 1.29 | 73.81 ± 1.67 | 71.40 ± 3.13 | 72.50 ± 1.59 | 80.16 ± 2.12 | 71.37 ± 4.75 | 72.76 ± 1.68 |
| dblp | 5 | 67.08 ± 0.72 | 66.17 ± 3.43 | 67.75 ± 0.22 | 85.62 ± 0.13 | 84.21 ± 2.84 | 85.27 ± 0.77 | 88.61 ± 0.66 | 82.67 ± 3.88 | 83.00 ± 0.61 |
| Amazon_electronics | 1 | 77.94 ± 1.32 | 78.03 ± 3.13 | 73.35 ± 0.24 | 80.54 ± 1.00 | 82.14 ± 2.97 | 74.97 ± 0.37 | 86.66 ± 0.81 | 77.18 ± 2.48 | 70.45 ± 1.26 |
| Amazon_electronics | 5 | 90.35 ± 0.46 | 90.90 ± 2.59 | 87.01 ± 0.42 | 90.91 ± 0.23 | 91.44 ± 2.57 | 87.44 ± 0.03 | 93.97 ± 0.28 | 87.49 ± 2.71 | 82.58 ± 0.34 |

### 표 B2 — Δ_G = acc(`emb`, G) − acc(`diff`, G) (%p)
| 데이터셋 | shot | Δ_base | Δ_valid | Δ_test | Δ_base − Δ_test |
|---|---|---|---|---|---|
| Amazon_clothing | 1 | +5.71 ± 1.98 | -6.07 ± 3.22 | -8.25 ± 1.53 | +13.96 ± 1.21 |
| Amazon_clothing | 5 | +3.60 ± 3.10 | -5.46 ± 1.08 | -4.70 ± 0.81 | +8.30 ± 3.81 |
| dblp | 1 | +6.35 ± 0.49 | -0.03 ± 1.70 | +0.26 ± 0.72 | +6.09 ± 1.14 |
| dblp | 5 | +2.99 ± 0.58 | -1.55 ± 1.36 | -2.28 ± 0.21 | +5.27 ± 0.78 |
| Amazon_electronics | 1 | +6.12 ± 1.74 | -4.95 ± 2.11 | -4.51 ± 1.27 | +10.63 ± 0.46 |
| Amazon_electronics | 5 | +3.06 ± 0.10 | -3.96 ± 1.46 | -4.86 ± 0.32 | +7.92 ± 0.42 |

### 표 C — 구분 축 coverage (`emb`, base 프로토타입 차이 PCA 90% 부분공간 P, 잔차 비율 ‖(I−P)d‖²/‖d‖²)
| 데이터셋 | shot | r | 잔차(test 쌍) | 잔차(valid 쌍) | 잔차(무작위 1000) | (base/valid/test 클래스 수) |
|---|---|---|---|---|---|---|
| Amazon_clothing | 1 | 15.3 ± 0.6 | 0.2650 ± 0.0297 | 0.2804 ± 0.0555 | 0.7599 ± 0.0084 | 40/17/20 |
| Amazon_clothing | 5 | 14.7 ± 0.6 | 0.2067 ± 0.0692 | 0.2126 ± 0.0433 | 0.7731 ± 0.0091 | 40/17/20 |
| dblp | 1 | 9.3 ± 0.6 | 0.1131 ± 0.0096 | 0.1267 ± 0.0159 | 0.8558 ± 0.0106 | 80/27/30 |
| dblp | 5 | 9.7 ± 0.6 | 0.0952 ± 0.0096 | 0.1013 ± 0.0138 | 0.8494 ± 0.0101 | 80/27/30 |
| Amazon_electronics | 1 | 19.7 ± 0.6 | 0.1812 ± 0.0067 | 0.1775 ± 0.0302 | 0.6930 ± 0.0060 | 91/36/40 |
| Amazon_electronics | 5 | 18.7 ± 1.2 | 0.1389 ± 0.0250 | 0.1436 ± 0.0224 | 0.7093 ± 0.0158 | 91/36/40 |

### 참고 — G1·G2 기록
| 데이터셋 | shot | seed | `emb_epoch` | `best_epoch_valid` | 일치 | T06 `test_acc_at_best_valid` | T04 같은 seed |
|---|---|---|---|---|---|---|---|
| Amazon_clothing | 1 | 0 | 7 | 7 | 예 | 0.8048 | 0.8048 |
| Amazon_clothing | 1 | 1 | 3 | 3 | 예 | 0.8056 | 0.8056 |
| Amazon_clothing | 1 | 2 | 2 | 2 | 예 | 0.7824 | 0.7824 |
| Amazon_clothing | 5 | 0 | 7 | 7 | 예 | 0.9008 | 0.9008 |
| Amazon_clothing | 5 | 1 | 1 | 1 | 예 | 0.9072 | 0.9072 |
| Amazon_clothing | 5 | 2 | 10 | 10 | 예 | 0.9000 | 0.9000 |
| dblp | 1 | 0 | 7 | 7 | 예 | 0.7336 | 0.7336 |
| dblp | 1 | 1 | 7 | 7 | 예 | 0.7536 | 0.7536 |
| dblp | 1 | 2 | 10 | 10 | 예 | 0.7792 | 0.7808 |
| dblp | 5 | 0 | 10 | 10 | 예 | 0.8496 | 0.8504 |
| dblp | 5 | 1 | 9 | 9 | 예 | 0.8408 | 0.8408 |
| dblp | 5 | 2 | 9 | 9 | 예 | 0.8256 | 0.8256 |
| Amazon_electronics | 1 | 0 | 9 | 9 | 예 | 0.7264 | 0.7264 |
| Amazon_electronics | 1 | 1 | 5 | 5 | 예 | 0.7464 | 0.7464 |
| Amazon_electronics | 1 | 2 | 6 | 6 | 예 | 0.7200 | 0.7200 |
| Amazon_electronics | 5 | 0 | 4 | 4 | 예 | 0.8736 | 0.8736 |
| Amazon_electronics | 5 | 1 | 9 | 9 | 예 | 0.8536 | 0.8536 |
| Amazon_electronics | 5 | 2 | 8 | 8 | 예 | 0.8816 | 0.8816 |

## 5. 사전 등록 대비 (P40–P41 원문 수정 없음)
| 예측 | 내용 | 관측 | 일치·어긋남 |
|---|---|---|---|
| P40 | 표 B2 (Δ_base − Δ_test) > 0 인 (데이터셋, shot) ≥ 4/6 | 6/6 (+13.96, +8.30, +6.09, +5.27, +10.63, +7.92) | 일치 |
| P41 | 표 B1 acc(`emb`, test) < acc(`diff`, test) 인 (데이터셋, shot) ≥ 2/6 | 5/6 | 일치 |

## 6. 이상 징후
1. T06 지시서는 브랜치 커밋 `0d7fb9e`에서 `reports/T06_coverage_gap_diag.md` 경로로 들어 있었다. 이 경로는 이 보고서 경로와 같다. `git mv`로 `instructions/T06_coverage_gap_diag.md`로 옮겨 커밋했다(`b74467e`, 내용 변경 없음).
2. `diff` 공간의 Â는 PyG `gcn_norm(edge_index, add_self_loops=True)`로 만들었다. GCNConv가 쓰는 것과 같은 함수·같은 간선이고, `load_data`의 `adj`는 쓰지 않았다. 세 데이터셋의 간선은 대칭(대칭 비율 1.000)이다.
3. `raw` 공간은 `utils.load_data`가 반환한 `X`를 그대로 썼다. `load_data`는 Amazon·dblp에서 valid 분할에 전역 `random`을 호출하지만, 분석은 그 분할을 쓰지 않고 run의 `split.json`을 쓴다.
4. 표 B1의 에피소드는 G마다 한 번 뽑아 공간 3종에 공유했다(지시서에 공유 여부 미기재 — 가장 단순한 쪽). 클래스당 노드 수 최소 101로, K + 5개 추출에서 제외된 클래스는 없다.
5. 표 C의 PCA는 base 클래스 **순서쌍**(i ≠ j) 차이 벡터의 SVD로 계산했다. 순서쌍 집합은 평균이 0이라 중심화한 PCA와 같다. r은 누적 분산 비율이 0.9 이상이 되는 최소 개수다. test·valid 잔차는 비순서쌍(i < j)으로 계산했다.
6. cosine 계산의 노름 하한은 1e-12다. `raw`·`diff`·`emb`(확인한 run) 모두 노름이 0인 행은 0개였다.
7. 표 A·B2의 sd 표기 형식(부호 제거)을 한 번 고쳐 다시 생성했다. 재실행 값은 이전 실행과 같다.
8. T04 대비 `test_acc_at_best_valid`가 다른 run은 2개다(G2 행). 이번 run은 `--mem` 없이 실행했다.
9. 표 B2의 Δ_test가 양수인 셀: dblp 1-shot (+0.26 ± 0.72). 나머지 5셀은 음수.
10. 표 C 무작위 기준선 잔차(0.69–0.86)는 (64 − r)/64와 근접하다(예: Amazon_clothing 1-shot r = 15.3 → 0.761).

## 부기 (2026-10-06)
- 이 보고서의 파일명을 `reports/T06_coverage_gap_diag.md`에서 `reports/T06_coverage_gap_diag_report.md`로 바꿨다(사람 지시).
- 지시서 `instructions/T06_coverage_gap_diag.md`와 파일명이 같아 혼동을 피하기 위해서다.
- 지시서 §6의 보고 경로 문구는 수정하지 않았다. 본문 내용 변경 없음.
