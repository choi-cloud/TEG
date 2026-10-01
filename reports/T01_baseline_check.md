# T01 보고 — 원본 TEG 기준 실행 (2026-10-01)

## 1. 변경 파일과 위치
- 코드 변경 없음. 2단계에서 수정 전 상태로 이미 물리 GPU 2번에서 실행되어 3단계 수정(`embedder.py:21` 삭제)을 하지 않았다.
- `b2aa4a8` docs: add CLAUDE.md, instructions T01-T04, methodology v0 (`.gitignore`에 `results/` 추가 포함)
- 이 보고서 커밋에 태그 `t01`.
- 로그: `results/T01/` (gitignore). `*.log`(tee 원본), `*.log.ts`(줄별 unix 시각), `*.log.time`(시작·종료 시각).

## 2. 실행 명령과 출력

### 표 1 — 데이터 파일
| 파일 | 크기 (bytes) |
|---|---|
| dataset/Amazon_clothing/Amazon_clothing_network | 2,036,322 |
| dataset/Amazon_clothing/Amazon_clothing_train.mat | 6,155,520 |
| dataset/Amazon_clothing/Amazon_clothing_test.mat | 3,808,608 |
| dataset/Amazon_electronics/Amazon_electronics_network | 999,514 |
| dataset/Amazon_electronics/Amazon_electronics_train.mat | 16,218,608 |
| dataset/Amazon_electronics/Amazon_electronics_test.mat | 4,971,904 |
| dataset/dblp/dblp_network | 6,635,708 |
| dataset/dblp/dblp_train.mat | 17,711,600 |
| dataset/dblp/dblp_test.mat | 5,728,224 |

### 표 2 — GPU 지정
명령: `CUDA_VISIBLE_DEVICES=2 python main.py --dataset Amazon_clothing --way 5 --shot 1 --epochs 1 --seed 0 --device 0`
확인: `nvidia-smi -i <i> --query-compute-apps=pid,gpu_bus_id,gpu_uuid` (i = 0,1,2,4,5,6)로 PID가 올라간 GPU 조회.

| 상태 | 검출된 nvidia-smi index | bus id | uuid | 종료 |
|---|---|---|---|---|
| 수정 전 | 2 | 00000000:50:00.0 | GPU-2b6846d8-a8a0-c83b-dbf5-8bb183352d29 | exit 0 |
| 수정 후 | — (수정 안 함) | — | — | — |

물리 GPU 2번: nvidia-smi index 2, bus id 00000000:50:00.0. 일치.

참고 — `CUDA_VISIBLE_DEVICES` 값별로 torch가 보는 장치 (`torch.cuda.get_device_properties(0).uuid`):

| CVD | uuid | nvidia-smi index (bus id) |
|---|---|---|
| 0 | 4b1a9f61… | 0 (17:00.0) |
| 1 | 510575ba… | 1 (3D:00.0) |
| 2 | 2b6846d8… | 2 (50:00.0) |
| 3 | e2a9ef79… | **4** (99:00.0) |
| 4 | b428f48d… | **5** (BD:00.0) |
| 5 | ec7b3ebe… | **6** (E1:00.0) |
| 6 | 장치 없음 | — |
| GPU-e2a9ef79-… (uuid 지정) | e2a9ef79… | 4 (99:00.0) |

### 표 3 — 기준 실행 (5-way 1-shot, seed 0, epochs 10)
명령: `CUDA_VISIBLE_DEVICES={0,1,2} python -u main.py --dataset <ds> --way 5 --shot 1 --seed 0 --num_seed 1 --device 0 2>&1 | tee results/T01/<ds>_5w1s_seed0.log` (3개 동시 실행)

| epoch | clothing train | valid | test | electronics train | valid | test | dblp train | valid | test |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6376 | 0.5520 | 0.6248 | 0.5936 | 0.6912 | 0.5560 | 0.5584 | 0.5208 | 0.5424 |
| 1 | 0.6312 | 0.7080 | 0.7960 | 0.5040 | 0.7776 | 0.6448 | 0.5424 | 0.6984 | 0.6648 |
| 2 | 0.7904 | 0.7160 | 0.7792 | 0.6960 | 0.8224 | 0.6920 | 0.6592 | 0.6632 | 0.6968 |
| 3 | 0.8664 | 0.7536 | 0.8336 | 0.7264 | 0.8088 | 0.7000 | 0.7368 | 0.7152 | 0.7168 |
| 4 | 0.8272 | 0.7440 | 0.8128 | 0.7888 | 0.8256 | 0.7432 | 0.7792 | 0.7440 | 0.7320 |
| 5 | 0.8728 | 0.7256 | 0.8120 | 0.8208 | 0.8104 | 0.7352 | 0.8048 | 0.7600 | 0.6904 |
| 6 | 0.8456 | 0.7352 | 0.8360 | 0.7880 | 0.8256 | 0.7400 | 0.7784 | 0.7864 | 0.7640 |
| 7 | 0.8808 | 0.7960 | 0.8048 | 0.8112 | 0.8240 | 0.7280 | 0.7664 | 0.8008 | 0.7336 |
| 8 | 0.8712 | 0.7688 | 0.8184 | 0.7800 | 0.8224 | 0.7368 | 0.7888 | 0.7496 | 0.7576 |
| 9 | 0.8696 | 0.7392 | 0.8272 | 0.8256 | 0.8496 | 0.7264 | 0.8136 | 0.7552 | 0.7600 |
| 10 | 0.8816 | 0.7728 | 0.8160 | 0.8232 | 0.8056 | 0.7784 | 0.8080 | 0.7784 | 0.7528 |

| 데이터셋 | GPU | Acc_Test_At_Best_Valid (F1) | 에폭 | 전체 (s) | 구조 특징 "Generating"~"Done." (s) | "Done."~첫 tqdm 출력 (s) |
|---|---|---|---|---|---|---|
| Amazon_clothing | 0 | 0.8048 (0.7990) | 7 | 23.87 | 0.013 | 1.344 |
| Amazon_electronics | 1 | 0.7264 (0.7153) | 9 | 26.10 | 0.020 | 1.312 |
| dblp | 2 | 0.7336 (0.7198) | 7 | 25.57 | 0.021 | 2.745 |

### 표 4 — 결정성 (Amazon_clothing 5w1s seed 0, 물리 GPU 0)
| 비교 | 결과 |
|---|---|
| 에폭 0~10 `acc_train`/`acc_valid`/`acc_test` 33개 값 | 전부 같음 |
| `Acc_Test_At_Best_Valid`, F1, 에폭 | 같음 (0.8048, 0.7990, epoch 7) |
| 재실행 소요 시간 | 전체 21.88 s, 구조 특징 0.014 s, "Done."~첫 tqdm 1.204 s |

## 3. 완료 조건 대비
| 조건 | 결과 | 충족 |
|---|---|---|
| 데이터 9개 파일 존재 | 9/9 존재 | 충족 |
| `CUDA_VISIBLE_DEVICES=2 ... --device 0`이 물리 GPU 2번에서 실행 | bus id 00000000:50:00.0 (수정 전) | 충족 |
| 세 데이터셋 run 오류 없이 종료, `Acc_Test_At_Best_Valid` 출력 | 3/3 exit 0, 출력 있음 | 충족 |
| 결정성 비교 결과와 run별 소요 시간 보고 | 같음, 표 3·4 | 충족 |

## 4. 이상 징후
1. `nvidia-smi`의 전체 조회(`--query-gpu`, `--query-compute-apps`)가 GPU 3번(0000:63:00.0) "Unknown Error"로 실패한다. `-i <index>`로 GPU별 조회 시 동작한다.
2. CUDA 런타임은 장치를 6개만 열거한다(GPU 3번 제외). `CUDA_VISIBLE_DEVICES=3/4/5`는 nvidia-smi index 4/5/6에 대응하고, `=6`은 장치 없음(표 2 참고). CLAUDE.md §4의 "`CUDA_VISIBLE_DEVICES=<물리번호>`" 형식은 0·1·2에서만 nvidia-smi 번호와 일치한다. uuid 지정(`CUDA_VISIBLE_DEVICES=GPU-…`)은 해당 GPU에 올라간다.
3. `embedder.py:21`의 `os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)`는 그대로 있으나 외부 지정(2)이 유지되었다.
4. 구조 특징 구간: `embedder.py`는 앵커 엣지 생성 후 "Done."을 출력하고, 최단거리 계산(`embedder.py:62-80`)은 그 뒤에 수행한다. 지시서 구간("Generating"~"Done.")은 이 계산을 포함하지 않는다. "Done."~첫 tqdm 출력 구간을 별도 열로 기록했다.
5. 줄별 시각 기록을 위해 `python` 대신 `python -u`로 실행하고, tee 뒤에 타임스탬프 파이프(`ts.py`, 리포 밖 scratchpad)를 붙였다. 학습 코드·인자는 지시서와 같다.
6. 2단계 첫 시도에서 nvidia-smi 전체 조회 실패(1번)로 GPU 검출을 못 했고, 같은 명령을 다시 실행하면서 `results/T01/gpu_check_before.log`를 덮어썼다. 첫 시도 로그는 남아 있지 않다. 두 시도 모두 exit 0.
7. "Virtual Anchor Node [k] samples less than 1" 경고: clothing k=15,16, electronics·dblp k=16. 
8. 각 run의 정확도 행은 에폭 0~10의 11개다. 에폭 0은 `model.py`에서 학습 없이 eval 모드로 계산된다.
9. 2단계 run(epochs 1)의 에폭 0·1 값(train 0.6376 / valid 0.7080 / test 0.7960 at epoch 1)이 4단계 Amazon_clothing 에폭 0·1 값과 같다.
